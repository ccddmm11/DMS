#!/usr/bin/env python3
"""Standalone validation for src/memory/mutation.py + the full DMS main loop
in src/androidworld_integration/dms_agent_adapter.py.

Does NOT require the emulator or vLLM: the environment and the Planner/
Actor/Verifier LLM roles are all replaced with deterministic stubs so we
can exercise Algorithm 1's full control flow (Retrieve -> risk gate ->
epsilon-Mutation -> Replay/Actor-generation -> Verify -> memory
creation/evolutionary-replacement/strike-pruning) end to end.

Usage:
  python scripts/test_dms_mutation_loop.py
"""

import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from android_world.env import representation_utils

from src.agent import app_intent
from src.agent import loop_guard
from src.agent.actor import ActorStepOutput
from src.agent.planner import Planner
from src.agent.planner import PlannerOutput
from src.agent.planner import SubPlan
from src.agent.planner import coarsen_sub_plans
from src.agent.verifier import VerifierOutput
from src.androidworld_integration.dms_agent_adapter import DMSAgent
from src.baselines.static_memory_agent import StaticMemoryAgent
from src.memory import mutation
from src.memory.memory_bank import MemoryBank
from src.memory.memory_unit import MemoryUnit
from src.memory.memory_unit import TrajectoryStep
from src.memory.risk import BayesianRiskConfig
from src.memory.risk import RiskRegulator

TEST_STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "_tmp_dms_mutation_loop_test",
)


def make_memory(precondition, goal, n_actions, source_task="TestTask"):
  trajectory = [
      TrajectoryStep(
          reason=f"step {i}",
          action={"action_type": "click", "index": i},
          target_element_desc=f"element-{i}",
      )
      for i in range(n_actions)
  ]
  return MemoryUnit(
      precondition=precondition, goal=goal, trajectory=trajectory,
      success=True, description=f"Created from {source_task}",
      source_task=source_task,
  )


def ui_element(text, is_visible=True):
  return representation_utils.UIElement(text=text, is_visible=is_visible)


def fresh_dir(path):
  """Removes `path` if it already exists (e.g. left over from a previous
  failed run that never reached its own cleanup), then returns it."""
  if os.path.exists(path):
    shutil.rmtree(path)
  return path


def test_task_app_guardrails():
  print("[0/7] task app context: declared app launch + Planner prompt...")

  class FakeLauncherEnv:
    foreground_activity_name = "com.android.launcher3/.Launcher"

  decision = app_intent.initial_task_app_fast_path_decision(
      FakeLauncherEnv(), ["camera"], has_started_task_navigation=False
  )
  assert decision == ("open", "camera")
  prompt = Planner(llm=None)._build_prompt(
      "Take one photo.", [], "launcher UI", ["camera"]
  )
  assert "Declared target app(s) for this task: camera" in prompt
  assert "Photos is not a substitute for Camera" in prompt
  assert "Memory-aware granularity" in prompt
  coarse = coarsen_sub_plans(
      [SubPlan("Contacts home", "click the '+' button")],
      "Create and save Maya Wang's contact with the requested phone number.",
  )
  assert len(coarse) == 1 and coarse[0].goal.startswith("Create and save")
  navigation_then_coarse = coarsen_sub_plans(
      [
          SubPlan("Launcher", "open the Contacts app"),
          SubPlan("Contacts home", "click the '+' button"),
      ],
      "Create and save Maya Wang's contact with the requested phone number.",
  )
  assert [p.goal for p in navigation_then_coarse] == [
      "open the Contacts app",
      "Create and save Maya Wang's contact with the requested phone number.",
  ]
  print("    OK: target app metadata forces Camera launch context, not Photos.")


# ---------------------------------------------------------------------------
# Part 1: mutation.py pure-logic unit checks.
# ---------------------------------------------------------------------------
def test_decide_retrieval_and_reuse():
  print("[1/6] decide_retrieval_and_reuse(): risk-gating + epsilon-Mutation...")
  bank = MemoryBank(fresh_dir(TEST_STORE_DIR + "_p1"))
  risk = RiskRegulator(bank, BayesianRiskConfig())
  config = mutation.MutationConfig(epsilon=0.2, retrieval_score_threshold=0.3)

  # (a) No memory at all -> no candidate, never reuse.
  decision = mutation.decide_retrieval_and_reuse(
      bank, risk, "Home screen is open.", "Turn on wifi.", config,
      random.Random(0),
  )
  assert decision.candidate is None and not decision.do_reuse
  print("    OK: empty bank -> no candidate.")

  safe_memory = bank.add(make_memory(
      "Settings home screen is open.", "Turn wifi on by tapping the toggle.", 2
  ))

  # (b) Safe memory + high roll (> epsilon) -> DoReuse.
  decision = mutation.decide_retrieval_and_reuse(
      bank, risk, "Settings home screen is open.", "Turn wifi on.", config,
      random.Random(),  # will roll something; force determinism below instead
  )
  # Force the exact roll deterministically via a scripted RNG.
  class ScriptedRandom:
    def __init__(self, values):
      self.values = list(values)
    def random(self):
      return self.values.pop(0)

  decision = mutation.decide_retrieval_and_reuse(
      bank, risk, "Settings home screen is open.", "Turn wifi on.", config,
      ScriptedRandom([0.9]),
  )
  assert decision.candidate is not None and decision.candidate.memory_id == safe_memory.memory_id
  assert decision.do_reuse and not decision.is_mutation and not decision.is_risk_suppressed
  print("    OK: safe memory + roll(0.9) > epsilon(0.2) -> DoReuse=True.")

  # (c) Safe memory + low roll (<= epsilon) -> Mutation (explore instead).
  decision = mutation.decide_retrieval_and_reuse(
      bank, risk, "Settings home screen is open.", "Turn wifi on.", config,
      ScriptedRandom([0.05]),
  )
  assert decision.candidate is not None and not decision.do_reuse and decision.is_mutation
  print("    OK: safe memory + roll(0.05) <= epsilon(0.2) -> Mutation (DoReuse=False).")

  # (d) Risky memory (risk_score above dynamic threshold) -> suppressed.
  threshold = risk.get_dynamic_threshold()
  bank.update_risk_stats(
      safe_memory.memory_id, failure_count=10, success_count=0,
      risk_mu=threshold + 0.3, risk_score=threshold + 0.3,
  )
  decision = mutation.decide_retrieval_and_reuse(
      bank, risk, "Settings home screen is open.", "Turn wifi on.", config,
      ScriptedRandom([0.9]),  # even a high roll shouldn't matter now.
  )
  assert decision.candidate is not None and not decision.do_reuse
  assert decision.is_risk_suppressed and not decision.is_mutation
  print("    OK: risk_score > tau_risk -> is_risk_suppressed=True regardless of epsilon roll.")

  shutil.rmtree(TEST_STORE_DIR + "_p1")


def test_replay_trajectory():
  print("[2/6] replay_trajectory(): re-grounding drifted indexes + abort...")

  class FakeEnv:
    def __init__(self, ui_elements_sequence):
      self.logical_screen_size = (1080, 2400)
      self._sequence = list(ui_elements_sequence)
      self._pixels = np.zeros((4, 4, 3), dtype=np.uint8)
      self.executed_actions = []

    def get_state(self, wait_to_stabilize=False):
      elements = self._sequence.pop(0) if len(self._sequence) > 1 else self._sequence[0]
      class _State:
        pass
      s = _State()
      s.ui_elements = elements
      s.pixels = self._pixels
      return s

    def execute_action(self, action):
      self.executed_actions.append(action)

    def reset(self, go_home=False):
      del go_home

  # Original recording: click on "Save" at index 0. On replay, "Save" has
  # drifted to index 2 (two unrelated elements now precede it).
  trajectory = [
      TrajectoryStep(reason="click save", action={"action_type": "click", "index": 0},
                     target_element_desc="Save"),
  ]
  drifted_elements = [ui_element("Cancel"), ui_element("Discard"), ui_element("Save")]
  env = FakeEnv([drifted_elements, drifted_elements])
  result = mutation.replay_trajectory(trajectory, env, wait_after_action_seconds=0.0)
  assert result.success_execution
  assert env.executed_actions[0].index == 2, (
      f"Expected re-grounded index 2, got {env.executed_actions[0].index}"
  )
  print("    OK: index-drifted 'Save' element correctly re-grounded (0 -> 2).")

  # Abort case: target description matches nothing AND original index is
  # out of range for the current (much shorter) UI element list.
  trajectory_bad = [
      TrajectoryStep(reason="click missing", action={"action_type": "click", "index": 5},
                      target_element_desc="Nonexistent Button"),
  ]
  env2 = FakeEnv([[ui_element("OK")]])
  result2 = mutation.replay_trajectory(trajectory_bad, env2, wait_after_action_seconds=0.0)
  assert not result2.success_execution
  assert not env2.executed_actions
  print("    OK: un-groundable action correctly aborts replay (success_execution=False).")


def test_decide_memory_update():
  print("[3/6] decide_memory_update(): create / evolutionary-replace / filter...")
  bank = MemoryBank(fresh_dir(TEST_STORE_DIR + "_p3"))

  # (a) |tau|=1 -> filtered, no bank mutation.
  outcome = mutation.decide_memory_update(
      bank, "p", "g", [TrajectoryStep(reason="r", action={"action_type": "click", "index": 0})],
      candidate=None,
  )
  assert outcome.action == "skipped_too_short" and len(bank) == 0
  print("    OK: |tau|=1 filtered per Sec 3.2.1.")

  # (b) No candidate, |tau|>=2 -> creates a new memory.
  fresh_tau = [
      TrajectoryStep(reason="r0", action={"action_type": "click", "index": 0},
                     target_element_desc="first button"),
      TrajectoryStep(reason="r1", action={"action_type": "click", "index": 1},
                     target_element_desc="second button"),
  ]
  outcome = mutation.decide_memory_update(bank, "p", "g", fresh_tau, candidate=None)
  assert outcome.action == "created" and len(bank) == 1
  print("    OK: no candidate -> CreateMemory (Algorithm 1 line 20).")

  # (c) Candidate exists, new trajectory SHORTER -> in-place evolutionary
  # replacement (same memory_id, trajectory overwritten).
  candidate = bank.add(make_memory("p2", "g2", 4))
  shorter_tau = [
      TrajectoryStep(reason="r0", action={"action_type": "click", "index": 0},
                     target_element_desc="first button"),
      TrajectoryStep(reason="r1", action={"action_type": "click", "index": 1},
                     target_element_desc="second button"),
  ]
  outcome = mutation.decide_memory_update(bank, "p2", "g2", shorter_tau, candidate=candidate)
  assert outcome.action == "replaced" and outcome.memory_id == candidate.memory_id
  reloaded = bank.get(candidate.memory_id, load_trajectory=True)
  assert len(reloaded.trajectory) == 2
  print("    OK: shorter mutation trajectory triggers in-place evolutionary replacement.")

  # (d) Candidate exists, new trajectory NOT shorter -> a separate new
  # memory is created; the candidate is left untouched.
  candidate2 = bank.add(make_memory("p3", "g3", 2))
  not_shorter_tau = [
      TrajectoryStep(reason="r0", action={"action_type": "click", "index": 0},
                     target_element_desc="first button"),
      TrajectoryStep(reason="r1", action={"action_type": "click", "index": 1},
                     target_element_desc="second button"),
      TrajectoryStep(reason="r2", action={"action_type": "click", "index": 2},
                     target_element_desc="third button"),
  ]
  size_before = len(bank)
  outcome = mutation.decide_memory_update(bank, "p3", "g3", not_shorter_tau, candidate=candidate2)
  assert outcome.action == "created" and len(bank) == size_before + 1
  untouched = bank.get(candidate2.memory_id, load_trajectory=True)
  assert len(untouched.trajectory) == 2
  print("    OK: non-improving mutation creates a new memory, leaves candidate untouched.")

  # (e) A repeated unanchored exploration loop is not a reusable memory.
  repeated_scrolls = [
      TrajectoryStep(
          reason="searching",
          action={"action_type": "scroll", "direction": "up"},
      )
      for _ in range(4)
  ]
  outcome = mutation.decide_memory_update(
      bank, "p", "find settings", repeated_scrolls, candidate=None
  )
  assert outcome.action == "skipped_repeated_navigation"
  print("    OK: repeated unanchored scroll loop is rejected before persistence.")

  # (f) Index actions without a semantic target cannot be safely re-grounded.
  ungroundable_tau = [
      TrajectoryStep(reason="r0", action={"action_type": "click", "index": 0}),
      TrajectoryStep(reason="r1", action={"action_type": "click", "index": 1}),
  ]
  outcome = mutation.decide_memory_update(
      bank, "p", "unsafe clicks", ungroundable_tau, candidate=None
  )
  assert outcome.action == "skipped_unreplayable_index"
  print("    OK: index trajectory without target descriptors is rejected.")

  # (g) Repeating an action without any recorded UI change is an Actor loop,
  # not a reusable trajectory. This also protects against slow replay loops.
  stagnant_actions = [
      TrajectoryStep(
          reason="press Enter again",
          action={"action_type": "keyboard_enter"},
          state_signature="unchanged-ui",
      )
      for _ in range(3)
  ]
  outcome = mutation.decide_memory_update(
      bank, "p", "confirm", stagnant_actions, candidate=None
  )
  assert outcome.action == "skipped_stagnant_actions"
  assert mutation._has_repeated_stagnant_action(stagnant_actions)
  breaker = loop_guard.StagnantActionBreaker(max_repeats=2)
  action = {"action_type": "click", "index": 1}
  assert not breaker.record_and_check("same-ui", action)
  assert breaker.record_and_check("same-ui", action)
  assert not breaker.record_and_check("changed-ui", action)
  print("    OK: unchanged-state repeated actions are rejected before replay.")

  shutil.rmtree(TEST_STORE_DIR + "_p3")


# ---------------------------------------------------------------------------
# Part 2: full DMSAgent.step() state-machine wiring, via stubs.
# ---------------------------------------------------------------------------
class StubPlanner:
  def __init__(self, outputs):
    self.outputs = list(outputs)
    self.calls = 0

  def plan(self, goal, history, ui_elements_str, screenshots, task_apps=None):
    del task_apps
    self.calls += 1
    return self.outputs.pop(0)


class StubActor:
  def __init__(self, outputs):
    self.outputs = list(outputs)
    self.calls = 0

  def act(self, sub_task_str, history, ui_elements_str, screenshots):
    self.calls += 1
    return self.outputs.pop(0)


class StubVerifier:
  def __init__(self, outputs):
    self.outputs = list(outputs)
    self.calls = 0

  def verify(self, goal, history, screenshot):
    self.calls += 1
    return self.outputs.pop(0)


class FakeAndroidEnv:
  """Minimal stand-in for `android_world.env.interface.AsyncEnv`."""

  def __init__(self, ui_elements):
    self.logical_screen_size = (1080, 2400)
    self.orientation = 0
    self.physical_frame_boundary = (0, 0, 1080, 2400)
    self._ui_elements = ui_elements
    self._pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    self.executed_actions = []

  def reset(self, go_home=False):
    pass

  def hide_automation_ui(self):
    pass

  def get_state(self, wait_to_stabilize=False):
    class _State:
      pass
    s = _State()
    s.ui_elements = self._ui_elements
    s.pixels = self._pixels
    return s

  def execute_action(self, action):
    self.executed_actions.append(action)


class ScriptedRandom:
  def __init__(self, values):
    self.values = list(values)

  def random(self):
    return self.values.pop(0)


def make_agent(store_dir, ui_elements, epsilon=0.15):
  env = FakeAndroidEnv(ui_elements)
  agent = DMSAgent(
      env, llm=None, memory_store_dir=store_dir,
      mutation_config=mutation.MutationConfig(epsilon=epsilon, retrieval_score_threshold=0.3),
      wait_after_action_seconds=0.0,
  )
  agent.transition_pause = 0.0
  return agent, env


def test_full_loop_reuse_then_mutation():
  print("[4/6] DMSAgent.step(): end-to-end Reuse -> success bookkeeping...")
  store_dir = fresh_dir(TEST_STORE_DIR + "_p4")
  ui_elements = [ui_element("WiFi toggle"), ui_element("Confirm"), ui_element("Done")]

  seed_memory = make_memory(
      "Settings home screen is open.", "Turn wifi on by tapping the toggle.", 3,
  )
  seed_bank = MemoryBank(store_dir)
  seed_memory = seed_bank.add(seed_memory)
  del seed_bank

  agent, env = make_agent(store_dir, ui_elements)
  agent.planner = StubPlanner([
      PlannerOutput(done=False, sub_plans=[
          SubPlan("Settings home screen is open.", "Turn wifi on by tapping the toggle.")
      ]),
  ])
  agent.verifier = StubVerifier([
      VerifierOutput(verified_success=True, reason="History shows toggle tapped."),
  ])
  agent._rng = ScriptedRandom([0.9])  # > epsilon(0.15) -> DoReuse.

  goal = "Turn on wifi."
  # DMSAgent chains Plan -> Dispatch -> (full) Replay -> Verify -> finish
  # within a SINGLE step() call whenever nothing requires an LLM-generated
  # atomic action in between (same chaining convention as Baseline A's
  # Plan -> first Actor action; see dms_agent_adapter.py's `step()`
  # docstring). With one sub-plan and an immediate reuse decision, one
  # call is enough to complete the whole sub-task.
  result = agent.step(goal)
  assert result.data["phase"] == "replay"
  assert result.data["retrieval"]["do_reuse"] is True
  assert len(env.executed_actions) == 3, "Expected all 3 stored actions replayed."
  assert agent.current_sub_task is None, "Sub-task should be finished after replay+verify."

  reloaded = MemoryBank(store_dir).get(seed_memory.memory_id, load_trajectory=False)
  assert reloaded.reuse_count == 1
  assert seed_memory.memory_id in agent.episode_active_memory_ids
  assert agent.usage.replay_attempts == 1 and agent.usage.replay_successes == 1
  assert agent.usage.replayed_actions_executed == 3
  print("    OK: replay executed all 3 stored actions in one step() call;"
        f" reuse_count=1, MRR={agent.usage.memory_reuse_rate:.2f}.")

  shutil.rmtree(store_dir)


def test_full_loop_mutation_evolves_memory():
  print("[5/6] DMSAgent.step(): epsilon-Mutation -> in-place evolutionary"
        " replacement...")
  store_dir = fresh_dir(TEST_STORE_DIR + "_p5")
  ui_elements = [ui_element("WiFi toggle"), ui_element("Confirm")]

  seed_bank = MemoryBank(store_dir)
  seed_memory = seed_bank.add(make_memory(
      "Settings home screen is open.", "Turn wifi on by tapping the toggle.", 3,
  ))
  del seed_bank

  agent, env = make_agent(store_dir, ui_elements)
  agent.planner = StubPlanner([
      PlannerOutput(done=False, sub_plans=[
          SubPlan("Settings home screen is open.", "Turn wifi on by tapping the toggle.")
      ]),
  ])
  # Fresh generation: two clicks, then declare complete (=> a 2-action
  # trajectory recorded, shorter than the seed memory's 3 actions, but
  # still >= MIN_TRAJECTORY_LENGTH_TO_STORE so it's eligible to compete).
  agent.actor = StubActor([
      ActorStepOutput(reason="tap toggle", action_json_str='{"action_type": "click", "index": 0}'),
      ActorStepOutput(reason="tap confirm", action_json_str='{"action_type": "click", "index": 1}'),
      ActorStepOutput(reason="done", action_json_str='{"action_type": "status", "goal_status": "complete"}'),
  ])
  agent.verifier = StubVerifier([
      VerifierOutput(verified_success=True, reason="Toggle was tapped."),
  ])
  agent._rng = ScriptedRandom([0.05])  # <= epsilon(0.15) -> Mutation.

  goal = "Turn on wifi."
  # Call 1 chains Plan -> Dispatch (Mutation decided, since roll <= epsilon)
  # -> the 1st fresh Actor action, all in one step() call.
  result1 = agent.step(goal)
  assert result1.data["retrieval"]["is_mutation"] is True
  assert result1.data["phase"] == "act"
  agent.step(goal)  # 2nd fresh Actor action.
  agent.step(goal)  # 3rd action: status complete -> Verify -> finish.

  assert agent.current_sub_task is None
  assert agent.usage.memories_replaced == 1
  bank_after = MemoryBank(store_dir)
  assert len(bank_after) == 1, "Mutation should REPLACE, not add a second memory."
  reloaded = bank_after.get(seed_memory.memory_id, load_trajectory=True)
  assert len(reloaded.trajectory) == 2, (
      f"Expected the shorter 2-action trajectory to win, got"
      f" {len(reloaded.trajectory)} actions."
  )
  assert [step.action["index"] for step in reloaded.trajectory] == [0, 1]
  print("    OK: mutation trajectory (2 actions) beat the seed memory (3"
        " actions) and was installed in place (bank size still 1).")

  shutil.rmtree(store_dir)


def test_terminal_success_persists_active_trajectory():
  print("[6/7] Ground-truth terminal success persists active trajectories...")
  trajectory = [
      TrajectoryStep(
          reason="first action",
          action={"action_type": "click", "index": 0},
          target_element_desc="first",
          state_signature="state-before",
      ),
      TrajectoryStep(
          reason="second action",
          action={"action_type": "click", "index": 1},
          target_element_desc="second",
          state_signature="state-after",
      ),
  ]

  dms_store = fresh_dir(TEST_STORE_DIR + "_terminal_dms")
  dms, _env = make_agent(dms_store, [ui_element("first"), ui_element("second")])
  dms.current_sub_task = SubPlan("screen is ready", "complete the task")
  dms.sub_task_trajectory = trajectory
  dms.finalize_task(True)
  assert len(dms.bank) == 1 and dms.usage.memories_created == 1

  static_store = fresh_dir(TEST_STORE_DIR + "_terminal_static")
  static = StaticMemoryAgent(
      FakeAndroidEnv([ui_element("first"), ui_element("second")]),
      llm=None,
      memory_store_dir=static_store,
      wait_after_action_seconds=0.0,
  )
  static.current_sub_task = SubPlan("screen is ready", "complete the task")
  static.sub_task_trajectory = trajectory
  static.finalize_task(True)
  assert len(static.bank) == 1 and static.usage.memories_created == 1
  print("    OK: evaluator-short-circuited success remains reusable memory.")

  shutil.rmtree(dms_store)
  shutil.rmtree(static_store)


def test_live_stagnant_action_forces_replan():
  print("[7/8] Repeated unchanged-state actions force a replan...")
  store_dir = fresh_dir(TEST_STORE_DIR + "_stagnant_action")
  agent, env = make_agent(store_dir, [ui_element("Add")])
  agent.planner = StubPlanner([
      PlannerOutput(done=False, sub_plans=[
          SubPlan("Contacts home", "Create and save the requested contact.")
      ]),
  ])
  agent.actor = StubActor([
      ActorStepOutput(
          reason="tap Add",
          action_json_str='{"action_type": "click", "index": 0}',
      ),
      ActorStepOutput(
          reason="tap Add again",
          action_json_str='{"action_type": "click", "index": 0}',
      ),
  ])

  agent.step("Create the requested contact.")
  agent.step("Create the requested contact.")
  assert agent.current_sub_task is None and not agent.sub_plan_queue
  assert len(env.executed_actions) == 2
  assert "Stagnant: repeated the same action" in agent.task_history[-1]
  print("    OK: same action + unchanged UI fails the sub-task immediately.")

  shutil.rmtree(store_dir)


def test_strike_based_pruning():
  print("[8/8] Repeated replay failures -> K_limit strikes -> memory pruned...")
  store_dir = fresh_dir(TEST_STORE_DIR + "_p6")

  agent, _env = make_agent(store_dir, [ui_element("X")])
  memory = agent.bank.add(make_memory("p", "g", 2))
  assert agent.mutation_config.k_limit == 3

  for i in range(agent.mutation_config.k_limit):
    still_present = agent.bank.get(memory.memory_id, load_trajectory=False)
    assert still_present is not None, f"Memory pruned too early (iteration {i})."
    agent.current_sub_task = SubPlan("p", "g")  # Re-set: cleared by each failure.
    agent._fail_current_sub_task_via_reuse(memory, f"forced failure {i}")

  assert agent.bank.get(memory.memory_id, load_trajectory=False) is None
  assert agent.usage.memories_pruned_by_strikes == 1
  print(f"    OK: memory pruned after {agent.mutation_config.k_limit} verification"
        " strikes (Algorithm 1 line 25-27 / Appendix D.1 K=3).")

  shutil.rmtree(store_dir)


def test_task_level_memory_write_and_replay():
  print("[9/9] Task-level memory: fast-path success -> round-2 replay...")
  from src.androidworld_integration.dms_agent_adapter import _TASK_LEVEL_PRECONDITION

  store_dir = fresh_dir(TEST_STORE_DIR + "_tasklevel")
  ui_elements = [ui_element("shutter"), ui_element("done")]
  agent, _env = make_agent(store_dir, ui_elements, epsilon=0.15)
  agent._rng = ScriptedRandom([0.9])  # round-2 roll > epsilon -> DoReuse.

  goal = "Take one photo."
  # Round 0: simulate a fast-path-solved episode (the sub-task memory-write
  # path never fires) by recording the fresh actions directly into the
  # episode trajectory, then confirm via ground truth.
  agent._episode_task_goal = goal
  agent._record_episode_step(
      "open Camera", {"action_type": "open_app", "app_name": "Camera"},
      target_element_desc=None,
  )
  agent._record_episode_step(
      "click shutter", {"action_type": "click", "index": 0},
      target_element_desc="shutter",
  )
  agent.finalize_task(True)
  assert len(agent.bank) == 1, f"task-level memory not written, bank={len(agent.bank)}"
  assert agent.usage.memories_created == 1
  mem = agent.bank.all_memories()[0]
  assert mem.precondition == _TASK_LEVEL_PRECONDITION
  assert mem.goal == goal
  print("    OK: round-0 fast-path episode persisted as a task-level memory"
        " (bank=1, precondition=task-level).")

  # Round 1: a fresh episode on the SAME task. The task-level retrieval at
  # the top of step() must hit and replay the stored trajectory instead of
  # re-solving via Planner / fast paths. A replay makes no LLM calls, so no
  # Planner/Actor/Verifier stubs are needed.
  agent._reset_episode_state()
  agent._rng = ScriptedRandom([0.9])
  result = agent.step(goal)
  assert result.data["phase"] == "task_replay", (
      f"expected phase=task_replay, got {result.data.get('phase')!r}")
  assert agent.usage.retrieval_hits == 1
  assert agent.usage.replay_attempts == 1
  assert agent.usage.replayed_actions_executed == 2, (
      f"expected 2 replayed actions, got {agent.usage.replayed_actions_executed}")
  assert agent.usage.memory_reuse_rate > 0.0
  # Resolve the deferred replay against ground truth (runner does this with
  # task.is_successful; here we assert the positive outcome path).
  agent.finalize_task(True)
  assert agent.usage.replay_successes == 1
  reloaded = agent.bank.get(mem.memory_id, load_trajectory=False)
  assert reloaded.reuse_count == 1
  print("    OK: round-1 task-level retrieval -> replay -> reuse_count=1,"
        f" MRR={agent.usage.memory_reuse_rate:.2f}.")

  shutil.rmtree(store_dir)


def main():
  if os.path.exists(TEST_STORE_DIR):
    shutil.rmtree(TEST_STORE_DIR)

  test_task_app_guardrails()
  test_decide_retrieval_and_reuse()
  test_replay_trajectory()
  test_decide_memory_update()
  test_full_loop_reuse_then_mutation()
  test_full_loop_mutation_evolves_memory()
  test_terminal_success_persists_active_trajectory()
  test_live_stagnant_action_forces_replan()
  test_strike_based_pruning()
  test_task_level_memory_write_and_replay()

  print("\nAll DMS mutation + main-loop checks passed.")


if __name__ == "__main__":
  main()
