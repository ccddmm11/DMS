#!/usr/bin/env python3
"""Standalone validation for src/baselines/static_memory_agent.py (Baseline B).

Does NOT require the emulator or vLLM: the environment and the Planner/
Actor/Verifier LLM roles are all replaced with deterministic stubs, same
approach as `scripts/test_dms_mutation_loop.py`. The key property under
test here is the ABSENCE of DMS's self-regulation machinery: retrieval
hits are always replayed unconditionally (no risk gate, no epsilon), and
memories are never pruned or revised no matter how many times a replay
fails -- the bank only ever grows via plain append.

Usage:
  python scripts/test_static_memory_agent.py
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from android_world.env import representation_utils

from src.agent.actor import ActorStepOutput
from src.agent.planner import PlannerOutput
from src.agent.planner import SubPlan
from src.agent.verifier import VerifierOutput
from src.baselines.static_memory_agent import StaticMemoryAgent
from src.memory.memory_bank import MemoryBank
from src.memory.memory_unit import MemoryUnit
from src.memory.memory_unit import TrajectoryStep

TEST_STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "_tmp_static_memory_agent_test",
)


def fresh_dir(path):
  if os.path.exists(path):
    shutil.rmtree(path)
  return path


def make_memory(precondition, goal, n_actions, source_task="TestTask"):
  trajectory = [
      TrajectoryStep(
          reason=f"step {i}", action={"action_type": "click", "index": i},
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


class StubPlanner:
  def __init__(self, outputs):
    self.outputs = list(outputs)

  def plan(self, goal, history, ui_elements_str, screenshots):
    return self.outputs.pop(0)


class StubActor:
  def __init__(self, outputs):
    self.outputs = list(outputs)

  def act(self, sub_task_str, history, ui_elements_str, screenshots):
    return self.outputs.pop(0)


class StubVerifier:
  def __init__(self, outputs):
    self.outputs = list(outputs)

  def verify(self, goal, history, screenshot):
    return self.outputs.pop(0)


class FakeAndroidEnv:
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


def make_agent(store_dir, ui_elements):
  env = FakeAndroidEnv(ui_elements)
  agent = StaticMemoryAgent(
      env, llm=None, memory_store_dir=store_dir, wait_after_action_seconds=0.0,
  )
  agent.transition_pause = 0.0
  return agent, env


def test_no_self_regulation_machinery():
  print("[1/4] Baseline B has no risk/self-regulation objects at all...")
  store_dir = fresh_dir(TEST_STORE_DIR + "_p1")
  agent, _env = make_agent(store_dir, [ui_element("X")])
  assert not hasattr(agent, "risk_regulator")
  assert not hasattr(agent, "self_regulator")
  assert not hasattr(agent, "mutation_config")
  print("    OK: no risk_regulator / self_regulator / mutation_config attributes.")
  shutil.rmtree(store_dir)


def test_retrieval_hit_always_replayed():
  print("[2/4] Retrieval hit -> ALWAYS replayed unconditionally (no epsilon/risk)...")
  store_dir = fresh_dir(TEST_STORE_DIR + "_p2")
  ui_elements = [ui_element("WiFi toggle"), ui_element("Confirm"), ui_element("Done")]

  seed_bank = MemoryBank(store_dir)
  seed_memory = seed_bank.add(make_memory(
      "Settings home screen is open.", "Turn wifi on by tapping the toggle.", 3,
  ))
  # Even a memory with an artificially extreme "risk-like" verification
  # strike count must still be reused: Baseline B has no risk model at all.
  seed_bank.record_verification_failure(seed_memory.memory_id)
  seed_bank.record_verification_failure(seed_memory.memory_id)
  seed_bank.record_verification_failure(seed_memory.memory_id)
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

  result = agent.step("Turn on wifi.")
  assert result.data["phase"] == "replay"
  assert result.data["retrieval"]["do_reuse"] is True
  assert len(env.executed_actions) == 3
  assert agent.current_sub_task is None

  reloaded = MemoryBank(store_dir).get(seed_memory.memory_id, load_trajectory=False)
  assert reloaded.reuse_count == 1
  assert reloaded.verification_strikes == 3, (
      "Strikes should still be readable (set by our seeding above) but"
      " Baseline B itself must never act on them."
  )
  print("    OK: reused despite pre-existing 'strikes' -- Baseline B has no"
        " strike-based suppression/pruning logic to consult them.")
  shutil.rmtree(store_dir)


def test_failed_replay_never_prunes_or_strikes():
  print("[3/4] Failed replay -> sub-task fails, but memory is left"
        " completely untouched (no strikes, no pruning)...")
  store_dir = fresh_dir(TEST_STORE_DIR + "_p3")
  ui_elements = [ui_element("A"), ui_element("B")]

  seed_bank = MemoryBank(store_dir)
  seed_memory = seed_bank.add(make_memory("Precond.", "Goal.", 2))
  del seed_bank

  agent, _env = make_agent(store_dir, ui_elements)

  for _ in range(5):
    # Repeat the SAME failing sub-task 5 times (well past DMS's K_limit=3)
    # to prove Baseline B truly never prunes, no matter how many failures.
    agent.planner = StubPlanner([
        PlannerOutput(done=False, sub_plans=[SubPlan("Precond.", "Goal.")]),
    ])
    agent.verifier = StubVerifier([
        VerifierOutput(verified_success=False, reason="Still contradicted."),
    ])
    result = agent.step("Some goal.")
    assert result.data["phase"] == "replay"
    assert agent.current_sub_task is None  # Sub-task failed -> replan.

    still_present = agent.bank.get(seed_memory.memory_id, load_trajectory=False)
    assert still_present is not None, "Baseline B must NEVER remove a memory."
    assert still_present.verification_strikes == 0, (
        "Baseline B must never increment verification strikes either."
    )
  print("    OK: memory survived 5 consecutive replay failures untouched"
        " (present, strikes=0) -- confirms the 'no pruning' ablation property.")
  shutil.rmtree(store_dir)


def test_fresh_generation_only_appends():
  print("[4/4] Fresh generation on a miss -> pure append, never replaces...")
  store_dir = fresh_dir(TEST_STORE_DIR + "_p4")
  ui_elements = [ui_element("Toggle")]

  agent, _env = make_agent(store_dir, ui_elements)
  agent.planner = StubPlanner([
      PlannerOutput(done=False, sub_plans=[SubPlan(
          "Settings app home screen is open.",
          "Turn wifi on by tapping the toggle.",
      )]),
  ])
  agent.actor = StubActor([
      ActorStepOutput(reason="tap", action_json_str='{"action_type": "click", "index": 0}'),
      ActorStepOutput(reason="tap2", action_json_str='{"action_type": "click", "index": 0}'),
      ActorStepOutput(reason="done", action_json_str='{"action_type": "status", "goal_status": "complete"}'),
  ])
  agent.verifier = StubVerifier([
      VerifierOutput(verified_success=True, reason="ok"),
  ])

  agent.step("Goal 1.")  # Plan + dispatch(miss) + 1st action.
  agent.step("Goal 1.")  # 2nd action.
  agent.step("Goal 1.")  # 3rd action: status complete -> verify -> finish.
  assert len(agent.bank) == 1
  assert agent.usage.memories_created == 1

  # A second, DIFFERENT sub-task with another miss should APPEND a second
  # memory (never overwrite/replace the first one -- Baseline B has no
  # evolutionary-replacement mechanism at all).
  agent.planner = StubPlanner([
      PlannerOutput(done=False, sub_plans=[SubPlan(
          "Markor app note list is open.",
          "Create a new note titled 'Shopping List'.",
      )]),
  ])
  agent.actor = StubActor([
      ActorStepOutput(reason="tap", action_json_str='{"action_type": "click", "index": 0}'),
      ActorStepOutput(reason="tap2", action_json_str='{"action_type": "click", "index": 0}'),
      ActorStepOutput(reason="done", action_json_str='{"action_type": "status", "goal_status": "complete"}'),
  ])
  agent.verifier = StubVerifier([
      VerifierOutput(verified_success=True, reason="ok"),
  ])
  agent.step("Goal 2.")
  agent.step("Goal 2.")
  agent.step("Goal 2.")
  assert len(agent.bank) == 2, "Expected pure append: bank size should grow to 2."
  assert agent.usage.memories_created == 2
  print("    OK: two independent misses -> two independently created"
        " memories (bank size 2), no replacement.")
  shutil.rmtree(store_dir)


def main():
  if os.path.exists(TEST_STORE_DIR):
    shutil.rmtree(TEST_STORE_DIR)

  test_no_self_regulation_machinery()
  test_retrieval_hit_always_replayed()
  test_failed_replay_never_prunes_or_strikes()
  test_fresh_generation_only_appends()

  print("\nAll Baseline B (static append-only memory) checks passed.")


if __name__ == "__main__":
  main()
