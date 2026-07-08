# DMS reproduction project — Baseline B: static append-only memory agent.
#
# The middle ground between Baseline A (PA-Lite, zero memory) and the full
# DMS agent: same hierarchical Planner-Actor-Verifier loop, and it DOES
# retrieve + blindly replay stored trajectories on a hit (so it gets the
# same latency/token savings DMS gets on a successful reuse), but it has
# NONE of DMS's self-regulating machinery:
#
#   - No epsilon-Mutation: a retrieval hit is ALWAYS replayed (no
#     exploration, no in-place evolutionary replacement -- so a memory
#     entry, once created, is never revised).
#   - No Bayesian risk gating: there is no notion of a memory's risk
#     score / dynamic threshold; any hit above the retrieval-score
#     threshold is trusted and reused unconditionally.
#   - No Survival-Value / Elbow-Method pruning or capacity regulation.
#   - No verification-strike counting or pruning: a failed replay simply
#     fails the sub-task (forcing a replan) but never removes or
#     penalizes the memory that was replayed.
#
# The Memory Bank therefore only ever GROWS (Sec 3.2.1's CreateMemory,
# gated by the same |tau|=1 filter) and never shrinks or gets revised --
# this isolates, in the 3-condition comparison (A vs. B vs. DMS), exactly
# how much of DMS's advantage comes from memory *existing* at all (A -> B)
# versus from the *self-regulation + risk feedback + mutation* machinery
# on top of it (B -> DMS).
#
#   while task not done and global_step < max_steps:
#     P = Planner(state, task)
#     for p_i in P:
#       m = Retrieve(Bank, p_i)                 # Dual-Factor score, no risk gate
#       if m is not None:
#         tau = m.trajectory; R_sub = Replay(tau)   # no LLM calls, unconditional
#       else:
#         tau = Actor.generate_from_scratch(state, p_i)
#       R_sub = Verifier(p_i, tau, state')
#       if R_sub == SUCCESS:
#         if reused: n_m += 1   (plain reuse-count bookkeeping only)
#         else: CreateMemory(p_i, tau)              # pure append, never replaces
#       else:
#         discard remaining P -> replan             # memory itself untouched
#
# Reuses `src/memory/mutation.py`'s `replay_trajectory` (identical replay
# mechanics to DMS) and `decide_memory_update` (called with `candidate=
# None`, which always takes its "create a new memory" branch -- exactly
# the append-only behavior we want here, with zero code duplication).

from __future__ import annotations

import dataclasses
import time
from typing import Any, Optional

from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.env import interface
from android_world.env import json_action

from src.agent import ui_utils
from src.agent.actor import Actor
from src.agent.planner import Planner
from src.agent.planner import SubPlan
from src.agent.verifier import Verifier
from src.memory import mutation
from src.memory.memory_bank import MemoryBank
from src.memory.memory_bank import RetrievalResult
from src.memory.memory_unit import MemoryUnit
from src.memory.memory_unit import TrajectoryStep

_INDEX_BASED_ACTIONS = ("click", "long_press", "input_text", "scroll")

DEFAULT_RETRIEVAL_SCORE_THRESHOLD = 0.3


@dataclasses.dataclass
class StaticMemoryUsageStats:
  """Aggregated LLM usage + memory counters for one episode."""

  planner_calls: int = 0
  actor_calls: int = 0
  verifier_calls: int = 0
  atomic_actions_executed: int = 0     # Actions taken via fresh generation.
  replayed_actions_executed: int = 0   # Actions taken via memory replay.
  retrieval_attempts: int = 0
  retrieval_hits: int = 0
  replay_attempts: int = 0
  replay_successes: int = 0
  memories_created: int = 0
  prompt_tokens: int = 0
  completion_tokens: int = 0

  def add(self, raw_response: Any) -> None:
    if isinstance(raw_response, dict):
      usage = raw_response.get("usage") or {}
      self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
      self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)

  @property
  def memory_reuse_rate(self) -> float:
    total = self.atomic_actions_executed + self.replayed_actions_executed
    if total == 0:
      return 0.0
    return self.replayed_actions_executed / total


class StaticMemoryAgent(base_agent.EnvironmentInteractingAgent):
  """Baseline B: hierarchical Planner-Actor-Verifier + a static,
  append-only Memory Bank (retrieval + unconditional replay, no
  self-regulation / risk feedback / mutation)."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer.MultimodalLlmWrapper,
      memory_store_dir: str,
      name: str = "Static-Memory",
      max_actor_steps_per_subtask: int = 8,
      wait_after_action_seconds: float = 2.0,
      retrieval_score_threshold: float = DEFAULT_RETRIEVAL_SCORE_THRESHOLD,
  ):
    """Initializes the Baseline B agent.

    Args:
      env: The AndroidWorld environment.
      llm: A multimodal LLM wrapper shared across Planner/Actor/Verifier.
      memory_store_dir: Directory backing this run's persistent MemoryBank
        (survives across `reset()` calls / episodes / rounds).
      name: Agent name, used in logs/results.
      max_actor_steps_per_subtask: Local step limit `MaxA` for the Actor.
      wait_after_action_seconds: Seconds to sleep after an atomic action.
      retrieval_score_threshold: Minimum Dual-Factor score to consider a
        retrieval a "hit" at all (same knob as DMS's
        `MutationConfig.retrieval_score_threshold`, kept identical across
        conditions B/DMS for a fair comparison).
    """
    super().__init__(env, name)
    self.planner = Planner(llm)
    self.actor = Actor(llm)
    self.verifier = Verifier(llm)
    self.bank = MemoryBank(memory_store_dir)

    self.max_actor_steps_per_subtask = max_actor_steps_per_subtask
    self.wait_after_action_seconds = wait_after_action_seconds
    self.retrieval_score_threshold = retrieval_score_threshold
    self._completed_task_count = 0
    self._current_task_name: Optional[str] = None

    self.additional_guidelines = None  # kept for API parity with M3A/T3A.
    self._reset_episode_state()

  # -- API parity with M3A/T3A ---------------------------------------------
  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def reset(self, go_home_on_reset: bool = False) -> None:
    super().reset(go_home_on_reset)
    self.env.hide_automation_ui()
    self.bank.tick(1.0)
    self._reset_episode_state()

  # -- DMS-condition-parity hooks (no-ops beyond bookkeeping: Baseline B
  # has no Global Feedback Regulation Stage / Self-Regulation to run). --
  def start_new_task(self, task_name: Optional[str] = None) -> None:
    self._current_task_name = task_name

  def finalize_task(self, task_succeeded: bool) -> None:
    del task_succeeded  # Unused: no risk model / regulation in Baseline B.
    self._completed_task_count += 1

  # -- Internal episode state ----------------------------------------------
  def _reset_episode_state(self) -> None:
    self.task_history: list[str] = []
    self.sub_plan_queue: list[SubPlan] = []
    self.current_sub_task: Optional[SubPlan] = None
    self.sub_task_history: list[str] = []
    self.sub_task_trajectory: list[TrajectoryStep] = []
    self.sub_task_step_count = 0
    self.sub_task_candidate: Optional[MemoryUnit] = None
    self.replan_cycles = 0
    self.usage = StaticMemoryUsageStats()

  def _advance_to_next_sub_task_slot(self) -> None:
    self.current_sub_task = None
    self.sub_task_candidate = None
    self.sub_task_history = []
    self.sub_task_trajectory = []
    self.sub_task_step_count = 0

  def _fail_current_sub_task(self, reason: str) -> None:
    assert self.current_sub_task is not None
    self.task_history.append(
        f"Sub-task [{self.current_sub_task.as_prompt_str()}] FAILED:"
        f" {reason}"
    )
    self._advance_to_next_sub_task_slot()
    self.sub_plan_queue = []  # PlanFailed <- TRUE; discard rest of this plan.

  def _succeed_current_sub_task_via_reuse(self, memory: MemoryUnit) -> None:
    self.bank.record_reuse(memory.memory_id, persist=True)
    self.task_history.append(
        f"Sub-task [{self.current_sub_task.as_prompt_str()}] COMPLETED"
        " (replayed from memory)."
    )
    self._advance_to_next_sub_task_slot()

  def _succeed_current_sub_task_via_fresh_generation(self) -> None:
    # `candidate=None`: Baseline B never revises an existing memory, it
    # only ever appends -- this is the ONLY caller of `decide_memory_update`
    # here, and it is only reached when retrieval found no hit at all.
    outcome = mutation.decide_memory_update(
        self.bank,
        self.current_sub_task.precondition,
        self.current_sub_task.goal,
        self.sub_task_trajectory,
        candidate=None,
        source_task=self._current_task_name,
    )
    if outcome.action == "created":
      self.usage.memories_created += 1
    self.task_history.append(
        f"Sub-task [{self.current_sub_task.as_prompt_str()}] COMPLETED"
        f" (fresh generation, memory_update={outcome.action})."
    )
    self._advance_to_next_sub_task_slot()

  def _verify_and_finish_fresh_sub_task(self, final_screenshot) -> None:
    self.usage.verifier_calls += 1
    verifier_output = self.verifier.verify(
        self.current_sub_task.goal, self.sub_task_history, final_screenshot
    )
    self.usage.add(verifier_output.raw_response)
    if verifier_output.verified_success:
      self._succeed_current_sub_task_via_fresh_generation()
    else:
      self._fail_current_sub_task(
          f"Verifier rejected fresh trajectory ({verifier_output.reason})."
      )

  def _dispatch_next_sub_task(self) -> Optional[RetrievalResult]:
    """Pops the next queued sub-task and looks up a retrieval candidate
    (no risk gate, no epsilon: a hit is always reused)."""
    self.current_sub_task = self.sub_plan_queue.pop(0)
    self.sub_task_history = []
    self.sub_task_trajectory = []
    self.sub_task_step_count = 0

    self.usage.retrieval_attempts += 1
    results = self.bank.retrieve(
        self.current_sub_task.precondition,
        self.current_sub_task.goal,
        top_k=1,
        score_threshold=self.retrieval_score_threshold,
    )
    result = results[0] if results and results[0].memory.trajectory else None
    if result is not None:
      self.usage.retrieval_hits += 1
    self.sub_task_candidate = result.memory if result is not None else None
    return result

  # -- Main loop -------------------------------------------------------------
  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    step_data: dict[str, Any] = {
        "phase": None,
        "sub_task": None,
        "planner_message": None,
        "retrieval": None,
        "action_reason": None,
        "action_output_json": None,
        "raw_screenshot": None,
    }

    state = self.get_post_transition_state()
    logical_screen_size = self.env.logical_screen_size
    orientation = self.env.orientation
    physical_frame_boundary = self.env.physical_frame_boundary
    ui_elements = state.ui_elements
    ui_elements_str = ui_utils.describe_ui_elements(
        ui_elements, logical_screen_size
    )
    raw_screenshot = state.pixels.copy()
    step_data["raw_screenshot"] = raw_screenshot
    som_screenshot = ui_utils.build_som_screenshot(
        raw_screenshot,
        ui_elements,
        logical_screen_size,
        physical_frame_boundary,
        orientation,
    )

    # --- Planning phase. ---
    if self.current_sub_task is None and not self.sub_plan_queue:
      step_data["phase"] = "plan"
      self.usage.planner_calls += 1
      self.replan_cycles += 1
      planner_output = self.planner.plan(
          goal, self.task_history, ui_elements_str,
          [raw_screenshot, som_screenshot],
      )
      self.usage.add(planner_output.raw_response)
      step_data["planner_message"] = planner_output.message

      if planner_output.done:
        return base_agent.AgentInteractionResult(True, step_data)

      if not planner_output.sub_plans:
        self.task_history.append(
            "[Planner] Produced no valid sub-plans this cycle "
            f"({planner_output.message or 'parse failure'})."
        )
        return base_agent.AgentInteractionResult(False, step_data)

      self.sub_plan_queue = list(planner_output.sub_plans)

    # --- Dispatch phase: pop next sub-task + retrieval lookup, then
    # either a full Replay (same call) or fall through to fresh Actor
    # generation (same call, first atomic action). ---
    if self.current_sub_task is None:
      step_data["phase"] = "dispatch"
      result = self._dispatch_next_sub_task()
      step_data["sub_task"] = self.current_sub_task.to_dict()
      step_data["retrieval"] = {
          "hit": result is not None,
          "score": result.score if result is not None else 0.0,
          "do_reuse": result is not None,
      }

      if result is not None:
        step_data["phase"] = "replay"
        self.usage.replay_attempts += 1
        return self._execute_replay(step_data, result.memory)
      # Else: fall through to fresh Actor generation below, same call.

    # --- Fresh Actor generation phase: one atomic action this call. ---
    step_data["phase"] = "act"
    step_data["sub_task"] = self.current_sub_task.to_dict()
    self.usage.actor_calls += 1
    self.sub_task_step_count += 1

    actor_output = self.actor.act(
        self.current_sub_task.as_prompt_str(),
        self.sub_task_history,
        ui_elements_str,
        [raw_screenshot, som_screenshot],
    )
    self.usage.add(actor_output.raw_response)

    if not actor_output.parse_ok:
      self.sub_task_history.append(
          "Action selection output was not in the correct format; no"
          " action performed."
      )
      if self.sub_task_step_count >= self.max_actor_steps_per_subtask:
        self._fail_current_sub_task(
            "Exceeded local step limit with repeated format errors."
        )
      return base_agent.AgentInteractionResult(False, step_data)

    reason, action_str = actor_output.reason, actor_output.action_json_str
    step_data["action_reason"] = reason

    try:
      converted_action = json_action.JSONAction(
          **agent_utils.extract_json(action_str)
      )
      step_data["action_output_json"] = converted_action
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.sub_task_history.append(
          f"Reason: {reason} Action: {action_str} -> FAILED to parse into a"
          f" valid action ({e})."
      )
      if self.sub_task_step_count >= self.max_actor_steps_per_subtask:
        self._fail_current_sub_task(
            "Exceeded local step limit with repeated invalid actions."
        )
      return base_agent.AgentInteractionResult(False, step_data)

    if converted_action.action_type == "status":
      self.sub_task_history.append(
          f"Reason: {reason} Action: declared sub-task"
          f" {converted_action.goal_status}."
      )
      after_state = self.env.get_state(wait_to_stabilize=False)
      self._verify_and_finish_fresh_sub_task(after_state.pixels)
      return base_agent.AgentInteractionResult(False, step_data)

    num_ui_elements = len(ui_elements)
    if (
        converted_action.action_type in _INDEX_BASED_ACTIONS
        and converted_action.index is not None
        and converted_action.index >= num_ui_elements
    ):
      self.sub_task_history.append(
          f"Reason: {reason} Action: {action_str} -> FAILED: index out of"
          f" range (UI element list only has {num_ui_elements} elements)."
      )
      if self.sub_task_step_count >= self.max_actor_steps_per_subtask:
        self._fail_current_sub_task(
            "Exceeded local step limit with repeated out-of-range indices."
        )
      return base_agent.AgentInteractionResult(False, step_data)

    target_desc = None
    if (
        converted_action.action_type in _INDEX_BASED_ACTIONS
        and converted_action.index is not None
    ):
      target_desc = ui_utils.describe_target_element(
          ui_elements, converted_action.index
      )

    try:
      self.env.execute_action(converted_action)
      self.sub_task_history.append(f"Reason: {reason} Action: {action_str}")
      self.sub_task_trajectory.append(
          TrajectoryStep(
              reason=reason or "",
              action=converted_action.as_dict(),
              target_element_desc=target_desc,
          )
      )
      self.usage.atomic_actions_executed += 1
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.sub_task_history.append(
          f"Reason: {reason} Action: {action_str} -> FAILED to execute"
          f" ({e})."
      )

    time.sleep(self.wait_after_action_seconds)

    if self.sub_task_step_count >= self.max_actor_steps_per_subtask:
      after_state = self.env.get_state(wait_to_stabilize=False)
      self._verify_and_finish_fresh_sub_task(after_state.pixels)

    return base_agent.AgentInteractionResult(False, step_data)

  # -- Replay execution (unconditional on a retrieval hit) ------------------
  def _execute_replay(
      self, step_data: dict[str, Any], candidate: MemoryUnit
  ) -> base_agent.AgentInteractionResult:
    replay_result = mutation.replay_trajectory(
        candidate.trajectory, self.env, self.wait_after_action_seconds
    )
    step_data["replay_steps"] = replay_result.steps_replayed
    self.usage.replayed_actions_executed += replay_result.steps_replayed

    if not replay_result.success_execution:
      self._fail_current_sub_task(
          "Replay aborted: could not re-ground stored action "
          "(memory untouched -- Baseline B never prunes/strikes)."
      )
      return base_agent.AgentInteractionResult(False, step_data)

    self.usage.verifier_calls += 1
    verifier_output = self.verifier.verify(
        self.current_sub_task.goal, replay_result.history,
        replay_result.final_screenshot,
    )
    self.usage.add(verifier_output.raw_response)
    step_data["action_reason"] = "Replayed stored trajectory from memory."

    if verifier_output.verified_success:
      self.usage.replay_successes += 1
      self._succeed_current_sub_task_via_reuse(candidate)
    else:
      self._fail_current_sub_task(
          f"Replay rejected by Verifier ({verifier_output.reason}); memory"
          " left untouched (no strikes/pruning in Baseline B)."
      )

    return base_agent.AgentInteractionResult(False, step_data)
