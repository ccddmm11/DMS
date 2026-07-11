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
import hashlib
import time
from typing import Any, Optional

from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.env import interface
from android_world.env import json_action

from src.agent import action_utils
from src.agent import app_intent
from src.agent import loop_guard
from src.agent import system_ui_intent
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
    self._task_apps: list[str] = []

    self.additional_guidelines = None  # kept for API parity with M3A/T3A.
    self._repetition_breaker = loop_guard.RepetitionBreaker()
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
  def start_new_task(
      self, task_name: Optional[str] = None, task_apps: Optional[list[str]] = None
  ) -> None:
    self._current_task_name = task_name
    self._task_apps = list(task_apps or [])

  def finalize_task(self, task_succeeded: bool) -> None:
    # The runner's ground-truth evaluator can end an episode immediately
    # after the action that solves it. Persist the active fresh trajectory
    # before reset discards it; otherwise Baseline B never accumulates
    # memories for successful terminal sub-tasks.
    if (
        task_succeeded
        and self.current_sub_task is not None
        and self.sub_task_trajectory
    ):
      self._succeed_current_sub_task_via_fresh_generation()
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
    self._repetition_breaker.reset()
    self._task_navigation_started = False
    self._planner_completion_rejections = 0
    self._system_toggle_route_labels: list[str] = []

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
    stall_warning = loop_guard.stall_warning_if_zero_action(
        self.current_sub_task.goal, len(self.sub_task_trajectory)
    )
    if stall_warning:
      self.task_history.append(stall_warning)
    self._advance_to_next_sub_task_slot()

  def _advance_after_fast_path(self, msg: str) -> None:
    """Marks the current sub-task COMPLETED via a deterministic fast path
    (no memory write or Verifier) and frees the slot."""
    assert self.current_sub_task is not None
    self.task_history.append(
        f"Sub-task [{self.current_sub_task.as_prompt_str()}] COMPLETED: {msg}"
    )
    self._advance_to_next_sub_task_slot()

  def _maybe_open_app_fast_path(
      self, step_data: dict[str, Any]
  ) -> Optional[base_agent.AgentInteractionResult]:
    """#3: deterministic handling of an explicit "open <app>" sub-goal. See
    `src/agent/app_intent.py`. Returns a result to short-circuit `step()`, or
    None to proceed with the normal Actor."""
    if self.sub_task_step_count != 0:
      return None
    decision = app_intent.open_app_fast_path_decision(
        self.env, self.current_sub_task.goal
    )
    if decision is None:
      return None
    kind, app_name = decision
    step_data["phase"] = "open_app_fast_path"
    step_data["sub_task"] = self.current_sub_task.to_dict()
    if kind == "already":
      step_data["action_reason"] = f"Already in the '{app_name}' app."
      self._advance_after_fast_path(
          f"already in the '{app_name}' app; no navigation needed."
      )
      return base_agent.AgentInteractionResult(False, step_data)
    try:
      self.env.execute_action(
          json_action.JSONAction(action_type="open_app", app_name=app_name)
      )
      self.usage.atomic_actions_executed += 1
      step_data["action_reason"] = f"Deterministic open_app({app_name})."
      time.sleep(self.wait_after_action_seconds)
      self._advance_after_fast_path(
          f"opened the '{app_name}' app directly (deterministic fast path)."
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._fail_current_sub_task(
          f"could not open app '{app_name}' directly ({e})."
      )
    return base_agent.AgentInteractionResult(False, step_data)

  def _maybe_open_initial_task_app_fast_path(
      self, step_data: dict[str, Any]
  ) -> Optional[base_agent.AgentInteractionResult]:
    """Uses task-suite app metadata only for the launcher-to-app prefix."""
    decision = app_intent.initial_task_app_fast_path_decision(
        self.env, self._task_apps, self._task_navigation_started
    )
    if decision is None:
      return None
    kind, app_name = decision
    self._task_navigation_started = True
    step_data["phase"] = "initial_task_app_fast_path"
    if kind == "already":
      step_data["action_reason"] = f"Already in declared task app '{app_name}'."
      self.task_history.append(
          f"[Task setup] Already in declared target app '{app_name}'."
      )
      return base_agent.AgentInteractionResult(False, step_data)
    try:
      self.env.execute_action(
          json_action.JSONAction(action_type="open_app", app_name=app_name)
      )
      self.usage.atomic_actions_executed += 1
      step_data["action_reason"] = (
          f"Opened declared target app '{app_name}' from task-suite metadata."
      )
      self.task_history.append(
          f"[Task setup] Opened declared target app '{app_name}' directly."
      )
      time.sleep(self.wait_after_action_seconds)
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.task_history.append(
          f"[Task setup] Could not open declared target app '{app_name}': {e}."
      )
    return base_agent.AgentInteractionResult(False, step_data)

  def _maybe_open_quick_settings_fast_path(
      self, step_data: dict[str, Any], task_goal: str
  ) -> Optional[base_agent.AgentInteractionResult]:
    """Opens Quick Settings for a narrow launcher/system-toggle gesture."""
    if self.sub_task_step_count != 0:
      return None
    decision = system_ui_intent.quick_settings_fast_path_decision(
        self.env, self.current_sub_task.goal, task_goal
    )
    if decision is None:
      return None
    step_data["phase"] = "system_navigation_fast_path"
    step_data["sub_task"] = self.current_sub_task.to_dict()
    try:
      if decision == "open_settings":
        self.env.execute_action(
            json_action.JSONAction(action_type="open_app", app_name="Settings")
        )
        progress_message = "opened Settings directly for a system-toggle task."
        action_reason = (
            "Deterministic open_app(Settings) for a launcher system-toggle"
            " task; Settings exposes labeled controls more reliably than"
            " generic Quick Settings switches."
        )
      else:
        self.env.execute_action(
            json_action.JSONAction(action_type="swipe", direction="down")
        )
        progress_message = (
            "opened Quick Settings with a deterministic top-edge pull-down."
        )
        action_reason = (
            "Deterministic top-edge pull-down to open Quick Settings."
        )
      self.usage.atomic_actions_executed += 1
      step_data["action_reason"] = action_reason
      time.sleep(self.wait_after_action_seconds)
      self._advance_after_fast_path(progress_message)
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._fail_current_sub_task(
          f"could not open Quick Settings directly ({e})."
      )
    return base_agent.AgentInteractionResult(False, step_data)

  def _maybe_advance_system_toggle_fast_path(
      self,
      step_data: dict[str, Any],
      task_goal: str,
      ui_elements,
      logical_screen_size: tuple[int, int],
  ) -> Optional[base_agent.AgentInteractionResult]:
    decision = system_ui_intent.next_labeled_system_toggle_action(
        self.env, task_goal, ui_elements, logical_screen_size,
        tuple(self._system_toggle_route_labels),
    )
    if decision is None:
      return None
    action_dict, label = decision
    action_dict, _ = action_utils.reground_action(
        action_dict, ui_elements, label, f"click '{label}'",
        logical_screen_size,
    )
    if action_dict.get("index") == -1:
      fallback_index = ui_utils.find_element_by_description(
          ui_elements, label, logical_screen_size
      )
      if fallback_index is not None:
        action_dict["index"] = fallback_index
    step_data["phase"] = "system_toggle_fast_path"
    try:
      self.env.execute_action(json_action.JSONAction(**action_dict))
      self.usage.atomic_actions_executed += 1
      step_data["action_reason"] = (
          f"Clicked labeled Android Settings route '{label}' for system toggle."
      )
      self.task_history.append(
          f"[System toggle setup] Clicked labeled Settings target '{label}'."
      )
      self._system_toggle_route_labels.append(label)
      time.sleep(self.wait_after_action_seconds)
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.task_history.append(
          f"[System toggle setup] Could not click Settings target '{label}': {e}."
      )
    return base_agent.AgentInteractionResult(False, step_data)

  def _verify_and_finish_fresh_sub_task(
      self, final_screenshot, observation_degraded: bool = False
  ) -> None:
    if observation_degraded:
      # See `ui_utils.get_robust_state`: the Actor was effectively blind
      # (degenerate a11y tree), so its claimed history is not trustworthy
      # grounding. Fail closed rather than let the Verifier's default
      # History-First trust rubber-stamp a hallucinated success.
      self._fail_current_sub_task(
          "Skipped verification: environment observation was degraded"
          " (a11y tree still empty after retries)."
      )
      return
    self.usage.verifier_calls += 1
    verifier_output = self.verifier.verify(
        self.current_sub_task.goal, self.sub_task_history, final_screenshot
    )
    self.usage.add(verifier_output.raw_response)
    # #6: veto a zero-action "success" on an interaction-implying sub-goal
    # (the "found it = done" exploit) before the softer repetition breaker.
    if (
        verifier_output.verified_success
        and len(self.sub_task_trajectory) == 0
        and loop_guard.goal_requires_interaction(self.current_sub_task.goal)
    ):
      self._fail_current_sub_task(
          "Rejected zero-action completion: the sub-goal requires a concrete"
          " UI interaction (click/type/toggle/...), but the Actor performed"
          " no action; forcing a real interaction / replan."
      )
      return
    if verifier_output.verified_success and self._repetition_breaker.record_and_check(
        self.current_sub_task.goal, len(self.sub_task_trajectory)
    ):
      self._fail_current_sub_task(
          "Stalled: repeated the same zero-action sub-goal completion"
          f" {self._repetition_breaker.max_repeats}x in a row without any"
          " real interaction; forcing failure to break the loop and force"
          " a genuine replan."
      )
    elif verifier_output.verified_success:
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
    if result is not None and (
        mutation.trajectory_contains_answer(result.memory.trajectory)
        or mutation._has_unreplayable_index_action(result.memory.trajectory)
        or mutation._has_repeated_unanchored_navigation(
            result.memory.trajectory
        )
        or mutation._has_repeated_stagnant_action(result.memory.trajectory)
    ):
      # Baseline B has no strike/pruning mechanism, but it must still refuse
      # legacy trajectories that are unsafe or provably stagnant. Otherwise a
      # single historical Actor loop can consume minutes on every replay hit.
      result = None
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

    state, state_degraded = ui_utils.get_robust_state(self.env)
    step_data["state_degraded"] = state_degraded
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

    initial_task_app_result = self._maybe_open_initial_task_app_fast_path(
        step_data
    )
    if initial_task_app_result is not None:
      return initial_task_app_result
    system_toggle_result = self._maybe_advance_system_toggle_fast_path(
        step_data, goal, ui_elements, logical_screen_size
    )
    if system_toggle_result is not None:
      return system_toggle_result

    # --- Planning phase. ---
    if self.current_sub_task is None and not self.sub_plan_queue:
      step_data["phase"] = "plan"
      self.usage.planner_calls += 1
      self.replan_cycles += 1
      planner_output = self.planner.plan(
          goal, self.task_history, ui_elements_str,
          [raw_screenshot, som_screenshot], task_apps=self._task_apps,
      )
      self.usage.add(planner_output.raw_response)
      step_data["planner_message"] = planner_output.message

      if not planner_output.parse_ok and planner_output.raw_response is None:
        # A bounded vLLM transport failure is not recoverable by immediately
        # submitting the same image-heavy planner request again. End this
        # episode so the runner persists a resume-safe failed cell instead of
        # appearing stuck for max_steps × request_timeout.
        self.task_history.append(
            "[Planner transport failure] Ending episode for forward progress:"
            f" {planner_output.message}"
        )
        return base_agent.AgentInteractionResult(True, step_data)

      if planner_output.done:
        # Ground-truth evaluator termination (in the runner) is authoritative;
        # do not end an episode on a weak Planner's self-report.
        self._planner_completion_rejections += 1
        self.task_history.append(
            "[Planner completion rejected] AndroidWorld evaluator has not"
            f" satisfied goal={goal!r}; declared_target_apps={self._task_apps};"
            f" foreground_package={app_intent.current_app_package(self.env)!r};"
            f" completion_message={planner_output.message!r};"
            f" rejection_count={self._planner_completion_rejections}."
            " Return a concrete recovery interaction that changes the missing"
            " task state; do not repeat completion."
        )
        return base_agent.AgentInteractionResult(False, step_data)

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

    # --- #3: deterministic open-app fast path (no LLM) before the Actor. ---
    fast_path_result = self._maybe_open_app_fast_path(step_data)
    if fast_path_result is not None:
      return fast_path_result
    fast_path_result = self._maybe_open_quick_settings_fast_path(
        step_data, goal
    )
    if fast_path_result is not None:
      return fast_path_result

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
      # #4: sanitize recoverable schema noise and re-ground the pointing
      # index to the target the Actor named before building the action.
      action_dict = action_utils.normalize_action_dict(
          agent_utils.extract_json(action_str)
      )
      action_dict, reground_note = action_utils.reground_action(
          action_dict, ui_elements, reason,
          self.current_sub_task.goal, logical_screen_size,
      )
      if reground_note:
        reason = f"{reason} [{reground_note}]"
        step_data["action_reason"] = reason
      converted_action = json_action.JSONAction(**action_dict)
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
      if not self.sub_task_history:
        self.sub_task_history.append(
            loop_guard.annotate_zero_action_completion(
                reason, converted_action.goal_status
            )
        )
      else:
        self.sub_task_history.append(
            f"Reason: {reason} Action: declared sub-task"
            f" {converted_action.goal_status}."
        )
      # `infeasible` denotes an execution failure, not evidence of goal
      # completion. Do not let the History-First Verifier rubber-stamp it.
      if converted_action.goal_status == "infeasible":
        self._fail_current_sub_task(
            f"Actor declared sub-task infeasible: {reason}"
        )
        return base_agent.AgentInteractionResult(False, step_data)
      after_state, after_degraded = ui_utils.get_robust_state(self.env)
      self._verify_and_finish_fresh_sub_task(
          after_state.pixels, observation_degraded=after_degraded
      )
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
              state_signature=hashlib.sha256(
                  ui_elements_str.encode("utf-8")
              ).hexdigest(),
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
      after_state, after_degraded = ui_utils.get_robust_state(self.env)
      self._verify_and_finish_fresh_sub_task(
          after_state.pixels, observation_degraded=after_degraded
      )

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

    step_data["action_reason"] = "Replayed stored trajectory from memory."
    if replay_result.observation_degraded:
      self._fail_current_sub_task(
          "Skipped verification: environment observation was degraded"
          " during replay (a11y tree still empty after retries); memory"
          " left untouched (no strikes/pruning in Baseline B)."
      )
      return base_agent.AgentInteractionResult(False, step_data)

    self.usage.verifier_calls += 1
    verifier_output = self.verifier.verify(
        self.current_sub_task.goal, replay_result.history,
        replay_result.final_screenshot,
    )
    self.usage.add(verifier_output.raw_response)

    if verifier_output.verified_success:
      self.usage.replay_successes += 1
      self._succeed_current_sub_task_via_reuse(candidate)
    else:
      self._fail_current_sub_task(
          f"Replay rejected by Verifier ({verifier_output.reason}); memory"
          " left untouched (no strikes/pruning in Baseline B)."
      )

    return base_agent.AgentInteractionResult(False, step_data)
