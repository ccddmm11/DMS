# DMS reproduction project — Baseline A: PA-Lite (memory-free Planner-Actor).
#
# This is our independent implementation of the paper's "PA-Lite" baseline
# (Table 1 / Sec 3.1): a canonical, memory-free Planner-Actor agent that
# plugs into AndroidWorld's `EnvironmentInteractingAgent` interface exactly
# like the official M3A/T3A baselines, so it can be dropped into
# `run.py` / `suite_utils.run` unmodified.
#
# Control flow mirrors Algorithm 1 (Appendix D) with the memory-specific
# steps (Retrieve/CreateMemory/Survival-Value bookkeeping) removed, since
# Baseline A has zero memory:
#   while task not done and global_step < max_steps:
#     P = Planner(state, task)              # <=5 sub-plans
#     for p_i in P:
#       tau = Actor.generate_from_scratch(state, p_i)   # always fresh, no reuse
#       R_sub = Verifier(p_i, tau, state')
#       if R_sub == FAIL: discard remaining P, break -> replan
#     if all sub-plans succeeded: back to Planner to check global completion
#
# Because AndroidWorld's harness drives the agent with repeated `step()`
# calls (one atomic env action per call, exactly like M3A), we implement
# this nested loop as an internal state machine advanced by one Actor
# action (or one Planner call) per `step()` invocation. This makes the
# hierarchical Planner-Actor-Verifier loop fully compatible with the
# existing `suite_utils.run` runner and per-step accounting used by
# `minimal_task_runner.py` / our own eval harness.

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

_INDEX_BASED_ACTIONS = ("click", "long_press", "input_text", "scroll")


@dataclasses.dataclass
class UsageStats:
  """Aggregated LLM usage for one episode, used by the eval harness."""

  planner_calls: int = 0
  actor_calls: int = 0
  verifier_calls: int = 0
  atomic_actions_executed: int = 0
  prompt_tokens: int = 0
  completion_tokens: int = 0

  def add(self, raw_response: Any) -> None:
    if isinstance(raw_response, dict):
      usage = raw_response.get("usage") or {}
      self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
      self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)


class PALiteAgent(base_agent.EnvironmentInteractingAgent):
  """Baseline A: memory-free hierarchical Planner-Actor-Verifier agent."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer.MultimodalLlmWrapper,
      name: str = "PA-Lite",
      max_actor_steps_per_subtask: int = 8,
      wait_after_action_seconds: float = 2.0,
  ):
    """Initializes the PA-Lite agent.

    Args:
      env: The AndroidWorld environment.
      llm: A multimodal LLM wrapper (e.g. our local QwenVLWrapper), shared
        across the Planner/Actor/Verifier roles, matching the paper's
        "unified backbone" (Univ.) baselines in Table 1.
      name: Agent name, used in logs/results.
      max_actor_steps_per_subtask: Local step limit `MaxA` (Sec 3.1) for the
        Actor before we force a verification + (re)plan.
      wait_after_action_seconds: Seconds to sleep after an atomic action for
        the screen to stabilize (same convention as M3A).
    """
    super().__init__(env, name)
    self.planner = Planner(llm)
    self.actor = Actor(llm)
    self.verifier = Verifier(llm)
    self.max_actor_steps_per_subtask = max_actor_steps_per_subtask
    self.wait_after_action_seconds = wait_after_action_seconds
    self.additional_guidelines = None  # kept for API parity with M3A/T3A.
    self._repetition_breaker = loop_guard.RepetitionBreaker()
    self._stagnant_action_breaker = loop_guard.StagnantActionBreaker()
    self._current_task_name: Optional[str] = None
    self._task_apps: list[str] = []
    self._reset_episode_state()

  # -- API parity with M3A/T3A -----------------------------------------
  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def reset(self, go_home_on_reset: bool = False) -> None:
    super().reset(go_home_on_reset)
    self.env.hide_automation_ui()
    self._reset_episode_state()

  # -- No-op parity hooks with Baseline B / DMS, so the eval harness can
  # call `start_new_task`/`finalize_task` uniformly across all three
  # conditions without special-casing Baseline A (which has no memory and
  # therefore nothing to tick/regulate). --------------------------------
  def start_new_task(
      self, task_name: Optional[str] = None, task_apps: Optional[list[str]] = None
  ) -> None:
    self._current_task_name = task_name
    self._task_apps = list(task_apps or [])

  def finalize_task(self, task_succeeded: bool) -> None:
    del task_succeeded

  # -- Internal episode state --------------------------------------------
  def _reset_episode_state(self) -> None:
    self.task_history: list[str] = []
    self.sub_plan_queue: list[SubPlan] = []
    self.current_sub_task: Optional[SubPlan] = None
    self.sub_task_history: list[str] = []
    self.sub_task_step_count = 0
    self.replan_cycles = 0
    self.usage = UsageStats()
    self._repetition_breaker.reset()
    self._stagnant_action_breaker.reset()
    self._task_navigation_started = False
    self._planner_completion_rejections = 0
    self._system_toggle_route_labels: list[str] = []

  def _finish_sub_task(self, success: bool, reason: str) -> None:
    assert self.current_sub_task is not None
    actions_taken = max(0, len(self.sub_task_history) - 1)
    if success and self._repetition_breaker.record_and_check(
        self.current_sub_task.goal, actions_taken
    ):
      success = False
      reason = (
          "Stalled: repeated the same zero-action sub-goal completion"
          f" {self._repetition_breaker.max_repeats}x in a row without any"
          " real interaction; forcing failure to break the loop and force"
          " a genuine replan."
      )
    tag = "COMPLETED" if success else "FAILED"
    self.task_history.append(
        f"Sub-task [{self.current_sub_task.as_prompt_str()}] {tag}:"
        f" {reason}"
    )
    if success:
      # `sub_task_history` still holds this cycle's entries at this point
      # (reset happens further below) -- exactly 1 entry means it was JUST
      # the zero-action "declared complete" line (see
      # `loop_guard.annotate_zero_action_completion`), i.e. no real
      # interaction happened this sub-task.
      stall_warning = loop_guard.stall_warning_if_zero_action(
          self.current_sub_task.goal, actions_taken=actions_taken
      )
      if stall_warning:
        self.task_history.append(stall_warning)
    self.current_sub_task = None
    if not success:
      # Algorithm 1: discard the rest of this plan cycle and force a full
      # replan next step() call.
      self.sub_plan_queue = []
      return
    if self.sub_plan_queue:
      self.current_sub_task = self.sub_plan_queue.pop(0)
      self.sub_task_history = []
      self.sub_task_step_count = 0
    # Else: queue is also empty -> next step() call re-invokes the Planner,
    # which gets the chance to declare overall task completion.

  def _advance_after_fast_path(self, msg: str) -> None:
    """Marks the current sub-task COMPLETED via a deterministic fast path
    (no memory, Verifier, or repetition-breaker accounting)."""
    assert self.current_sub_task is not None
    self.task_history.append(
        f"Sub-task [{self.current_sub_task.as_prompt_str()}] COMPLETED: {msg}"
    )
    self.current_sub_task = None
    if self.sub_plan_queue:
      self.current_sub_task = self.sub_plan_queue.pop(0)
      self.sub_task_history = []
      self.sub_task_step_count = 0

  def _maybe_open_app_fast_path(
      self, step_data: dict[str, Any]
  ) -> Optional[base_agent.AgentInteractionResult]:
    """#3: deterministic handling of an explicit "open <app>" sub-goal at the
    start of a sub-task. Returns an AgentInteractionResult to short-circuit
    this `step()` call, or None to proceed with the normal Actor."""
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
    # kind == "open"
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
      self._finish_sub_task(
          False, f"could not open app '{app_name}' directly ({e})."
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
      self._finish_sub_task(
          False, f"could not open Quick Settings directly ({e})."
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

  def _verify_and_finish_sub_task(
      self, final_screenshot, observation_degraded: bool = False
  ) -> None:
    if observation_degraded:
      # See `ui_utils.get_robust_state`: the Actor was effectively blind
      # (degenerate a11y tree), so its claimed history is not trustworthy
      # grounding. Fail closed rather than let the Verifier's default
      # History-First trust rubber-stamp a hallucinated success.
      self._finish_sub_task(
          False,
          "Skipped verification: environment observation was degraded"
          " (a11y tree still empty after retries).",
      )
      return
    self.usage.verifier_calls += 1
    verifier_output = self.verifier.verify(
        self.current_sub_task.goal, self.sub_task_history, final_screenshot
    )
    self.usage.add(verifier_output.raw_response)
    # #6: veto a zero-action "success" on an interaction-implying sub-goal
    # (the "found it = done" exploit) so it fails on first occurrence.
    actions_taken = max(0, len(self.sub_task_history) - 1)
    if (
        verifier_output.verified_success
        and actions_taken == 0
        and loop_guard.goal_requires_interaction(self.current_sub_task.goal)
    ):
      self._finish_sub_task(
          False,
          "Rejected zero-action completion: the sub-goal requires a concrete"
          " UI interaction, but no action was performed.",
      )
      return
    self._finish_sub_task(verifier_output.verified_success,
                           verifier_output.reason)

  # -- Main loop -----------------------------------------------------------
  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    step_data: dict[str, Any] = {
        "phase": None,
        "sub_task": None,
        "planner_message": None,
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

    # --- Planning phase: only entered when there is no active sub-task. ---
    if self.current_sub_task is None:
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
        # Retrying the identical failed VLM request on every outer step only
        # turns a transient server outage into a stalled evaluation worker.
        self.task_history.append(
            "[Planner transport failure] Ending episode for forward progress:"
            f" {planner_output.message}"
        )
        return base_agent.AgentInteractionResult(True, step_data)

      if planner_output.done:
        # The evaluation runner supplies AndroidWorld's ground-truth
        # termination callback. A 7B Planner's self-report is not reliable
        # enough to end an episode: it can mistake an unlabeled system tile
        # for an enabled setting (e.g. SystemWifiTurnOn). Keep the episode
        # alive so the evaluator alone can terminate a real success.
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
      self.current_sub_task = self.sub_plan_queue.pop(0)
      self.sub_task_history = []
      self.sub_task_step_count = 0

    # --- #3: deterministic open-app fast path (no LLM) before the Actor. ---
    fast_path_result = self._maybe_open_app_fast_path(step_data)
    if fast_path_result is not None:
      return fast_path_result
    fast_path_result = self._maybe_open_quick_settings_fast_path(
        step_data, goal
    )
    if fast_path_result is not None:
      return fast_path_result

    # --- Execution phase: one Actor-generated atomic action. ---
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
        self._finish_sub_task(
            False, "Exceeded local step limit with repeated format errors."
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
        self._finish_sub_task(
            False, "Exceeded local step limit with repeated invalid actions."
        )
      return base_agent.AgentInteractionResult(False, step_data)

    # Sub-task-level status declaration: intercepted, never sent to `env`.
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
      # `infeasible` is the Actor reporting that it could not execute this
      # sub-task. It must force PlanFailed immediately, not be presented to
      # the History-First Verifier as evidence that the goal was achieved.
      if converted_action.goal_status == "infeasible":
        self._finish_sub_task(
            False, f"Actor declared sub-task infeasible: {reason}"
        )
        return base_agent.AgentInteractionResult(False, step_data)
      after_state, after_degraded = ui_utils.get_robust_state(self.env)
      self._verify_and_finish_sub_task(
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
        self._finish_sub_task(
            False,
            "Exceeded local step limit with repeated out-of-range indices.",
        )
      return base_agent.AgentInteractionResult(False, step_data)

    action_executed = False
    try:
      self.env.execute_action(converted_action)
      self.sub_task_history.append(f"Reason: {reason} Action: {action_str}")
      self.usage.atomic_actions_executed += 1
      action_executed = True
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.sub_task_history.append(
          f"Reason: {reason} Action: {action_str} -> FAILED to execute"
          f" ({e})."
      )

    time.sleep(self.wait_after_action_seconds)

    state_signature = hashlib.sha256(
        ui_elements_str.encode("utf-8")
    ).hexdigest()
    if action_executed and self._stagnant_action_breaker.record_and_check(
        state_signature, converted_action.as_dict()
    ):
      self._finish_sub_task(
          False,
          "Stagnant: repeated the same action while the UI state was"
          " unchanged; forcing a replan.",
      )
      return base_agent.AgentInteractionResult(False, step_data)

    if self.sub_task_step_count >= self.max_actor_steps_per_subtask:
      after_state, after_degraded = ui_utils.get_robust_state(self.env)
      self._verify_and_finish_sub_task(
          after_state.pixels, observation_degraded=after_degraded
      )

    return base_agent.AgentInteractionResult(False, step_data)
