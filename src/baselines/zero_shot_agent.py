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
  def start_new_task(self, task_name: Optional[str] = None) -> None:
    del task_name

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

  def _finish_sub_task(self, success: bool, reason: str) -> None:
    assert self.current_sub_task is not None
    tag = "COMPLETED" if success else "FAILED"
    self.task_history.append(
        f"Sub-task [{self.current_sub_task.as_prompt_str()}] {tag}:"
        f" {reason}"
    )
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

  def _verify_and_finish_sub_task(self, final_screenshot) -> None:
    self.usage.verifier_calls += 1
    verifier_output = self.verifier.verify(
        self.current_sub_task.goal, self.sub_task_history, final_screenshot
    )
    self.usage.add(verifier_output.raw_response)
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

    # --- Planning phase: only entered when there is no active sub-task. ---
    if self.current_sub_task is None:
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
      self.current_sub_task = self.sub_plan_queue.pop(0)
      self.sub_task_history = []
      self.sub_task_step_count = 0

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
        self._finish_sub_task(
            False, "Exceeded local step limit with repeated invalid actions."
        )
      return base_agent.AgentInteractionResult(False, step_data)

    # Sub-task-level status declaration: intercepted, never sent to `env`.
    if converted_action.action_type == "status":
      self.sub_task_history.append(
          f"Reason: {reason} Action: declared sub-task"
          f" {converted_action.goal_status}."
      )
      after_state = self.env.get_state(wait_to_stabilize=False)
      self._verify_and_finish_sub_task(after_state.pixels)
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

    try:
      self.env.execute_action(converted_action)
      self.sub_task_history.append(f"Reason: {reason} Action: {action_str}")
      self.usage.atomic_actions_executed += 1
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.sub_task_history.append(
          f"Reason: {reason} Action: {action_str} -> FAILED to execute"
          f" ({e})."
      )

    time.sleep(self.wait_after_action_seconds)

    if self.sub_task_step_count >= self.max_actor_steps_per_subtask:
      after_state = self.env.get_state(wait_to_stabilize=False)
      self._verify_and_finish_sub_task(after_state.pixels)

    return base_agent.AgentInteractionResult(False, step_data)
