# DMS reproduction project — Actor role.
#
# Reimplements the paper's canonical Actor (Sec 3.1 "Execution Phase" and
# Appendix H, Fig 11 "Codeact Prompt"). The Actor receives ONE sub-task
# p_i = <Precondition, Goal> from the Planner and generates atomic actions
# `a_t = A(o_t, q, p_i)` step by step until the sub-task is resolved or a
# local step limit is reached.
#
# We keep the *semantics* of the paper's Actor prompt -- strict literal
# execution (anti-overreach), error-loop prevention, one-screen-per-action
# discipline -- but express actions in AndroidWorld's index-based JSON
# action schema (android_world/env/json_action.py), reusing the same
# action space as the official M3A baseline, rather than the paper's raw
# tap(x, y) CodeAct python snippets. This keeps the Actor's grounding
# consistent with what we already validated works for our 7B VLM
# (see results/smoke_test/).

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np
from android_world.agents import infer
from android_world.agents import m3a_utils

ACTOR_PROMPT_PREFIX = (
    "You are an Actor agent that executes ONE Android UI action at a time to"
    " accomplish a specific sub-task assigned by a Planner. You do NOT need"
    " to reason about the overall multi-step task -- only about the"
    " sub-task below.\n\n"
    "## CRITICAL: STRICT LITERAL EXECUTION (ANTI-OVERREACH)\n"
    "You are FORBIDDEN from performing any action not explicitly required by"
    ' the sub-task goal. If the goal says "type text", do NOT also click'
    ' "Send". If the goal says "select", do NOT also click "OK". Once the'
    " requested action is done, immediately declare the sub-task complete --"
    " do not add unrequested cleanup/confirmation steps.\n\n"
    "## ERROR LOOP PREVENTION\n"
    "Check the sub-task history below before acting. You are STRICTLY"
    " FORBIDDEN from repeating an action that already failed or produced no"
    " visible change on the screen. If an action did not work, you MUST"
    " pivot to a different action (e.g. scroll to find another element, use"
    " a different index) or declare the sub-task infeasible.\n\n"
    "At each step you are given the current screenshot (including the"
    " original screenshot and the same screenshot with bounding boxes and"
    " numeric indexes added to some UI elements) and a list of detailed"
    " information for the UI elements. You must choose ONE action from the"
    " following list (action description followed by the JSON format):\n"
    "- Click/tap on an element on the screen, use the numeric index to"
    " indicate which element:"
    ' `{{"action_type": "click", "index": <target_index>}}`.\n'
    "- Long press on an element on the screen, use the numeric index:"
    ' `{{"action_type": "long_press", "index": <target_index>}}`.\n'
    "- Type text into a text field (this action contains clicking the text"
    " field, typing in the text and pressing enter, so no need to click on"
    " the target field first), use the numeric index to indicate the target"
    " text field:"
    ' `{{"action_type": "input_text", "text": <text_input>,'
    ' "index": <target_index>}}`\n'
    '- Press the Enter key: `{{"action_type": "keyboard_enter"}}`\n'
    '- Navigate to the home screen: `{{"action_type": "navigate_home"}}`\n'
    '- Navigate back: `{{"action_type": "navigate_back"}}`\n'
    "- Scroll the screen or a scrollable UI element in one of the four"
    " directions, use the same numeric index as above if you want to scroll"
    " a specific UI element, leave it empty to scroll the whole screen:"
    ' `{{"action_type": "scroll", "direction": <up, down, left, right>,'
    ' "index": <optional_target_index>}}`\n'
    "- Open an app (nothing will happen if the app is not installed):"
    ' `{{"action_type": "open_app", "app_name": <name>}}`\n'
    '- Wait for the screen to update: `{{"action_type": "wait"}}`\n'
    "- Answer the user's question (use this when the sub-task goal is to"
    " report information you found, e.g. a date/count/name):"
    ' `{{"action_type": "answer", "text": "<answer_text>"}}`\n'
    "- Declare THIS SUB-TASK complete (i.e. its goal has been achieved):"
    ' `{{"action_type": "status", "goal_status": "complete"}}`\n'
    "- Declare THIS SUB-TASK infeasible (i.e. it cannot be achieved, e.g."
    " target element genuinely does not exist after exploring):"
    ' `{{"action_type": "status", "goal_status": "infeasible"}}`\n\n'
)

ACTOR_PROMPT_TEMPLATE = (
    ACTOR_PROMPT_PREFIX
    + "Your assigned sub-task (from the Planner) is:\n{sub_task}\n\n"
    "Here is what you have done so far FOR THIS SUB-TASK ONLY (empty if you"
    " just started it):\n{history}\n\n"
    "Here is a list of detailed information for the UI elements visible in"
    " the current screenshot (numeric indexes are consistent with the marks"
    " on the labeled screenshot):\n{ui_elements}\n\n"
    "Now output an action from the above list in the correct JSON format,"
    " following the reason why you do that. Your answer should look like:\n"
    "Reason: ...\nAction: {{\"action_type\":...}}\n\n"
    "Your Answer:\n"
)


@dataclasses.dataclass
class ActorStepOutput:
  """Result of one Actor invocation (one atomic action decision)."""

  reason: Optional[str] = None
  action_json_str: Optional[str] = None
  raw_text: Optional[str] = None
  raw_response: Any = None
  parse_ok: bool = True


class Actor:
  """Low-level Actor: sub-task + observation -> one atomic JSON action."""

  def __init__(self, llm: infer.MultimodalLlmWrapper):
    self.llm = llm

  def _build_prompt(
      self, sub_task_str: str, history: list[str], ui_elements: str
  ) -> str:
    history_str = (
        "\n".join(history)
        if history
        else "You just started this sub-task, no action performed yet."
    )
    return ACTOR_PROMPT_TEMPLATE.format(
        sub_task=sub_task_str,
        history=history_str,
        ui_elements=ui_elements if ui_elements else "Not available",
    )

  def act(
      self,
      sub_task_str: str,
      history: list[str],
      ui_elements: str,
      screenshots: list[np.ndarray],
  ) -> ActorStepOutput:
    """Calls the Actor LLM and parses the (reason, action) pair."""
    prompt = self._build_prompt(sub_task_str, history, ui_elements)
    raw_text, is_safe, raw_response = self.llm.predict_mm(prompt, screenshots)

    if not raw_response or is_safe is False:  # pylint: disable=g-bool-id-comparison
      return ActorStepOutput(
          reason=None,
          action_json_str=None,
          raw_text=raw_text,
          raw_response=raw_response,
          parse_ok=False,
      )

    reason, action = m3a_utils.parse_reason_action_output(raw_text)
    return ActorStepOutput(
        reason=reason,
        action_json_str=action,
        raw_text=raw_text,
        raw_response=raw_response,
        parse_ok=bool(reason and action),
    )
