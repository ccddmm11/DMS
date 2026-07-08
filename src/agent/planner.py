# DMS reproduction project — Planner role.
#
# Reimplements the paper's canonical Planner (Sec 3.1 "Planning Phase" and
# Appendix H, Fig 10 "Planner Prompt"). We keep the same *semantics*:
#   - Decompose the (remaining) high-level task into <=5 short sub-plans.
#   - Each sub-plan is a `Precondition -> Goal` pair (the same structured
#     format used later as the memory key p=<p_pre, p_goal>, Sec 3.2.1),
#     so Baseline A / B / DMS can all share this Planner unmodified.
#   - Task history persists across replanning cycles and is fed back in.
#   - The Planner alone decides overall task completion / infeasibility.
#
# We express the Planner's output as compact JSON (parsed with
# android_world's own `agent_utils.extract_json`) instead of the paper's
# CodeAct tool-call convention, since our Actor operates over
# AndroidWorld's index-based JSON action space rather than raw
# tap(x, y) python code.

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np
from android_world.agents import infer

from src.agent.json_utils import extract_balanced_json

MAX_SUB_PLANS_PER_CYCLE = 5


@dataclasses.dataclass
class SubPlan:
  """A single sub-task, structured as <Precondition, Goal> (paper Sec 3.2.1)."""

  precondition: str
  goal: str

  def as_prompt_str(self) -> str:
    return f"Precondition: {self.precondition} Goal: {self.goal}"

  def to_dict(self) -> dict[str, str]:
    return {"precondition": self.precondition, "goal": self.goal}


@dataclasses.dataclass
class PlannerOutput:
  """Result of one Planner invocation."""

  done: bool = False
  goal_status: str = "in_progress"  # complete | infeasible | in_progress
  message: str = ""
  sub_plans: list[SubPlan] = dataclasses.field(default_factory=list)
  raw_text: Optional[str] = None
  raw_response: Any = None
  parse_ok: bool = True


PLANNER_PROMPT_TEMPLATE = (
    "You are an Android Task Planner. Your job is to create short, functional"
    " plans (1-5 steps) to achieve a user's goal on an Android device, then"
    " hand each step to a low-level Actor that will execute atomic UI"
    " actions to fulfill it.\n\n"
    "**Inputs you receive:**\n"
    "1. The user's overall goal.\n"
    "2. The current device state: a screenshot (with numeric index marks on"
    " visible UI elements) and a JSON list of detailed information for those"
    " UI elements.\n"
    "3. Complete task history: a record of ALL sub-plans that have been"
    " completed or failed so far in this session. This history persists"
    " across replanning cycles and is never lost.\n\n"
    "**Your task:** Given the goal, current state and task history, devise"
    " the next 1-5 functional steps to make progress towards the goal. Focus"
    " on WHAT to achieve, not HOW (the Actor decides the atomic taps/"
    " scrolls/typing). Planning fewer steps at a time improves accuracy,"
    " since the screen state can change after each step is executed. If the"
    " user's goal is a question (e.g. asking for a date, a count, a name),"
    " make sure one of your sub-plans has a goal telling the Actor to"
    " report/answer with the specific information once it has been"
    " located on screen.\n\n"
    "**Step format:** Each step in \"sub_plans\" MUST be an object with two"
    " fields:\n"
    '  - "precondition": the expected starting screen/state for this step'
    ' (use "None" if not critical, e.g. for the very first step of a new'
    " sequence).\n"
    '  - "goal": the concrete, single functional objective for this step.\n\n'
    "**Termination:** After your planned steps are executed, you will be"
    " invoked again with the new device state. At that point:\n"
    '  - If the OVERALL user goal is now complete, set "done": true and'
    ' "goal_status": "complete".\n'
    "  - If the overall goal is infeasible (e.g. missing information,"
    ' impossible request), set "done": true and "goal_status":'
    ' "infeasible".\n'
    '  - Otherwise, set "done": false and provide the next 1-5 items in'
    ' "sub_plans".\n\n'
    "**Output format:** Respond ONLY with a single JSON object (no other"
    " text before or after):\n"
    '{{"done": <bool>, "goal_status": "<complete|infeasible|in_progress>",'
    ' "message": "<short summary/answer/reason>", "sub_plans":'
    ' [{{"precondition": "...", "goal": "..."}}, ...]}}\n\n'
    "The current user goal/request is: {goal}\n\n"
    "Here is the complete task history so far (empty if this is the first"
    " planning cycle):\n{history}\n\n"
    "Here is a list of detailed information for the UI elements visible in"
    " the current screenshot (numeric indexes match the marks on the"
    " screenshot):\n{ui_elements}\n\n"
    "Now output your decision in the exact JSON format described above.\n"
    "Your Answer:\n"
)


class Planner:
  """High-level Planner: task -> <=5 sub-plans, or overall done/infeasible."""

  def __init__(self, llm: infer.MultimodalLlmWrapper):
    self.llm = llm

  def _build_prompt(
      self, goal: str, history: list[str], ui_elements: str
  ) -> str:
    history_str = (
        "\n".join(history) if history else "No sub-plans attempted yet."
    )
    return PLANNER_PROMPT_TEMPLATE.format(
        goal=goal,
        history=history_str,
        ui_elements=ui_elements if ui_elements else "Not available",
    )

  def plan(
      self,
      goal: str,
      history: list[str],
      ui_elements: str,
      screenshots: list[np.ndarray],
  ) -> PlannerOutput:
    """Calls the Planner LLM and parses its structured decision."""
    prompt = self._build_prompt(goal, history, ui_elements)
    raw_text, is_safe, raw_response = self.llm.predict_mm(prompt, screenshots)

    if not raw_response or is_safe is False:  # pylint: disable=g-bool-id-comparison
      return PlannerOutput(
          done=False,
          goal_status="in_progress",
          message="LLM call failed or was blocked by safety filter.",
          sub_plans=[],
          raw_text=raw_text,
          raw_response=raw_response,
          parse_ok=False,
      )

    parsed = extract_balanced_json(raw_text)
    if parsed is None or ("sub_plans" not in parsed and not parsed.get("done")):
      return PlannerOutput(
          done=False,
          goal_status="in_progress",
          message="Failed to parse Planner JSON output.",
          sub_plans=[],
          raw_text=raw_text,
          raw_response=raw_response,
          parse_ok=False,
      )

    sub_plans = []
    for item in parsed.get("sub_plans", []) or []:
      try:
        sub_plans.append(SubPlan(
            precondition=str(item.get("precondition", "None")),
            goal=str(item["goal"]),
        ))
      except (KeyError, TypeError):
        continue
    sub_plans = sub_plans[:MAX_SUB_PLANS_PER_CYCLE]

    return PlannerOutput(
        done=bool(parsed.get("done", False)),
        goal_status=str(parsed.get("goal_status", "in_progress")),
        message=str(parsed.get("message", "")),
        sub_plans=sub_plans,
        raw_text=raw_text,
        raw_response=raw_response,
        parse_ok=True,
    )
