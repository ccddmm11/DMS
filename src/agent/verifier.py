# DMS reproduction project — Verifier role.
#
# Faithful reimplementation of the paper's Verifier (Appendix D, Fig 9):
# "History-First" verification with a "Visual Veto" as a secondary
# contradiction check. The Verifier decides whether a just-executed
# sub-task actually achieved its <Precondition, Goal> objective, which
# feeds Algorithm 1's `Execute(tau) -> R_sub` result used for both the
# Planner-Actor control flow (continue vs. replan) and, later, DMS's
# Survival Value / Bayesian risk bookkeeping (K_i strikes, F_i failures).
#
# Prompt text follows Fig. 9 of the paper near-verbatim, only substituting
# "Original Goal"/"Execution History" placeholders and JSON key names to
# match our (reason, action) log format instead of (Thought, Code).

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np
from android_world.agents import infer

from src.agent.json_utils import extract_balanced_json

VERIFIER_PROMPT_TEMPLATE = (
    "Role: You are an expert Android Task Verifier. Your job is to"
    " determine if the agent's execution history successfully achieved the"
    " user's goal.\n\n"
    "Input Information:\n"
    "1. Original Goal: The user's original objective for this sub-task.\n"
    "2. Execution History: The (Reason, Action) steps the agent claims it"
    " just performed. This is your PRIMARY source of truth.\n"
    "3. Final Screenshot: The ground truth screenshot. This is your"
    " SECONDARY check for contradictions.\n\n"
    "YOUR VERIFICATION LOGIC (History-First):\n"
    "1. Analyze History (Trust): Read the Execution History. Did the agent"
    " perform the logical actions required to complete the Original Goal?"
    ' (e.g. for "Save recording", did the agent click(\'Save\')?)\n'
    "2. Assume Success: If the history looks correct, your default verdict"
    ' is {{"verified_success": true}}.\n'
    "3. Visual Veto (Contradiction Check): Now, look at the Final"
    " Screenshot. Does this screenshot explicitly contradict the agent's"
    " claim of success?\n"
    '   - Contradiction (-> Fail): The screenshot shows an error message'
    ' (e.g. "Password incorrect").\n'
    "   - Contradiction (-> Fail): The screenshot shows the agent is in the"
    " wrong application.\n"
    '   - Contradiction (-> Fail): The goal was "Dismiss the \'OK\''
    ' dialog," but the screenshot clearly shows the \'OK\' dialog is still'
    " visible.\n"
    '   - NO Contradiction (-> Success): The goal was "Dismiss the \'OK\''
    ' dialog," and the screenshot shows the dialog is gone. This confirms'
    " the history.\n"
    '   - NO Contradiction (-> Success): The goal was "Click the \'Save\''
    ' button," and the screenshot shows the app has moved to a different'
    " screen. This confirms the history.\n\n"
    "Key Rule: You must default to True (success) if the history is sound"
    " AND the screenshot does not provide strong, undeniable proof of"
    " failure.\n\n"
    "Original Goal: {goal}\n\n"
    "Execution History:\n{history}\n\n"
    'Output Format: Respond ONLY with the JSON object: {{"verified_success":'
    ' <bool>, "reason": "<string>"}}\n\n'
    "Your Answer:\n"
)


@dataclasses.dataclass
class VerifierOutput:
  verified_success: bool = False
  reason: str = ""
  raw_text: Optional[str] = None
  raw_response: Any = None
  parse_ok: bool = True


class Verifier:
  """History-first + visual-veto verifier for sub-task completion."""

  def __init__(self, llm: infer.MultimodalLlmWrapper):
    self.llm = llm

  def _build_prompt(self, goal: str, history: list[str]) -> str:
    history_str = "\n".join(history) if history else "(no actions performed)"
    return VERIFIER_PROMPT_TEMPLATE.format(goal=goal, history=history_str)

  def verify(
      self,
      goal: str,
      history: list[str],
      final_screenshot: np.ndarray,
      default_on_parse_failure: bool = True,
  ) -> VerifierOutput:
    """Runs the History-First + Visual-Veto verification for a sub-task.

    Args:
      goal: The sub-task's goal text (its `p_goal`).
      history: List of "Reason: ...; Action: ..." strings actually executed.
      final_screenshot: Screenshot taken after the last executed action.
      default_on_parse_failure: If the LLM output cannot be parsed, whether
        to default to success=True (paper's "Key Rule": trust history absent
        strong contrary evidence) rather than failing the sub-task outright
        due to a formatting problem unrelated to actual task execution.

    Returns:
      VerifierOutput with the verdict and raw LLM artifacts.
    """
    prompt = self._build_prompt(goal, history)
    raw_text, is_safe, raw_response = self.llm.predict_mm(
        prompt, [final_screenshot]
    )

    if not raw_response or is_safe is False:  # pylint: disable=g-bool-id-comparison
      return VerifierOutput(
          verified_success=default_on_parse_failure,
          reason="Verifier LLM call failed or was blocked.",
          raw_text=raw_text,
          raw_response=raw_response,
          parse_ok=False,
      )

    parsed = extract_balanced_json(raw_text)
    if parsed is None or "verified_success" not in parsed:
      return VerifierOutput(
          verified_success=default_on_parse_failure,
          reason="Failed to parse Verifier JSON output.",
          raw_text=raw_text,
          raw_response=raw_response,
          parse_ok=False,
      )

    return VerifierOutput(
        verified_success=bool(parsed["verified_success"]),
        reason=str(parsed.get("reason", "")),
        raw_text=raw_text,
        raw_response=raw_response,
        parse_ok=True,
    )
