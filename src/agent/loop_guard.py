# DMS reproduction project — lightweight engineering guardrails against two
# failure patterns observed empirically with a 7B Planner-Actor-Verifier
# loop (see results/debug/zero_success_rate_diagnosis.md):
#
#   1. The Actor can "trivially" satisfy a passive/observational sub-goal
#      (e.g. "scroll down to find X") without taking a single real action,
#      whenever X already happens to be visible on the starting screen --
#      technically compliant with its own "strict literal execution"
#      instruction (Sec `ACTOR_PROMPT_PREFIX`'s anti-overreach rule), but
#      this starves the Verifier's History-First check of any real evidence
#      of interaction, letting it default to "success" with nothing to show
#      for it.
#   2. The Planner can then get stuck re-issuing a (near-)identical
#      sub-goal cycle after cycle once that sub-goal is marked COMPLETED,
#      instead of advancing to the next concrete interaction (e.g. clicking
#      the element it just "found"), because nothing in `task_history`
#      signals that no real progress was made -- from the Planner's point
#      of view, the screen looks the same and the history says "COMPLETED",
#      so it just repeats itself.
#
# Neither of these is part of the paper's Algorithm 1 -- they are honest-
# history + repetition-detection nudges we add on top for a materially
# weaker (7B) backbone than the paper's. Both act purely by enriching what
# the Planner/Verifier *see* in text (never by hard-overriding a verdict),
# to stay as close as possible to the paper's original Planner/Verifier
# decision logic while closing an exploit that a strict-literal-execution
# Actor + a History-First Verifier otherwise fall into together.

from __future__ import annotations

import json
import re
from typing import Optional

# Sub-goal verbs that imply a real state-changing UI interaction. A sub-goal
# built around one of these that the Actor declares COMPLETE with ZERO actions
# taken is almost always the "found it = done" exploit (issue #6), not a
# genuinely already-satisfied state -- so the Verifier's "success" is vetoed
# and the sub-task is failed to force a concrete interaction / replan. Passive
# verbs ("find", "locate", "scroll", "check", "verify") are deliberately
# EXCLUDED: those can legitimately need zero actions, and their repeat loops
# are already handled by `RepetitionBreaker`.
_INTERACTION_VERBS = (
    "click", "tap", "open", "launch", "toggle", "switch on", "switch off",
    "turn on", "turn off", "enable", "disable", "type", "enter", "input",
    "fill", "select", "add", "create", "delete", "remove", "send", "save",
    "set", "press", "choose", "play", "pause", "record", "install",
    "uninstall", "dismiss", "submit", "confirm", "rename", "move", "copy",
    "download", "upload", "share", "edit", "update", "attach", "mark",
)

_INTERACTION_VERB_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(verb) for verb in _INTERACTION_VERBS) + r")\b",
    flags=re.IGNORECASE,
)


def goal_requires_interaction(goal: Optional[str]) -> bool:
  """True iff the sub-goal text implies a concrete state-changing interaction
  (see `_INTERACTION_VERBS`). Used to veto a zero-action "success" (issue #6)."""
  if not goal:
    return False
  return bool(_INTERACTION_VERB_RE.search(goal))


_STALL_WARNING_TEMPLATE = (
    '[SYSTEM NOTE] The sub-goal "{goal}" was just marked COMPLETED, but no'
    " interactive action (click/type/toggle/etc.) was performed to reach"
    " that state -- it was already true on the starting screen, so nothing"
    " actually changed. Do NOT issue this same (or a near-identical)"
    " sub-goal again -- that would repeat the exact same non-progress."
    " Instead, specify the next CONCRETE interaction needed (e.g. \"click"
    " the '...' option\", \"toggle the '...' switch\") to actually move the"
    " task forward."
)


def annotate_zero_action_completion(reason: Optional[str], goal_status: str) -> str:
  """History string for a sub-task `status` declaration made on the Actor's
  very FIRST attempt for that sub-task (i.e. with no preceding real
  action). Makes this fact explicit in the text fed to the Verifier,
  instead of the generic "declared sub-task complete", so its History-First
  check has an honest signal to judge whether "no interaction needed" is
  actually plausible for this goal."""
  return (
      f"Reason: {reason} Action: declared sub-task {goal_status} WITHOUT"
      " performing any action first (this was the Actor's first response"
      " for this sub-task -- it judged the goal already satisfied by the"
      " starting screen alone, with zero clicks/scrolls/etc. taken)."
  )


def stall_warning_if_zero_action(
    goal: str, actions_taken: int
) -> Optional[str]:
  """Returns a `task_history` entry warning the Planner about a just-
  completed sub-goal that required zero real actions to reach (see
  `annotate_zero_action_completion`), or `None` if at least one real action
  was taken this cycle. Meant to be appended to `task_history` right after
  such a completion, so the NEXT Planner call sees an explicit nudge
  against repeating the same passive instruction."""
  if actions_taken > 0:
    return None
  return _STALL_WARNING_TEMPLATE.format(goal=goal)


class RepetitionBreaker:
  """A prompt-only nudge (`stall_warning_if_zero_action`) turned out, in
  practice, to NOT reliably stop a 7B Planner from re-issuing the exact
  same zero-action sub-goal cycle after cycle (see
  `results/debug/zero_success_rate_diagnosis.md`'s regression trace: the
  Planner kept repeating "scroll up to reveal more options" 10 times in a
  row even with the `[SYSTEM NOTE]` warning present in its history). This
  is a hard, code-level circuit breaker for that case: it tracks
  consecutive sub-task completions with (a) zero real actions taken and
  (b) (near-)identical goal text, and reports when the SAME passive/no-op
  cycle would repeat for the `max_repeats`-th time in a row -- signalling
  the caller to convert what would otherwise be a "COMPLETED" verdict into
  a hard FAILURE instead, which forces `PlanFailed <- TRUE` (Algorithm 1)
  and a genuine replan, rather than burning the whole step budget in an
  unproductive loop.
  """

  def __init__(self, max_repeats: int = 2):
    self.max_repeats = max_repeats
    self._last_goal: Optional[str] = None
    self._streak = 0

  def reset(self) -> None:
    """Call at the start of each new episode/task."""
    self._last_goal = None
    self._streak = 0

  def record_and_check(self, goal: str, actions_taken: int) -> bool:
    """Call once per sub-task completion (BEFORE honoring a "success"
    verdict). Returns True iff this completion should instead be forced
    into a FAILURE to break a detected stall loop."""
    if actions_taken > 0:
      self._last_goal = None
      self._streak = 0
      return False

    normalized = " ".join(goal.strip().lower().split())
    if normalized == self._last_goal:
      self._streak += 1
    else:
      self._last_goal = normalized
      self._streak = 1

    if self._streak >= self.max_repeats:
      self._last_goal = None
      self._streak = 0
      return True
    return False


class StagnantActionBreaker:
  """Breaks repeated identical actions against an unchanged UI state.

  This is the live counterpart of memory write hygiene's stagnant-trajectory
  check. It prevents an Actor from consuming a full local step budget on the
  same click/scroll while preserving the same policy for all conditions.
  """

  def __init__(self, max_repeats: int = 2):
    self.max_repeats = max_repeats
    self._last_key: Optional[tuple[str, str]] = None
    self._streak = 0

  def reset(self) -> None:
    self._last_key = None
    self._streak = 0

  def record_and_check(
      self, state_signature: str, action: dict
  ) -> bool:
    """Returns True when the repeated action should force a replan."""
    key = (state_signature, json.dumps(action, sort_keys=True))
    if key == self._last_key:
      self._streak += 1
    else:
      self._last_key = key
      self._streak = 1
    if self._streak >= self.max_repeats:
      self.reset()
      return True
    return False
