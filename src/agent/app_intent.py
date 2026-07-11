# DMS reproduction project — deterministic "open app" fast path (engineering
# improvement #3, adapted for a weak 7B backbone; NOT part of the paper's
# Algorithm 1).
#
# Empirically (see results/debug/zero_success_rate_diagnosis.md and the peer
# reproductions), a 7B Planner-Actor often fails the trivial "get into the
# right app" prefix of a task: it decomposes "Take one photo" into "Open the
# Photos gallery" (wrong app), or it keeps re-issuing "open Settings" while
# already inside Settings, burning the whole step budget before any real
# interaction happens. Both peer repros solve this the same way we do here:
#
#   1. If a sub-goal is an explicit, single-clause "open/launch <app>"
#      instruction AND that app maps to a known AndroidWorld launch activity,
#      execute a deterministic `open_app` action directly (no LLM Actor call),
#      instead of letting the Actor guess taps.
#   2. If the target app is ALREADY in the foreground, treat the sub-goal as
#      trivially satisfied (veto) rather than re-opening / churning the Actor.
#
# This only ever fires for unambiguous open-app sub-goals (multi-clause goals
# like "open task.html in Chrome" are deliberately skipped), so it cannot
# silently swallow real interaction steps.

from __future__ import annotations

import re
from typing import Optional

from android_world.env import adb_utils

# Leading verb(s) that denote an "enter this app" intent.
_OPEN_VERB_RE = re.compile(
    r"^\s*(?:please\s+)?"
    r"(?:open|launch|start|re-?open|go\s+to|navigate\s+to|open\s+up|switch\s+to)\s+"
    r"(?:the\s+)?(.+?)\s*$",
    re.IGNORECASE,
)
# Trailing filler after the app name ("... app", "... application", "... screen").
_TRAILING_NOISE_RE = re.compile(
    r"\s*\b(?:application|app|screen|page|menu)\b\.?\s*$", re.IGNORECASE
)


def extract_open_app_target(goal: str) -> Optional[str]:
  """Returns the app name iff `goal` is an unambiguous "open <app>" sub-goal
  that maps to a known AndroidWorld launch activity, else None.

  Deliberately conservative: multi-clause goals (containing "and"/"then"/a
  comma/semicolon) are rejected so we never fast-path away a step that also
  requires real in-app interaction (e.g. "open Chrome and click the button").
  """
  if not goal:
    return None
  text = goal.strip()
  low = text.lower()
  if " and " in low or " then " in low or "," in text or ";" in text:
    return None

  match = _OPEN_VERB_RE.match(text)
  if not match:
    return None

  candidate = match.group(1).strip().strip("\"'.")
  candidate = _TRAILING_NOISE_RE.sub("", candidate).strip().strip("\"'.")
  if not candidate:
    return None
  # Only accept names AndroidWorld can actually launch (its own app registry).
  if adb_utils.get_adb_activity(candidate) is None:
    return None
  return candidate


def target_app_package(app_name: str) -> Optional[str]:
  """Maps an app name to its Android package (via AndroidWorld's registry)."""
  activity = adb_utils.get_adb_activity(app_name)
  if not activity:
    return None
  return activity.split("/")[0]


def current_app_package(env) -> Optional[str]:
  """Best-effort current foreground package, or None if it can't be read."""
  try:
    activity = env.foreground_activity_name
  except Exception:  # pylint: disable=broad-exception-caught
    return None
  if not activity:
    return None
  return activity.split("/")[0]


def is_target_app_foreground(env, app_name: str) -> bool:
  """True iff `app_name`'s package is currently in the foreground."""
  target = target_app_package(app_name)
  if not target:
    return False
  return current_app_package(env) == target


def open_app_fast_path_decision(env, goal: str) -> Optional[tuple[str, str]]:
  """Classifies a sub-goal for the deterministic open-app fast path.

  Returns:
    None                      -- not an unambiguous open-app sub-goal; the
                                 caller should proceed with the normal Actor.
    ("already", app_name)     -- target app is already in the foreground;
                                 the sub-goal can be marked done with no action.
    ("open", app_name)        -- caller should execute open_app(app_name).
  """
  app = extract_open_app_target(goal)
  if app is None:
    return None
  if is_target_app_foreground(env, app):
    return ("already", app)
  return ("open", app)


def initial_task_app_fast_path_decision(
    env, task_apps: list[str], has_started_task_navigation: bool
) -> Optional[tuple[str, str]]:
  """Returns a generic first-navigation action from task-suite metadata.

  AndroidWorld task metadata identifies the app family that owns the task.
  When an episode begins at the launcher, opening that declared primary app is
  a deterministic, task-agnostic setup action. This prevents a weak Planner
  from substituting a related but incorrect app (for example, Photos for a
  Camera task). Multi-app tasks intentionally use only their first declared
  app as the initial entry point; subsequent cross-app routing remains under
  the Planner's control.
  """
  if has_started_task_navigation or not task_apps or not _is_launcher_foreground(env):
    return None
  app_name = task_apps[0]
  if adb_utils.get_adb_activity(app_name) is None:
    return None
  if is_target_app_foreground(env, app_name):
    return ("already", app_name)
  return ("open", app_name)


def _is_launcher_foreground(env) -> bool:
  package = current_app_package(env) or ""
  return "launcher" in package.lower() or "quickstep" in package.lower()
