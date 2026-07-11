# DMS reproduction project — deterministic System UI gesture fast path.
#
# A 7B Actor repeatedly treats a physical pull-down from the launcher as a
# normal accessibility-tree "scroll up" operation, then declares it impossible
# because the launcher exposes no scrollable node. AndroidWorld implements the
# notification / Quick Settings shade as a `swipe` action (not `scroll`).
#
# This is deliberately narrow: it fires only from the launcher. An explicit
# Quick-Settings sub-goal uses that gesture, but an ambiguous generic
# "scroll up" sub-goal for a system toggle opens Settings directly. The latter
# is more reliable: the collapsed shade exposes generic, unlabeled switches,
# whereas Settings exposes an accessible "Network & internet" / Bluetooth row.

from __future__ import annotations

import re
from typing import Optional

from android_world.env import representation_utils

from src.agent import ui_utils

_LAUNCHER_PACKAGE_MARKERS = ("launcher", "quickstep")
_QUICK_SETTINGS_MARKERS = (
    "quick settings",
    "notification shade",
    "control center",
)
_SYSTEM_TOGGLE_MARKERS = (
    "wifi",
    "wi-fi",
    "bluetooth",
    "airplane mode",
    "flashlight",
    "do not disturb",
    "mobile data",
    "hotspot",
)
_SYSTEM_TOGGLE_PATHS = {
    "wifi": ("Network & internet", "Internet", "Wi-Fi"),
    "wi-fi": ("Network & internet", "Internet", "Wi-Fi"),
    "bluetooth": ("Connected devices", "Connection preferences", "Bluetooth"),
}
_GENERIC_SCROLL_UP_RE = re.compile(
    r"^\s*(?:scroll|swipe|pull)\s+up\s*(?:to|for)?\s*$",
    flags=re.IGNORECASE,
)


def _normalized(text: Optional[str]) -> str:
  return " ".join((text or "").lower().split())


def _foreground_package(env) -> str:
  try:
    activity = env.foreground_activity_name or ""
  except Exception:  # pylint: disable=broad-exception-caught
    return ""
  return activity.split("/", maxsplit=1)[0].lower()


def _is_launcher_foreground(env) -> bool:
  package = _foreground_package(env)
  return any(marker in package for marker in _LAUNCHER_PACKAGE_MARKERS)


def _mentions_any(text: str, markers: tuple[str, ...]) -> bool:
  return any(marker in text for marker in markers)


def quick_settings_fast_path_decision(
    env, sub_goal: str, task_goal: str
) -> Optional[str]:
  """Returns a deterministic system-navigation action when one is needed.

  ``"swipe_down"`` requires
  ``JSONAction(action_type="swipe", direction="down")``; ``"open_settings"``
  requires ``JSONAction(action_type="open_app", app_name="Settings")``.
  Returns None for all ordinary in-app scrolls and non-launcher contexts.
  """
  if not _is_launcher_foreground(env):
    return None

  normalized_sub_goal = _normalized(sub_goal)
  normalized_task_goal = _normalized(task_goal)
  explicit_quick_settings = _mentions_any(
      normalized_sub_goal, _QUICK_SETTINGS_MARKERS
  )
  generic_scroll_for_system_toggle = (
      bool(_GENERIC_SCROLL_UP_RE.fullmatch(normalized_sub_goal))
      and _mentions_any(normalized_task_goal, _SYSTEM_TOGGLE_MARKERS)
  )
  if explicit_quick_settings:
    return "swipe_down"
  if generic_scroll_for_system_toggle:
    return "open_settings"
  return None


def next_labeled_system_toggle_action(
    env,
    task_goal: str,
    ui_elements: list[representation_utils.UIElement],
    screen_size: tuple[int, int],
    completed_labels: tuple[str, ...] = (),
) -> Optional[tuple[dict[str, object], str]]:
  """Returns the next visible Settings-row click for a named system toggle.

  This is a reusable Android Settings adapter rather than a task-name rule:
  the declared task goal selects a stable, labeled route, and no action is
  returned until the relevant row is visible in Settings.
  """
  if "settings" not in _foreground_package(env):
    return None
  normalized_goal = _normalized(task_goal)
  path = next(
      (candidate for marker, candidate in _SYSTEM_TOGGLE_PATHS.items()
       if marker in normalized_goal),
      None,
  )
  if path is None:
    return None
  last_completed = completed_labels[-1] if completed_labels else None
  start_index = path.index(last_completed) + 1 if last_completed in path else 0
  for label in path[start_index:]:
    index = ui_utils.find_element_by_description(ui_elements, label, screen_size)
    if index is not None:
      # Use an intentionally invalid index, then let the shared re-grounder
      # resolve the unique actionable row or its label's bounding-box center.
      # The first text match can be a non-interactive child of that row.
      del index
      return ({"action_type": "click", "index": -1}, label)
  return None
