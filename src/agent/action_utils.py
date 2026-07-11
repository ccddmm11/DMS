# DMS reproduction project — Actor action normalization + grounding repair
# (engineering improvement #4, adapted for a weak 7B backbone; NOT part of
# the paper's Algorithm 1).
#
# Two failure modes observed with the 7B Actor (see the peer reproductions'
# action_parser / android_actor and results/debug/validate_wifi_after_123.log):
#
#   1. Schema noise that HARD-CRASHES `json_action.JSONAction(**d)` and wastes
#      whole steps: e.g. `{"action_type":"scroll","direction":"up","index":""}`
#      (empty-string index -> `int("")` ValueError), `type` instead of
#      `input_text`, `swipe` instead of `scroll`, float/string coordinates.
#      `normalize_action_dict` sanitizes these before construction.
#
#   2. Grounding drift: the Actor names the right target in its Reason/Goal but
#      picks a wrong / out-of-range / non-clickable index. `reground_action`
#      re-grounds the index to the unique visible element matching the target
#      token, or falls back to a bbox-center coordinate click when the named
#      target has no clickable index (mirrors chencen's
#      `_maybe_correct_pointing_payload`).

from __future__ import annotations

import re
from typing import Any, Optional

from android_world.agents import m3a_utils
from android_world.env import representation_utils

_POINTING_ACTIONS = ("click", "long_press", "input_text")

_ACTION_TYPE_ALIASES = {
    "type": "input_text",
    "enter_text": "input_text",
    "fill_text": "input_text",
    "set_text": "input_text",
    "insert_text": "input_text",
    "tap": "click",
    "touch": "click",
    "press": "click",
    "swipe": "scroll",
    "drag": "scroll",
    "back": "navigate_back",
    "go_back": "navigate_back",
    "home": "navigate_home",
    "go_home": "navigate_home",
    "launch_app": "open_app",
    "start_app": "open_app",
    "finish": "status",
}

_DIRECTION_ALIASES = {
    "forward": "down",
    "backward": "up",
    "vertical": "down",
    "horizontal": "right",
    "top": "up",
    "bottom": "down",
}

# Only these keys are accepted by `json_action.JSONAction`.
_ALLOWED_KEYS = {
    "action_type", "index", "x", "y", "text", "direction",
    "goal_status", "app_name", "keycode", "clear_text",
}

_WIFI_ALIASES = {"wifi", "wi-fi"}


def _coerce_int(value: Any) -> Optional[int]:
  """Best-effort int coercion; returns None if not interpretable."""
  if value is None or isinstance(value, bool):
    return None
  if isinstance(value, int):
    return value
  if isinstance(value, float):
    return int(round(value))
  if isinstance(value, str):
    text = value.strip()
    if not text:
      return None
    try:
      return int(round(float(text)))
    except ValueError:
      nums = re.findall(r"-?\d+(?:\.\d+)?", text)
      if len(nums) == 1:
        return int(round(float(nums[0])))
  return None


def normalize_action_dict(raw: dict[str, Any]) -> dict[str, Any]:
  """Sanitizes a model-produced action dict so `JSONAction(**d)` won't crash
  on recoverable schema noise (empty index, aliases, loose coordinates)."""
  d = dict(raw or {})
  action_type = str(d.get("action_type", "")).strip().lower()
  action_type = _ACTION_TYPE_ALIASES.get(action_type, action_type)
  d["action_type"] = action_type

  # `content` is a common Actor synonym for `text`.
  if not d.get("text") and d.get("content"):
    d["text"] = d["content"]

  # Index: coerce or drop (empty/invalid string index is the crash source).
  if "index" in d:
    idx = _coerce_int(d.get("index"))
    if idx is None:
      d.pop("index")
    else:
      d["index"] = idx

  # Coordinates: coerce or drop.
  for key in ("x", "y"):
    if key in d:
      coord = _coerce_int(d.get(key))
      if coord is None:
        d.pop(key)
      else:
        d[key] = coord

  # JSONAction forbids index + x/y together: prefer index.
  if d.get("index") is not None and (d.get("x") is not None or d.get("y") is not None):
    d.pop("x", None)
    d.pop("y", None)

  if action_type == "scroll":
    direction = str(d.get("direction", "")).strip().lower()
    d["direction"] = _DIRECTION_ALIASES.get(direction, direction)

  if action_type == "status":
    gs = str(d.get("goal_status", "")).strip().lower()
    # legacy `finish` action often carries success/reason instead.
    if gs not in ("complete", "infeasible"):
      d["goal_status"] = "complete" if gs in ("", "success", "successful", "done", "true") else "infeasible"

  for key in list(d.keys()):
    if key not in _ALLOWED_KEYS:
      d.pop(key)
  return d


def _norm_text(text: Any) -> str:
  return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _element_texts(element: representation_utils.UIElement) -> list[str]:
  return [
      _norm_text(element.text),
      _norm_text(element.content_description),
      _norm_text(element.hint_text),
  ]


def _matches_token(element: representation_utils.UIElement, token: str) -> bool:
  token = _norm_text(token)
  if not token:
    return False
  alias_set = _WIFI_ALIASES if token in _WIFI_ALIASES else {token}
  for value in _element_texts(element):
    if not value:
      continue
    for alias in alias_set:
      if alias == value or alias in value or value in alias:
        return True
  return False


def _is_actionable(
    element: representation_utils.UIElement,
    require_clickable: bool,
    require_editable: bool,
) -> bool:
  if require_editable and not element.is_editable:
    return False
  if require_clickable and not (
      element.is_clickable or element.is_long_clickable
  ):
    return False
  return True


def _quoted_tokens(text: str) -> list[str]:
  return [m.strip() for m in re.findall(r"['\"]([^'\"]+)['\"]", text or "") if m.strip()]


def _extract_goal(goal_or_subtask: str) -> str:
  match = re.search(
      r"Goal\s*:\s*(.+?)(?:$|\n)", goal_or_subtask, flags=re.IGNORECASE | re.DOTALL
  )
  return match.group(1).strip() if match else (goal_or_subtask or "").strip()


def _target_token(
    goal: str,
    reason: str,
    current_target: Optional[representation_utils.UIElement],
) -> Optional[str]:
  """Best-effort target label from quoted text in the goal/reason, then a few
  well-known Settings tokens, then the currently-pointed element's own label."""
  goal_text = _extract_goal(goal)
  for candidate in _quoted_tokens(goal_text) + _quoted_tokens(reason):
    normalized = _norm_text(candidate)
    if normalized:
      return normalized
  combined = _norm_text(" ".join(filter(None, [goal_text, reason])))
  for token in (
      "network & internet", "wi-fi", "wifi", "bluetooth", "settings",
      "airplane mode", "location", "contacts", "phone",
  ):
    if token in combined:
      return _norm_text(token)
  if current_target is not None:
    for value in _element_texts(current_target):
      if value:
        return value
  return None


def reground_action(
    action: dict[str, Any],
    ui_elements: list[representation_utils.UIElement],
    reason: str,
    goal: str,
    screen_size: tuple[int, int],
) -> tuple[dict[str, Any], Optional[str]]:
  """Re-grounds an index-based pointing action to the visible element that
  matches the Actor's stated target, or to a bbox-center coordinate click when
  the target label has no clickable index. Returns (action, note)."""
  action_type = action.get("action_type")
  if action_type not in _POINTING_ACTIONS:
    return action, None

  has_index = action.get("index") is not None
  has_xy = action.get("x") is not None and action.get("y") is not None
  if has_xy and not has_index:
    return action, None  # explicit coordinates: trust the Actor.

  n = len(ui_elements)
  require_editable = action_type == "input_text"
  require_clickable = action_type in ("click", "long_press")

  def visible(i: int) -> bool:
    return 0 <= i < n and m3a_utils.validate_ui_element(
        ui_elements[i], screen_size
    )

  idx = action.get("index")
  current = ui_elements[idx] if (has_index and 0 <= idx < n) else None
  token = _target_token(goal, reason, current)

  current_ok = (
      has_index
      and visible(idx)
      and _is_actionable(ui_elements[idx], require_clickable, require_editable)
  )
  if current_ok and (not token or _matches_token(ui_elements[idx], token)):
    return action, None

  if not token:
    return action, None

  matches = [i for i in range(n) if visible(i) and _matches_token(ui_elements[i], token)]
  actionable = [
      i for i in matches
      if _is_actionable(ui_elements[i], require_clickable, require_editable)
  ]

  new_action = dict(action)
  if len(actionable) == 1:
    new_action["index"] = actionable[0]
    new_action.pop("x", None)
    new_action.pop("y", None)
    return new_action, (
        f"re-grounded {action_type} index {action.get('index')!r} -> {actionable[0]}"
        f" (unique visible '{token}' target)."
    )

  # Non-clickable label / control with a unique match: coordinate fallback
  # (only for click/long_press; input_text needs a real editable index).
  if require_clickable and len(matches) == 1:
    element = ui_elements[matches[0]]
    if element.bbox_pixels is not None:
      cx, cy = element.bbox_pixels.center
      new_action.pop("index", None)
      new_action["x"] = int(round(cx))
      new_action["y"] = int(round(cy))
      return new_action, (
          f"re-grounded {action_type} onto bbox-center coordinates of the unique"
          f" visible '{token}' target (no clickable index exposed)."
      )

  return action, None
