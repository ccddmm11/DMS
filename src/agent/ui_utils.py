# DMS reproduction project — shared UI/state formatting helpers.
#
# Small re-implementation of the private helpers in
# android_world/agents/m3a.py (`_generate_ui_element_description` /
# `_generate_ui_elements_description_list`) so our own Planner/Actor/
# Verifier-based agents (Baseline A/B, DMS) can share one canonical
# "UI element list -> text description" formatter without depending on
# android_world's private module-level functions.

from __future__ import annotations

import logging
import time
from typing import Optional

from android_world.agents import m3a_utils
from android_world.env import interface
from android_world.env import representation_utils

_logger = logging.getLogger(__name__)


def describe_ui_element(
    ui_element: representation_utils.UIElement, index: int
) -> str:
  """Formats a single UI element into a compact JSON-like description."""
  desc = f'UI element {index}: {{"index": {index}, '
  if ui_element.text:
    desc += f'"text": "{ui_element.text}", '
  if ui_element.content_description:
    desc += f'"content_description": "{ui_element.content_description}", '
  if ui_element.hint_text:
    desc += f'"hint_text": "{ui_element.hint_text}", '
  if ui_element.tooltip:
    desc += f'"tooltip": "{ui_element.tooltip}", '
  desc += f'"is_clickable": {"True" if ui_element.is_clickable else "False"}, '
  desc += (
      '"is_long_clickable":'
      f' {"True" if ui_element.is_long_clickable else "False"}, '
  )
  desc += f'"is_editable": {"True" if ui_element.is_editable else "False"}, '
  if ui_element.is_scrollable:
    desc += '"is_scrollable": True, '
  if ui_element.is_focusable:
    desc += '"is_focusable": True, '
  desc += f'"is_selected": {"True" if ui_element.is_selected else "False"}, '
  desc += f'"is_checked": {"True" if ui_element.is_checked else "False"}, '
  return desc[:-2] + "}"


def describe_ui_elements(
    ui_elements: list[representation_utils.UIElement],
    screen_width_height_px: tuple[int, int],
) -> str:
  """Formats the full (filtered) UI element list for a Planner/Actor prompt."""
  lines = []
  for index, ui_element in enumerate(ui_elements):
    if m3a_utils.validate_ui_element(ui_element, screen_width_height_px):
      lines.append(describe_ui_element(ui_element, index))
  return "\n".join(lines)


def describe_target_element(
    ui_elements: list[representation_utils.UIElement], index: Optional[int]
) -> Optional[str]:
  """Best-effort short text descriptor for a UI element, used to re-ground
  an index-based action against a *different* UI element list later (e.g.
  when blindly replaying a stored DMS memory trajectory whose recorded
  indexes may no longer line up with the current screen's element order).
  """
  if index is None or index < 0 or index >= len(ui_elements):
    return None
  element = ui_elements[index]
  return element.text or element.content_description or element.hint_text or None


def find_element_by_description(
    ui_elements: list[representation_utils.UIElement],
    description: Optional[str],
    screen_width_height_px: tuple[int, int],
) -> Optional[int]:
  """Finds the index of a *currently visible* UI element best matching
  `description` (as produced by `describe_target_element`). Prefers an
  exact case-insensitive match on text/content_description/hint_text,
  falling back to substring containment. Returns None if no candidate
  element is found among the currently valid (visible) elements."""
  if not description:
    return None
  target = description.strip().lower()
  if not target:
    return None

  substring_match = None
  for index, element in enumerate(ui_elements):
    if not m3a_utils.validate_ui_element(element, screen_width_height_px):
      continue
    candidates = [element.text, element.content_description, element.hint_text]
    for candidate in candidates:
      if not candidate:
        continue
      candidate_lower = candidate.strip().lower()
      if candidate_lower == target:
        return index
      if substring_match is None and (
          target in candidate_lower or candidate_lower in target
      ):
        substring_match = index
  return substring_match


def get_robust_state(
    env: interface.AsyncEnv,
    min_elements: int = 2,
    max_retries: int = 4,
    retry_sleep_seconds: float = 1.5,
    reset_on_failure: bool = True,
) -> tuple[interface.State, bool]:
  """`env.get_state()` guarded against AndroidWorld's occasional a11y-tree
  degeneration bug (see `results/debug/zero_success_rate_diagnosis.md`):
  the accessibility-forwarder app's gRPC handshake can race on a fresh env
  session and return a single-node placeholder forest (just the root
  container, no icons/text/clickables) that then never refreshes for the
  rest of the episode -- silently blinding the agent for its whole
  duration. `env.get_state(wait_to_stabilize=True)` does NOT catch this:
  a frozen degenerate tree is trivially "stable" by that check's own
  definition (same element list across repeated polls).

  We instead detect degeneracy heuristically (suspiciously few UI
  elements) and retry with backoff; if it's still degenerate half-way
  through the retry budget we force one full `env.reset(go_home=True)`
  (which re-runs AndroidWorld's AccessibilityForwarder setup/broadcast
  sequence from scratch) before continuing to retry.

  Returns:
    (state, is_degenerate) -- `is_degenerate` is True if we exhausted all
    retries (incl. the forced reset) while still seeing a suspiciously
    empty UI, so callers can react (e.g. the Verifier must not trust a
    "success" claim grounded in a blind observation -- see call sites in
    `dms_agent_adapter.py` / `static_memory_agent.py` / `zero_shot_agent.py`).
  """
  state = env.get_state(wait_to_stabilize=False)
  attempts = 0
  did_reset = False
  while len(state.ui_elements) < min_elements and attempts < max_retries:
    attempts += 1
    _logger.warning(
        "get_robust_state: degenerate UI tree (%d elements), retry %d/%d.",
        len(state.ui_elements), attempts, max_retries,
    )
    time.sleep(retry_sleep_seconds)
    if (
        reset_on_failure
        and not did_reset
        and attempts >= max(1, max_retries // 2)
    ):
      did_reset = True
      _logger.warning(
          "get_robust_state: still degenerate after %d retries, forcing"
          " env.reset(go_home=True) to re-init the a11y forwarder.",
          attempts,
      )
      env.reset(go_home=True)
    state = env.get_state(wait_to_stabilize=False)

  is_degenerate = len(state.ui_elements) < min_elements
  if is_degenerate:
    _logger.warning(
        "get_robust_state: giving up after %d retries, still only %d UI"
        " element(s). Proceeding with a possibly-blind observation.",
        attempts, len(state.ui_elements),
    )
  return state, is_degenerate


def build_som_screenshot(
    raw_screenshot,
    ui_elements: list[representation_utils.UIElement],
    logical_screen_size: tuple[int, int],
    physical_frame_boundary: tuple[int, int, int, int],
    orientation: int,
):
  """Returns a copy of `raw_screenshot` with Set-of-Mark index boxes drawn."""
  som_screenshot = raw_screenshot.copy()
  for index, ui_element in enumerate(ui_elements):
    if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
      m3a_utils.add_ui_element_mark(
          som_screenshot,
          ui_element,
          index,
          logical_screen_size,
          physical_frame_boundary,
          orientation,
      )
  return som_screenshot
