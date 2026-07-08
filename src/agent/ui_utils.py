# DMS reproduction project — shared UI/state formatting helpers.
#
# Small re-implementation of the private helpers in
# android_world/agents/m3a.py (`_generate_ui_element_description` /
# `_generate_ui_elements_description_list`) so our own Planner/Actor/
# Verifier-based agents (Baseline A/B, DMS) can share one canonical
# "UI element list -> text description" formatter without depending on
# android_world's private module-level functions.

from __future__ import annotations

from typing import Optional

from android_world.agents import m3a_utils
from android_world.env import representation_utils


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
