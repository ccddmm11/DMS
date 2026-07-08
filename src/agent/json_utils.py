# DMS reproduction project — robust JSON extraction for LLM outputs.
#
# android_world's own `agent_utils.extract_json` uses a non-greedy regex
# (`\{.*?\}`) which only works for the FLAT, single-level action JSON used
# by M3A/T3A (e.g. `{"action_type": "click", "index": 5}`). Our Planner
# emits a NESTED JSON object (a "sub_plans" list of `{precondition, goal}`
# dicts), so the non-greedy regex stops at the first inner `}` and always
# fails to parse. This module implements a brace-matching extractor that
# correctly handles nesting and quoted strings, and falls back to
# `agent_utils.extract_json` for robustness.

from __future__ import annotations

import ast
import json
from typing import Any, Optional


def extract_balanced_json(s: str) -> Optional[dict[str, Any]]:
  """Extracts the first balanced top-level `{...}` object from `s`.

  Unlike a simple non-greedy regex, this correctly matches nested braces
  (e.g. a "sub_plans" list of dicts) and ignores braces that appear inside
  quoted strings.

  Args:
    s: Raw LLM text output, possibly with prose before/after the JSON.

  Returns:
    The parsed dict, or None if no valid JSON object could be found.
  """
  if not s:
    return None
  start = s.find("{")
  while start != -1:
    depth = 0
    in_string = False
    escape = False
    quote_char = ""
    for i in range(start, len(s)):
      c = s[i]
      if in_string:
        if escape:
          escape = False
        elif c == "\\":
          escape = True
        elif c == quote_char:
          in_string = False
        continue
      if c in ("\"", "'"):
        in_string = True
        quote_char = c
      elif c == "{":
        depth += 1
      elif c == "}":
        depth -= 1
        if depth == 0:
          candidate = s[start : i + 1]
          parsed = _try_parse(candidate)
          if parsed is not None:
            return parsed
          break  # Malformed candidate at this start; try the next '{'.
    start = s.find("{", start + 1)
  return None


def _try_parse(candidate: str) -> Optional[dict[str, Any]]:
  try:
    return json.loads(candidate)
  except (json.JSONDecodeError, ValueError):
    pass
  try:
    result = ast.literal_eval(candidate)
    return result if isinstance(result, dict) else None
  except (SyntaxError, ValueError):
    return None
