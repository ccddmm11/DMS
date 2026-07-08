# DMS reproduction project — episode-level records + round-level aggregation.
#
# One `EpisodeRecord` is appended (as one JSON line) per (condition, round,
# task) cell, immediately after that episode finishes -- so a crashed/killed
# worker never loses more than the single in-flight episode, and
# `load_completed_cells` lets a restarted worker skip everything already
# recorded (resume-safe).
#
# `aggregate` turns a flat list of `EpisodeRecord` into the per
# (condition, round) summary table the report needs:
#   - SR   (Success Rate):            mean(success) over that cell's episodes
#   - SRR  (Successful Replay Rate):  replay_successes / replay_attempts
#   - MRR  (Memory Reuse Rate):       replayed_actions / (replayed+fresh actions)
#   - avg atomic steps, avg tokens (prompt+completion), avg wall-clock
#   - memory bank size at the end of that round (last episode's snapshot)

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Optional


@dataclasses.dataclass
class EpisodeRecord:
  """One (condition, round, task) episode's full outcome + usage stats."""

  condition: str
  round_idx: int
  task_name: str
  difficulty: str
  tags: list[str]
  complexity: float
  goal: str
  seed: int
  max_steps: int
  success: bool
  agent_done: bool
  atomic_steps: int
  wall_clock_seconds: float
  timestamp: str

  planner_calls: int = 0
  actor_calls: int = 0
  verifier_calls: int = 0
  replan_cycles: int = 0
  prompt_tokens: int = 0
  completion_tokens: int = 0

  # Memory-mechanism counters; None for Baseline A (zero_shot), which has
  # no memory at all. 0 (not None) for Baseline B / DMS when simply unused
  # this episode.
  retrieval_attempts: Optional[int] = None
  retrieval_hits: Optional[int] = None
  replay_attempts: Optional[int] = None
  replay_successes: Optional[int] = None
  replayed_actions_executed: Optional[int] = None
  fresh_actions_executed: Optional[int] = None
  memory_reuse_rate: Optional[float] = None
  memory_bank_size_after: Optional[int] = None

  # DMS-only self-regulation counters (None for Baseline A/B).
  mutation_attempts: Optional[int] = None
  memories_created: Optional[int] = None
  memories_replaced: Optional[int] = None
  memories_pruned_by_strikes: Optional[int] = None
  regulation_action: Optional[str] = None  # "prune" | "expand" | "noop" | None

  error: Optional[str] = None

  def to_json_line(self) -> str:
    return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


def append_record(path: str, record: EpisodeRecord) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "a", encoding="utf-8") as f:
    f.write(record.to_json_line() + "\n")


def load_records(path: str) -> list[dict[str, Any]]:
  if not os.path.exists(path):
    return []
  records = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        records.append(json.loads(line))
  return records


def load_completed_cells(path: str) -> set[tuple[int, str]]:
  """Returns the set of (round_idx, task_name) cells already recorded, so a
  restarted worker can skip them (resume-safe checkpointing)."""
  return {(r["round_idx"], r["task_name"]) for r in load_records(path)}


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
  if not denominator:
    return None
  return numerator / denominator


def aggregate_by_round(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Groups by (condition, round_idx) and computes SR/SRR/MRR/efficiency."""
  groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
  for r in records:
    key = (r["condition"], r["round_idx"])
    groups.setdefault(key, []).append(r)

  summary = []
  for (condition, round_idx), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    n = len(rows)
    success_count = sum(1 for r in rows if r["success"])
    total_tokens = [r["prompt_tokens"] + r["completion_tokens"] for r in rows]
    total_replay_attempts = sum(r.get("replay_attempts") or 0 for r in rows)
    total_replay_successes = sum(r.get("replay_successes") or 0 for r in rows)
    total_replayed_actions = sum(r.get("replayed_actions_executed") or 0 for r in rows)
    total_fresh_actions = sum(r.get("fresh_actions_executed") or 0 for r in rows)
    total_actions = total_replayed_actions + total_fresh_actions
    bank_sizes = [r["memory_bank_size_after"] for r in rows
                  if r.get("memory_bank_size_after") is not None]
    pruned = [r.get("memories_pruned_by_strikes") for r in rows
              if r.get("memories_pruned_by_strikes") is not None]

    summary.append({
        "condition": condition,
        "round_idx": round_idx,
        "n_episodes": n,
        "success_rate": success_count / n if n else None,
        "successful_replay_rate": _safe_div(total_replay_successes, total_replay_attempts),
        "memory_reuse_rate": _safe_div(total_replayed_actions, total_actions),
        "avg_atomic_steps": sum(r["atomic_steps"] for r in rows) / n if n else None,
        "avg_tokens": sum(total_tokens) / n if n else None,
        "avg_wall_clock_seconds": sum(r["wall_clock_seconds"] for r in rows) / n if n else None,
        "memory_bank_size": bank_sizes[-1] if bank_sizes else None,
        "memories_pruned_by_strikes_cum": sum(pruned) if pruned else None,
        "n_errors": sum(1 for r in rows if r.get("error")),
    })
  return summary


def write_csv(rows: list[dict[str, Any]], path: str) -> None:
  if not rows:
    return
  os.makedirs(os.path.dirname(path), exist_ok=True)
  fieldnames = list(rows[0].keys())
  import csv
  with open(path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)
