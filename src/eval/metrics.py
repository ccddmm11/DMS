# DMS reproduction project — episode-level records + round-level aggregation.
#
# One `EpisodeRecord` is appended (as one JSON line) per (condition, round,
# task) cell, immediately after that episode finishes -- so a crashed/killed
# worker never loses more than the single in-flight episode, and
# `load_completed_cells` lets a restarted worker skip everything already
# recorded (resume-safe).
#
# `aggregate_by_round` turns a flat list of `EpisodeRecord` into the per
# (condition, round) summary table the report needs:
#   - SR    (Success Rate, paper Appendix E.1):
#       mean(success) over that cell's episodes.
#   - subtask_replay_verified_rate (NOT the paper's SRR): how often a
#       replayed sub-task passed the LLM Verifier.
#   - episode_success_after_replay_rate: ground-truth AndroidWorld task
#       success among episodes that attempted at least one replay.
#   - MRR   (Memory Reuse Rate, paper Appendix E.3):
#       sum(replayed_actions) / sum(replayed+fresh actions) over that cell.
#   - avg atomic steps, avg tokens (prompt+completion), avg wall-clock.
#   - memory bank size at the end of that round (last episode's snapshot).
#
# `success_retention_rate` (SRR, paper Appendix E.2) is a DIFFERENT,
# temporal-sequence metric -- P(x_{t+1}=1 | x_t=1) pooled over every task's
# across-round outcome sequence for one condition -- and is computed
# separately by `aggregate_srr_by_condition` (it needs the FULL multi-round
# history per task, not a single (condition, round) cell).

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
  # Explicit labels prevent a verifier-approved replay from being confused
  # with end-to-end task success.
  subtask_replay_attempts: Optional[int] = None
  subtask_replay_verified: Optional[int] = None
  episode_replay_attempted: Optional[bool] = None
  episode_success_after_replay: Optional[bool] = None
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
    total_subtask_replay_attempts = sum(
        r.get("subtask_replay_attempts", r.get("replay_attempts")) or 0
        for r in rows
    )
    total_subtask_replay_verified = sum(
        r.get("subtask_replay_verified", r.get("replay_successes")) or 0
        for r in rows
    )
    replayed_episode_outcomes = [
        r.get("episode_success_after_replay", r.get("success"))
        for r in rows
        if r.get("episode_replay_attempted", (r.get("replay_attempts") or 0) > 0)
    ]
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
        "subtask_replay_verified_rate": _safe_div(
            total_subtask_replay_verified, total_subtask_replay_attempts
        ),
        "episode_success_after_replay_rate": (
            sum(bool(v) for v in replayed_episode_outcomes)
            / len(replayed_episode_outcomes)
            if replayed_episode_outcomes else None
        ),
        "memory_reuse_rate": _safe_div(total_replayed_actions, total_actions),
        "avg_atomic_steps": sum(r["atomic_steps"] for r in rows) / n if n else None,
        "avg_tokens": sum(total_tokens) / n if n else None,
        "avg_wall_clock_seconds": sum(r["wall_clock_seconds"] for r in rows) / n if n else None,
        "memory_bank_size": bank_sizes[-1] if bank_sizes else None,
        "memories_pruned_by_strikes_cum": sum(pruned) if pruned else None,
        "n_errors": sum(1 for r in rows if r.get("error")),
    })
  return summary


def success_retention_rate(sequences: list[list[bool]]) -> Optional[float]:
  """Paper Appendix E.2's Success Retention Rate (SRR):

    SRR = P(x_{t+1}=1 | x_t=1)
        = sum_t I(x_t=1 and x_{t+1}=1) / sum_t I(x_t=1)

  computed over one or more outcome sequences `x = {x_1, ..., x_N}` (here,
  one sequence per task = that task's success/fail outcome in round 1, 2,
  ..., N, in round order). Consecutive-pair transitions from ALL provided
  sequences are pooled into a single numerator/denominator, matching
  Figure 5b's single SRR number per condition/model (not one SRR per task).
  Sequences shorter than 2 rounds contribute no transitions and are
  effectively ignored. Returns None if no task had >=2 rounds of data (or
  no task ever succeeded, i.e. the denominator is 0).
  """
  numerator = 0
  denominator = 0
  for seq in sequences:
    for t in range(len(seq) - 1):
      if seq[t]:
        denominator += 1
        if seq[t + 1]:
          numerator += 1
  return _safe_div(numerator, denominator)


def aggregate_srr_by_condition(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Groups by condition, builds each task's round-ordered success sequence,
  and computes the pooled SRR (paper Appendix E.2 / Figure 5b) per
  condition. Also returns `n_tasks_with_transitions` (tasks that had >=2
  rounds recorded) for transparency."""
  by_condition: dict[str, dict[str, dict[int, bool]]] = {}
  for r in records:
    cond = r["condition"]
    task = r["task_name"]
    by_condition.setdefault(cond, {}).setdefault(task, {})[r["round_idx"]] = bool(r["success"])

  summary = []
  for condition, per_task in sorted(by_condition.items()):
    sequences = []
    for task, round_to_success in per_task.items():
      ordered_rounds = sorted(round_to_success.keys())
      sequences.append([round_to_success[r] for r in ordered_rounds])
    n_with_transitions = sum(1 for seq in sequences if len(seq) >= 2)
    summary.append({
        "condition": condition,
        "n_tasks": len(sequences),
        "n_tasks_with_transitions": n_with_transitions,
        "success_retention_rate": success_retention_rate(sequences),
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
