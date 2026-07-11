# DMS reproduction project — report chart generation.
#
# Consumes the per-(condition, round) summary produced by
# `metrics.aggregate_by_round` (+ the per-condition SRR summary from
# `metrics.aggregate_srr_by_condition`) and renders the figures the final
# technical report needs, loosely mirroring the paper's own presentation
# (Sec 4.3-4.6, Figures 4-8):
#   - Success Rate vs. execution round, one line per condition
#     (paper Figure 5a).
#   - Memory Reuse Rate vs. execution round, Baseline B vs. DMS
#     (paper Sec 4.3 / Figure 4).
#   - Token / atomic-step efficiency vs. execution round, one line per
#     condition (paper Sec 4.6 / Figure 7, "does memory reuse cut cost").
#   - Memory Bank size vs. execution round, Baseline B vs. DMS (paper
#     Sec 4.5 / Figure 6 -- Baseline B has no pruning so should keep
#     growing while DMS should plateau/oscillate once self-regulation
#     kicks in).
#   - Success Retention Rate (SRR) bar chart, one bar per condition
#     (paper Figure 5b).
#
# All functions are pure (matplotlib Agg backend, no display needed) and
# take the exact `list[dict]` shapes `metrics.py` already produces, so they
# can be exercised on tiny smoke-test data today and re-run unmodified on
# the full 3x29x5 experiment matrix later.

from __future__ import annotations

import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

_CONDITION_STYLE = {
    "zero_shot": {"label": "Baseline A (PA-Lite, zero-shot)", "color": "#888888", "marker": "o"},
    "static_memory": {"label": "Baseline B (Static Memory)", "color": "#1f77b4", "marker": "s"},
    "dms": {"label": "DMS (Ours)", "color": "#d62728", "marker": "^"},
}


def _style(condition: str) -> dict[str, Any]:
    return _CONDITION_STYLE.get(
        condition, {"label": condition, "color": None, "marker": "o"}
    )


def _by_condition(summary_by_round: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in summary_by_round:
        out.setdefault(row["condition"], []).append(row)
    for rows in out.values():
        rows.sort(key=lambda r: r["round_idx"])
    return out


def _save(fig, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_success_rate_by_round(summary_by_round: list[dict[str, Any]], out_path: str) -> None:
    """Paper Figure 5a analogue: SR (%) vs. execution round, per condition."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for condition, rows in _by_condition(summary_by_round).items():
        style = _style(condition)
        xs = [r["round_idx"] + 1 for r in rows]
        ys = [100.0 * r["success_rate"] if r["success_rate"] is not None else None for r in rows]
        ax.plot(xs, ys, marker=style["marker"], color=style["color"], label=style["label"])
    ax.set_xlabel("Execution Round")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate vs. Execution Round")
    ax.set_xticks(sorted({r["round_idx"] + 1 for r in summary_by_round}))
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_path)


def plot_memory_reuse_rate_by_round(summary_by_round: list[dict[str, Any]], out_path: str) -> None:
    """Paper Sec 4.3 analogue: MRR (%) vs. execution round, memory conditions only."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for condition, rows in _by_condition(summary_by_round).items():
        if condition == "zero_shot":
            continue  # No memory at all -> MRR is undefined/always 0.
        style = _style(condition)
        xs = [r["round_idx"] + 1 for r in rows]
        ys = [100.0 * r["memory_reuse_rate"] if r["memory_reuse_rate"] is not None else None
              for r in rows]
        ax.plot(xs, ys, marker=style["marker"], color=style["color"], label=style["label"])
    ax.set_xlabel("Execution Round")
    ax.set_ylabel("Memory Reuse Rate (%)")
    ax.set_title("Memory Reuse Rate vs. Execution Round")
    ax.set_xticks(sorted({r["round_idx"] + 1 for r in summary_by_round}))
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_path)


def plot_efficiency_by_round(summary_by_round: list[dict[str, Any]], out_path: str) -> None:
    """Paper Sec 4.6 analogue: avg tokens/episode (left) and avg atomic
    steps/episode (right) vs. execution round, per condition."""
    fig, (ax_tokens, ax_steps) = plt.subplots(1, 2, figsize=(11, 4))
    for condition, rows in _by_condition(summary_by_round).items():
        style = _style(condition)
        xs = [r["round_idx"] + 1 for r in rows]
        ax_tokens.plot(xs, [r["avg_tokens"] for r in rows], marker=style["marker"],
                        color=style["color"], label=style["label"])
        ax_steps.plot(xs, [r["avg_atomic_steps"] for r in rows], marker=style["marker"],
                       color=style["color"], label=style["label"])
    for ax, title, ylabel in (
        (ax_tokens, "Avg. Tokens / Episode", "Prompt + Completion Tokens"),
        (ax_steps, "Avg. Atomic Steps / Episode", "Atomic Steps"),
    ):
        ax.set_xlabel("Execution Round")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(sorted({r["round_idx"] + 1 for r in summary_by_round}))
        ax.grid(alpha=0.3)
    ax_tokens.legend()
    _save(fig, out_path)


def plot_memory_bank_size_by_round(summary_by_round: list[dict[str, Any]], out_path: str) -> None:
    """Paper Sec 4.5 analogue: Memory Bank size (entry count) vs. round,
    Baseline B (unbounded growth) vs. DMS (self-regulated)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for condition, rows in _by_condition(summary_by_round).items():
        if condition == "zero_shot":
            continue  # No memory bank.
        style = _style(condition)
        xs = [r["round_idx"] + 1 for r in rows]
        ys = [r["memory_bank_size"] for r in rows]
        ax.plot(xs, ys, marker=style["marker"], color=style["color"], label=style["label"])
    ax.set_xlabel("Execution Round")
    ax.set_ylabel("Memory Bank Size (# entries)")
    ax.set_title("Memory Bank Size vs. Execution Round")
    ax.set_xticks(sorted({r["round_idx"] + 1 for r in summary_by_round}))
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_path)


def plot_srr_bar(srr_by_condition: list[dict[str, Any]], out_path: str) -> None:
    """Paper Figure 5b analogue: one bar per condition for the pooled SRR."""
    fig, ax = plt.subplots(figsize=(5, 4))
    labels, values, colors = [], [], []
    for row in srr_by_condition:
        style = _style(row["condition"])
        labels.append(style["label"])
        values.append(100.0 * row["success_retention_rate"]
                       if row["success_retention_rate"] is not None else 0.0)
        colors.append(style["color"])
    bars = ax.bar(labels, values, color=colors)
    for bar, row in zip(bars, srr_by_condition):
        v = row["success_retention_rate"]
        text = f"{v*100:.1f}%" if v is not None else "n/a"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text,
                ha="center", va="bottom")
    ax.set_ylabel("Success Retention Rate (%)")
    ax.set_title("Stability: Success Retention Rate (SRR) by Condition")
    ax.set_ylim(0, 105)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_path)


def generate_all_plots(
    summary_by_round: list[dict[str, Any]],
    srr_by_condition: list[dict[str, Any]],
    out_dir: str,
) -> list[str]:
    """Generates every report figure into `out_dir`; returns the written paths."""
    paths = {
        "success_rate_by_round.png": lambda p: plot_success_rate_by_round(summary_by_round, p),
        "memory_reuse_rate_by_round.png": lambda p: plot_memory_reuse_rate_by_round(summary_by_round, p),
        "efficiency_by_round.png": lambda p: plot_efficiency_by_round(summary_by_round, p),
        "memory_bank_size_by_round.png": lambda p: plot_memory_bank_size_by_round(summary_by_round, p),
        "srr_by_condition.png": lambda p: plot_srr_bar(srr_by_condition, p),
    }
    written = []
    for filename, fn in paths.items():
        out_path = os.path.join(out_dir, filename)
        fn(out_path)
        written.append(out_path)
    return written
