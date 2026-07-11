#!/usr/bin/env python3
"""Aggregates one eval-harness run's `<condition>.jsonl` files into a single
per (condition, round) summary CSV (SR / SRR / MRR / tokens / steps /
memory size), plus (unless `--no_plots`) the report figures under
`<run_dir>/plots/`. Safe to run at any point mid-experiment (e.g. to check
progress) since it just re-reads whatever JSONL lines exist so far.

Usage:
  python scripts/aggregate_eval_results.py --run_dir results/eval/run_001
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import metrics
from src.eval import plots


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run_dir", required=True)
  parser.add_argument("--out_csv", default=None)
  parser.add_argument("--no_plots", action="store_true",
                       help="Skip generating the report PNG figures.")
  args = parser.parse_args()

  all_records = []
  jsonl_paths = sorted(glob.glob(os.path.join(args.run_dir, "*.jsonl")))
  if not jsonl_paths:
    print(f"No .jsonl files found in {args.run_dir}")
    return 1

  for path in jsonl_paths:
    records = metrics.load_records(path)
    print(f"  {os.path.basename(path)}: {len(records)} episodes")
    all_records.extend(records)

  summary = metrics.aggregate_by_round(all_records)
  out_csv = args.out_csv or os.path.join(args.run_dir, "summary_by_round.csv")
  metrics.write_csv(summary, out_csv)
  print(f"\nWrote {len(summary)} (condition, round) rows to {out_csv}\n")

  header = (f"{'condition':<15}{'round':>6}{'n':>5}{'SR':>7}"
            f"{'SubVfy':>8}{'EpReplay':>9}{'MRR':>7}{'steps':>8}"
            f"{'tokens':>9}{'bank':>7}{'errs':>6}")
  print(header)
  print("-" * len(header))
  for row in summary:
    def fmt(v, pct=False):
      if v is None:
        return "  n/a"
      return f"{v*100:5.1f}%" if pct else f"{v:6.1f}"
    print(
        f"{row['condition']:<15}{row['round_idx']:>6}{row['n_episodes']:>5}"
        f"{fmt(row['success_rate'], pct=True):>7}"
        f"{fmt(row['subtask_replay_verified_rate'], pct=True):>8}"
        f"{fmt(row['episode_success_after_replay_rate'], pct=True):>9}"
        f"{fmt(row['memory_reuse_rate'], pct=True):>7}"
        f"{fmt(row['avg_atomic_steps']):>8}"
        f"{fmt(row['avg_tokens']):>9}"
        f"{(str(row['memory_bank_size']) if row['memory_bank_size'] is not None else 'n/a'):>7}"
        f"{row['n_errors']:>6}"
    )

  # Paper Appendix E.2 / Figure 5b: SRR is a pooled cross-round temporal
  # metric (P(success at round t+1 | success at round t)), NOT a per-round
  # cell -- reported once per condition, separately from the table above.
  srr_summary = metrics.aggregate_srr_by_condition(all_records)
  srr_csv = os.path.join(args.run_dir, "summary_srr_by_condition.csv")
  metrics.write_csv(srr_summary, srr_csv)
  print(f"\nSuccess Retention Rate (SRR, paper Appendix E.2) -- wrote {srr_csv}\n")
  srr_header = f"{'condition':<15}{'n_tasks':>9}{'n_w/transitions':>16}{'SRR':>8}"
  print(srr_header)
  print("-" * len(srr_header))
  for row in srr_summary:
    srr = row["success_retention_rate"]
    srr_str = f"{srr*100:6.1f}%" if srr is not None else "    n/a"
    print(f"{row['condition']:<15}{row['n_tasks']:>9}{row['n_tasks_with_transitions']:>16}{srr_str:>8}")

  if not args.no_plots:
    plots_dir = os.path.join(args.run_dir, "plots")
    written = plots.generate_all_plots(summary, srr_summary, plots_dir)
    print(f"\nWrote {len(written)} figures to {plots_dir}/")
  return 0


if __name__ == "__main__":
  sys.exit(main())
