#!/usr/bin/env python3
"""Aggregates one eval-harness run's `<condition>.jsonl` files into a single
per (condition, round) summary CSV (SR / SRR / MRR / tokens / steps /
memory size). Safe to run at any point mid-experiment (e.g. to check
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


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run_dir", required=True)
  parser.add_argument("--out_csv", default=None)
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

  header = f"{'condition':<15}{'round':>6}{'n':>5}{'SR':>7}{'SRR':>7}{'MRR':>7}{'steps':>8}{'tokens':>9}{'bank':>7}{'errs':>6}"
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
        f"{fmt(row['successful_replay_rate'], pct=True):>7}"
        f"{fmt(row['memory_reuse_rate'], pct=True):>7}"
        f"{fmt(row['avg_atomic_steps']):>8}"
        f"{fmt(row['avg_tokens']):>9}"
        f"{(str(row['memory_bank_size']) if row['memory_bank_size'] is not None else 'n/a'):>7}"
        f"{row['n_errors']:>6}"
    )
  return 0


if __name__ == "__main__":
  sys.exit(main())
