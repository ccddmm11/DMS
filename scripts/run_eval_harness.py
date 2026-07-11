#!/usr/bin/env python3
"""Top-level multi-emulator parallel eval orchestrator.

Launches one `scripts/run_eval_worker.py` subprocess PER CONDITION
(zero_shot / static_memory / dms), each pinned to its own already-running
AndroidWorld emulator instance, so all three conditions' `rounds x
task_suite` sweeps execute concurrently (wall-clock time approx. = the
slowest single condition, not the sum of all three).

Concurrency-safety note: `static_memory` and `dms` each own exactly ONE
persistent, evolving Memory Bank directory that is mutated via a
read-modify-write-whole-file pattern (`src/memory/memory_bank.py` has no
cross-process file locking) -- so each of those two conditions MUST stay
pinned to exactly one worker process for the whole run. `zero_shot`
(Baseline A) has no memory / no shared state at all, so it is embarrassingly
parallel: with the 4th (otherwise spare) emulator instance, its task list is
round-robin-partitioned across `--zero_shot_workers` (default 2) independent
worker processes, each writing to its own `zero_shot_w<i>.jsonl` (the
aggregator globs `*.jsonl`, so this is transparent downstream).

Default emulator assignment (see `scripts/setup_androidworld.sh` /
`scripts/launch_emulator_pool.sh` for how these 4 parallel instances were
brought up):
  zero_shot (worker 0) -> console_port=5554, grpc_port=8554  (dms_emulator)
  static_memory        -> console_port=5556, grpc_port=8555  (dms_emulator_1)
  dms                   -> console_port=5558, grpc_port=8556  (dms_emulator_2)
  zero_shot (worker 1) -> console_port=5560, grpc_port=8557  (dms_emulator_3)

Each condition's Memory Bank (Baseline B / DMS) is a fresh, condition-local
directory under `--output_dir/memory_stores/<condition>` unless
`--resume` points back at a previous run's `--output_dir`, in which case
each worker resumes both its persistent memory bank AND its
`<condition>.jsonl` episode log from exactly where it left off.

Usage:
  # Full experiment matrix (3 conditions x 29 tasks x 5 rounds):
  python scripts/run_eval_harness.py --output_dir results/eval/run_001

  # Harness smoke test (2 tasks x 1 round x 3 conditions, ~minutes not hours):
  python scripts/run_eval_harness.py --output_dir results/eval/smoke \
      --rounds 1 --tasks CameraTakePhoto,ClockTimerEntry --zero_shot_workers 1
"""

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# One dedicated emulator per single-worker condition, plus a pool of extra
# instances available to fan out `zero_shot`'s embarrassingly-parallel work.
_SINGLE_WORKER_EMULATORS = {
    "static_memory": {"console_port": 5556, "grpc_port": 8555},
    "dms": {"console_port": 5558, "grpc_port": 8556},
}
_ZERO_SHOT_EMULATOR_POOL = [
    {"console_port": 5554, "grpc_port": 8554},
    {"console_port": 5560, "grpc_port": 8557},
]


def _terminate_worker_group(proc: subprocess.Popen) -> None:
  """Stops a worker and its AndroidWorld helper children together."""
  try:
    os.killpg(proc.pid, signal.SIGTERM)
  except ProcessLookupError:
    return


def _load_full_task_names() -> list[str]:
  sys.path.insert(0, _REPO_ROOT)
  from src.eval.task_suite import load_task_suite  # local import: keeps CLI startup fast
  return [t.name for t in load_task_suite().tasks]


def _partition_round_robin(items: list[str], n: int) -> list[list[str]]:
  buckets: list[list[str]] = [[] for _ in range(n)]
  for i, item in enumerate(items):
    buckets[i % n].append(item)
  return buckets


def _build_worker_specs(
    conditions: list[str], zero_shot_workers: int, tasks_arg: str | None,
) -> list[dict]:
  """Returns a list of {label, condition, console_port, grpc_port, tasks}."""
  specs = []
  for condition in conditions:
    if condition == "zero_shot":
      pool = _ZERO_SHOT_EMULATOR_POOL[:zero_shot_workers]
      if len(pool) < zero_shot_workers:
        raise ValueError(
            f"--zero_shot_workers={zero_shot_workers} exceeds the "
            f"{len(_ZERO_SHOT_EMULATOR_POOL)}-instance zero_shot emulator pool."
        )
      task_names = tasks_arg.split(",") if tasks_arg else _load_full_task_names()
      partitions = _partition_round_robin(task_names, zero_shot_workers)
      for i, (emu, part_tasks) in enumerate(zip(pool, partitions)):
        if not part_tasks:
          continue
        specs.append({
            "label": f"zero_shot_w{i}" if zero_shot_workers > 1 else "zero_shot",
            "condition": "zero_shot",
            "console_port": emu["console_port"],
            "grpc_port": emu["grpc_port"],
            "tasks": ",".join(part_tasks),
        })
    else:
      emu = _SINGLE_WORKER_EMULATORS[condition]
      specs.append({
          "label": condition,
          "condition": condition,
          "console_port": emu["console_port"],
          "grpc_port": emu["grpc_port"],
          "tasks": tasks_arg,
      })
  return specs


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_dir", required=True)
  parser.add_argument("--conditions", default="zero_shot,static_memory,dms")
  parser.add_argument("--rounds", type=int, default=5)
  parser.add_argument("--base_seed", type=int, default=20260707)
  parser.add_argument("--tasks", default=None,
                       help="Comma-separated task names to restrict to (default: full 29-task suite).")
  parser.add_argument("--round_limit_episodes", type=int, default=None)
  parser.add_argument("--vllm_base_url", default="http://localhost:8000/v1")
  parser.add_argument("--adb_path", default=os.path.join(
      _REPO_ROOT, "android_sdk_host", "platform-tools", "adb"))
  parser.add_argument("--poll_seconds", type=float, default=30.0)
  parser.add_argument("--zero_shot_workers", type=int, default=2,
                       help="How many parallel workers/emulators to fan Baseline A "
                            "(memory-free, no shared state) out across (max "
                            f"{len(_ZERO_SHOT_EMULATOR_POOL)}, since static_memory/dms "
                            "each require exactly 1 dedicated worker for their shared "
                            "evolving Memory Bank).")
  args = parser.parse_args()

  conditions = args.conditions.split(",")
  valid_conditions = {"zero_shot", *_SINGLE_WORKER_EMULATORS}
  for c in conditions:
    if c not in valid_conditions:
      raise ValueError(f"Unknown condition {c!r}; choices: {sorted(valid_conditions)}")

  worker_specs = _build_worker_specs(conditions, args.zero_shot_workers, args.tasks)

  output_dir = os.path.abspath(args.output_dir)
  logs_dir = os.path.join(output_dir, "logs")
  memory_root = os.path.join(output_dir, "memory_stores")
  os.makedirs(logs_dir, exist_ok=True)
  os.makedirs(memory_root, exist_ok=True)

  manifest = {
      "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
      "output_dir": output_dir,
      "conditions": conditions,
      "rounds": args.rounds,
      "base_seed": args.base_seed,
      "tasks": args.tasks,
      "zero_shot_workers": args.zero_shot_workers,
      "workers": worker_specs,
  }
  with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
  print(f"[harness] manifest written to {output_dir}/manifest.json")
  print(f"[harness] conditions={conditions} rounds={args.rounds} "
        f"tasks={args.tasks or 'ALL (29)'} workers={[w['label'] for w in worker_specs]}")

  procs = {}
  log_files = {}
  for spec in worker_specs:
    label = spec["label"]
    output_jsonl = os.path.join(output_dir, f"{label}.jsonl")
    log_path = os.path.join(logs_dir, f"{label}.log")
    cmd = [
        sys.executable,
        os.path.join(_REPO_ROOT, "scripts", "run_eval_worker.py"),
        "--condition", spec["condition"],
        "--console_port", str(spec["console_port"]),
        "--grpc_port", str(spec["grpc_port"]),
        "--adb_path", args.adb_path,
        "--rounds", str(args.rounds),
        "--base_seed", str(args.base_seed),
        "--output_jsonl", output_jsonl,
        "--memory_root", memory_root,
        "--vllm_base_url", args.vllm_base_url,
        "--log_file", log_path,
    ]
    if spec["tasks"]:
      cmd += ["--tasks", spec["tasks"]]
    if args.round_limit_episodes:
      cmd += ["--round_limit_episodes", str(args.round_limit_episodes)]

    print(f"[harness] launching {label} (condition={spec['condition']}) on "
          f"console_port={spec['console_port']} -> log={log_path}")
    log_f = open(log_path, "a", encoding="utf-8")
    log_files[label] = log_f
    procs[label] = subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=_REPO_ROOT,
        start_new_session=True,
    )
    time.sleep(3.0)  # stagger startup so LLM/embedder cold-start doesn't spike at once

  print(f"[harness] all {len(procs)} workers launched. Polling every "
        f"{args.poll_seconds:.0f}s ...")
  t0 = time.time()
  try:
    while procs:
      time.sleep(args.poll_seconds)
      finished = []
      for label, proc in procs.items():
        ret = proc.poll()
        if ret is not None:
          finished.append((label, ret))
      for label, ret in finished:
        status = "OK" if ret == 0 else f"EXIT_CODE={ret}"
        print(f"[harness] [{time.time()-t0:7.0f}s] worker '{label}' finished: {status}")
        if ret != 0:
          # A crashed worker otherwise leaves its logcat/a11y helpers alive,
          # and a resumed worker then competes for the same emulator.
          _terminate_worker_group(procs[label])
        del procs[label]
      if procs:
        print(f"[harness] [{time.time()-t0:7.0f}s] still running: {list(procs.keys())}")
  except KeyboardInterrupt:
    print("[harness] KeyboardInterrupt: terminating remaining workers ...")
    for proc in procs.values():
      _terminate_worker_group(proc)
    for proc in procs.values():
      try:
        proc.wait(timeout=30)
      except subprocess.TimeoutExpired:
        try:
          os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
          pass
    raise
  finally:
    for f in log_files.values():
      f.close()

  print(f"[harness] all workers finished in {time.time()-t0:.0f}s. "
        f"Results in {output_dir}/*.jsonl")
  return 0


if __name__ == "__main__":
  sys.exit(main())
