#!/usr/bin/env python3
"""Top-level multi-emulator parallel eval orchestrator.

Launches one `scripts/run_eval_worker.py` subprocess PER CONDITION
(zero_shot / static_memory / dms), each pinned to its own already-running
AndroidWorld emulator instance, so all three conditions' `rounds x
task_suite` sweeps execute concurrently (wall-clock time approx. = the
slowest single condition, not the sum of all three).

Default emulator assignment (see `scripts/setup_androidworld.sh` /
`scripts/Dockerfile.emulator` for how these 4 parallel instances were
brought up):
  zero_shot      -> console_port=5554, grpc_port=8554  (dms_emulator)
  static_memory  -> console_port=5556, grpc_port=8555  (dms_emulator_1)
  dms            -> console_port=5558, grpc_port=8556  (dms_emulator_2)
  (spare)        -> console_port=5560, grpc_port=8557  (dms_emulator_3)

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
      --rounds 1 --tasks CameraTakePhoto,ClockTimerEntry
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DEFAULT_EMULATORS = {
    "zero_shot": {"console_port": 5554, "grpc_port": 8554},
    "static_memory": {"console_port": 5556, "grpc_port": 8555},
    "dms": {"console_port": 5558, "grpc_port": 8556},
}


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
  args = parser.parse_args()

  conditions = args.conditions.split(",")
  for c in conditions:
    if c not in _DEFAULT_EMULATORS:
      raise ValueError(f"Unknown condition {c!r}; choices: {list(_DEFAULT_EMULATORS)}")

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
      "emulators": {c: _DEFAULT_EMULATORS[c] for c in conditions},
  }
  with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
  print(f"[harness] manifest written to {output_dir}/manifest.json")
  print(f"[harness] conditions={conditions} rounds={args.rounds} "
        f"tasks={args.tasks or 'ALL (29)'}")

  procs = {}
  log_files = {}
  for condition in conditions:
    emu = _DEFAULT_EMULATORS[condition]
    output_jsonl = os.path.join(output_dir, f"{condition}.jsonl")
    log_path = os.path.join(logs_dir, f"{condition}.log")
    cmd = [
        sys.executable,
        os.path.join(_REPO_ROOT, "scripts", "run_eval_worker.py"),
        "--condition", condition,
        "--console_port", str(emu["console_port"]),
        "--grpc_port", str(emu["grpc_port"]),
        "--adb_path", args.adb_path,
        "--rounds", str(args.rounds),
        "--base_seed", str(args.base_seed),
        "--output_jsonl", output_jsonl,
        "--memory_root", memory_root,
        "--vllm_base_url", args.vllm_base_url,
        "--log_file", log_path,
    ]
    if args.tasks:
      cmd += ["--tasks", args.tasks]
    if args.round_limit_episodes:
      cmd += ["--round_limit_episodes", str(args.round_limit_episodes)]

    print(f"[harness] launching {condition} on console_port={emu['console_port']} "
          f"-> log={log_path}")
    log_f = open(log_path, "a", encoding="utf-8")
    log_files[condition] = log_f
    procs[condition] = subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=_REPO_ROOT,
    )
    time.sleep(3.0)  # stagger startup so LLM/embedder cold-start doesn't spike at once

  print(f"[harness] all {len(procs)} workers launched. Polling every "
        f"{args.poll_seconds:.0f}s ...")
  t0 = time.time()
  try:
    while procs:
      time.sleep(args.poll_seconds)
      finished = []
      for condition, proc in procs.items():
        ret = proc.poll()
        if ret is not None:
          finished.append((condition, ret))
      for condition, ret in finished:
        status = "OK" if ret == 0 else f"EXIT_CODE={ret}"
        print(f"[harness] [{time.time()-t0:7.0f}s] worker '{condition}' finished: {status}")
        del procs[condition]
      if procs:
        print(f"[harness] [{time.time()-t0:7.0f}s] still running: {list(procs.keys())}")
  except KeyboardInterrupt:
    print("[harness] KeyboardInterrupt: terminating remaining workers ...")
    for proc in procs.values():
      proc.terminate()
    for proc in procs.values():
      proc.wait(timeout=30)
    raise
  finally:
    for f in log_files.values():
      f.close()

  print(f"[harness] all workers finished in {time.time()-t0:.0f}s. "
        f"Results in {output_dir}/*.jsonl")
  return 0


if __name__ == "__main__":
  sys.exit(main())
