#!/usr/bin/env python3
"""Single-condition, single-emulator eval worker.

Connects to ONE already-running AndroidWorld emulator instance (identified
by `--console_port`/`--grpc_port`), builds ONE of the three agents
(zero_shot / static_memory / dms), and runs it through
`--rounds x task_suite` episodes, appending one JSON line per episode to
`--output_jsonl` (resume-safe: rerunning with the same `--output_jsonl`
skips already-completed (round, task) cells).

This is meant to be launched as a subprocess by
`scripts/run_eval_harness.py` (one process per condition, each pinned to a
distinct emulator), but can also be run standalone for debugging/smoke
testing a single condition:

  python scripts/run_eval_worker.py --condition dms --console_port 5554 \
      --grpc_port 8554 --rounds 1 --tasks CameraTakePhoto,ClockTimerEntry \
      --output_jsonl results/eval/smoke/dms.jsonl
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "android_world",
    ),
)

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = "none"

from absl import logging as absl_logging

absl_logging.set_verbosity(absl_logging.WARNING)

from android_world.env import env_launcher

from src.baselines.static_memory_agent import StaticMemoryAgent
from src.baselines.zero_shot_agent import PALiteAgent
from src.androidworld_integration.dms_agent_adapter import DMSAgent
from src.eval.runner import RunConfig, run_condition
from src.eval.task_suite import load_task_suite
from src.vlm.qwen_vl_client import QwenVLConfig, QwenVLWrapper

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build_agent(condition: str, env, llm, memory_root: str, rng_seed: int):
  if condition == "zero_shot":
    return PALiteAgent(env, llm)
  if condition == "static_memory":
    return StaticMemoryAgent(env, llm, memory_store_dir=os.path.join(memory_root, "static_memory"))
  if condition == "dms":
    return DMSAgent(
        env, llm, memory_store_dir=os.path.join(memory_root, "dms"),
        rng_seed=rng_seed,
    )
  raise ValueError(f"Unknown condition: {condition}")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--condition", required=True,
                       choices=["zero_shot", "static_memory", "dms"])
  parser.add_argument("--console_port", type=int, required=True)
  parser.add_argument("--grpc_port", type=int, required=True)
  parser.add_argument("--adb_path", default=os.path.join(
      _REPO_ROOT, "android_sdk_host", "platform-tools", "adb"))
  parser.add_argument("--rounds", type=int, default=5)
  parser.add_argument("--base_seed", type=int, default=20260707)
  parser.add_argument("--tasks", default=None,
                       help="Comma-separated task names to restrict to (default: full suite).")
  parser.add_argument("--round_limit_episodes", type=int, default=None,
                       help="Cap episodes per round (debugging/smoke test only).")
  parser.add_argument("--output_jsonl", required=True)
  parser.add_argument("--memory_root", default=os.path.join(_REPO_ROOT, "results", "memory_stores"))
  parser.add_argument("--vllm_base_url", default="http://localhost:8000/v1")
  parser.add_argument("--log_file", default=None)
  args = parser.parse_args()

  handlers = [logging.StreamHandler(sys.stdout)]
  if args.log_file:
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    handlers.append(logging.FileHandler(args.log_file))
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] %(message)s",
      handlers=handlers,
      force=True,
  )
  logger = logging.getLogger("dms_eval")
  logger.info("Worker starting: condition=%s console_port=%d grpc_port=%d rounds=%d",
              args.condition, args.console_port, args.grpc_port, args.rounds)

  env = env_launcher.load_and_setup_env(
      console_port=args.console_port,
      emulator_setup=False,
      freeze_datetime=True,
      adb_path=args.adb_path,
      grpc_port=args.grpc_port,
  )
  env.reset(go_home=True)

  suite = load_task_suite()
  task_filter = args.tasks.split(",") if args.tasks else None

  llm = QwenVLWrapper(QwenVLConfig(base_url=args.vllm_base_url))
  agent_seed = args.base_seed + {"zero_shot": 0, "static_memory": 1, "dms": 2}[args.condition]

  def agent_factory():
    return _build_agent(args.condition, env, llm, args.memory_root, agent_seed)

  config = RunConfig(
      condition=args.condition,
      output_jsonl=args.output_jsonl,
      rounds=args.rounds,
      base_seed=args.base_seed,
      task_filter=task_filter,
      round_limit_episodes=args.round_limit_episodes,
  )

  try:
    run_condition(env, agent_factory, suite, config)
  finally:
    env.close()

  logger.info("Worker finished: condition=%s output=%s", args.condition, args.output_jsonl)
  return 0


if __name__ == "__main__":
  sys.exit(main())
