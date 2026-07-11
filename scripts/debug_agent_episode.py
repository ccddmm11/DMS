#!/usr/bin/env python3
"""Regression-test helper: run the REAL agent classes (PALiteAgent /
StaticMemoryAgent / DMSAgent) end-to-end on one deterministic (task, round)
cell exactly like the eval harness does, printing per-step phase/sub_task/
action/task_history so we can verify the loop_guard fixes actually change
behavior on a previously-reproduced stall case.

Usage:
  python scripts/debug_agent_episode.py --condition zero_shot \
      --task SystemWifiTurnOn --round 5 --console_port 5560 --grpc_port 8557
"""

import argparse
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

from src.eval.task_instance import instantiate_task
from src.eval.task_suite import load_task_suite
from src.vlm.qwen_vl_client import QwenVLConfig, QwenVLWrapper


def build_agent(condition, env, llm, memory_dir):
  if condition == "zero_shot":
    from src.baselines.zero_shot_agent import PALiteAgent
    return PALiteAgent(env, llm)
  if condition == "static_memory":
    from src.baselines.static_memory_agent import StaticMemoryAgent
    return StaticMemoryAgent(env, llm, memory_dir)
  if condition == "dms":
    from src.androidworld_integration.dms_agent_adapter import DMSAgent
    return DMSAgent(env, llm, memory_dir)
  raise ValueError(condition)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--condition", default="zero_shot",
                       choices=["zero_shot", "static_memory", "dms"])
  parser.add_argument("--task", required=True)
  parser.add_argument("--round", type=int, default=0)
  parser.add_argument("--base_seed", type=int, default=20260707)
  parser.add_argument("--console_port", type=int, default=5560)
  parser.add_argument("--grpc_port", type=int, default=8557)
  parser.add_argument("--max_steps", type=int, default=10)
  parser.add_argument("--vllm_base_url", default="http://localhost:8000/v1")
  parser.add_argument("--memory_dir", default="/tmp/debug_agent_memory")
  args = parser.parse_args()

  suite = load_task_suite()
  spec = next((t for t in suite.tasks if t.name == args.task), None)
  if spec is None:
    raise ValueError(f"Task {args.task} not in suite.")

  print(f"[setup] connecting to emulator console_port={args.console_port} ...")
  env = env_launcher.load_and_setup_env(
      console_port=args.console_port,
      emulator_setup=False,
      freeze_datetime=True,
      adb_path="android_sdk_host/platform-tools/adb",
      grpc_port=args.grpc_port,
  )
  env.reset(go_home=True)

  task = instantiate_task(spec, args.round, args.base_seed, env=env)
  task.initialize_task(env)
  print(f"[task] name={spec.name} goal={task.goal!r} params={task.params}")

  llm = QwenVLWrapper(QwenVLConfig(base_url=args.vllm_base_url))
  agent = build_agent(args.condition, env, llm, args.memory_dir)
  agent.reset(go_home_on_reset=False)
  if hasattr(agent, "start_new_task"):
    agent.start_new_task(spec.name, spec.apps)

  is_done = False
  for step_idx in range(args.max_steps):
    response = agent.step(task.goal)
    d = response.data
    print(
        f"\n[step {step_idx + 1}/{args.max_steps}] phase={d.get('phase')} "
        f"sub_task={d.get('sub_task')} "
        f"planner_message={d.get('planner_message')!r} "
        f"action_reason={d.get('action_reason')!r} "
        f"action={d.get('action_output_json')}"
    )
    if getattr(agent, "sub_task_history", None):
      print(f"    last_sub_task_history_entry={agent.sub_task_history[-1]!r}")
    # Mirror `src.eval.runner`'s evaluator-driven early termination. Without
    # this, a successful low-level state change can be followed by many
    # pointless Planner calls while the 7B model repeatedly self-declares
    # completion, obscuring the action that actually solved the task.
    if task.is_successful(env) == 1:
      print("    evaluator_success=True; ending debug episode.")
      is_done = True
      break
    if response.done:
      is_done = True
      break

  print("\n[task_history]")
  for line in agent.task_history:
    print(" -", line)

  success = task.is_successful(env) == 1
  print(f"\n[FINAL] agent_done={is_done} task.is_successful={success}")
  if hasattr(agent, "finalize_task"):
    agent.finalize_task(success)
  task.tear_down(env)
  env.close()


if __name__ == "__main__":
  main()
