#!/usr/bin/env python3
"""Smoke test: run our Baseline A (PA-Lite, memory-free Planner-Actor-Verifier)
agent, powered by the local vLLM-served Qwen2.5-VL-7B-Instruct, on a single
AndroidWorld task end-to-end.

This mirrors scripts/run_m3a_smoke_test.py but swaps `m3a.M3A` for
`src.baselines.zero_shot_agent.PALiteAgent`, so it proves the full
hierarchical Planner -> Actor -> Verifier loop: emulator screenshot -> local
VLM planning -> sub-task execution -> verification -> replanning -> task
success verification.

Usage:
  python scripts/run_palite_smoke_test.py --task SystemWifiTurnOn
  python scripts/run_palite_smoke_test.py  # random task
"""

import argparse
import os
import random
import sys
import time

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

from android_world import registry
from android_world.env import env_launcher

from src.baselines.zero_shot_agent import PALiteAgent
from src.vlm.qwen_vl_client import QwenVLWrapper, QwenVLConfig


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default=None, help="Specific task name to run.")
  parser.add_argument("--adb_path", default=os.path.join(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
      "android_sdk_host", "platform-tools", "adb"))
  parser.add_argument("--console_port", type=int, default=5554)
  parser.add_argument("--max_steps", type=int, default=None)
  parser.add_argument("--vllm_base_url", default="http://localhost:8000/v1")
  parser.add_argument("--seed", type=int, default=None)
  args = parser.parse_args()

  if args.seed is not None:
    random.seed(args.seed)

  print(f"[1/5] Connecting to AndroidWorld emulator via adb={args.adb_path} "
        f"console_port={args.console_port} ...")
  env = env_launcher.load_and_setup_env(
      console_port=args.console_port,
      emulator_setup=False,
      freeze_datetime=True,
      adb_path=args.adb_path,
  )
  env.reset(go_home=True)
  print("    OK: environment connected and reset to home screen.")

  print("[2/5] Selecting task...")
  task_registry = registry.TaskRegistry()
  aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
  if args.task:
    if args.task not in aw_registry:
      raise ValueError(f"Task {args.task} not found in registry (116 tasks).")
    task_type = aw_registry[args.task]
  else:
    task_type = random.choice(list(aw_registry.values()))
  params = task_type.generate_random_params()
  task = task_type(params)
  task.initialize_task(env)
  print(f"    Goal: {task.goal}")

  print("[3/5] Initializing PA-Lite agent with local Qwen2.5-VL-7B-Instruct "
        f"(vLLM @ {args.vllm_base_url}) ...")
  llm = QwenVLWrapper(QwenVLConfig(base_url=args.vllm_base_url))
  agent = PALiteAgent(env, llm)
  agent.reset(go_home_on_reset=False)

  max_steps = args.max_steps or int(task.complexity * 10)
  print(f"[4/5] Running agent loop (max {max_steps} atomic steps) ...")
  is_done = False
  t0 = time.time()
  step_idx = 0
  for step_idx in range(max_steps):
    step_t0 = time.time()
    response = agent.step(task.goal)
    step_data = response.data
    phase = step_data.get("phase")
    if phase == "plan":
      msg = f"planner_message={step_data.get('planner_message')!r}"
    else:
      sub_task = step_data.get("sub_task")
      action_str = step_data.get("action_output_json")
      msg = f"sub_task={sub_task} action={action_str}"
    print(f"    step {step_idx + 1}/{max_steps} [{phase}]: {msg} "
          f"({time.time() - step_t0:.1f}s)")
    if response.done:
      is_done = True
      break

  elapsed = time.time() - t0
  print("[5/5] Verifying task success ...")
  agent_successful = is_done and task.is_successful(env) == 1
  status = "Task Successful \u2705" if agent_successful else "Task Failed \u274c"
  print(f"{status}; goal='{task.goal}'; atomic_steps={step_idx + 1}; "
        f"elapsed={elapsed:.1f}s")
  print(f"    usage: planner_calls={agent.usage.planner_calls} "
        f"actor_calls={agent.usage.actor_calls} "
        f"verifier_calls={agent.usage.verifier_calls} "
        f"atomic_actions={agent.usage.atomic_actions_executed} "
        f"prompt_tokens={agent.usage.prompt_tokens} "
        f"completion_tokens={agent.usage.completion_tokens}")

  env.close()
  return 0 if agent_successful else 1


if __name__ == "__main__":
  sys.exit(main())
