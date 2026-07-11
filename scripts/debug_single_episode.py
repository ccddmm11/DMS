#!/usr/bin/env python3
"""Debug helper: replays ONE (task, round) cell exactly like the eval
harness would (same deterministic seed via `src.eval.task_instance`), but
prints full Planner/Actor/Verifier reasoning + raw model text at every
step, to diagnose why success rate is 0% across all three conditions.

Usage:
  python scripts/debug_single_episode.py --task CameraTakePhoto --round 0 \
      --console_port 5560 --grpc_port 8557
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

from src.agent.actor import Actor
from src.agent.planner import Planner
from src.agent.verifier import Verifier
from src.agent import ui_utils
from src.eval.task_instance import instantiate_task
from src.eval.task_suite import load_task_suite
from src.vlm.qwen_vl_client import QwenVLConfig, QwenVLWrapper


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", required=True)
  parser.add_argument("--round", type=int, default=0)
  parser.add_argument("--base_seed", type=int, default=20260707)
  parser.add_argument("--console_port", type=int, default=5560)
  parser.add_argument("--grpc_port", type=int, default=8557)
  parser.add_argument("--max_steps", type=int, default=10)
  parser.add_argument("--vllm_base_url", default="http://localhost:8000/v1")
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
  planner = Planner(llm)
  actor = Actor(llm)
  verifier = Verifier(llm)

  task_history = []
  sub_plan_queue = []
  current_sub_task = None
  sub_task_history = []

  for step_idx in range(args.max_steps):
    state, state_degraded = ui_utils.get_robust_state(env)
    if state_degraded:
      print("[STATE] WARNING: still degenerate after retries!")
    ui_elements = state.ui_elements
    from android_world.agents import m3a_utils as _m3a_utils
    _valid_n = sum(
        1 for _e in ui_elements
        if _m3a_utils.validate_ui_element(_e, env.logical_screen_size)
    )
    print(f"[STATE] raw={len(ui_elements)} valid={_valid_n}")
    ui_elements_str = ui_utils.describe_ui_elements(ui_elements, env.logical_screen_size)
    raw_screenshot = state.pixels.copy()
    som_screenshot = ui_utils.build_som_screenshot(
        raw_screenshot, ui_elements, env.logical_screen_size,
        env.physical_frame_boundary, env.orientation,
    )

    print(f"\n===== STEP {step_idx+1}/{args.max_steps} =====")
    print(f"[ui_elements] {ui_elements_str[:800]}")

    if current_sub_task is None and not sub_plan_queue:
      print("[PLANNER] calling ...")
      out = planner.plan(task.goal, task_history, ui_elements_str, [raw_screenshot, som_screenshot])
      print(f"[PLANNER] parse_ok={out.parse_ok} done={out.done} goal_status={out.goal_status}")
      print(f"[PLANNER] message={out.message!r}")
      print(f"[PLANNER] raw_text={out.raw_text!r}")
      print(f"[PLANNER] sub_plans={[sp.to_dict() for sp in out.sub_plans]}")
      if out.done:
        print(f"[RESULT] Planner declared done (status={out.goal_status}) at step {step_idx+1}.")
        break
      if not out.sub_plans:
        task_history.append(f"[Planner] no valid sub-plans ({out.message or 'parse failure'}).")
        continue
      sub_plan_queue = list(out.sub_plans)

    if current_sub_task is None:
      current_sub_task = sub_plan_queue.pop(0)
      sub_task_history = []
      print(f"[DISPATCH] current_sub_task={current_sub_task.to_dict()}")

    print("[ACTOR] calling ...")
    actor_out = actor.act(
        current_sub_task.as_prompt_str(), sub_task_history, ui_elements_str,
        [raw_screenshot, som_screenshot],
    )
    print(f"[ACTOR] parse_ok={actor_out.parse_ok} reason={actor_out.reason!r}")
    print(f"[ACTOR] action_json_str={actor_out.action_json_str!r}")
    print(f"[ACTOR] raw_text={getattr(actor_out, 'raw_text', None)!r}")

    if not actor_out.parse_ok:
      sub_task_history.append("Action selection output was not in the correct format.")
      continue

    from android_world.agents import agent_utils
    from android_world.env import json_action
    try:
      converted_action = json_action.JSONAction(**agent_utils.extract_json(actor_out.action_json_str))
    except Exception as e:
      print(f"[ACTOR] FAILED to parse into JSONAction: {e}")
      sub_task_history.append(f"Action: {actor_out.action_json_str} -> FAILED to parse ({e}).")
      continue

    if converted_action.action_type == "status":
      print(f"[ACTOR] declared sub-task status={converted_action.goal_status}")
      after_state, after_degraded = ui_utils.get_robust_state(env)
      if after_degraded:
        print("[VERIFIER] SKIPPED: observation degraded, would fail closed.")
        task_history.append(f"Sub-task [{current_sub_task.as_prompt_str()}] FAILED: observation degraded.")
        sub_plan_queue = []
        current_sub_task = None
        continue
      print("[VERIFIER] calling ...")
      v_out = verifier.verify(current_sub_task.goal, sub_task_history + [f"Reason: {actor_out.reason} Action: declared {converted_action.goal_status}."], after_state.pixels)
      print(f"[VERIFIER] verified_success={v_out.verified_success} reason={v_out.reason!r}")
      print(f"[VERIFIER] raw_text={getattr(v_out, 'raw_text', None)!r}")
      if v_out.verified_success:
        task_history.append(f"Sub-task [{current_sub_task.as_prompt_str()}] COMPLETED.")
      else:
        task_history.append(f"Sub-task [{current_sub_task.as_prompt_str()}] FAILED: {v_out.reason}")
        sub_plan_queue = []
      current_sub_task = None
      continue

    print(f"[EXECUTE] {converted_action.action_type} index={converted_action.index}")
    try:
      env.execute_action(converted_action)
      sub_task_history.append(f"Reason: {actor_out.reason} Action: {actor_out.action_json_str}")
    except Exception as e:
      print(f"[EXECUTE] FAILED: {e}")
      sub_task_history.append(f"Action: {actor_out.action_json_str} -> FAILED to execute ({e}).")

    import time
    time.sleep(2.0)

  print("\n[FINAL] checking ground-truth task.is_successful() ...")
  success = task.is_successful(env)
  print(f"[FINAL] task.is_successful() = {success}")
  task.tear_down(env)
  env.close()


if __name__ == "__main__":
  main()
