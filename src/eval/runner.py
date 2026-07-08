# DMS reproduction project — single-condition, single-emulator eval loop.
#
# One call to `run_condition(...)` owns exactly one AndroidWorld environment
# (one emulator instance, one console/gRPC port pair) and exactly one agent
# (one of Baseline A / Baseline B / DMS), and drives it through
# `rounds x len(task_suite)` episodes, in "round-major" order (round 0's 29
# tasks, then round 1's 29 tasks, ...) so that the x-axis of the report
# ("evolutionary round") is a well-defined checkpoint of the SAME persistent
# Memory Bank having seen every task exactly once per prior round -- this is
# what lets Baseline B / DMS's memory actually accumulate and evolve across
# rounds, matching the paper's "evolutionary round" protocol (Sec 4.1/4.5).
#
# Each episode is wrapped in its own try/except (mirroring AndroidWorld's own
# `suite_utils._run_task`): a single task crashing (LLM timeout, app crash,
# unparseable action, ...) is logged as an `error` episode and the run moves
# on to the next cell, rather than losing the entire multi-hour run. Progress
# is persisted one JSON line at a time (`metrics.append_record`), so a killed
# / restarted worker resumes exactly where it left off via
# `metrics.load_completed_cells`.

from __future__ import annotations

import dataclasses
import datetime
import logging
import time
import traceback
from typing import Any, Callable, Optional

from android_world.env import interface
from android_world.episode_runner import run_episode

from src.eval import metrics
from src.eval.task_instance import instantiate_task
from src.eval.task_suite import TaskSpec, TaskSuite

logger = logging.getLogger("dms_eval")

# AndroidWorld's own convention (`suite_utils._allocate_step_budget`).
STEPS_PER_COMPLEXITY_UNIT = 10
MIN_STEPS = 10


@dataclasses.dataclass
class RunConfig:
  condition: str                 # "zero_shot" | "static_memory" | "dms"
  output_jsonl: str
  rounds: int
  base_seed: int = 20260707
  task_filter: Optional[list[str]] = None  # limit to these task names, else all
  round_limit_episodes: Optional[int] = None  # for smoke-testing the harness
  episode_timeout_seconds: float = 900.0     # soft guard, checked between episodes


def _step_budget(complexity: float) -> int:
  return max(MIN_STEPS, int(STEPS_PER_COMPLEXITY_UNIT * complexity))


def _usage_dict(agent: Any) -> dict[str, Any]:
  usage = getattr(agent, "usage", None)
  if usage is None:
    return {}
  return dataclasses.asdict(usage)


def run_condition(
    env: interface.AsyncEnv,
    agent_factory: Callable[[], Any],
    suite: TaskSuite,
    config: RunConfig,
) -> None:
  """Runs one condition end-to-end on an already-connected `env`.

  Args:
    env: A live AndroidWorld environment (already `load_and_setup_env`'d by
      the caller, pinned to one emulator instance).
    agent_factory: Zero-arg callable that constructs the agent bound to
      `env` (so callers control the memory_store_dir / hyperparameters).
    suite: The resolved task suite (see `src.eval.task_suite`).
    config: Run parameters (condition name, output path, rounds, seed, ...).
  """
  agent = agent_factory()

  tasks = suite.tasks
  if config.task_filter:
    wanted = set(config.task_filter)
    tasks = [t for t in tasks if t.name in wanted]
    if not tasks:
      raise ValueError(f"task_filter {config.task_filter} matched no tasks in suite.")

  completed = metrics.load_completed_cells(config.output_jsonl)
  if completed:
    logger.info(
        "[%s] Resuming: %d cells already completed in %s.",
        config.condition, len(completed), config.output_jsonl,
    )

  n_this_round = 0
  for round_idx in range(config.rounds):
    n_this_round = 0
    for spec in tasks:
      if config.round_limit_episodes is not None and n_this_round >= config.round_limit_episodes:
        break
      cell = (round_idx, spec.name)
      if cell in completed:
        continue
      _run_one_episode(env, agent, spec, round_idx, config)
      n_this_round += 1


def _run_one_episode(
    env: interface.AsyncEnv,
    agent: Any,
    spec: TaskSpec,
    round_idx: int,
    config: RunConfig,
) -> None:
  t0 = time.time()
  task = instantiate_task(spec, round_idx, config.base_seed, env=env)
  max_steps = _step_budget(spec.complexity)

  logger.info(
      "[%s] round=%d task=%s goal=%r max_steps=%d",
      config.condition, round_idx, spec.name, task.goal, max_steps,
  )

  error_msg = None
  success = False
  agent_done = False
  atomic_steps = 0
  usage_before = _usage_dict(agent)

  try:
    task.initialize_task(env)
    agent.start_new_task(spec.name)
    episode_result = run_episode(
        goal=task.goal,
        agent=agent,
        max_n_steps=max_steps,
        start_on_home_screen=getattr(task, "start_on_home_screen", False),
    )
    agent_done = bool(episode_result.done)
    atomic_steps = len(episode_result.step_data.get("phase", [])) if episode_result.step_data else 0
    task_success_signal = task.is_successful(env)
    success = agent_done and task_success_signal == 1
  except Exception as e:  # pylint: disable=broad-exception-caught
    error_msg = f"{type(e).__name__}: {e}"
    logger.exception("[%s] round=%d task=%s CRASHED: %s", config.condition, round_idx, spec.name, e)
    traceback.print_exc()

  regulation_action = None
  try:
    regulation_result = agent.finalize_task(success)
    if regulation_result is not None:
      regulation_action = getattr(regulation_result, "action", None)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.exception("[%s] finalize_task failed: %s", config.condition, e)
    if error_msg is None:
      error_msg = f"finalize_task {type(e).__name__}: {e}"

  try:
    task.tear_down(env)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.exception("[%s] tear_down failed: %s", config.condition, e)

  usage_after = _usage_dict(agent)
  usage_delta = {
      k: usage_after.get(k, 0) - usage_before.get(k, 0)
      for k in usage_after
      if isinstance(usage_after.get(k), (int, float))
  }
  bank = getattr(agent, "bank", None)
  memory_bank_size_after = len(bank) if bank is not None else None

  replayed = usage_delta.get("replayed_actions_executed")
  fresh = usage_delta.get("atomic_actions_executed")
  mrr = None
  if replayed is not None and fresh is not None and (replayed + fresh) > 0:
    mrr = replayed / (replayed + fresh)

  record = metrics.EpisodeRecord(
      condition=config.condition,
      round_idx=round_idx,
      task_name=spec.name,
      difficulty=spec.difficulty,
      tags=list(spec.tags),
      complexity=spec.complexity,
      goal=task.goal,
      seed=task.params.get("seed", -1),
      max_steps=max_steps,
      success=bool(success),
      agent_done=agent_done,
      atomic_steps=atomic_steps,
      wall_clock_seconds=time.time() - t0,
      timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
      planner_calls=usage_delta.get("planner_calls", 0),
      actor_calls=usage_delta.get("actor_calls", 0),
      verifier_calls=usage_delta.get("verifier_calls", 0),
      replan_cycles=(getattr(agent, "replan_cycles", 0)),
      prompt_tokens=usage_delta.get("prompt_tokens", 0),
      completion_tokens=usage_delta.get("completion_tokens", 0),
      retrieval_attempts=usage_delta.get("retrieval_attempts"),
      retrieval_hits=usage_delta.get("retrieval_hits"),
      replay_attempts=usage_delta.get("replay_attempts"),
      replay_successes=usage_delta.get("replay_successes"),
      replayed_actions_executed=usage_delta.get("replayed_actions_executed"),
      fresh_actions_executed=usage_delta.get("atomic_actions_executed"),
      memory_reuse_rate=mrr,
      memory_bank_size_after=memory_bank_size_after,
      mutation_attempts=usage_delta.get("mutation_attempts"),
      memories_created=usage_delta.get("memories_created"),
      memories_replaced=usage_delta.get("memories_replaced"),
      memories_pruned_by_strikes=usage_delta.get("memories_pruned_by_strikes"),
      regulation_action=regulation_action,
      error=error_msg,
  )
  metrics.append_record(config.output_jsonl, record)

  status = "OK" if success else ("ERROR" if error_msg else "FAIL")
  logger.info(
      "[%s] round=%d task=%s -> %s (steps=%d, %.1fs, bank=%s)",
      config.condition, round_idx, spec.name, status, atomic_steps,
      record.wall_clock_seconds, memory_bank_size_after,
  )
