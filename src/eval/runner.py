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

# Step budget per episode. We keep AndroidWorld's complexity-proportional
# scaling (`suite_utils._allocate_step_budget`), but raise the floor from 10
# to 30: with a 7B backbone most of the budget is spent recovering from
# navigation/grounding mistakes, and a 10-step floor left no room to actually
# finish even simple multi-hop tasks (matching the peer reproductions, which
# use a fixed 30-step budget).
STEPS_PER_COMPLEXITY_UNIT = 10
MIN_STEPS = 30


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

  try:
    task.initialize_task(env)
    agent.start_new_task(spec.name, spec.apps)

    # #1: use AndroidWorld's ground-truth evaluator to END the episode as soon
    # as the task is actually done, instead of relying on the agent's Planner
    # to self-declare completion (which a 7B Planner almost never does -- see
    # results/debug/zero_success_rate_diagnosis.md). `run_episode` calls this
    # after every atomic step and terminates with done=True on a truthy value.
    # This is an engineering deviation from the paper's Verifier-driven
    # completion, adopted (like the peer reproductions) to make the metric
    # reflect real task success rather than the agent's self-report.
    def _evaluator_termination_fn(_env: interface.AsyncEnv) -> float:
      try:
        return 1.0 if task.is_successful(_env) == 1 else 0.0
      except Exception:  # pylint: disable=broad-exception-caught
        return 0.0

    episode_result = run_episode(
        goal=task.goal,
        agent=agent,
        max_n_steps=max_steps,
        start_on_home_screen=getattr(task, "start_on_home_screen", False),
        termination_fn=_evaluator_termination_fn,
    )
    # `agent_done` now means "episode ended early" (agent self-declared OR the
    # evaluator fired), while `success` is the ground-truth evaluator verdict
    # and is no longer gated on the agent declaring done.
    agent_done = bool(episode_result.done)
    atomic_steps = len(episode_result.step_data.get("phase", [])) if episode_result.step_data else 0
    task_success_signal = task.is_successful(env)
    success = task_success_signal == 1
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

  # NOTE: `agent.usage` is NOT a running total -- `agent.reset()` (called as
  # the first thing inside `episode_runner.run_episode()`, i.e. already
  # executed by the time we get here, including on the exception path)
  # always replaces it with a fresh, zeroed usage object for this episode.
  # So no before/after delta is needed (or correct -- a delta would instead
  # subtract the *previous* episode's final totals from THIS episode's,
  # producing nonsensical negative counts).
  usage_delta = _usage_dict(agent)
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
      subtask_replay_attempts=usage_delta.get("replay_attempts"),
      subtask_replay_verified=usage_delta.get("replay_successes"),
      episode_replay_attempted=bool(usage_delta.get("replay_attempts", 0)),
      episode_success_after_replay=(
          bool(success) if usage_delta.get("replay_attempts", 0) else None
      ),
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
