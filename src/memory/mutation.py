# DMS reproduction project — epsilon-Mutation and Evolutionary Replacement
# (paper Sec 3.2.2), plus the mechanics of "blindly" replaying a stored
# trajectory against a live AndroidWorld environment.
#
#   pi_exec(st) = Replay(tau_retrieved)     with prob. 1 - epsilon
#               = Actor(st, q) (Mutation)   with prob. epsilon
#
# If the mutation trajectory tau' succeeds AND is more efficient
# (|tau'| < |tau|), it triggers an IN-PLACE EVOLUTIONARY UPDATE that
# overwrites the existing memory entry rather than creating a new one.
#
# Algorithm 1's pseudocode (Appendix D.2) only shows the two coarse
# branches (`DoReuse` vs. not); the finer "in-place evolutionary update"
# behavior is only specified in the Sec 3.2.2 prose. We implement it here
# as `decide_memory_update`, applied whenever a fresh (non-reused)
# execution succeeds AND there was a retrieval `candidate` that got
# skipped this round (whether via the epsilon roll or via risk
# suppression) -- see `dms_agent_adapter.py` for how this is wired into
# the main loop.

from __future__ import annotations

import dataclasses
import json
import random
import time
from typing import Optional

import numpy as np
from android_world.env import interface
from android_world.env import json_action

from src.agent import ui_utils
from src.memory.memory_bank import MemoryBank
from src.memory.memory_unit import MemoryUnit
from src.memory.memory_unit import TrajectoryStep
from src.memory.risk import RiskRegulator

_INDEX_BASED_ACTIONS = ("click", "long_press", "input_text", "scroll")

# Minimum trajectory length to keep as a reusable memory (Sec 3.2.1: "To
# prevent memory fragmentation, trajectories with |tau|=1 ... are filtered
# out.").
MIN_TRAJECTORY_LENGTH_TO_STORE = 2

# #5 memory write hygiene. Action types that carry NO reusable GUI interaction:
# `answer` embeds an episode-specific reply (e.g. a screen-brightness reading,
# a contact's number) that would be replayed verbatim into a DIFFERENT episode
# and produce a wrong answer, so answer-bearing trajectories must never be
# stored or blindly replayed; `wait`/`status` are non-interactive fillers, so a
# trajectory made ONLY of them is not a reusable skill either.
_ANSWER_ACTION_TYPE = "answer"
_NON_INTERACTIVE_ACTION_TYPES = ("status", "wait", _ANSWER_ACTION_TYPE)


def _step_action_type(step: TrajectoryStep) -> Optional[str]:
  action = step.action or {}
  return action.get("action_type")


def trajectory_contains_answer(trajectory: list[TrajectoryStep]) -> bool:
  """True iff the trajectory issues an `answer` action (episode-specific ->
  not generalizable, so neither storable nor blindly replayable)."""
  return any(
      _step_action_type(step) == _ANSWER_ACTION_TYPE for step in trajectory
  )


def _has_meaningful_interaction(trajectory: list[TrajectoryStep]) -> bool:
  return any(
      _step_action_type(step) not in _NON_INTERACTIVE_ACTION_TYPES
      for step in trajectory
  )


def _has_unreplayable_index_action(trajectory: list[TrajectoryStep]) -> bool:
  """Index actions must carry a semantic target for replay re-grounding."""
  return any(
      _step_action_type(step) in _INDEX_BASED_ACTIONS
      and step.action.get("index") is not None
      and not step.target_element_desc
      for step in trajectory
  )


def _has_repeated_unanchored_navigation(
    trajectory: list[TrajectoryStep], max_repeats: int = 3
) -> bool:
  """Rejects discovery loops rather than persisting them as reusable skill.

  A few same-direction scrolls can be necessary to locate content, but a long
  run of unanchored identical scrolls has no semantic target to re-ground and
  was the exact source of the preflight's eight-scroll false memory.
  """
  previous_signature = None
  repeats = 0
  for step in trajectory:
    action = step.action or {}
    if action.get("action_type") != "scroll" or action.get("index") is not None:
      previous_signature = None
      repeats = 0
      continue
    signature = (action.get("action_type"), action.get("direction"))
    if signature == previous_signature:
      repeats += 1
    else:
      previous_signature = signature
      repeats = 1
    if repeats > max_repeats:
      return True
  return False


def _has_repeated_stagnant_action(
    trajectory: list[TrajectoryStep], max_repeats: int = 2
) -> bool:
  """Reject identical actions repeated while the recorded UI is unchanged.

  This indicates an Actor loop (for example, repeatedly pressing Enter after
  it was accepted). Replaying it turns one bad trajectory into many slow ADB
  calls on every retrieval hit.
  """
  previous_key = None
  repeats = 0
  for step in trajectory:
    if not step.state_signature:
      previous_key = None
      repeats = 0
      continue
    key = (
        step.state_signature,
        json.dumps(step.action or {}, sort_keys=True),
    )
    if key == previous_key:
      repeats += 1
    else:
      previous_key = key
      repeats = 1
    if repeats > max_repeats:
      return True
  return False


# Appendix D.1: "we set K = 3" -- the verification-depth / accumulated
# verification-strikes limit before an obsolete memory is pruned
# (Algorithm 1 line 26: `if Km >= Klimit`).
DEFAULT_K_LIMIT = 3


@dataclasses.dataclass
class MutationConfig:
  """Hyperparameters for retrieval-gating + epsilon-Mutation.

  `epsilon` is described only qualitatively ("a small probability") in
  the paper, with no numeric value given in Appendix B; we pick a modest
  default and document the choice. `retrieval_score_threshold` (the
  minimum Dual-Factor score to consider a retrieval a "hit" at all,
  Algorithm 1 line 9's `Retrieve(M, pi)` returning non-None) is likewise
  not given numerically.
  """

  epsilon: float = 0.15
  retrieval_score_threshold: float = 0.3
  k_limit: int = DEFAULT_K_LIMIT


@dataclasses.dataclass
class RetrievalDecision:
  candidate: Optional[MemoryUnit]
  score: float
  do_reuse: bool
  is_mutation: bool  # candidate existed & was safe, but epsilon-roll skipped it.
  is_risk_suppressed: bool


def decide_retrieval_and_reuse(
    bank: MemoryBank,
    risk_regulator: RiskRegulator,
    precondition_query: str,
    goal_query: str,
    config: Optional[MutationConfig] = None,
    rng: Optional[random.Random] = None,
) -> RetrievalDecision:
  """Algorithm 1 line 9-13: Retrieve(M, pi), then gate reuse by risk + eps.

  `m != None and rho_m < tau_risk and Random() > epsilon => DoReuse`
  """
  config = config or MutationConfig()
  rng = rng or random

  results = bank.retrieve(
      precondition_query,
      goal_query,
      top_k=1,
      score_threshold=config.retrieval_score_threshold,
  )
  if not results or not results[0].memory.trajectory:
    return RetrievalDecision(None, 0.0, False, False, False)

  candidate = results[0].memory
  # Defend against legacy memories written before the trajectory-quality
  # filters existed: none of these can be replayed safely or efficiently.
  if (
      trajectory_contains_answer(candidate.trajectory)
      or _has_unreplayable_index_action(candidate.trajectory)
      or _has_repeated_unanchored_navigation(candidate.trajectory)
      or _has_repeated_stagnant_action(candidate.trajectory)
  ):
    return RetrievalDecision(None, 0.0, False, False, False)
  score = results[0].score
  is_safe = candidate.risk_score <= risk_regulator.get_dynamic_threshold()
  roll = rng.random()
  do_reuse = is_safe and roll > config.epsilon

  return RetrievalDecision(
      candidate=candidate,
      score=score,
      do_reuse=do_reuse,
      is_mutation=is_safe and not do_reuse,
      is_risk_suppressed=not is_safe,
  )


@dataclasses.dataclass
class ReplayResult:
  success_execution: bool  # False only if replay had to abort (bad grounding).
  history: list[str]       # Display strings, for the Verifier prompt.
  final_screenshot: Optional[np.ndarray]
  steps_replayed: int
  observation_degraded: bool = False  # See `ui_utils.get_robust_state`.


def replay_trajectory(
    trajectory: list[TrajectoryStep],
    env: interface.AsyncEnv,
    wait_after_action_seconds: float = 2.0,
) -> ReplayResult:
  """Blindly replays a stored trajectory (Sec 3.2.2 "Replay(tau_retrieved)"),
  re-grounding index-based actions against the CURRENT UI element list via
  each step's recorded `target_element_desc` (since raw indexes can drift
  between the original recording and now, even on a structurally similar
  screen). No LLM calls are made -- this is what gives DMS its latency/
  token-cost advantage on a retrieval hit (Sec 4.6).

  Aborts early (returns `success_execution=False`) if an index-based
  action's target element cannot be re-grounded at all.
  """
  history: list[str] = []
  final_screenshot = None
  steps_replayed = 0
  any_degraded = False

  for step in trajectory:
    state, degraded = ui_utils.get_robust_state(env)
    any_degraded = any_degraded or degraded
    ui_elements = state.ui_elements
    logical_screen_size = env.logical_screen_size

    action_dict = dict(step.action)
    action_type = action_dict.get("action_type")

    if action_type in _INDEX_BASED_ACTIONS and action_dict.get("index") is not None:
      regrounded_index = ui_utils.find_element_by_description(
          ui_elements, step.target_element_desc, logical_screen_size
      )
      if regrounded_index is not None:
        action_dict["index"] = regrounded_index
      elif action_dict["index"] >= len(ui_elements):
        # Original index no longer valid and no textual match found:
        # replay cannot proceed faithfully.
        return ReplayResult(
            False, history, final_screenshot, steps_replayed, any_degraded
        )

    try:
      action = json_action.JSONAction(**action_dict)
    except Exception:  # pylint: disable=broad-exception-caught
      return ReplayResult(
          False, history, final_screenshot, steps_replayed, any_degraded
      )

    if action.action_type == "status":
      history.append(f"Reason: {step.reason} Action: declared sub-task"
                      f" {action.goal_status}.")
      steps_replayed += 1
      continue

    try:
      env.execute_action(action)
      history.append(f"Reason: {step.reason} Action: {action.json_str()}")
    except Exception as e:  # pylint: disable=broad-exception-caught
      history.append(f"Reason: {step.reason} Action: {action.json_str()} ->"
                      f" FAILED to execute ({e}).")
      return ReplayResult(
          False, history, final_screenshot, steps_replayed, any_degraded
      )

    time.sleep(wait_after_action_seconds)
    steps_replayed += 1

  final_state, final_degraded = ui_utils.get_robust_state(env)
  any_degraded = any_degraded or final_degraded
  final_screenshot = final_state.pixels.copy()
  return ReplayResult(True, history, final_screenshot, steps_replayed, any_degraded)


@dataclasses.dataclass
class MemoryUpdateOutcome:
  # "created" | "replaced" | "skipped_too_short" | "skipped_answer_task"
  # | "skipped_no_interaction" | "skipped_unreplayable_index"
  # | "skipped_repeated_navigation" | "skipped_stagnant_actions" | "noop"
  action: str
  memory_id: Optional[str] = None


def decide_memory_update(
    bank: MemoryBank,
    precondition: str,
    goal: str,
    new_trajectory: list[TrajectoryStep],
    candidate: Optional[MemoryUnit],
    source_task: Optional[str] = None,
) -> MemoryUpdateOutcome:
  """Algorithm 1 line 19-21 (`CreateMemory`) + Sec 3.2.2's in-place
  evolutionary-update refinement: if a retrieval `candidate` existed for
  this sub-task but was skipped this round (mutation or risk-suppression),
  and the fresh trajectory succeeded and is strictly more efficient
  (shorter), overwrite the candidate in place; otherwise create a brand
  new memory entry (subject to the |tau|=1 filter, Sec 3.2.1).
  """
  if len(new_trajectory) < MIN_TRAJECTORY_LENGTH_TO_STORE:
    return MemoryUpdateOutcome("skipped_too_short")

  # #5 write hygiene: don't persist episode-specific answer trajectories or
  # ones with no real interaction (all status/wait) -- they are not reusable
  # skills and only pollute the bank / mislead future retrieval.
  if trajectory_contains_answer(new_trajectory):
    return MemoryUpdateOutcome("skipped_answer_task")
  if not _has_meaningful_interaction(new_trajectory):
    return MemoryUpdateOutcome("skipped_no_interaction")
  if _has_unreplayable_index_action(new_trajectory):
    return MemoryUpdateOutcome("skipped_unreplayable_index")
  if _has_repeated_unanchored_navigation(new_trajectory):
    return MemoryUpdateOutcome("skipped_repeated_navigation")
  if _has_repeated_stagnant_action(new_trajectory):
    return MemoryUpdateOutcome("skipped_stagnant_actions")

  if candidate is not None and len(new_trajectory) < len(candidate.trajectory):
    bank.replace_trajectory(candidate.memory_id, new_trajectory)
    return MemoryUpdateOutcome("replaced", candidate.memory_id)

  new_memory = MemoryUnit(
      precondition=precondition,
      goal=goal,
      trajectory=new_trajectory,
      success=True,
      description="Created via fresh Actor generation.",
      source_task=source_task,
      success_count=1.0,  # Algorithm 1 line 20: CreateMemory(..., S<-1, ...).
  )
  bank.add(new_memory)
  return MemoryUpdateOutcome("created", new_memory.memory_id)
