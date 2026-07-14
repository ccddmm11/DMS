# DMS reproduction project — full Darwinian Memory System agent.
#
# This is the complete system (Algorithm 1, Appendix D.2): the same
# hierarchical Planner-Actor-Verifier loop as Baseline A (`PALiteAgent`),
# but with every sub-task first going through Dual-Factor Retrieval +
# Bayesian risk gating + epsilon-Mutation before falling back to fresh
# Actor generation, and with the Memory Bank continuously evolving via
# Survival-Value/Elbow-Method self-regulation and Bayesian risk feedback:
#
#   while task not done and global_step < max_steps:
#     P = Planner(state, task)
#     for p_i in P:
#       m = Retrieve(Bank, p_i)                       # Dual-Factor score
#       DoReuse = m is not None and risk_ok(m) and Random() > epsilon
#       if DoReuse:
#         tau = m.trajectory; R_sub = Replay(tau)      # no LLM calls
#       else:
#         tau = Actor.generate_from_scratch(state, p_i)  # "Mutation" if m
#                                                          # existed but was
#                                                          # skipped
#       R_sub = Verifier(p_i, tau, state')
#       if R_sub == SUCCESS:
#         if DoReuse: n_m += 1; S_m += 1 (Bayesian)
#         else: CreateMemory(p_i, tau) OR, if `m` existed and tau is
#               shorter, in-place EVOLUTIONARY REPLACEMENT of `m`.
#       else:
#         if DoReuse: K_m += 1; prune m if K_m >= K_limit
#         discard remaining P -> replan
#   Global Feedback Regulation Stage: F_i += 1 for all L_active memories
#   iff the whole task failed; refresh risk scores; periodic Self-
#   Regulation (Elbow-Method prune/expand) over the whole bank.
#
# Because AndroidWorld's harness drives the agent with repeated `step()`
# calls (one atomic env action per call, like M3A/PA-Lite), we implement
# the nested loop as an internal state machine. A REPLAY (no LLM calls,
# just deterministic env actions) is cheap enough to execute in full
# within a single `step()` call -- this is itself a faithful reflection
# of DMS's latency/token advantage on a memory hit (Sec 4.6). Fresh Actor
# generation instead advances one atomic action per `step()` call, same
# step-budget accounting as Baseline A/B for a fair comparison.
#
# NOTE: unlike `reset()` (called automatically by AndroidWorld's harness
# before every episode), `finalize_task(success)` is a DMS-specific hook
# that the CALLING eval harness must invoke explicitly once ground-truth
# task success is known (`task.is_successful(env)`), since the agent
# itself cannot always know the true outcome. This drives Algorithm 1's
# "Global Feedback Regulation Stage" (Bayesian F_i updates + periodic
# Elbow-Method self-regulation).

from __future__ import annotations

import dataclasses
import hashlib
import random
import time
from typing import Any, Optional

from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.env import interface
from android_world.env import json_action

from src.agent import action_utils
from src.agent import app_intent
from src.agent import loop_guard
from src.agent import system_ui_intent
from src.agent import ui_utils
from src.agent.actor import Actor
from src.agent.planner import Planner
from src.agent.planner import SubPlan
from src.agent.verifier import Verifier
from src.memory import mutation
from src.memory.memory_bank import MemoryBank
from src.memory.memory_unit import MemoryUnit
from src.memory.memory_unit import TrajectoryStep
from src.memory.mutation import MutationConfig
from src.memory.risk import BayesianRiskConfig
from src.memory.risk import RiskRegulator
from src.memory.survival import RegulationConfig
from src.memory.survival import RegulationResult
from src.memory.survival import SelfRegulator
from src.memory.survival import SurvivalValueConfig

_INDEX_BASED_ACTIONS = ("click", "long_press", "input_text", "scroll")

# Whole-task memory precondition lives in `src/memory/mutation.py` so both the
# DMS agent and Baseline B share one source of truth (see its docstring there).
_TASK_LEVEL_PRECONDITION = mutation.TASK_LEVEL_PRECONDITION


@dataclasses.dataclass
class DMSUsageStats:
  """Aggregated LLM usage + memory-mechanism counters for one episode."""

  planner_calls: int = 0
  actor_calls: int = 0
  verifier_calls: int = 0
  atomic_actions_executed: int = 0     # Actions taken via fresh generation.
  replayed_actions_executed: int = 0   # Actions taken via memory replay.
  retrieval_attempts: int = 0
  retrieval_hits: int = 0
  replay_attempts: int = 0
  replay_successes: int = 0
  mutation_attempts: int = 0
  memories_created: int = 0
  memories_replaced: int = 0
  memories_pruned_by_strikes: int = 0
  prompt_tokens: int = 0
  completion_tokens: int = 0

  def add(self, raw_response: Any) -> None:
    if isinstance(raw_response, dict):
      usage = raw_response.get("usage") or {}
      self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
      self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)

  def record_memory_update(self, action: str) -> None:
    if action == "created":
      self.memories_created += 1
    elif action == "replaced":
      self.memories_replaced += 1

  @property
  def memory_reuse_rate(self) -> float:
    """MRR: fraction of ATOMIC actions this episode that came from a
    replayed memory rather than fresh Actor generation."""
    total = self.atomic_actions_executed + self.replayed_actions_executed
    if total == 0:
      return 0.0
    return self.replayed_actions_executed / total


class DMSAgent(base_agent.EnvironmentInteractingAgent):
  """The full Darwinian Memory System agent (Algorithm 1)."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer.MultimodalLlmWrapper,
      memory_store_dir: str,
      name: str = "DMS",
      max_actor_steps_per_subtask: int = 8,
      wait_after_action_seconds: float = 2.0,
      mutation_config: Optional[MutationConfig] = None,
      survival_config: Optional[SurvivalValueConfig] = None,
      regulation_config: Optional[RegulationConfig] = None,
      risk_config: Optional[BayesianRiskConfig] = None,
      regulate_every_n_tasks: int = 1,
      rng_seed: Optional[int] = None,
  ):
    """Initializes the DMS agent.

    Args:
      env: The AndroidWorld environment.
      llm: A multimodal LLM wrapper shared across Planner/Actor/Verifier.
      memory_store_dir: Directory backing this run's persistent MemoryBank
        (survives across `reset()` calls / episodes / rounds -- only
        `src/eval` orchestration decides whether to point two conditions
        at the same or different directories).
      name: Agent name, used in logs/results.
      max_actor_steps_per_subtask: Local step limit `MaxA` for the Actor
        before we force a verification + (re)plan (fresh-generation path
        only; replay has its own natural length = len(stored trajectory)).
      wait_after_action_seconds: Seconds to sleep after an atomic action.
      mutation_config: epsilon / retrieval-hit-threshold / K_limit.
      survival_config: S(m_i) formula hyperparameters (Sec 3.2.3).
      regulation_config: Elbow-Method capacity hyperparameters.
      risk_config: Bayesian risk model hyperparameters (Sec 3.2.4).
      regulate_every_n_tasks: Run Self-Regulation (Elbow prune/expand)
        once every N completed tasks (via `finalize_task`), not
        necessarily every single task, to save compute.
      rng_seed: Seed for the epsilon-Mutation dice roll (reproducibility).
    """
    super().__init__(env, name)
    self.planner = Planner(llm)
    self.actor = Actor(llm)
    self.verifier = Verifier(llm)

    self.bank = MemoryBank(memory_store_dir)
    self.self_regulator = SelfRegulator(
        self.bank, survival_config, regulation_config
    )
    self.risk_regulator = RiskRegulator(self.bank, risk_config)
    self.mutation_config = mutation_config or MutationConfig()

    self.max_actor_steps_per_subtask = max_actor_steps_per_subtask
    self.wait_after_action_seconds = wait_after_action_seconds
    self.regulate_every_n_tasks = regulate_every_n_tasks
    self._rng = random.Random(rng_seed)
    self._completed_task_count = 0
    self._current_task_name: Optional[str] = None
    self._task_apps: list[str] = []

    self.additional_guidelines = None  # kept for API parity with M3A/T3A.
    self._repetition_breaker = loop_guard.RepetitionBreaker()
    self._stagnant_action_breaker = loop_guard.StagnantActionBreaker()
    self._reset_episode_state()

  # -- API parity with M3A/T3A ---------------------------------------------
  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def reset(self, go_home_on_reset: bool = False) -> None:
    super().reset(go_home_on_reset)
    self.env.hide_automation_ui()
    self.bank.tick(1.0)  # Sec 3.2.3's Delta_t is in logical (task) steps.
    self._reset_episode_state()

  # -- DMS-specific hooks (the eval harness MUST call these; see header) --
  def start_new_task(
      self, task_name: Optional[str] = None, task_apps: Optional[list[str]] = None
  ) -> None:
    self._current_task_name = task_name
    self._task_apps = list(task_apps or [])

  def finalize_task(self, task_succeeded: bool) -> Optional[RegulationResult]:
    """Algorithm 1 lines 40-46 (Global Feedback Regulation Stage) + a
    periodic Self-Regulation (Elbow-Method) pass. Returns the
    `RegulationResult` if a regulation cycle ran this call, else None."""
    # AndroidWorld's ground-truth termination callback runs immediately after
    # the final Actor action, before the next agent step can invoke the LLM
    # Verifier. Ground-truth success is stronger evidence than that verifier,
    # so persist the active fresh trajectory here rather than losing every
    # successful terminal sub-task to evaluator short-circuiting.
    if (
        task_succeeded
        and self.current_sub_task is not None
        and self.sub_task_trajectory
    ):
      self._succeed_current_sub_task_via_fresh_generation()

    # Resolve a deferred task-level replay against ground truth. The replay
    # itself ran inside one `step()` call (executing the whole stored
    # trajectory), so the LLM Verifier never saw the final state -- we let the
    # AndroidWorld evaluator be the truth and update reuse/strike bookkeeping
    # here, exactly as the runner already does for episode termination.
    pending = self._task_level_replay_pending
    if pending is not None:
      self._task_level_replay_pending = None
      if task_succeeded:
        self.usage.replay_successes += 1
        self.risk_regulator.record_reuse_success(pending.memory_id, persist=False)
        self.bank.record_reuse(pending.memory_id, persist=True)
      else:
        strikes = self.bank.record_verification_failure(
            pending.memory_id, persist=False
        )
        if strikes >= self.mutation_config.k_limit:
          self.bank.remove(pending.memory_id, persist=True)
          self.usage.memories_pruned_by_strikes += 1
        else:
          self.bank.save()

    # Task-level memory creation for episodes the fast paths / Actor solved
    # fresh (no sub-task memory was written via the path above, and no pure
    # replay happened -- `episode_trajectory` is empty for a pure-replay
    # success). This is what makes DMS accumulate experience on the simple
    # tasks that the 7B fast paths solve deterministically. If a retrieval
    # `candidate` existed but was skipped (epsilon-mutation / risk suppression
    # / a replay that aborted and was then re-solved fresh), `decide_memory_update`
    # performs the in-place evolutionary replacement when the fresh trajectory
    # is shorter (Sec 3.2.2).
    if (
        task_succeeded
        and self.usage.memories_created == 0
        and self.usage.memories_replaced == 0
        and self.episode_trajectory
    ):
      outcome = mutation.decide_memory_update(
          self.bank,
          _TASK_LEVEL_PRECONDITION,
          self._episode_task_goal,
          self.episode_trajectory,
          self._task_level_candidate,
          source_task=self._current_task_name,
      )
      self.usage.record_memory_update(outcome.action)
      if outcome.memory_id is not None:
        self.episode_active_memory_ids.add(outcome.memory_id)
        if outcome.action == "replaced":
          # The mutation's trajectory "won" against the skipped candidate;
          # treat it like a confirmed-good reuse for risk bookkeeping too.
          self.risk_regulator.record_reuse_success(outcome.memory_id, persist=False)
          self.bank.record_reuse(outcome.memory_id, persist=True)

    self.risk_regulator.record_global_task_outcome(
        list(self.episode_active_memory_ids), task_succeeded
    )
    self._completed_task_count += 1
    if self._completed_task_count % self.regulate_every_n_tasks == 0:
      return self.self_regulator.regulate()
    return None

  # -- Internal episode state ----------------------------------------------
  def _reset_episode_state(self) -> None:
    self.task_history: list[str] = []
    self.sub_plan_queue: list[SubPlan] = []
    self.current_sub_task: Optional[SubPlan] = None
    self.sub_task_history: list[str] = []
    self.sub_task_trajectory: list[TrajectoryStep] = []
    self.sub_task_step_count = 0
    self.sub_task_candidate: Optional[MemoryUnit] = None
    self.episode_active_memory_ids: set[str] = set()
    self.replan_cycles = 0
    self.usage = DMSUsageStats()
    self._repetition_breaker.reset()
    self._stagnant_action_breaker.reset()
    self._task_navigation_started = False
    self._planner_completion_rejections = 0
    self._system_toggle_route_labels: list[str] = []

    # -- Task-level memory tracking (see _TASK_LEVEL_PRECONDITION docstring).
    # `episode_trajectory` collects EVERY fresh atomic action this episode
    # (fast-path + Actor), so a fast-path-solved success can be persisted as
    # one task-level memory in `finalize_task`. `_task_replay_attempted` gates
    # the once-per-episode task-level retrieval. `_task_level_candidate` is
    # the retrieval hit (if any) carried into `finalize_task` for epsilon
    # mutation's in-place evolutionary replacement. `_task_level_replay_pending`
    # defers a task-level replay's reuse/strike bookkeeping to ground truth.
    self.episode_trajectory: list[TrajectoryStep] = []
    self._episode_task_goal: str = ""
    self._task_replay_attempted: bool = False
    self._task_level_candidate: Optional[MemoryUnit] = None
    self._task_level_replay_pending: Optional[MemoryUnit] = None
    self._current_state_signature: str = ""

  def _advance_to_next_sub_task_slot(self) -> None:
    """Clears the "active sub-task" slot; the NEXT `step()` call will pop
    the next queued sub-task (and run its retrieval decision) or, if the
    queue is empty, re-invoke the Planner."""
    self.current_sub_task = None
    self.sub_task_candidate = None
    self.sub_task_history = []
    self.sub_task_trajectory = []
    self.sub_task_step_count = 0

  def _record_episode_step(
      self,
      reason: Optional[str],
      action_dict: dict[str, Any],
      target_element_desc: Optional[str],
  ) -> None:
    """Appends one fresh atomic action to the whole-episode trajectory.

    Fast-path and Actor actions alike are recorded here so that an episode
    the deterministic fast paths solve (which never enters the sub-task
    memory-write path) can still be persisted as a task-level memory in
    `finalize_task`. Replay actions are NOT recorded here -- a pure-replay
    success already has its memory and must not be re-persisted.
    """
    self.episode_trajectory.append(
        TrajectoryStep(
            reason=reason or "",
            action=dict(action_dict or {}),
            target_element_desc=target_element_desc,
            state_signature=self._current_state_signature,
        )
    )

  def _fail_current_sub_task(self, reason: str) -> None:
    assert self.current_sub_task is not None
    self.task_history.append(
        f"Sub-task [{self.current_sub_task.as_prompt_str()}] FAILED:"
        f" {reason}"
    )
    self._advance_to_next_sub_task_slot()
    # Algorithm 1: PlanFailed <- TRUE; discard rest of this plan cycle.
    self.sub_plan_queue = []

  def _succeed_current_sub_task_via_reuse(self, memory: MemoryUnit) -> None:
    self.risk_regulator.record_reuse_success(memory.memory_id, persist=False)
    self.bank.record_reuse(memory.memory_id, persist=True)
    self.episode_active_memory_ids.add(memory.memory_id)
    self.task_history.append(
        f"Sub-task [{self.current_sub_task.as_prompt_str()}] COMPLETED"
        " (replayed from memory)."
    )
    self._advance_to_next_sub_task_slot()

  def _fail_current_sub_task_via_reuse(self, memory: MemoryUnit, reason: str) -> None:
    strikes = self.bank.record_verification_failure(memory.memory_id, persist=False)
    self.episode_active_memory_ids.add(memory.memory_id)
    if strikes >= self.mutation_config.k_limit:
      self.bank.remove(memory.memory_id, persist=True)
      self.usage.memories_pruned_by_strikes += 1
    else:
      self.bank.save()
    self._fail_current_sub_task(f"replay rejected by Verifier ({reason}).")

  def _succeed_current_sub_task_via_fresh_generation(self) -> None:
    outcome = mutation.decide_memory_update(
        self.bank,
        self.current_sub_task.precondition,
        self.current_sub_task.goal,
        self.sub_task_trajectory,
        self.sub_task_candidate,
        source_task=self._current_task_name,
    )
    self.usage.record_memory_update(outcome.action)
    if outcome.memory_id is not None:
      self.episode_active_memory_ids.add(outcome.memory_id)
      if outcome.action == "replaced":
        # The mutation's trajectory "won" against the skipped candidate;
        # treat it like a confirmed-good reuse for risk bookkeeping too.
        self.risk_regulator.record_reuse_success(outcome.memory_id, persist=False)
        self.bank.record_reuse(outcome.memory_id, persist=True)
    self.task_history.append(
        f"Sub-task [{self.current_sub_task.as_prompt_str()}] COMPLETED"
        f" (fresh generation, memory_update={outcome.action})."
    )
    stall_warning = loop_guard.stall_warning_if_zero_action(
        self.current_sub_task.goal, len(self.sub_task_trajectory)
    )
    if stall_warning:
      self.task_history.append(stall_warning)
    self._advance_to_next_sub_task_slot()

  def _advance_after_fast_path(self, msg: str) -> None:
    """Marks the current sub-task COMPLETED via a deterministic fast path
    (no memory write or Verifier) and frees the slot."""
    assert self.current_sub_task is not None
    self.task_history.append(
        f"Sub-task [{self.current_sub_task.as_prompt_str()}] COMPLETED: {msg}"
    )
    self._advance_to_next_sub_task_slot()

  def _maybe_open_app_fast_path(
      self, step_data: dict[str, Any]
  ) -> Optional[base_agent.AgentInteractionResult]:
    """#3: deterministic handling of an explicit "open <app>" sub-goal. See
    `src/agent/app_intent.py`. Returns a result to short-circuit `step()`, or
    None to proceed with the normal Actor."""
    if self.sub_task_step_count != 0:
      return None
    decision = app_intent.open_app_fast_path_decision(
        self.env, self.current_sub_task.goal
    )
    if decision is None:
      return None
    kind, app_name = decision
    step_data["phase"] = "open_app_fast_path"
    step_data["sub_task"] = self.current_sub_task.to_dict()
    if kind == "already":
      step_data["action_reason"] = f"Already in the '{app_name}' app."
      self._advance_after_fast_path(
          f"already in the '{app_name}' app; no navigation needed."
      )
      return base_agent.AgentInteractionResult(False, step_data)
    try:
      self.env.execute_action(
          json_action.JSONAction(action_type="open_app", app_name=app_name)
      )
      self.usage.atomic_actions_executed += 1
      step_data["action_reason"] = f"Deterministic open_app({app_name})."
      self._record_episode_step(
          f"open_app({app_name})",
          {"action_type": "open_app", "app_name": app_name},
          target_element_desc=None,
      )
      time.sleep(self.wait_after_action_seconds)
      self._advance_after_fast_path(
          f"opened the '{app_name}' app directly (deterministic fast path)."
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._fail_current_sub_task(
          f"could not open app '{app_name}' directly ({e})."
      )
    return base_agent.AgentInteractionResult(False, step_data)

  def _maybe_open_initial_task_app_fast_path(
      self, step_data: dict[str, Any]
  ) -> Optional[base_agent.AgentInteractionResult]:
    """Uses task-suite app metadata only for the launcher-to-app prefix."""
    decision = app_intent.initial_task_app_fast_path_decision(
        self.env, self._task_apps, self._task_navigation_started
    )
    if decision is None:
      return None
    kind, app_name = decision
    self._task_navigation_started = True
    step_data["phase"] = "initial_task_app_fast_path"
    if kind == "already":
      step_data["action_reason"] = f"Already in declared task app '{app_name}'."
      self.task_history.append(
          f"[Task setup] Already in declared target app '{app_name}'."
      )
      return base_agent.AgentInteractionResult(False, step_data)
    try:
      self.env.execute_action(
          json_action.JSONAction(action_type="open_app", app_name=app_name)
      )
      self.usage.atomic_actions_executed += 1
      step_data["action_reason"] = (
          f"Opened declared target app '{app_name}' from task-suite metadata."
      )
      self.task_history.append(
          f"[Task setup] Opened declared target app '{app_name}' directly."
      )
      self._record_episode_step(
          f"open declared target app '{app_name}'",
          {"action_type": "open_app", "app_name": app_name},
          target_element_desc=None,
      )
      time.sleep(self.wait_after_action_seconds)
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.task_history.append(
          f"[Task setup] Could not open declared target app '{app_name}': {e}."
      )
    return base_agent.AgentInteractionResult(False, step_data)

  def _maybe_open_quick_settings_fast_path(
      self, step_data: dict[str, Any], task_goal: str
  ) -> Optional[base_agent.AgentInteractionResult]:
    """Opens Quick Settings for a narrow launcher/system-toggle gesture."""
    if self.sub_task_step_count != 0:
      return None
    decision = system_ui_intent.quick_settings_fast_path_decision(
        self.env, self.current_sub_task.goal, task_goal
    )
    if decision is None:
      return None
    step_data["phase"] = "system_navigation_fast_path"
    step_data["sub_task"] = self.current_sub_task.to_dict()
    try:
      if decision == "open_settings":
        self.env.execute_action(
            json_action.JSONAction(action_type="open_app", app_name="Settings")
        )
        progress_message = "opened Settings directly for a system-toggle task."
        action_reason = (
            "Deterministic open_app(Settings) for a launcher system-toggle"
            " task; Settings exposes labeled controls more reliably than"
            " generic Quick Settings switches."
        )
        self._record_episode_step(
            "open Settings for system-toggle task",
            {"action_type": "open_app", "app_name": "Settings"},
            target_element_desc=None,
        )
      else:
        self.env.execute_action(
            json_action.JSONAction(action_type="swipe", direction="down")
        )
        progress_message = (
            "opened Quick Settings with a deterministic top-edge pull-down."
        )
        action_reason = (
            "Deterministic top-edge pull-down to open Quick Settings."
        )
        self._record_episode_step(
            "pull down Quick Settings",
            {"action_type": "swipe", "direction": "down"},
            target_element_desc=None,
        )
      self.usage.atomic_actions_executed += 1
      step_data["action_reason"] = action_reason
      time.sleep(self.wait_after_action_seconds)
      self._advance_after_fast_path(progress_message)
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._fail_current_sub_task(
          f"could not open Quick Settings directly ({e})."
      )
    return base_agent.AgentInteractionResult(False, step_data)

  def _maybe_advance_system_toggle_fast_path(
      self,
      step_data: dict[str, Any],
      task_goal: str,
      ui_elements,
      logical_screen_size: tuple[int, int],
  ) -> Optional[base_agent.AgentInteractionResult]:
    decision = system_ui_intent.next_labeled_system_toggle_action(
        self.env, task_goal, ui_elements, logical_screen_size,
        tuple(self._system_toggle_route_labels),
    )
    if decision is None:
      return None
    action_dict, label = decision
    action_dict, _ = action_utils.reground_action(
        action_dict, ui_elements, label, f"click '{label}'",
        logical_screen_size,
    )
    if action_dict.get("index") == -1:
      fallback_index = ui_utils.find_element_by_description(
          ui_elements, label, logical_screen_size
      )
      if fallback_index is not None:
        action_dict["index"] = fallback_index
    step_data["phase"] = "system_toggle_fast_path"
    try:
      self.env.execute_action(json_action.JSONAction(**action_dict))
      self.usage.atomic_actions_executed += 1
      step_data["action_reason"] = (
          f"Clicked labeled Android Settings route '{label}' for system toggle."
      )
      self.task_history.append(
          f"[System toggle setup] Clicked labeled Settings target '{label}'."
      )
      self._system_toggle_route_labels.append(label)
      self._record_episode_step(
          f"click labeled Settings target '{label}'",
          action_dict,
          target_element_desc=label,
      )
      time.sleep(self.wait_after_action_seconds)
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.task_history.append(
          f"[System toggle setup] Could not click Settings target '{label}': {e}."
      )
    return base_agent.AgentInteractionResult(False, step_data)

  def _verify_and_finish_fresh_sub_task(
      self, final_screenshot, observation_degraded: bool = False
  ) -> None:
    if observation_degraded:
      # The a11y/UI observation used to ground this sub-task's actions (and
      # this verification screenshot) was degenerate (see
      # `ui_utils.get_robust_state`) -- the Actor was effectively blind and
      # any "history" it produced is not trustworthy grounding. Skip the
      # Verifier's default History-First trust and fail closed instead of
      # risking a hallucinated success that would otherwise loop forever.
      self._fail_current_sub_task(
          "Skipped verification: environment observation was degraded"
          " (a11y tree still empty after retries)."
      )
      return
    self.usage.verifier_calls += 1
    verifier_output = self.verifier.verify(
        self.current_sub_task.goal, self.sub_task_history, final_screenshot
    )
    self.usage.add(verifier_output.raw_response)
    # #6: a zero-action "success" on a sub-goal whose text implies a real
    # interaction (click/toggle/type/...) is the "found it = done" exploit,
    # not a genuinely pre-satisfied state. Veto it up front (before the
    # softer repetition breaker) so the very first occurrence forces a
    # concrete interaction / replan instead of a hallucinated completion.
    if (
        verifier_output.verified_success
        and len(self.sub_task_trajectory) == 0
        and loop_guard.goal_requires_interaction(self.current_sub_task.goal)
    ):
      self._fail_current_sub_task(
          "Rejected zero-action completion: the sub-goal requires a concrete"
          " UI interaction (click/type/toggle/...), but the Actor performed"
          " no action; forcing a real interaction / replan."
      )
      return
    if verifier_output.verified_success and self._repetition_breaker.record_and_check(
        self.current_sub_task.goal, len(self.sub_task_trajectory)
    ):
      self._fail_current_sub_task(
          "Stalled: repeated the same zero-action sub-goal completion"
          f" {self._repetition_breaker.max_repeats}x in a row without any"
          " real interaction; forcing failure to break the loop and force"
          " a genuine replan."
      )
    elif verifier_output.verified_success:
      self._succeed_current_sub_task_via_fresh_generation()
    else:
      self._fail_current_sub_task(
          f"Verifier rejected fresh trajectory ({verifier_output.reason})."
      )

  def _dispatch_next_sub_task(self) -> mutation.RetrievalDecision:
    """Pops the next queued sub-task and runs its retrieval/risk/epsilon
    decision (Algorithm 1 lines 9-13). Does NOT execute anything yet."""
    self.current_sub_task = self.sub_plan_queue.pop(0)
    self.sub_task_history = []
    self.sub_task_trajectory = []
    self.sub_task_step_count = 0

    decision = mutation.decide_retrieval_and_reuse(
        self.bank,
        self.risk_regulator,
        self.current_sub_task.precondition,
        self.current_sub_task.goal,
        self.mutation_config,
        self._rng,
    )
    self.usage.retrieval_attempts += 1
    if decision.candidate is not None:
      self.usage.retrieval_hits += 1
      if not decision.do_reuse:
        self.usage.mutation_attempts += 1
    self.sub_task_candidate = decision.candidate
    return decision

  # -- Main loop -------------------------------------------------------------
  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    step_data: dict[str, Any] = {
        "phase": None,
        "sub_task": None,
        "planner_message": None,
        "retrieval": None,
        "action_reason": None,
        "action_output_json": None,
        "raw_screenshot": None,
    }

    state, state_degraded = ui_utils.get_robust_state(self.env)
    step_data["state_degraded"] = state_degraded
    logical_screen_size = self.env.logical_screen_size
    orientation = self.env.orientation
    physical_frame_boundary = self.env.physical_frame_boundary
    ui_elements = state.ui_elements
    ui_elements_str = ui_utils.describe_ui_elements(
        ui_elements, logical_screen_size
    )
    raw_screenshot = state.pixels.copy()
    step_data["raw_screenshot"] = raw_screenshot
    som_screenshot = ui_utils.build_som_screenshot(
        raw_screenshot,
        ui_elements,
        logical_screen_size,
        physical_frame_boundary,
        orientation,
    )

    # Snapshot the pre-action UI signature once per step so every fast-path /
    # Actor action recorded into `episode_trajectory` shares a consistent
    # state_signature (used by the stagnant-trajectory write hygiene filter).
    self._current_state_signature = hashlib.sha256(
        ui_elements_str.encode("utf-8")
    ).hexdigest()
    # The eval harness drives the agent with `step(goal)` per atomic action;
    # the goal string is identical across all steps of one episode, so capture
    # it for the task-level memory key on the first step.
    self._episode_task_goal = goal

    # -- DMS task-level Retrieve->Replay (Algorithm 1 lines 9-13 lifted to
    # whole-task granularity for fast-path-solved tasks). Attempted exactly
    # once per episode, BEFORE the deterministic fast paths preempt, so a
    # stored whole-task memory can be replayed instead of re-solved. On round
    # 0 the bank is empty and this is a no-op; from round 1 on, a hit short-
    # circuits the fast paths / Planner via replay. --
    if (
        self.current_sub_task is None
        and not self.sub_plan_queue
        and not self._task_replay_attempted
        and self._episode_task_goal
    ):
      self._task_replay_attempted = True
      task_decision = mutation.decide_retrieval_and_reuse(
          self.bank,
          self.risk_regulator,
          _TASK_LEVEL_PRECONDITION,
          self._episode_task_goal,
          self.mutation_config,
          self._rng,
          precondition_must_equal=_TASK_LEVEL_PRECONDITION,
      )
      self.usage.retrieval_attempts += 1
      if task_decision.candidate is not None:
        self.usage.retrieval_hits += 1
        self._task_level_candidate = task_decision.candidate
        if task_decision.do_reuse:
          self.current_sub_task = SubPlan(
              precondition=_TASK_LEVEL_PRECONDITION,
              goal=self._episode_task_goal,
          )
          self.sub_task_candidate = task_decision.candidate
          step_data["phase"] = "task_replay"
          step_data["sub_task"] = self.current_sub_task.to_dict()
          step_data["retrieval"] = {
              "hit": True,
              "score": task_decision.score,
              "do_reuse": True,
              "is_mutation": False,
              "is_risk_suppressed": False,
          }
          self.usage.replay_attempts += 1
          return self._execute_task_level_replay(
              step_data, task_decision.candidate
          )
        # Candidate existed but was skipped (epsilon-mutation or risk
        # suppression): fall through to fresh generation; `_task_level_candidate`
        # is kept for the evolutionary-replacement path in `finalize_task`.
        self.usage.mutation_attempts += 1

    initial_task_app_result = self._maybe_open_initial_task_app_fast_path(
        step_data
    )
    if initial_task_app_result is not None:
      return initial_task_app_result
    system_toggle_result = self._maybe_advance_system_toggle_fast_path(
        step_data, goal, ui_elements, logical_screen_size
    )
    if system_toggle_result is not None:
      return system_toggle_result

    # --- Planning phase: only entered when there is no active sub-task
    # AND no queued sub-tasks left from the previous plan cycle. ---
    if self.current_sub_task is None and not self.sub_plan_queue:
      step_data["phase"] = "plan"
      self.usage.planner_calls += 1
      self.replan_cycles += 1
      planner_output = self.planner.plan(
          goal, self.task_history, ui_elements_str,
          [raw_screenshot, som_screenshot], task_apps=self._task_apps,
      )
      self.usage.add(planner_output.raw_response)
      step_data["planner_message"] = planner_output.message

      if not planner_output.parse_ok and planner_output.raw_response is None:
        # Do not repeatedly submit an identical failed multimodal request.
        # Finishing lets the evaluator record this cell and the resume-safe
        # runner can move to the next task.
        self.task_history.append(
            "[Planner transport failure] Ending episode for forward progress:"
            f" {planner_output.message}"
        )
        return base_agent.AgentInteractionResult(True, step_data)

      if planner_output.done:
        # Ground-truth evaluator termination (in the runner) is authoritative;
        # do not end an episode on a weak Planner's self-report.
        self._planner_completion_rejections += 1
        self.task_history.append(
            "[Planner completion rejected] AndroidWorld evaluator has not"
            f" satisfied goal={goal!r}; declared_target_apps={self._task_apps};"
            f" foreground_package={app_intent.current_app_package(self.env)!r};"
            f" completion_message={planner_output.message!r};"
            f" rejection_count={self._planner_completion_rejections}."
            " Return a concrete recovery interaction that changes the missing"
            " task state; do not repeat completion."
        )
        return base_agent.AgentInteractionResult(False, step_data)

      if not planner_output.sub_plans:
        self.task_history.append(
            "[Planner] Produced no valid sub-plans this cycle "
            f"({planner_output.message or 'parse failure'})."
        )
        return base_agent.AgentInteractionResult(False, step_data)

      self.sub_plan_queue = list(planner_output.sub_plans)

    # --- Dispatch phase: pop next sub-task + retrieval/risk/eps decision,
    # in the SAME call as either a REPLAY (executed fully here, since it
    # makes no LLM calls) or the first atomic step of fresh generation. ---
    if self.current_sub_task is None:
      step_data["phase"] = "dispatch"
      decision = self._dispatch_next_sub_task()
      step_data["sub_task"] = self.current_sub_task.to_dict()
      step_data["retrieval"] = {
          "hit": decision.candidate is not None,
          "score": decision.score,
          "do_reuse": decision.do_reuse,
          "is_mutation": decision.is_mutation,
          "is_risk_suppressed": decision.is_risk_suppressed,
      }

      if decision.do_reuse:
        step_data["phase"] = "replay"
        self.usage.replay_attempts += 1
        return self._execute_replay(step_data, decision.candidate)
      # Else: fall through to fresh Actor generation below, same call.

    # --- #3: deterministic open-app fast path (no LLM) before the Actor. ---
    fast_path_result = self._maybe_open_app_fast_path(step_data)
    if fast_path_result is not None:
      return fast_path_result
    fast_path_result = self._maybe_open_quick_settings_fast_path(
        step_data, goal
    )
    if fast_path_result is not None:
      return fast_path_result

    # --- Fresh Actor generation phase: one atomic action this call. ---
    step_data["phase"] = "act"
    step_data["sub_task"] = self.current_sub_task.to_dict()
    self.usage.actor_calls += 1
    self.sub_task_step_count += 1

    actor_output = self.actor.act(
        self.current_sub_task.as_prompt_str(),
        self.sub_task_history,
        ui_elements_str,
        [raw_screenshot, som_screenshot],
    )
    self.usage.add(actor_output.raw_response)

    if not actor_output.parse_ok:
      self.sub_task_history.append(
          "Action selection output was not in the correct format; no"
          " action performed."
      )
      if self.sub_task_step_count >= self.max_actor_steps_per_subtask:
        self._fail_current_sub_task(
            "Exceeded local step limit with repeated format errors."
        )
      return base_agent.AgentInteractionResult(False, step_data)

    reason, action_str = actor_output.reason, actor_output.action_json_str
    step_data["action_reason"] = reason

    try:
      # #4: sanitize recoverable schema noise (empty index, type/swipe
      # aliases, loose coordinates) BEFORE JSONAction, then re-ground an
      # index-based pointing action to the target the Actor actually named.
      action_dict = action_utils.normalize_action_dict(
          agent_utils.extract_json(action_str)
      )
      action_dict, reground_note = action_utils.reground_action(
          action_dict, ui_elements, reason,
          self.current_sub_task.goal, logical_screen_size,
      )
      if reground_note:
        reason = f"{reason} [{reground_note}]"
        step_data["action_reason"] = reason
      converted_action = json_action.JSONAction(**action_dict)
      step_data["action_output_json"] = converted_action
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.sub_task_history.append(
          f"Reason: {reason} Action: {action_str} -> FAILED to parse into a"
          f" valid action ({e})."
      )
      if self.sub_task_step_count >= self.max_actor_steps_per_subtask:
        self._fail_current_sub_task(
            "Exceeded local step limit with repeated invalid actions."
        )
      return base_agent.AgentInteractionResult(False, step_data)

    # Sub-task-level status declaration: intercepted, never sent to `env`.
    if converted_action.action_type == "status":
      if not self.sub_task_history:
        self.sub_task_history.append(
            loop_guard.annotate_zero_action_completion(
                reason, converted_action.goal_status
            )
        )
      else:
        self.sub_task_history.append(
            f"Reason: {reason} Action: declared sub-task"
            f" {converted_action.goal_status}."
        )
      # `infeasible` denotes an execution failure, not evidence of goal
      # completion. Do not let the History-First Verifier rubber-stamp it.
      if converted_action.goal_status == "infeasible":
        self._fail_current_sub_task(
            f"Actor declared sub-task infeasible: {reason}"
        )
        return base_agent.AgentInteractionResult(False, step_data)
      after_state, after_degraded = ui_utils.get_robust_state(self.env)
      self._verify_and_finish_fresh_sub_task(
          after_state.pixels, observation_degraded=after_degraded
      )
      return base_agent.AgentInteractionResult(False, step_data)

    num_ui_elements = len(ui_elements)
    if (
        converted_action.action_type in _INDEX_BASED_ACTIONS
        and converted_action.index is not None
        and converted_action.index >= num_ui_elements
    ):
      self.sub_task_history.append(
          f"Reason: {reason} Action: {action_str} -> FAILED: index out of"
          f" range (UI element list only has {num_ui_elements} elements)."
      )
      if self.sub_task_step_count >= self.max_actor_steps_per_subtask:
        self._fail_current_sub_task(
            "Exceeded local step limit with repeated out-of-range indices."
        )
      return base_agent.AgentInteractionResult(False, step_data)

    target_desc = None
    if (
        converted_action.action_type in _INDEX_BASED_ACTIONS
        and converted_action.index is not None
    ):
      target_desc = ui_utils.describe_target_element(
          ui_elements, converted_action.index
      )

    action_executed = False
    try:
      self.env.execute_action(converted_action)
      self.sub_task_history.append(f"Reason: {reason} Action: {action_str}")
      self.sub_task_trajectory.append(
          TrajectoryStep(
              reason=reason or "",
              action=converted_action.as_dict(),
              target_element_desc=target_desc,
              state_signature=hashlib.sha256(
                  ui_elements_str.encode("utf-8")
              ).hexdigest(),
          )
      )
      self._record_episode_step(
          reason, converted_action.as_dict(), target_element_desc=target_desc
      )
      self.usage.atomic_actions_executed += 1
      action_executed = True
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.sub_task_history.append(
          f"Reason: {reason} Action: {action_str} -> FAILED to execute"
          f" ({e})."
      )

    time.sleep(self.wait_after_action_seconds)

    state_signature = hashlib.sha256(
        ui_elements_str.encode("utf-8")
    ).hexdigest()
    if action_executed and self._stagnant_action_breaker.record_and_check(
        state_signature, converted_action.as_dict()
    ):
      self._fail_current_sub_task(
          "Stagnant: repeated the same action while the UI state was"
          " unchanged; forcing a replan."
      )
      return base_agent.AgentInteractionResult(False, step_data)

    if self.sub_task_step_count >= self.max_actor_steps_per_subtask:
      after_state, after_degraded = ui_utils.get_robust_state(self.env)
      self._verify_and_finish_fresh_sub_task(
          after_state.pixels, observation_degraded=after_degraded
      )

    return base_agent.AgentInteractionResult(False, step_data)

  # -- Replay execution (Sec 3.2.2 "Replay(tau_retrieved)") ----------------
  def _execute_replay(
      self, step_data: dict[str, Any], candidate: MemoryUnit
  ) -> base_agent.AgentInteractionResult:
    replay_result = mutation.replay_trajectory(
        candidate.trajectory, self.env, self.wait_after_action_seconds
    )
    step_data["replay_steps"] = replay_result.steps_replayed
    self.usage.replayed_actions_executed += replay_result.steps_replayed

    if not replay_result.success_execution:
      self._fail_current_sub_task_via_reuse(
          candidate,
          "could not faithfully replay (re-grounding/execution failure).",
      )
      return base_agent.AgentInteractionResult(False, step_data)

    step_data["action_reason"] = "Replayed stored trajectory from memory."
    if replay_result.observation_degraded:
      # Environment-side failure, not the memory's fault -- do NOT charge a
      # verification strike against `candidate` (that would unfairly bias
      # its Survival Value / risk bookkeeping for an a11y glitch it had no
      # part in). Just fail this sub-task cleanly and force a replan.
      self.episode_active_memory_ids.add(candidate.memory_id)
      self._fail_current_sub_task(
          "Skipped verification: environment observation was degraded"
          " during replay (a11y tree still empty after retries)."
      )
      return base_agent.AgentInteractionResult(False, step_data)

    self.usage.verifier_calls += 1
    verifier_output = self.verifier.verify(
        self.current_sub_task.goal, replay_result.history,
        replay_result.final_screenshot,
    )
    self.usage.add(verifier_output.raw_response)

    if verifier_output.verified_success:
      self.usage.replay_successes += 1
      self._succeed_current_sub_task_via_reuse(candidate)
    else:
      self._fail_current_sub_task_via_reuse(candidate, verifier_output.reason)

    return base_agent.AgentInteractionResult(False, step_data)

  # -- Task-level replay (whole-task memory, ground-truth-deferred) ---------
  def _execute_task_level_replay(
      self, step_data: dict[str, Any], candidate: MemoryUnit
  ) -> base_agent.AgentInteractionResult:
    """Replays a whole-task memory (Sec 3.2.2 Replay(tau_retrieved)) but,
    unlike the sub-task replay above, defers the success/strike verdict to
    the AndroidWorld ground-truth evaluator rather than the 7B LLM Verifier.

    The sub-task replay's LLM Verifier is a reasonable judge of a single
    sub-goal, but for a WHOLE-task trajectory the 7B Verifier is an unreliable
    judge of the overall task goal (it false-negatives multi-step state like
    "is Wi-Fi now on"), and a false negative would charge an unfair strike
    against a memory that actually worked -- eventually pruning good
    memories. The runner already trusts ground truth for episode termination,
    so we do the same here: execute the replay, mark it pending, and let
    `finalize_task(task_succeeded)` resolve the bookkeeping.
    """
    replay_result = mutation.replay_trajectory(
        candidate.trajectory, self.env, self.wait_after_action_seconds
    )
    step_data["replay_steps"] = replay_result.steps_replayed
    self.usage.replayed_actions_executed += replay_result.steps_replayed
    self.episode_active_memory_ids.add(candidate.memory_id)
    self._task_level_replay_pending = candidate

    if not replay_result.success_execution:
      self.task_history.append(
          f"Sub-task [{self.current_sub_task.as_prompt_str()}] task-level"
          " replay aborted (could not re-ground a stored action against the"
          " current UI); falling back to fresh generation."
      )
    else:
      step_data["action_reason"] = (
          "Replayed whole-task memory; verifier deferred to the AndroidWorld"
          " ground-truth evaluator."
      )
      self.task_history.append(
          f"Sub-task [{self.current_sub_task.as_prompt_str()}] replayed from"
          " task-level memory (verifier deferred to ground-truth evaluator)."
      )
    self._advance_to_next_sub_task_slot()
    return base_agent.AgentInteractionResult(False, step_data)
