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
import random
import time
from typing import Any, Optional

from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.env import interface
from android_world.env import json_action

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

    self.additional_guidelines = None  # kept for API parity with M3A/T3A.
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
  def start_new_task(self, task_name: Optional[str] = None) -> None:
    self._current_task_name = task_name

  def finalize_task(self, task_succeeded: bool) -> Optional[RegulationResult]:
    """Algorithm 1 lines 40-46 (Global Feedback Regulation Stage) + a
    periodic Self-Regulation (Elbow-Method) pass. Returns the
    `RegulationResult` if a regulation cycle ran this call, else None."""
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

  def _advance_to_next_sub_task_slot(self) -> None:
    """Clears the "active sub-task" slot; the NEXT `step()` call will pop
    the next queued sub-task (and run its retrieval decision) or, if the
    queue is empty, re-invoke the Planner."""
    self.current_sub_task = None
    self.sub_task_candidate = None
    self.sub_task_history = []
    self.sub_task_trajectory = []
    self.sub_task_step_count = 0

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
    self._advance_to_next_sub_task_slot()

  def _verify_and_finish_fresh_sub_task(self, final_screenshot) -> None:
    self.usage.verifier_calls += 1
    verifier_output = self.verifier.verify(
        self.current_sub_task.goal, self.sub_task_history, final_screenshot
    )
    self.usage.add(verifier_output.raw_response)
    if verifier_output.verified_success:
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

    state = self.get_post_transition_state()
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

    # --- Planning phase: only entered when there is no active sub-task
    # AND no queued sub-tasks left from the previous plan cycle. ---
    if self.current_sub_task is None and not self.sub_plan_queue:
      step_data["phase"] = "plan"
      self.usage.planner_calls += 1
      self.replan_cycles += 1
      planner_output = self.planner.plan(
          goal, self.task_history, ui_elements_str,
          [raw_screenshot, som_screenshot],
      )
      self.usage.add(planner_output.raw_response)
      step_data["planner_message"] = planner_output.message

      if planner_output.done:
        return base_agent.AgentInteractionResult(True, step_data)

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
      converted_action = json_action.JSONAction(
          **agent_utils.extract_json(action_str)
      )
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
      self.sub_task_history.append(
          f"Reason: {reason} Action: declared sub-task"
          f" {converted_action.goal_status}."
      )
      after_state = self.env.get_state(wait_to_stabilize=False)
      self._verify_and_finish_fresh_sub_task(after_state.pixels)
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

    try:
      self.env.execute_action(converted_action)
      self.sub_task_history.append(f"Reason: {reason} Action: {action_str}")
      self.sub_task_trajectory.append(
          TrajectoryStep(
              reason=reason or "",
              action=converted_action.as_dict(),
              target_element_desc=target_desc,
          )
      )
      self.usage.atomic_actions_executed += 1
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.sub_task_history.append(
          f"Reason: {reason} Action: {action_str} -> FAILED to execute"
          f" ({e})."
      )

    time.sleep(self.wait_after_action_seconds)

    if self.sub_task_step_count >= self.max_actor_steps_per_subtask:
      after_state = self.env.get_state(wait_to_stabilize=False)
      self._verify_and_finish_fresh_sub_task(after_state.pixels)

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

    self.usage.verifier_calls += 1
    verifier_output = self.verifier.verify(
        self.current_sub_task.goal, replay_result.history,
        replay_result.final_screenshot,
    )
    self.usage.add(verifier_output.raw_response)
    step_data["action_reason"] = "Replayed stored trajectory from memory."

    if verifier_output.verified_success:
      self.usage.replay_successes += 1
      self._succeed_current_sub_task_via_reuse(candidate)
    else:
      self._fail_current_sub_task_via_reuse(candidate, verifier_output.reason)

    return base_agent.AgentInteractionResult(False, step_data)
