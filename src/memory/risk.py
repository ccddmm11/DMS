# DMS reproduction project — Risk Assessment and Feedback Regulation
# (paper Sec 3.2.4, and Algorithm 1's "Global Feedback Regulation Stage").
#
# Implements:
#   (1) Bayesian Reputation Modeling. Each plan/memory m_i has a latent
#       failure tendency theta_i in [0, 1]. Given F_i failures and S_i
#       successes, we use a Beta-Binomial conjugate prior smoothed towards
#       the system's current global failure rate T_global:
#
#         M           = alpha + beta                    (prior strength)
#         T_global    = alpha / (alpha + beta)           (global failure rate)
#         mu_i        = (F_i + M * T_global) / (F_i + S_i + M)
#
#   (2) Uncertainty-Aware Risk Scoring (Lower Confidence Bound):
#
#         sigma_i     = sqrt(mu_i * (1 - mu_i) / (F_i + S_i + M + 1))
#         T_i         = mu_i - sigma_i          (final "Risk Score")
#
#   (3) Dynamic Thresholding: the rejection threshold tightens as the
#       ecosystem's global failure rate rises:
#
#         tau         = tau_base * (1 - lambda * T_global)
#
#       Plans/memories with T_i > tau are suppressed (Algorithm 1 line 10:
#       reuse is only allowed when rho_m < tau_risk).
#
# `lambda` = 0.3 is given in Appendix B. The paper treats alpha/beta (or
# equivalently M and T_global) as given inputs to Algorithm 1, but T_global
# is explicitly described as "the global failure rate" -- i.e. an
# empirical, evolving statistic of the whole system, not a fixed constant.
# We therefore track it as a live running counter (`GlobalRiskStats`) over
# GLOBAL TASK outcomes (Algorithm 1's `R_task`), persisted alongside the
# MemoryBank. The prior strength M and tau_base are NOT given numerically
# in the paper (only lambda is); we pick modest defaults (M=4.0 pseudo
# observations, tau_base=0.5) sized for our smaller task suite and
# document this choice in the final report.

from __future__ import annotations

import dataclasses
import json
import math
import os
from typing import Optional

from src.memory.memory_bank import MemoryBank
from src.memory.memory_unit import MemoryUnit

_RISK_STATE_FILE = "risk_state.json"


@dataclasses.dataclass
class BayesianRiskConfig:
  """Hyperparameters for Bayesian reputation modeling + dynamic threshold."""

  prior_strength: float = 4.0      # M = alpha + beta (not given in paper).
  tau_base: float = 0.5            # Base rejection threshold (not given).
  lambda_sensitivity: float = 0.3  # Appendix B: lambda = 0.3.
  default_global_failure_rate: float = 0.5  # T_global before any data.


@dataclasses.dataclass
class GlobalRiskStats:
  """Running GLOBAL TASK-level failure/success counters -> T_global."""

  global_failures: float = 0.0
  global_successes: float = 0.0

  @property
  def total(self) -> float:
    return self.global_failures + self.global_successes

  def global_failure_rate(self, config: BayesianRiskConfig) -> float:
    if self.total <= 0:
      return config.default_global_failure_rate
    return self.global_failures / self.total

  def record_task_outcome(self, success: bool) -> None:
    if success:
      self.global_successes += 1
    else:
      self.global_failures += 1

  def to_dict(self) -> dict[str, float]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, d: dict[str, float]) -> "GlobalRiskStats":
    return cls(
        global_failures=float(d.get("global_failures", 0.0)),
        global_successes=float(d.get("global_successes", 0.0)),
    )


@dataclasses.dataclass
class RiskAssessment:
  mu: float
  sigma: float
  risk_score: float
  threshold: float

  @property
  def suppressed(self) -> bool:
    return self.risk_score > self.threshold


def compute_bayesian_risk(
    failure_count: float,
    success_count: float,
    global_stats: GlobalRiskStats,
    config: Optional[BayesianRiskConfig] = None,
) -> tuple[float, float, float]:
  """Returns (mu_i, sigma_i, risk_score T_i) per Sec 3.2.4."""
  config = config or BayesianRiskConfig()
  t_global = global_stats.global_failure_rate(config)
  prior_strength = config.prior_strength
  alpha = prior_strength * t_global
  n = failure_count + success_count

  mu = (failure_count + alpha) / (n + prior_strength)
  variance = max(mu * (1.0 - mu), 0.0) / (n + prior_strength + 1.0)
  sigma = math.sqrt(variance)
  risk_score = mu - sigma
  return mu, sigma, risk_score


def dynamic_threshold(
    global_stats: GlobalRiskStats, config: Optional[BayesianRiskConfig] = None
) -> float:
  """tau = tau_base * (1 - lambda * T_global) (Sec 3.2.4)."""
  config = config or BayesianRiskConfig()
  t_global = global_stats.global_failure_rate(config)
  return config.tau_base * (1.0 - config.lambda_sensitivity * t_global)


class RiskRegulator:
  """Owns the global failure-rate state + drives Bayesian risk scoring and
  dynamic-threshold suppression for one MemoryBank (Sec 3.2.4)."""

  def __init__(
      self, bank: MemoryBank, config: Optional[BayesianRiskConfig] = None
  ):
    self.bank = bank
    self.config = config or BayesianRiskConfig()
    self.global_stats = GlobalRiskStats()
    self._load_state()

  @property
  def _state_path(self) -> str:
    return os.path.join(self.bank.store_dir, _RISK_STATE_FILE)

  def _load_state(self) -> None:
    if os.path.exists(self._state_path):
      with open(self._state_path, "r", encoding="utf-8") as f:
        self.global_stats = GlobalRiskStats.from_dict(json.load(f))

  def _save_state(self) -> None:
    with open(self._state_path, "w", encoding="utf-8") as f:
      json.dump(self.global_stats.to_dict(), f)

  def get_dynamic_threshold(self) -> float:
    return dynamic_threshold(self.global_stats, self.config)

  def assess(self, memory: MemoryUnit) -> RiskAssessment:
    mu, sigma, risk_score = compute_bayesian_risk(
        memory.failure_count, memory.success_count, self.global_stats,
        self.config,
    )
    return RiskAssessment(mu, sigma, risk_score, self.get_dynamic_threshold())

  def refresh_risk(
      self, memory_id: str, persist: bool = True
  ) -> Optional[RiskAssessment]:
    """Recomputes + caches (mu_i, risk_score) for one memory."""
    memory = self.bank.get(memory_id, load_trajectory=False)
    if memory is None:
      return None
    assessment = self.assess(memory)
    self.bank.update_risk_stats(
        memory_id,
        failure_count=memory.failure_count,
        success_count=memory.success_count,
        risk_mu=assessment.mu,
        risk_score=assessment.risk_score,
        persist=persist,
    )
    return assessment

  def is_suppressed(self, memory_id: str) -> bool:
    """Whether a memory's cached risk_score currently exceeds tau. Callers
    should have called `refresh_risk`/`record_*` recently so the cached
    score reflects the latest global failure rate."""
    memory = self.bank.get(memory_id, load_trajectory=False)
    if memory is None:
      return False
    return memory.risk_score > self.get_dynamic_threshold()

  def record_reuse_success(self, memory_id: str, persist: bool = True) -> None:
    """Algorithm 1 line 18 (Bayesian side): a reused memory's sub-task
    succeeded -> S_i += 1, then refresh its risk score."""
    self.bank.increment_success_count(memory_id, persist=False)
    self.refresh_risk(memory_id, persist=persist)

  def record_global_task_outcome(
      self, active_memory_ids: list[str], success: bool
  ) -> dict[str, RiskAssessment]:
    """Algorithm 1 lines 40-46 'Global Feedback Regulation Stage': updates
    the global failure rate T_global, and -- only on a FAILED global task --
    charges every memory that was active this episode (`L_active`) with
    F_i += 1, then refreshes every active memory's (mu_i, risk_score)."""
    self.global_stats.record_task_outcome(success)
    self._save_state()

    if not success:
      for memory_id in active_memory_ids:
        self.bank.increment_failure_count(memory_id, persist=False)

    assessments = {}
    for memory_id in active_memory_ids:
      assessment = self.refresh_risk(memory_id, persist=False)
      if assessment is not None:
        assessments[memory_id] = assessment
    self.bank.save()
    return assessments
