# DMS reproduction project — Self-Regulation Strategy (paper Sec 3.2.3).
#
# Implements:
#   (1) The Survival Value S(m_i), a composite of Marginal Utility (with
#       Cold-Start Protection), Adaptive Temporal Decay, and a Reliability
#       Penalty:
#
#         U(n_i)              = ln(1 + n_i) + V_new
#         T_half(n_i)         = T_base + longevity_coeff * ln(1 + n_i)
#         D(delta_t, n_i)     = 1 / (1 + exp(beta * (delta_t - T_half(n_i))))
#         P(K_i)              = 1 / (1 + gamma * K_i)
#         S(m_i)              = U(n_i) * D(delta_t, n_i) * P(K_i)
#
#   (2) Adaptive Memory Regulation: the Elbow Method identifies the cutoff
#       index k* on the ranked survival-value curve f(k) = S(m_(k)) by
#       maximizing the discrete second-order gradient (curvature):
#
#         k* = argmax_k [f(k+1) - 2 f(k) + f(k-1)]
#
#       Pruning triggers once the bank reaches a preset capacity. If the
#       elbow value f(k*) is still >= the population mean (a "high-quality
#       saturation" signal), the system EXPANDS capacity instead of
#       pruning; otherwise it prunes the long tail beyond k*.
#
# Hyperparameters mirror Appendix B ("Hyperparameters Setting"): V_new=1.0,
# T_base=30.0, longevity_coeff=15.0 (reported there as "alpha", though the
# body text's T_half formula calls it "mu" -- we use a distinct name here
# to avoid clashing with the Bayesian risk mu_i introduced in Sec 3.2.4),
# beta=0.5, gamma=1.0. The absolute memory-bank capacity (Cmin/Cmax/step)
# is NOT given numerically in the paper (dataset/scale-dependent); we pick
# defaults sized for our ~29-task / 5-round / 3-condition experiment
# matrix and document this choice in the final report.

from __future__ import annotations

import dataclasses
import json
import math
import os
from typing import Optional

from src.memory.memory_bank import MemoryBank
from src.memory.memory_unit import MemoryUnit

_REGULATION_STATE_FILE = "regulation_state.json"


@dataclasses.dataclass
class SurvivalValueConfig:
  """Hyperparameters for S(m_i), defaults per Appendix B."""

  v_new: float = 1.0            # V_new: cold-start novelty bonus.
  t_base: float = 30.0          # T_base: base protection period.
  longevity_coeff: float = 15.0  # T_half's usage-sensitivity coefficient.
  beta: float = 0.5             # Decay steepness.
  gamma: float = 1.0            # Reliability-penalty severity.


@dataclasses.dataclass
class RegulationConfig:
  """Elbow-Method adaptive capacity regulation ("Adaptive Memory Regulation")."""

  initial_capacity: int = 60
  capacity_max: int = 240
  capacity_step: int = 20
  min_memories_for_elbow: int = 5


def compute_survival_value(
    memory: MemoryUnit,
    now: float,
    config: Optional[SurvivalValueConfig] = None,
) -> float:
  """S(m_i) = [ln(1+n_i) + V_new] * D(delta_t, n_i) * P(K_i)."""
  config = config or SurvivalValueConfig()
  n_i = memory.reuse_count
  delta_t = max(0.0, now - memory.last_used_at)

  utility = math.log1p(n_i) + config.v_new
  t_half = config.t_base + config.longevity_coeff * math.log1p(n_i)
  decay = 1.0 / (1.0 + math.exp(config.beta * (delta_t - t_half)))
  reliability = 1.0 / (1.0 + config.gamma * memory.verification_strikes)

  return utility * decay * reliability


def recompute_all(
    bank: MemoryBank,
    config: Optional[SurvivalValueConfig] = None,
    persist: bool = True,
) -> dict[str, float]:
  """Recomputes + caches S(m_i) for every memory currently in `bank`."""
  config = config or SurvivalValueConfig()
  now = bank.now
  values: dict[str, float] = {}
  for memory in bank.all_memories():
    value = compute_survival_value(memory, now, config)
    values[memory.memory_id] = value
    bank.update_survival_value(memory.memory_id, value, persist=False)
  if persist:
    bank.save()
  return values


def find_elbow_index(sorted_desc_values: list[float]) -> Optional[int]:
  """k* = argmax_k discrete curvature, over a descending-sorted value curve.

  For a curve that first declines gently, then drops off a "cliff" into a
  long tail, the discrete 2nd derivative peaks at the FIRST point of the
  tail (curvature measures the bend from steep-decline into shallow-decline,
  which happens right as the tail begins) rather than the last point of the
  head. We therefore treat the returned index k* as the start of the long
  tail: elements `[0, k*)` are the "core" memories to keep, `[k*, N)` is the
  long tail (candidate for pruning).

  Returns a 0-based index into `sorted_desc_values`. Must be an interior
  point (needs both a left and right neighbor to compute a 2nd derivative),
  so returns None if there are fewer than 3 values.
  """
  n = len(sorted_desc_values)
  if n < 3:
    return None
  best_idx, best_curvature = None, -math.inf
  for k in range(1, n - 1):
    curvature = (
        sorted_desc_values[k + 1]
        - 2 * sorted_desc_values[k]
        + sorted_desc_values[k - 1]
    )
    if curvature > best_curvature:
      best_curvature = curvature
      best_idx = k
  return best_idx


@dataclasses.dataclass
class RegulationResult:
  action: str  # "none" | "expand" | "prune"
  bank_size_before: int
  bank_size_after: int
  capacity_cap_before: int
  capacity_cap_after: int
  elbow_index: Optional[int] = None
  elbow_value: Optional[float] = None
  population_mean: Optional[float] = None
  pruned_memory_ids: list[str] = dataclasses.field(default_factory=list)


class SelfRegulator:
  """Owns the adaptive capacity cap + drives Elbow-Method pruning/expansion
  for one MemoryBank (Sec 3.2.3)."""

  def __init__(
      self,
      bank: MemoryBank,
      survival_config: Optional[SurvivalValueConfig] = None,
      regulation_config: Optional[RegulationConfig] = None,
  ):
    self.bank = bank
    self.survival_config = survival_config or SurvivalValueConfig()
    self.regulation_config = regulation_config or RegulationConfig()
    self.capacity_cap = self.regulation_config.initial_capacity
    self._load_state()

  @property
  def _state_path(self) -> str:
    return os.path.join(self.bank.store_dir, _REGULATION_STATE_FILE)

  def _load_state(self) -> None:
    if os.path.exists(self._state_path):
      with open(self._state_path, "r", encoding="utf-8") as f:
        data = json.load(f)
      self.capacity_cap = int(data.get("capacity_cap", self.capacity_cap))

  def _save_state(self) -> None:
    with open(self._state_path, "w", encoding="utf-8") as f:
      json.dump({"capacity_cap": self.capacity_cap}, f)

  def ranked_survival_curve(self) -> list[tuple[MemoryUnit, float]]:
    """Returns (memory, S(m_i)) pairs sorted by survival value, descending."""
    values = recompute_all(self.bank, self.survival_config)
    memories = self.bank.all_memories()
    ranked = sorted(memories, key=lambda m: values[m.memory_id], reverse=True)
    return [(m, values[m.memory_id]) for m in ranked]

  def regulate(self) -> RegulationResult:
    """Runs one Self-Regulation cycle: recompute S(m_i) for all memories,
    then (if at/over capacity) apply the Elbow Method to either prune the
    long tail or expand the capacity cap on high-quality saturation."""
    ranked = self.ranked_survival_curve()
    size_before = len(ranked)
    cap_before = self.capacity_cap

    if size_before < self.capacity_cap:
      return RegulationResult(
          action="none",
          bank_size_before=size_before,
          bank_size_after=size_before,
          capacity_cap_before=cap_before,
          capacity_cap_after=self.capacity_cap,
      )

    scores = [score for _, score in ranked]
    if len(scores) < self.regulation_config.min_memories_for_elbow:
      return RegulationResult(
          action="none",
          bank_size_before=size_before,
          bank_size_after=size_before,
          capacity_cap_before=cap_before,
          capacity_cap_after=self.capacity_cap,
      )

    elbow_idx = find_elbow_index(scores)
    if elbow_idx is None:
      return RegulationResult(
          action="none",
          bank_size_before=size_before,
          bank_size_after=size_before,
          capacity_cap_before=cap_before,
          capacity_cap_after=self.capacity_cap,
      )

    elbow_value = scores[elbow_idx]
    population_mean = sum(scores) / len(scores)

    if elbow_value >= population_mean:
      # High-quality saturation (f(k*) >= mean): expand rather than prune.
      self.capacity_cap = min(
          self.capacity_cap + self.regulation_config.capacity_step,
          self.regulation_config.capacity_max,
      )
      self._save_state()
      return RegulationResult(
          action="expand",
          bank_size_before=size_before,
          bank_size_after=size_before,
          capacity_cap_before=cap_before,
          capacity_cap_after=self.capacity_cap,
          elbow_index=elbow_idx,
          elbow_value=elbow_value,
          population_mean=population_mean,
      )

    # Otherwise: prune the long tail starting at the elbow.
    to_prune = [m for m, _ in ranked[elbow_idx:]]
    pruned_ids = [m.memory_id for m in to_prune]
    for memory in to_prune:
      self.bank.remove(memory.memory_id, persist=False)
    self.bank.save()

    return RegulationResult(
        action="prune",
        bank_size_before=size_before,
        bank_size_after=len(self.bank),
        capacity_cap_before=cap_before,
        capacity_cap_after=self.capacity_cap,
        elbow_index=elbow_idx,
        elbow_value=elbow_value,
        population_mean=population_mean,
        pruned_memory_ids=pruned_ids,
    )
