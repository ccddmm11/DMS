# DMS reproduction project — MemoryUnit: the atomic unit of the Memory Bank.
#
# Mirrors the memory entry defined in the paper (Sec 3.2.1 + Fig. 2):
#
#   m = (p, tau, s_meta)
#
# where `p = <Precondition, Goal>` is the natural-language plan serving as
# the semantic index, `tau = {(o_0, a_0), ..., (o_T, a_T)}` is the dense
# execution trajectory produced by the Actor, and `s_meta` carries metadata
# (success status, description).
#
# We additionally carry the bookkeeping fields shown in Fig. 2's memory
# card ("Reuse Count", "Created/Last used Time") plus the fields required
# by later mechanisms (Sec 3.2.3 Self-Regulation, Sec 3.2.4 Risk
# Assessment) so the on-disk schema is stable across the memory-core /
# survival-pruning / risk-feedback implementation steps -- only the
# *update logic* for those fields is added later, not the schema itself.

from __future__ import annotations

import dataclasses
import time
import uuid
from typing import Any, Optional


@dataclasses.dataclass
class TrajectoryStep:
  """One (o_t, a_t) pair in the dense execution trajectory tau.

  We do not persist raw screenshots (too heavy for long-term disk storage,
  and not needed for blind replay); instead we keep enough textual context
  to both (a) faithfully replay the action, and (b) re-ground an
  index-based action against a *new* UI element list if indices shift
  between the original recording and a later replay.
  """

  reason: str
  action: dict[str, Any]  # `json_action.JSONAction.as_dict()`-compatible.
  target_element_desc: Optional[str] = None
  state_signature: Optional[str] = None

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, d: dict[str, Any]) -> "TrajectoryStep":
    return cls(
        reason=d.get("reason", ""),
        action=d.get("action", {}),
        target_element_desc=d.get("target_element_desc"),
        state_signature=d.get("state_signature"),
    )


@dataclasses.dataclass
class MemoryUnit:
  """A single DMS memory entry m = (p, tau, s_meta)."""

  # --- Semantic index p = <precondition, goal> ---
  precondition: str
  goal: str

  # --- Dense trajectory tau ---
  trajectory: list[TrajectoryStep] = dataclasses.field(default_factory=list)

  # --- Metadata s_meta ---
  success: bool = True
  description: str = ""
  source_task: Optional[str] = None

  # --- Identity & bookkeeping (Fig. 2 memory card) ---
  memory_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
  created_at: float = 0.0    # Logical time (see MemoryBank's logical clock).
  last_used_at: float = 0.0
  reuse_count: int = 0       # n_i

  # --- Self-Regulation (Sec 3.2.3), filled in by the survival-pruning step.
  verification_strikes: int = 0   # K_i: accumulated verification failures.
  survival_value: float = 0.0     # S(m_i), cached from the last computation.

  # --- Risk Assessment (Sec 3.2.4), filled in by the risk-feedback step.
  failure_count: float = 0.0   # F_i: Beta-Binomial failure observations.
  success_count: float = 0.0   # S_i: Beta-Binomial success observations.
  risk_mu: float = 0.0         # mu_i: Bayesian-smoothed failure probability.
  risk_score: float = 0.0      # T_i = mu_i - sigma_i (LCB risk score).

  def as_prompt_str(self) -> str:
    return f"Precondition: {self.precondition} Goal: {self.goal}"

  def to_meta_dict(self) -> dict[str, Any]:
    """Metadata WITHOUT the dense trajectory (for the fast in-memory index)."""
    d = dataclasses.asdict(self)
    d.pop("trajectory")
    return d

  def to_full_dict(self) -> dict[str, Any]:
    """Full dict INCLUDING the dense trajectory (for on-disk persistence)."""
    d = self.to_meta_dict()
    d["trajectory"] = [step.to_dict() for step in self.trajectory]
    return d

  @classmethod
  def from_full_dict(cls, d: dict[str, Any]) -> "MemoryUnit":
    d = dict(d)
    trajectory = [
        TrajectoryStep.from_dict(step) for step in d.pop("trajectory", [])
    ]
    return cls(trajectory=trajectory, **d)

  @classmethod
  def from_meta_dict(
      cls, d: dict[str, Any], trajectory: Optional[list[TrajectoryStep]] = None
  ) -> "MemoryUnit":
    d = dict(d)
    return cls(trajectory=trajectory or [], **d)
