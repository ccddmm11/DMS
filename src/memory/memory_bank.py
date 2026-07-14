# DMS reproduction project — Memory Bank: persistence + Dual-Factor Retrieval.
#
# Implements the "decoupled storage architecture" described in Sec 3.2.2:
# high-dimensional trajectories tau are persisted on disk, while their
# semantic summaries p = <precondition, goal> are encoded as dense index
# vectors kept in memory for fast retrieval. Concretely:
#
#   <store_dir>/index.jsonl          one line per MemoryUnit's metadata
#                                     (NOT including the dense trajectory)
#   <store_dir>/trajectories/<id>.json   the dense execution trajectory,
#                                         loaded lazily only on a hit
#   <store_dir>/embeddings.npz       cached phi(precondition)/phi(goal)
#                                     vectors, keyed by memory_id
#   <store_dir>/clock.json           the bank's logical clock (Sec 3.2.3
#                                     uses *logical* time steps, not wall
#                                     clock time, for the decay term)
#
# Retrieval implements the Dual-Factor Similarity Metric (Sec 3.2.2):
#
#   Score(p_hat, p) = sim(phi(p_hat_pre), phi(p_pre)) * sim(phi(p_hat_goal), phi(p_goal))
#
# Bookkeeping fields relevant to Self-Regulation (Sec 3.2.3) and Risk
# Assessment (Sec 3.2.4) are updated via small dedicated methods here
# (`record_reuse`, `record_verification_failure`, `update_risk_stats`,
# `update_survival_value`) so that the survival-pruning / risk-feedback
# implementation steps only need to *call* this bank, not change its
# schema or persistence format.

from __future__ import annotations

import dataclasses
import json
import os
import threading
from typing import Optional

import numpy as np

from src.memory.embedder import Embedder
from src.memory.embedder import cosine_similarity
from src.memory.embedder import get_default_embedder
from src.memory.memory_unit import MemoryUnit
from src.memory.memory_unit import TrajectoryStep

_INDEX_FILE = "index.jsonl"
_EMBEDDINGS_FILE = "embeddings.npz"
_CLOCK_FILE = "clock.json"
_TRAJECTORIES_DIR = "trajectories"


@dataclasses.dataclass
class RetrievalResult:
  memory: MemoryUnit
  score: float
  precondition_similarity: float
  goal_similarity: float


class MemoryBank:
  """A single condition's memory store (e.g. one DMS/Baseline-B run)."""

  def __init__(
      self,
      store_dir: str,
      embedder: Optional[Embedder] = None,
      load_existing: bool = True,
  ):
    self.store_dir = store_dir
    os.makedirs(self.store_dir, exist_ok=True)
    os.makedirs(os.path.join(self.store_dir, _TRAJECTORIES_DIR), exist_ok=True)
    self.embedder = embedder or get_default_embedder()

    self._lock = threading.Lock()
    self._memories: dict[str, MemoryUnit] = {}
    self._precond_emb: dict[str, np.ndarray] = {}
    self._goal_emb: dict[str, np.ndarray] = {}
    self._logical_time: float = 0.0

    if load_existing:
      self._load()

  # -- Paths ---------------------------------------------------------------
  @property
  def _index_path(self) -> str:
    return os.path.join(self.store_dir, _INDEX_FILE)

  @property
  def _embeddings_path(self) -> str:
    return os.path.join(self.store_dir, _EMBEDDINGS_FILE)

  @property
  def _clock_path(self) -> str:
    return os.path.join(self.store_dir, _CLOCK_FILE)

  def _trajectory_path(self, memory_id: str) -> str:
    return os.path.join(self.store_dir, _TRAJECTORIES_DIR, f"{memory_id}.json")

  # -- Logical clock (Sec 3.2.3: Delta_t is in logical time steps) --------
  @property
  def now(self) -> float:
    return self._logical_time

  def tick(self, n: float = 1.0, persist: bool = True) -> float:
    self._logical_time += n
    if persist:
      self._persist_clock()
    return self._logical_time

  # -- Basic container protocol ---------------------------------------------
  def __len__(self) -> int:
    return len(self._memories)

  def all_memories(self) -> list[MemoryUnit]:
    return list(self._memories.values())

  def get(
      self, memory_id: str, load_trajectory: bool = True
  ) -> Optional[MemoryUnit]:
    memory = self._memories.get(memory_id)
    if memory is None:
      return None
    if load_trajectory and not memory.trajectory:
      memory.trajectory = self._load_trajectory(memory_id)
    return memory

  # -- Mutation --------------------------------------------------------------
  def add(self, memory: MemoryUnit, persist: bool = True) -> MemoryUnit:
    """Adds a new memory. Filters `|tau| == 1` per Sec 3.2.1 (see caller)."""
    with self._lock:
      if not memory.created_at:
        memory.created_at = self.now
      if not memory.last_used_at:
        memory.last_used_at = memory.created_at
      self._memories[memory.memory_id] = memory
      self._precond_emb[memory.memory_id] = self.embedder.embed(
          memory.precondition
      )
      self._goal_emb[memory.memory_id] = self.embedder.embed(memory.goal)
    if persist:
      self._persist_trajectory(memory)
      self._persist_index()
      self._persist_embeddings()
      self._persist_clock()
    return memory

  def remove(self, memory_id: str, persist: bool = True) -> None:
    with self._lock:
      self._memories.pop(memory_id, None)
      self._precond_emb.pop(memory_id, None)
      self._goal_emb.pop(memory_id, None)
    traj_path = self._trajectory_path(memory_id)
    if os.path.exists(traj_path):
      os.remove(traj_path)
    if persist:
      self._persist_index()
      self._persist_embeddings()

  def replace_trajectory(
      self, memory_id: str, new_trajectory: list[TrajectoryStep]
  ) -> None:
    """In-place evolutionary update (Sec 3.2.2 epsilon-Mutation): overwrite
    an existing entry's trajectory (e.g. with a shorter/more efficient one),
    keeping its identity, index vectors and bookkeeping intact."""
    memory = self.get(memory_id, load_trajectory=False)
    if memory is None:
      raise KeyError(f"No such memory: {memory_id}")
    memory.trajectory = new_trajectory
    self._persist_trajectory(memory)

  # -- Bookkeeping updates (consumed by later pruning/risk steps) ---------
  def record_reuse(self, memory_id: str, persist: bool = True) -> None:
    """Algorithm 1 line 18: successful reuse -> n_i += 1, refresh dormancy."""
    memory = self._memories.get(memory_id)
    if memory is None:
      return
    memory.reuse_count += 1
    memory.last_used_at = self.now
    if persist:
      self._persist_index()

  def record_verification_failure(
      self, memory_id: str, persist: bool = True
  ) -> int:
    """Algorithm 1 line 25: reused memory failed -> K_i += 1 (strikes)."""
    memory = self._memories.get(memory_id)
    if memory is None:
      return 0
    memory.verification_strikes += 1
    if persist:
      self._persist_index()
    return memory.verification_strikes

  def update_risk_stats(
      self,
      memory_id: str,
      failure_count: float,
      success_count: float,
      risk_mu: float,
      risk_score: float,
      persist: bool = True,
  ) -> None:
    memory = self._memories.get(memory_id)
    if memory is None:
      return
    memory.failure_count = failure_count
    memory.success_count = success_count
    memory.risk_mu = risk_mu
    memory.risk_score = risk_score
    if persist:
      self._persist_index()

  def increment_success_count(
      self, memory_id: str, amount: float = 1.0, persist: bool = True
  ) -> None:
    """Algorithm 1 line 18 (Bayesian side): S_i += 1 on confirmed reuse."""
    memory = self._memories.get(memory_id)
    if memory is None:
      return
    memory.success_count += amount
    if persist:
      self._persist_index()

  def increment_failure_count(
      self, memory_id: str, amount: float = 1.0, persist: bool = True
  ) -> None:
    """Algorithm 1 line 42: F_i += 1 for every active memory on global fail."""
    memory = self._memories.get(memory_id)
    if memory is None:
      return
    memory.failure_count += amount
    if persist:
      self._persist_index()

  def update_survival_value(
      self, memory_id: str, value: float, persist: bool = True
  ) -> None:
    memory = self._memories.get(memory_id)
    if memory is None:
      return
    memory.survival_value = value
    if persist:
      self._persist_index()

  # -- Dual-Factor Retrieval (Sec 3.2.2) ------------------------------------
  def retrieve(
      self,
      precondition_query: str,
      goal_query: str,
      top_k: int = 1,
      score_threshold: float = 0.0,
      load_trajectory: bool = True,
      precondition_must_equal: Optional[str] = None,
  ) -> list[RetrievalResult]:
    """Retrieves the top-k memories by the Dual-Factor Similarity Metric.

    Score(p_hat, p) = sim(phi(p_hat_pre), phi(p_pre)) * sim(phi(p_hat_goal), phi(p_goal))

    `precondition_must_equal`: when set, only memories whose stored
    `precondition` string equals this value are considered. Used by the
    task-level Retrieve->Replay path to match ONLY whole-task memories
    (keyed by the shared `_TASK_LEVEL_PRECONDITION`), never sub-plan
    memories whose trajectory covers just one sub-goal.

    Note: this only *finds* candidates; it does NOT mutate reuse_count /
    last_used_at (those are updated via `record_reuse` only once the
    caller has confirmed the retrieved trajectory was actually replayed
    successfully, matching Algorithm 1's semantics).
    """
    if not self._memories:
      return []
    pre_q = self.embedder.embed(precondition_query)
    goal_q = self.embedder.embed(goal_query)

    scored = []
    for memory_id in self._memories:
      if (
          precondition_must_equal is not None
          and self._memories[memory_id].precondition != precondition_must_equal
      ):
        continue
      pre_sim = cosine_similarity(pre_q, self._precond_emb[memory_id])
      goal_sim = cosine_similarity(goal_q, self._goal_emb[memory_id])
      scored.append((memory_id, pre_sim * goal_sim, pre_sim, goal_sim))
    scored.sort(key=lambda item: item[1], reverse=True)

    results = []
    for memory_id, score, pre_sim, goal_sim in scored[:top_k]:
      if score < score_threshold:
        continue
      memory = self.get(memory_id, load_trajectory=load_trajectory)
      results.append(RetrievalResult(memory, score, pre_sim, goal_sim))
    return results

  # -- Persistence -----------------------------------------------------------
  def save(self) -> None:
    self._persist_index()
    self._persist_embeddings()
    self._persist_clock()

  def _persist_index(self) -> None:
    tmp_path = self._index_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
      for memory in self._memories.values():
        f.write(json.dumps(memory.to_meta_dict()) + "\n")
    os.replace(tmp_path, self._index_path)

  def _persist_embeddings(self) -> None:
    if not self._precond_emb:
      if os.path.exists(self._embeddings_path):
        os.remove(self._embeddings_path)
      return
    payload = {}
    for memory_id, vec in self._precond_emb.items():
      payload[f"{memory_id}__pre"] = vec
    for memory_id, vec in self._goal_emb.items():
      payload[f"{memory_id}__goal"] = vec
    tmp_path = self._embeddings_path + ".tmp.npz"
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, self._embeddings_path)

  def _persist_clock(self) -> None:
    with open(self._clock_path, "w", encoding="utf-8") as f:
      json.dump({"logical_time": self._logical_time}, f)

  def _persist_trajectory(self, memory: MemoryUnit) -> None:
    payload = {"trajectory": [step.to_dict() for step in memory.trajectory]}
    with open(self._trajectory_path(memory.memory_id), "w", encoding="utf-8") as f:
      json.dump(payload, f)

  def _load_trajectory(self, memory_id: str) -> list[TrajectoryStep]:
    path = self._trajectory_path(memory_id)
    if not os.path.exists(path):
      return []
    with open(path, "r", encoding="utf-8") as f:
      payload = json.load(f)
    return [TrajectoryStep.from_dict(d) for d in payload.get("trajectory", [])]

  def _load(self) -> None:
    if os.path.exists(self._clock_path):
      with open(self._clock_path, "r", encoding="utf-8") as f:
        self._logical_time = float(json.load(f).get("logical_time", 0.0))

    if not os.path.exists(self._index_path):
      return

    with open(self._index_path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        meta = json.loads(line)
        memory = MemoryUnit.from_meta_dict(meta)
        self._memories[memory.memory_id] = memory

    if os.path.exists(self._embeddings_path):
      with np.load(self._embeddings_path) as data:
        for key in data.files:
          memory_id, kind = key.rsplit("__", 1)
          if kind == "pre":
            self._precond_emb[memory_id] = data[key]
          elif kind == "goal":
            self._goal_emb[memory_id] = data[key]

    # Backfill any embeddings missing from the cache (e.g. schema changes).
    for memory_id, memory in self._memories.items():
      if memory_id not in self._precond_emb:
        self._precond_emb[memory_id] = self.embedder.embed(memory.precondition)
      if memory_id not in self._goal_emb:
        self._goal_emb[memory_id] = self.embedder.embed(memory.goal)
