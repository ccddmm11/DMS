#!/usr/bin/env python3
"""Standalone validation for src/memory/survival.py (Sec 3.2.3
Self-Regulation Strategy: Survival Value + Elbow Method pruning/expansion).

Does NOT require the emulator or vLLM.

Usage:
  python scripts/test_survival_pruning.py
"""

import math
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.memory_bank import MemoryBank
from src.memory.memory_unit import MemoryUnit
from src.memory.memory_unit import TrajectoryStep
from src.memory.survival import RegulationConfig
from src.memory.survival import SelfRegulator
from src.memory.survival import SurvivalValueConfig
from src.memory.survival import compute_survival_value
from src.memory.survival import find_elbow_index

TEST_STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "_tmp_survival_test",
)


def make_memory(idx, reuse_count, last_used_at, strikes=0, app="TestApp"):
  return MemoryUnit(
      precondition=f"{app} screen {idx}",
      goal=f"Do thing {idx}",
      trajectory=[TrajectoryStep(reason="r", action={"action_type": "click", "index": idx})],
      reuse_count=reuse_count,
      last_used_at=last_used_at,
      verification_strikes=strikes,
      source_task=app,
  )


def test_formula_sanity():
  print("[1/4] Survival Value formula sanity checks...")
  cfg = SurvivalValueConfig()

  fresh = make_memory(0, reuse_count=0, last_used_at=0.0)
  s_fresh = compute_survival_value(fresh, now=0.0, config=cfg)
  expected_fresh = (math.log1p(0) + cfg.v_new) * 1.0 * 1.0  # decay~1 (delta_t=0), reliability=1
  print(f"    fresh memory (n=0, delta_t=0): S={s_fresh:.4f} (expected ~{expected_fresh:.4f})")
  assert abs(s_fresh - expected_fresh) < 1e-6

  # Cold-start protection: a fresh, never-reused memory should NOT already
  # be decayed to near-zero just because n_i=0 -- it survives until dormant
  # for a while (delta_t << T_base).
  assert s_fresh > 0.9, "Cold-start protection failed: fresh memory decayed too early."

  # Heavily reused + recently used memory should score much higher than a
  # dormant, never-reused one.
  popular = make_memory(1, reuse_count=20, last_used_at=100.0)
  s_popular = compute_survival_value(popular, now=100.0, config=cfg)
  dormant = make_memory(2, reuse_count=0, last_used_at=0.0)
  s_dormant = compute_survival_value(dormant, now=200.0, config=cfg)  # delta_t=200, way past T_half~30
  print(f"    popular & fresh-use (n=20, delta_t=0): S={s_popular:.4f}")
  print(f"    dormant (n=0, delta_t=200): S={s_dormant:.4f}")
  assert s_popular > s_dormant, "Popular/recent memory should outscore a long-dormant one."
  assert s_dormant < 0.1, "Long-dormant memory should have decayed close to 0."

  # Reliability penalty: repeated verification failures should sharply cut
  # the score even for an otherwise-popular memory.
  unreliable = make_memory(3, reuse_count=20, last_used_at=100.0, strikes=5)
  s_unreliable = compute_survival_value(unreliable, now=100.0, config=cfg)
  print(f"    popular but unreliable (n=20, delta_t=0, K=5): S={s_unreliable:.4f}")
  assert s_unreliable < s_popular / 2, "Reliability penalty did not sufficiently suppress score."
  print("    OK.")


def test_elbow_index():
  print("[2/4] Elbow index detection sanity checks...")
  # A curve with a sharp "cliff": 5 high values, then a long low-value tail.
  curve = [5.0, 4.9, 4.8, 4.7, 4.6, 0.3, 0.25, 0.2, 0.15, 0.1]
  idx = find_elbow_index(curve)
  print(f"    curve={curve}\n    elbow_index={idx} (value={curve[idx]})")
  # The curvature spike lands at the FIRST tail point (index 5, value 0.3):
  # elements [0, idx) are the "core" (kept), [idx, N) is the long tail.
  assert idx == 5, f"Expected elbow at the start of the tail (index 5), got {idx}."
  assert find_elbow_index([1.0, 0.5]) is None  # too short
  print("    OK.")


def test_regulator_prune_and_expand():
  print("[3/4] SelfRegulator: capacity-triggered pruning vs. expansion...")
  if os.path.exists(TEST_STORE_DIR):
    shutil.rmtree(TEST_STORE_DIR)

  # --- Scenario A: mixed quality -> should PRUNE the long tail. ---
  bank = MemoryBank(os.path.join(TEST_STORE_DIR, "prune_case"))
  bank.tick(200)  # advance logical clock so dormancy actually matters
  good_ids, bad_ids = [], []
  for i in range(8):
    m = bank.add(make_memory(i, reuse_count=15, last_used_at=195.0, app="GoodApp"))
    good_ids.append(m.memory_id)
  for i in range(8, 16):
    m = bank.add(make_memory(i, reuse_count=0, last_used_at=0.0, app="BadApp"))
    bad_ids.append(m.memory_id)
  assert len(bank) == 16

  regulator = SelfRegulator(
      bank,
      regulation_config=RegulationConfig(initial_capacity=10, capacity_max=100, capacity_step=10,
                                          min_memories_for_elbow=5),
  )
  result = regulator.regulate()
  print(f"    action={result.action} size {result.bank_size_before}->{result.bank_size_after} "
        f"cap {result.capacity_cap_before}->{result.capacity_cap_after} "
        f"elbow_value={result.elbow_value:.4f} pop_mean={result.population_mean:.4f}")
  assert result.action == "prune", f"Expected prune, got {result.action}"
  remaining_ids = {m.memory_id for m in bank.all_memories()}
  assert remaining_ids.issubset(set(good_ids)), (
      "Pruning should only remove BadApp (long-tail) memories, but some"
      f" GoodApp memories were lost. Remaining: {remaining_ids}"
  )
  assert not (remaining_ids & set(bad_ids)), "Some BadApp memories survived pruning."
  print(f"    OK: kept {len(remaining_ids)}/8 GoodApp memories, pruned all 8 BadApp memories.")

  # --- Scenario B: uniformly high-quality (smooth curve, no low tail /
  # "cliff") -> should EXPAND capacity rather than prune. We deliberately
  # avoid perfectly IDENTICAL survival values across all 12 memories: with
  # a true tie, argmax-curvature tie-breaking plus floating-point summation
  # noise on the population mean can flip `elbow_value >= mean` either way.
  # Instead we give a gentle, monotonic spread (varying reuse_count, all
  # well within the cold-start/decay protection window) so the curve has
  # uniformly low curvature and stays tightly clustered around its own
  # mean -- exactly the "high-quality saturation" scenario Sec 3.2.3
  # describes, without relying on an exact tie. ---
  bank2 = MemoryBank(os.path.join(TEST_STORE_DIR, "expand_case"))
  bank2.tick(50)
  for i in range(12):
    bank2.add(make_memory(i, reuse_count=10 + i, last_used_at=48.0, app="UniformApp"))
  regulator2 = SelfRegulator(
      bank2,
      regulation_config=RegulationConfig(initial_capacity=10, capacity_max=100, capacity_step=10,
                                          min_memories_for_elbow=5),
  )
  result2 = regulator2.regulate()
  print(f"    action={result2.action} size {result2.bank_size_before}->{result2.bank_size_after} "
        f"cap {result2.capacity_cap_before}->{result2.capacity_cap_after} "
        f"elbow_value={result2.elbow_value:.4f} pop_mean={result2.population_mean:.4f}")
  assert result2.action == "expand", f"Expected expand for a smooth, low-curvature curve, got {result2.action}"
  assert result2.capacity_cap_after == 20
  assert len(bank2) == 12, "Expansion must not remove any memories."
  print("    OK.")

  print("[4/4] Regulation state persistence...")
  del regulator2
  regulator2_reloaded = SelfRegulator(bank2)
  assert regulator2_reloaded.capacity_cap == 20, (
      f"Expected persisted capacity_cap=20, got {regulator2_reloaded.capacity_cap}"
  )
  print("    OK: capacity_cap persisted across SelfRegulator reload.")

  shutil.rmtree(TEST_STORE_DIR)


def main():
  test_formula_sanity()
  test_elbow_index()
  test_regulator_prune_and_expand()
  print("\nAll survival-pruning checks passed.")


if __name__ == "__main__":
  main()
