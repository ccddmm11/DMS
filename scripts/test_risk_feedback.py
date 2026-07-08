#!/usr/bin/env python3
"""Standalone validation for src/memory/risk.py (Sec 3.2.4 Risk Assessment
and Feedback Regulation: Bayesian reputation modeling, LCB risk scoring,
dynamic-threshold suppression).

Does NOT require the emulator or vLLM.

Usage:
  python scripts/test_risk_feedback.py
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.memory_bank import MemoryBank
from src.memory.memory_unit import MemoryUnit
from src.memory.risk import BayesianRiskConfig
from src.memory.risk import GlobalRiskStats
from src.memory.risk import RiskRegulator
from src.memory.risk import compute_bayesian_risk
from src.memory.risk import dynamic_threshold

TEST_STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "_tmp_risk_test",
)


def make_memory(idx, failure_count=0.0, success_count=0.0):
  return MemoryUnit(
      precondition=f"screen {idx}",
      goal=f"goal {idx}",
      failure_count=failure_count,
      success_count=success_count,
  )


def test_formula_sanity():
  print("[1/4] Bayesian risk formula sanity checks...")
  cfg = BayesianRiskConfig(prior_strength=4.0, tau_base=0.5, lambda_sensitivity=0.3,
                            default_global_failure_rate=0.5)
  stats = GlobalRiskStats()  # No data yet -> T_global falls back to default 0.5.

  # Cold start: F=S=0 -> mu should collapse exactly to T_global (no evidence
  # yet, so the Bayesian estimate is just the smoothing prior itself).
  mu0, sigma0, risk0 = compute_bayesian_risk(0, 0, stats, cfg)
  print(f"    cold start (F=0,S=0): mu={mu0:.4f} sigma={sigma0:.4f} risk={risk0:.4f}"
        f" (T_global={stats.global_failure_rate(cfg):.4f})")
  assert abs(mu0 - stats.global_failure_rate(cfg)) < 1e-9

  # More failures -> higher mu (more risk).
  mu_fail, _, risk_fail = compute_bayesian_risk(10, 0, stats, cfg)
  mu_success, _, risk_success = compute_bayesian_risk(0, 10, stats, cfg)
  print(f"    all failures (F=10,S=0): mu={mu_fail:.4f} risk={risk_fail:.4f}")
  print(f"    all successes (F=0,S=10): mu={mu_success:.4f} risk={risk_success:.4f}")
  assert mu_fail > mu0 > mu_success
  assert risk_fail > risk_success

  # Uncertainty-aware scoring: with few observations, sigma should be
  # larger (more uncertainty) than with many observations at a similar mu.
  _, sigma_few, _ = compute_bayesian_risk(1, 1, stats, cfg)
  _, sigma_many, _ = compute_bayesian_risk(50, 50, stats, cfg)
  print(f"    sigma (F=1,S=1)={sigma_few:.4f} vs sigma (F=50,S=50)={sigma_many:.4f}")
  assert sigma_few > sigma_many, "More observations should reduce posterior uncertainty."

  print("    OK.")


def test_dynamic_threshold():
  print("[2/4] Dynamic thresholding: tau tightens as T_global rises...")
  cfg = BayesianRiskConfig(tau_base=0.5, lambda_sensitivity=0.3)
  healthy = GlobalRiskStats(global_failures=1, global_successes=9)   # T_global=0.1
  unhealthy = GlobalRiskStats(global_failures=9, global_successes=1)  # T_global=0.9
  tau_healthy = dynamic_threshold(healthy, cfg)
  tau_unhealthy = dynamic_threshold(unhealthy, cfg)
  print(f"    T_global=0.1 -> tau={tau_healthy:.4f}; T_global=0.9 -> tau={tau_unhealthy:.4f}")
  assert tau_healthy > tau_unhealthy, "Higher global failure rate should tighten (lower) tau."
  print("    OK.")


def test_regulator_flow():
  print("[3/4] RiskRegulator: reuse-success / global-failure bookkeeping...")
  if os.path.exists(TEST_STORE_DIR):
    shutil.rmtree(TEST_STORE_DIR)
  bank = MemoryBank(TEST_STORE_DIR)
  reliable = bank.add(make_memory("reliable"))
  risky = bank.add(make_memory("risky"))

  regulator = RiskRegulator(bank, config=BayesianRiskConfig(prior_strength=2.0, tau_base=0.6))

  # `reliable` gets reused successfully many times across many tasks.
  for _ in range(8):
    regulator.record_reuse_success(reliable.memory_id)
    regulator.record_global_task_outcome([reliable.memory_id], success=True)

  # `risky` is active during several globally-FAILED tasks (never
  # successfully reused), so it should accumulate F_i and become risky.
  for _ in range(8):
    regulator.record_global_task_outcome([risky.memory_id], success=False)

  reliable_reloaded = bank.get(reliable.memory_id, load_trajectory=False)
  risky_reloaded = bank.get(risky.memory_id, load_trajectory=False)
  print(f"    reliable: F={reliable_reloaded.failure_count} S={reliable_reloaded.success_count} "
        f"mu={reliable_reloaded.risk_mu:.4f} risk={reliable_reloaded.risk_score:.4f}")
  print(f"    risky:    F={risky_reloaded.failure_count} S={risky_reloaded.success_count} "
        f"mu={risky_reloaded.risk_mu:.4f} risk={risky_reloaded.risk_score:.4f}")

  assert reliable_reloaded.success_count == 8
  assert reliable_reloaded.failure_count == 0
  assert risky_reloaded.failure_count == 8
  assert risky_reloaded.success_count == 0
  assert risky_reloaded.risk_score > reliable_reloaded.risk_score

  threshold = regulator.get_dynamic_threshold()
  print(f"    dynamic threshold tau={threshold:.4f}")
  assert not regulator.is_suppressed(reliable.memory_id), "Reliable memory should NOT be suppressed."
  assert regulator.is_suppressed(risky.memory_id), "Risky memory SHOULD be suppressed."
  print("    OK: reliable memory passes the risk gate, risky memory is suppressed.")

  print("[4/4] Persistence round-trip of global failure-rate state...")
  del regulator
  regulator2 = RiskRegulator(bank)
  assert regulator2.global_stats.global_failures == 8
  assert regulator2.global_stats.global_successes == 8
  print(f"    OK: reloaded global_stats={regulator2.global_stats}")

  shutil.rmtree(TEST_STORE_DIR)


def main():
  test_formula_sanity()
  test_dynamic_threshold()
  test_regulator_flow()
  print("\nAll risk-feedback checks passed.")


if __name__ == "__main__":
  main()
