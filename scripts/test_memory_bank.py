#!/usr/bin/env python3
"""Standalone validation for src/memory/{embedder,memory_unit,memory_bank}.py.

Does NOT require the emulator or vLLM -- only exercises the Dual-Factor
Retrieval math (Sec 3.2.2) and the disk persistence round-trip (Sec
3.2.1/3.2.2 "decoupled storage architecture") on a local CPU embedding
model.

Usage:
  python scripts/test_memory_bank.py
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.memory_bank import MemoryBank
from src.memory.memory_unit import MemoryUnit
from src.memory.memory_unit import TrajectoryStep

TEST_STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "_tmp_memory_bank_test",
)


def make_memory(precondition, goal, actions, source_task="TestTask"):
  trajectory = [
      TrajectoryStep(reason=f"step {i}", action=a, target_element_desc=None)
      for i, a in enumerate(actions)
  ]
  return MemoryUnit(
      precondition=precondition,
      goal=goal,
      trajectory=trajectory,
      success=True,
      description=f"Created from {source_task}",
      source_task=source_task,
  )


def main():
  if os.path.exists(TEST_STORE_DIR):
    shutil.rmtree(TEST_STORE_DIR)

  print("[1/5] Creating a fresh MemoryBank and adding 4 memories...")
  bank = MemoryBank(TEST_STORE_DIR)
  m1 = bank.add(make_memory(
      "Settings app home screen is open.",
      "Turn wifi on by tapping the Wi-Fi toggle.",
      [{"action_type": "click", "index": 3}, {"action_type": "click", "index": 5}],
  ))
  m2 = bank.add(make_memory(
      "Settings app home screen is open.",
      "Turn wifi off by tapping the Wi-Fi toggle.",
      [{"action_type": "click", "index": 3}, {"action_type": "click", "index": 5}],
  ))
  m3 = bank.add(make_memory(
      "Contacts app home screen is open.",
      "Create a new contact with the given name and phone number.",
      [{"action_type": "click", "index": 1}, {"action_type": "input_text", "index": 2, "text": "X"}],
  ))
  m4 = bank.add(make_memory(
      "Markor app note list is open.",
      "Create a new note with the given title.",
      [{"action_type": "click", "index": 0}],
  ))
  print(f"    Bank size: {len(bank)} (expected 4)")
  assert len(bank) == 4

  print("[2/5] Dual-Factor Retrieval sanity checks...")
  results = bank.retrieve(
      "Settings app home screen is open.",
      "Turn wifi on.",
      top_k=4,
  )
  for r in results:
    print(f"    score={r.score:.4f} pre_sim={r.precondition_similarity:.4f} "
          f"goal_sim={r.goal_similarity:.4f} -> {r.memory.as_prompt_str()}")
  assert results, "Expected at least one retrieval result."
  top = results[0]
  assert top.memory.memory_id == m1.memory_id, (
      f"Expected the 'turn wifi ON' memory to rank first, got: "
      f"{top.memory.as_prompt_str()}"
  )
  # The dual-factor score should punish precondition/goal mismatches: the
  # Contacts/Markor memories (different precondition AND goal) must score
  # noticeably lower than the two Settings/wifi memories.
  settings_ids = {m1.memory_id, m2.memory_id}
  other_ids = {m3.memory_id, m4.memory_id}
  settings_scores = [r.score for r in results if r.memory.memory_id in settings_ids]
  other_scores = [r.score for r in results if r.memory.memory_id in other_ids]
  assert min(settings_scores) > max(other_scores), (
      "Expected Settings/wifi memories to outscore unrelated memories: "
      f"settings={settings_scores} other={other_scores}"
  )
  print("    OK: dual-factor retrieval correctly ranks the matching memory"
        " first and unrelated app memories last.")

  print("[3/5] Bookkeeping updates (record_reuse / verification_failure /"
        " risk_stats / survival_value)...")
  bank.tick(5)
  bank.record_reuse(m1.memory_id)
  bank.record_verification_failure(m2.memory_id)
  bank.update_risk_stats(m2.memory_id, failure_count=1, success_count=0, risk_mu=0.4, risk_score=0.2)
  bank.update_survival_value(m1.memory_id, 3.14)
  reloaded_m1 = bank.get(m1.memory_id, load_trajectory=False)
  assert reloaded_m1.reuse_count == 1 and reloaded_m1.last_used_at == 5.0
  reloaded_m2 = bank.get(m2.memory_id, load_trajectory=False)
  assert reloaded_m2.verification_strikes == 1
  assert abs(reloaded_m2.risk_mu - 0.4) < 1e-9
  print("    OK.")

  print("[4/5] Persistence round-trip: reload bank from disk...")
  del bank
  bank2 = MemoryBank(TEST_STORE_DIR)
  assert len(bank2) == 4
  assert bank2.now == 5.0
  reloaded = bank2.get(m1.memory_id, load_trajectory=True)
  assert reloaded is not None
  assert reloaded.reuse_count == 1
  assert abs(reloaded.survival_value - 3.14) < 1e-6
  assert len(reloaded.trajectory) == 2
  assert reloaded.trajectory[0].action == {"action_type": "click", "index": 3}
  print(f"    OK: reloaded {len(bank2)} memories, trajectory lengths preserved,"
        " bookkeeping fields preserved, logical clock preserved (now="
        f"{bank2.now}).")

  print("[5/5] remove() + in-place trajectory replacement (epsilon-mutation"
        " hook)...")
  bank2.replace_trajectory(m4.memory_id, [
      TrajectoryStep(reason="shorter path", action={"action_type": "click", "index": 9}),
  ])
  reloaded_m4 = bank2.get(m4.memory_id, load_trajectory=True)
  assert len(reloaded_m4.trajectory) == 1
  bank2.remove(m3.memory_id)
  assert len(bank2) == 3
  print("    OK.")

  shutil.rmtree(TEST_STORE_DIR)
  print("\nAll memory-core checks passed.")


if __name__ == "__main__":
  main()
