# DMS reproduction project — deterministic, cross-condition task instantiation.
#
# The 3-condition comparison (Baseline A / Baseline B / DMS) is only fair if
# every condition faces the *exact same* randomized task instance for a given
# (task_name, round) cell -- otherwise a condition could get "lucky" with an
# easier random instantiation. We reuse AndroidWorld's own official seeding
# scheme (`android_world/suite_utils.py`'s `_get_instance_seed` /
# `_instantiate_task`) so that, for a fixed `base_seed`, calling
# `instantiate_task(spec, round_idx, base_seed)` from three independent
# worker processes (one per condition) yields byte-for-byte identical
# `task.params` (and therefore identical goals/target state).

from __future__ import annotations

import hashlib
import random
from typing import Optional

from android_world.env import interface
from android_world.task_evals import task_eval

from src.eval.task_suite import TaskSpec

_SEED_KEY = "seed"


def get_instance_seed(base_seed: int, task_name: str, round_idx: int) -> int:
  """Mirrors `suite_utils._get_instance_seed` (name, i) -> deterministic seed."""
  unique_seed_str = f"{base_seed}_{task_name}_{round_idx}"
  return int(hashlib.sha256(unique_seed_str.encode()).hexdigest(), 16) % (2**32)


def instantiate_task(
    spec: TaskSpec,
    round_idx: int,
    base_seed: int,
    env: Optional[interface.AsyncEnv] = None,
) -> task_eval.TaskEval:
  """Creates one instance of `spec.task_type` for this (task, round) cell.

  Identical `base_seed` + `spec.name` + `round_idx` across three separate
  processes (one per condition) always yields identical `params`, per
  AndroidWorld's own `_instantiate_task` convention.
  """
  instance_seed = get_instance_seed(base_seed, spec.name, round_idx)
  spec.task_type.set_device_time(env)
  random.seed(instance_seed)
  params = spec.task_type.generate_random_params()
  params[_SEED_KEY] = instance_seed
  return spec.task_type(params)
