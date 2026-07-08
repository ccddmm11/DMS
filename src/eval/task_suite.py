# DMS reproduction project — task suite loader.
#
# Loads `configs/task_suite.yaml` and resolves each entry against the live
# AndroidWorld task registry, so downstream code (Baseline A/B, DMS main
# loop, eval harness) has a single source of truth for "which 29 tasks,
# across which 20 apps, do we evaluate".

from __future__ import annotations

import dataclasses
import os
from typing import Any, Type

import yaml
from android_world import registry
from android_world.task_evals import task_eval

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs",
    "task_suite.yaml",
)


@dataclasses.dataclass
class TaskSpec:
  """A single selected AndroidWorld task, resolved to its live class."""

  name: str
  apps: list[str]
  complexity: float
  difficulty: str
  tags: list[str]
  task_type: Type[task_eval.TaskEval]


@dataclasses.dataclass
class TaskSuite:
  """The full selected task suite plus experiment metadata."""

  meta: dict[str, Any]
  tasks: list[TaskSpec]

  def by_difficulty(self, difficulty: str) -> list[TaskSpec]:
    return [t for t in self.tasks if t.difficulty == difficulty]

  def by_tag(self, tag: str) -> list[TaskSpec]:
    return [t for t in self.tasks if tag in t.tags]


def load_task_suite(config_path: str = _DEFAULT_CONFIG_PATH) -> TaskSuite:
  """Loads and resolves `configs/task_suite.yaml` against the live registry."""
  with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

  task_registry = registry.TaskRegistry()
  family = cfg.get("meta", {}).get("source_family", task_registry.ANDROID_WORLD_FAMILY)
  aw_registry = task_registry.get_registry(family)

  specs = []
  for entry in cfg["tasks"]:
    name = entry["name"]
    if name not in aw_registry:
      raise ValueError(
          f"Task '{name}' listed in {config_path} not found in AndroidWorld "
          f"'{family}' registry ({len(aw_registry)} tasks available)."
      )
    specs.append(TaskSpec(
        name=name,
        apps=list(entry.get("apps", [])),
        complexity=float(entry["complexity"]),
        difficulty=entry["difficulty"],
        tags=list(entry.get("tags", [])),
        task_type=aw_registry[name],
    ))

  return TaskSuite(meta=cfg.get("meta", {}), tasks=specs)


if __name__ == "__main__":
  suite = load_task_suite()
  print(f"Loaded {len(suite.tasks)} tasks "
        f"({suite.meta.get('selected_app_count')} apps), "
        f"conditions={suite.meta.get('conditions')}, "
        f"rounds={suite.meta.get('rounds_per_task')}")
  for tier in ("easy", "medium", "hard"):
    names = [t.name for t in suite.by_difficulty(tier)]
    print(f"  {tier} ({len(names)}): {names}")
  print(f"  composite/long_sequence: "
        f"{[t.name for t in suite.tasks if 'composite' in t.tags or 'long_sequence' in t.tags]}")
