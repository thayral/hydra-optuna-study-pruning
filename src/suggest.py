from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import optuna
from omegaconf import OmegaConf


@dataclass(frozen=True)
class SpaceItem:
    key: str
    suggest: str  # float|float_log|int|categorical
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    choices: List[Any] | None = None


def _parse_space(search_cfg) -> List[SpaceItem]:
    items = []
    for raw in search_cfg.space:
        raw = dict(raw)
        items.append(
            SpaceItem(
                key=raw["key"],
                suggest=raw["suggest"],
                low=raw.get("low"),
                high=raw.get("high"),
                step=raw.get("step"),
                choices=raw.get("choices"),
            )
        )
    return items


def suggest_into_cfg(cfg, search_cfg, trial: optuna.trial.Trial):
    """
    Returns a *new* cfg with suggested hyperparameters applied.
    """
    cfg2 = OmegaConf.to_container(cfg, resolve=True)
    cfg2 = OmegaConf.create(cfg2)

    for item in _parse_space(search_cfg):
        name = item.key.replace(".", "__")  # Optuna param names can't be nested nicely

        if item.suggest == "categorical":
            if not item.choices:
                raise ValueError(f"{item.key}: categorical requires choices")
            val = trial.suggest_categorical(name, list(item.choices))

        elif item.suggest == "int":
            if item.low is None or item.high is None:
                raise ValueError(f"{item.key}: int requires low/high")
            if item.step is None:
                val = trial.suggest_int(name, int(item.low), int(item.high))
            else:
                val = trial.suggest_int(name, int(item.low), int(item.high), step=int(item.step))

        elif item.suggest == "float":
            if item.low is None or item.high is None:
                raise ValueError(f"{item.key}: float requires low/high")
            if item.step is None:
                val = trial.suggest_float(name, float(item.low), float(item.high))
            else:
                val = trial.suggest_float(name, float(item.low), float(item.high), step=float(item.step))

        elif item.suggest == "float_log":
            if item.low is None or item.high is None:
                raise ValueError(f"{item.key}: float_log requires low/high")
            val = trial.suggest_float(name, float(item.low), float(item.high), log=True)

        else:
            raise ValueError(f"Unknown suggest type: {item.suggest}")

        OmegaConf.update(cfg2, item.key, val, merge=False)

    return cfg2


def apply_enqueue_overrides(cfg, overrides: Dict[str, Any]):
    """
    Apply a dict like {"train.lr": 1e-3, "model.name": "mlp"} into cfg and return a new cfg.
    """
    cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for k, v in overrides.items():
        OmegaConf.update(cfg2, k, v, merge=False)
    return cfg2