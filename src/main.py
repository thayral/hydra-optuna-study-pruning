import os
import random

import hydra
import optuna
import torch
from omegaconf import DictConfig

from suggest import suggest_into_cfg
from toy_train import train_with_pruning, direction_from_metric
from tqdm import tqdm

def seed_everything(seed: int) -> None:
    """Best-effort reproducibility across Python + PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sampler(cfg) -> optuna.samplers.BaseSampler:
    name = str(cfg.optuna.sampler.name).lower()
    seed = int(cfg.optuna.sampler.seed)

    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)

    raise ValueError(f"Unknown sampler: {name}")


def make_pruner(cfg) -> optuna.pruners.BasePruner:
    name = str(cfg.optuna.pruner.name).lower()

    if name == "none":
        return optuna.pruners.NopPruner()
    if name == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(cfg.optuna.pruner.n_startup_trials),
            n_warmup_steps=int(cfg.optuna.pruner.n_warmup_steps),
            interval_steps=int(cfg.optuna.pruner.interval_steps),
        )

    raise ValueError(f"Unknown pruner: {name}")


def is_demo_prune_exception(err: Exception) -> bool:
    """toy_train raises RuntimeError("PRUNED") to avoid importing optuna there."""
    return isinstance(err, RuntimeError) and str(err) == "PRUNED"




@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.exp.seed))

    sampler = make_sampler(cfg)
    pruner = make_pruner(cfg)

    metric_name = str(cfg.optuna.metric)
    direction = direction_from_metric(metric_name)

    storage = cfg.optuna.study.storage
    study = optuna.create_study(
        direction=direction,
        study_name=str(cfg.optuna.study.name),
        sampler=sampler,
        pruner=pruner,
        storage=str(storage) if storage else None,
        load_if_exists=bool(cfg.optuna.study.load_if_exists),
    )


    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Enqueue baseline trials (repro + sanity)
    for overrides in cfg.optuna.enqueue:
        overrides = dict(overrides)
        # Optuna expects param_name -> value, where names match our suggest() keys.
        # Our suggest() names are key paths with dots replaced by '__'.
        params = {k.replace(".", "__"): v for k, v in overrides.items()}
        study.enqueue_trial(params)


    def objective(trial) -> float:
        trial_cfg = suggest_into_cfg(cfg, cfg.search, trial)

        pruned = False
        try:
            metrics = train_with_pruning(trial_cfg, trial=trial)
            val = float(metrics[metric_name])
            return val

        except Exception as e:
            if is_demo_prune_exception(e):
                pruned = True
                raise optuna.TrialPruned() from None
            raise

        finally:
            #  best-epoch metrics from trial
            be = trial.user_attrs.get("best_epoch")
            be_loss = trial.user_attrs.get("best_epoch_val_loss")
            be_acc = trial.user_attrs.get("best_epoch_val_acc")

            if be_loss is None or be_acc is None:
                # e.g. pruned before first eval; keep it robust
                msg = "no metrics recorded"
            else:
                msg = f"epoch={be} | val_loss={be_loss:.4f} | val_acc={be_acc*100:.2f}%"

            status = "PRUNED" if pruned else "DONE"
            tqdm.write(f"Trial {trial.number} [{status}] | {msg}")

    study.optimize(
        objective,
        n_trials=int(cfg.optuna.n_trials),
        show_progress_bar=True,
    )



    # ===== LOGGING FINAL RESULTS ======
    best_trial = study.best_trial
    best_epoch = best_trial.user_attrs.get("best_epoch")
    best_epoch_val_loss = best_trial.user_attrs.get("best_epoch_val_loss")
    best_epoch_val_acc = best_trial.user_attrs.get("best_epoch_val_acc")

    print("\n=== Study finished ===")
    print(f"Best value ({metric_name}): {best_trial.value:.4f}")

    if best_epoch is not None:
        print(f"Best epoch: {best_epoch}")
    if best_epoch_val_loss is not None and best_epoch_val_acc is not None:
        print(
            f"Best-epoch metrics: val_loss={best_epoch_val_loss:.4f} | "
            f"val_acc={best_epoch_val_acc*100:.2f}%"
        )

    print("Best params:")
    for k, v in best_trial.params.items():
        print(f"  {k} = {v}")



if __name__ == "__main__":
    main()
