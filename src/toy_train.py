"""Tiny training loop used by the Optuna+Hydra pruning demo.

- Dataset: FashionMNIST (train split), with a small subset for speed.
- Models: a small MLP or a small CNN.
- Pruning: if an Optuna trial is provided, we report the chosen metric each epoch
  and raise RuntimeError("PRUNED") when `trial.should_prune()` triggers.

Notes:
- This file intentionally avoids importing Optuna so it can be reused in non-Optuna
  contexts.
"""

import sys

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import trange



METRIC_DIRECTION = {
    "val_loss": "minimize",
    "val_acc": "maximize",
}

def direction_from_metric(metric: str) :
    try:
        return METRIC_DIRECTION[metric]
    except KeyError as e:
        raise ValueError(
            f"Unknown metric={metric!r} (expected one of {list(METRIC_DIRECTION)})"
        ) from e
    



class SmallCNN(nn.Module):
    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
            nn.Dropout2d(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # "auto"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def eval_loss_and_acc(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    loss_sum = 0.0
    correct = 0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        loss = loss_fn(logits, y)
        loss_sum += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        n += x.size(0)

    mean_loss = loss_sum / max(n, 1)
    mean_acc = correct / max(n, 1)
    return mean_loss, mean_acc


def make_loaders(cfg, *, pin_memory: bool):
    data_root = str(cfg.data.root)
    batch_size = int(cfg.data.batch_size)
    num_workers = int(cfg.data.num_workers)
    seed = int(cfg.exp.seed)
    subset_size = int(cfg.data.subset_size)

    # For the demo we use FashionMNIST and keep the normalization in the config.
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(float(cfg.data.norm.FashionMNIST.mean),),
                std=(float(cfg.data.norm.FashionMNIST.std),),
            ),
        ]
    )

    full = datasets.FashionMNIST(
        root=data_root,
        train=True,
        download=True,
        transform=tfm,
    )

    # Small subset to keep the demo fast.
    full = torch.utils.data.Subset(full, range(subset_size))

    train_size = int(0.9 * len(full))
    val_size = len(full) - train_size

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full, [train_size, val_size], generator=g)

    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, val_loader


def build_model(cfg) -> nn.Module:
    name = str(cfg.model.name)
    if name == "cnn":
        return SmallCNN(dropout=float(cfg.model.dropout))
    raise ValueError(f"Unknown model.name={name}")

def train_with_pruning(cfg, trial=None):
    """Train and return best metrics.

    Returns a dict with the best validation metrics seen across epochs:
        {"val_loss": best_val_loss, "val_acc": best_val_acc}

    If `trial` is provided, reports `cfg.optuna.metric` each epoch and prunes when
    requested (raises RuntimeError("PRUNED")).
    """

    device = get_device(str(cfg.exp.device))
    torch.manual_seed(int(cfg.exp.seed))

    train_loader, val_loader = make_loaders(cfg, pin_memory=(device.type == "cuda"))

    torch.backends.cudnn.benchmark = True

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    loss_fn = nn.CrossEntropyLoss()

    metric_name = str(cfg.optuna.metric)
    if metric_name not in ("val_loss", "val_acc"):
        raise ValueError(
            f"Unsupported optuna.metric={metric_name} (expected 'val_loss' or 'val_acc')"
        )




    best_epoch_metrics = {"epoch": -1, "val_loss": math.inf, "val_acc": -math.inf}
    lookup_direction = {"val_acc": "maximize", "val_loss": "minimize"}
    direction = lookup_direction[metric_name]


    desc = f"trial {trial.number}" if trial is not None else "train"

    for epoch in trange(int(cfg.train.epochs), desc=desc, leave=False, file=sys.stderr):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        val_loss, val_acc = eval_loss_and_acc(model, val_loader, device)


        epoch_metrics = {
            "val_loss": float(val_loss),
            "val_acc": float(val_acc), 
        }


        if direction not in ("minimize", "maximize"):
            raise ValueError(f"Unsupported metric direction: {direction!r} (expected 'minimize' or 'maximize')")

        if direction == "minimize":
            is_best_epoch = epoch_metrics[metric_name] < best_epoch_metrics[metric_name]
        else:  # "maximize"
            is_best_epoch = epoch_metrics[metric_name] > best_epoch_metrics[metric_name]


        if is_best_epoch : 
            best_epoch_metrics["epoch"] = epoch
            best_epoch_metrics["val_loss"] = epoch_metrics["val_loss"]
            best_epoch_metrics["val_acc"] = epoch_metrics["val_acc"]



        if trial is not None:
            trial.report(epoch_metrics[metric_name], step=epoch)
            if trial.should_prune():
                # store best-so-far even if pruned
                trial.set_user_attr("best_epoch", best_epoch_metrics["epoch"])
                trial.set_user_attr("best_epoch_val_loss", best_epoch_metrics["val_loss"])
                trial.set_user_attr("best_epoch_val_acc", best_epoch_metrics["val_acc"])
                raise RuntimeError("PRUNED")

    if trial is not None:
        trial.set_user_attr("best_epoch", best_epoch_metrics["epoch"])
        trial.set_user_attr("best_epoch_val_loss", best_epoch_metrics["val_loss"])
        trial.set_user_attr("best_epoch_val_acc", best_epoch_metrics["val_acc"])

    return best_epoch_metrics