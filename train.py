# train.py
# Training loop, evaluation utilities, and confusion matrix plotting.

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import config


# ── Training loop ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Run one full training epoch.

    Returns:
        avg_loss: mean cross-entropy loss over all batches
        accuracy: fraction of correctly classified samples
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for specs, labels in loader:
        specs  = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(specs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * specs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += specs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model on a DataLoader.

    Returns:
        avg_loss, accuracy, all_preds, all_labels
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for specs, labels in loader:
        specs  = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(specs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * specs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += specs.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
    )


# ── Full training run ──────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = config.NUM_EPOCHS,
    lr: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    device: str = config.DEVICE,
    model_name: str = "model",
) -> dict:
    """Train a model and return a history dict with loss/accuracy curves.

    Implements:
      - Adam optimizer with weight decay (L2 regularisation)
      - ReduceLROnPlateau scheduler (halves LR if val loss stagnates)
      - Best-model checkpoint saved to  {model_name}_best.pt
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    best_val_acc = 0.0
    start_time   = time.time()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_t = time.time() - t0
        print(
            f"[{model_name}] Epoch {epoch:3d}/{num_epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"({epoch_t:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{model_name}_best.pt")

    total_time = time.time() - start_time
    print(f"\n[{model_name}] Training complete in {total_time:.1f}s  "
          f"Best val acc: {best_val_acc:.4f}\n")

    history["total_time_s"] = total_time
    history["best_val_acc"] = best_val_acc
    return history


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_training_curves(
    histories: dict[str, dict],
    save_path: str = "training_curves.png",
) -> None:
    """Plot loss and accuracy curves for all models side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors

    for idx, (name, hist) in enumerate(histories.items()):
        c = colors[idx]
        epochs = range(1, len(hist["train_loss"]) + 1)

        axes[0].plot(epochs, hist["train_loss"], color=c, linestyle="--", alpha=0.7)
        axes[0].plot(epochs, hist["val_loss"],   color=c, linestyle="-",  label=name)

        axes[1].plot(epochs, hist["train_acc"],  color=c, linestyle="--", alpha=0.7)
        axes[1].plot(epochs, hist["val_acc"],    color=c, linestyle="-",  label=name)

    for ax, title, ylabel in zip(
        axes,
        ["Loss (-- train, — val)", "Accuracy (-- train, — val)"],
        ["Cross-Entropy Loss", "Accuracy"],
    ):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves → {save_path}")


def plot_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    save_path: str | None = None,
) -> None:
    """Plot and optionally save a normalised confusion matrix."""
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=config.GENRES,
        yticklabels=config.GENRES,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_path = save_path or f"confusion_{model_name}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {save_path}")


def print_classification_report(
    preds: np.ndarray,
    labels: np.ndarray,
    model_name: str,
) -> None:
    print(f"\n── Classification Report: {model_name} ──")
    print(classification_report(labels, preds, target_names=config.GENRES))