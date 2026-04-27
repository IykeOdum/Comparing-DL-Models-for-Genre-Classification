# run_experiments.py
# Trains MLP, ShallowCNN, and DeepCNN on GTZAN and produces a comparison report.
#
# Usage:
#   python run_experiments.py
#
# Outputs (written to current directory):
#   training_curves.png         – loss + accuracy curves for all models
#   confusion_MLP.png           – per-model confusion matrices
#   confusion_ShallowCNN.png
#   confusion_DeepCNN.png
#   results_summary.txt         – test accuracy, parameter counts, training time

import torch
import torch.nn as nn

import config
from dataset import get_dataloaders
from models  import get_model, count_parameters
from train   import (
    train_model,
    evaluate,
    plot_training_curves,
    plot_confusion_matrix,
    print_classification_report,
)


def main():
    print(f"Device: {config.DEVICE}")
    print(f"Augmentation: {config.AUGMENT}\n")

    # ── Data ───────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders()

    # ── Experiment loop ────────────────────────────────────────────────────────
    model_names = ["MLP", "ShallowCNN", "DeepCNN"]
    histories   = {}
    test_results = {}

    criterion = nn.CrossEntropyLoss()

    for name in model_names:
        print(f"\n{'='*60}")
        print(f"  Training: {name}")
        print(f"{'='*60}")

        model = get_model(name)
        n_params = count_parameters(model)
        print(f"  Parameters: {n_params:,}\n")

        # Train
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.NUM_EPOCHS,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            device=config.DEVICE,
            model_name=name,
        )
        histories[name] = history

        # Load best checkpoint for final test evaluation
        model.load_state_dict(torch.load(f"{name}_best.pt", map_location=config.DEVICE))
        model = model.to(config.DEVICE)

        test_loss, test_acc, preds, labels = evaluate(
            model, test_loader, criterion, config.DEVICE
        )
        print(f"[{name}] Test acc: {test_acc:.4f}  Test loss: {test_loss:.4f}")

        test_results[name] = {
            "test_acc":   test_acc,
            "test_loss":  test_loss,
            "n_params":   n_params,
            "train_time": history["total_time_s"],
            "best_val_acc": history["best_val_acc"],
        }

        # Per-model diagnostics
        plot_confusion_matrix(preds, labels, name)
        print_classification_report(preds, labels, name)

    # ── Aggregate plots ────────────────────────────────────────────────────────
    plot_training_curves(histories)

    # ── Summary table ──────────────────────────────────────────────────────────
    summary_lines = [
        "=" * 65,
        f"{'Model':<14} {'Params':>10} {'Val Acc':>9} {'Test Acc':>9} {'Time (s)':>10}",
        "-" * 65,
    ]
    for name, r in test_results.items():
        summary_lines.append(
            f"{name:<14} {r['n_params']:>10,} {r['best_val_acc']:>9.4f} "
            f"{r['test_acc']:>9.4f} {r['train_time']:>10.1f}"
        )
    summary_lines.append("=" * 65)
    summary = "\n".join(summary_lines)

    print("\n" + summary)

    with open("results_summary.txt", "w") as f:
        f.write(summary + "\n")
    print("\nSaved results_summary.txt")


if __name__ == "__main__":
    main()