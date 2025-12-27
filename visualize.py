from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def _plot_metric(ax, history: List[Dict], key: str, label: str):
    epochs = [h["epoch"] for h in history]
    train_vals = [h["train"][key] for h in history]
    val_vals = [h["val"][key] for h in history]
    ax.plot(epochs, train_vals, label=f"train {label}")
    ax.plot(epochs, val_vals, label=f"val {label}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_training_histories(
    baseline_history: List[Dict],
    hybrid_history: List[Dict],
    out_path: str | Path = "plots",
):
    """Save side-by-side training curves for baseline vs hybrid."""
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, history in [("baseline", baseline_history), ("hybrid", hybrid_history)]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        _plot_metric(axes[0], history, "acc", "accuracy")
        _plot_metric(axes[1], history, "f1", "F1 score")
        _plot_metric(axes[2], history, "spk_per_sample", "spikes/sample")
        fig.suptitle(f"Training dynamics ({name})")
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_training.png", dpi=150)
        plt.close(fig)

    # Comparison charts with separated scales for readability
    b_acc, b_f1, b_spk = (
        baseline_history[-1]["val"]["acc"],
        baseline_history[-1]["val"]["f1"],
        baseline_history[-1]["val"]["spk_per_sample"],
    )
    h_acc, h_f1, h_spk = (
        hybrid_history[-1]["val"]["acc"],
        hybrid_history[-1]["val"]["f1"],
        hybrid_history[-1]["val"]["spk_per_sample"],
    )

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(7, 6))

    # Top: accuracy and F1 on [0,1]
    x = [0, 1]
    width = 0.35
    ax_top.bar([i - width / 2 for i in x], [b_acc, b_f1], width, label="baseline")
    ax_top.bar([i + width / 2 for i in x], [h_acc, h_f1], width, label="hybrid")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(["accuracy", "F1 score"])
    ax_top.set_ylim(0, 1.0)
    ax_top.set_ylabel("value")
    ax_top.set_title("Baseline vs Hybrid (validation)")
    ax_top.grid(True, axis="y", alpha=0.3)
    ax_top.legend()

    # Bottom: spikes/sample on its own scale
    ax_bottom.bar(["baseline", "hybrid"], [b_spk, h_spk], color=["C0", "C1"])
    ax_bottom.set_ylabel("spikes/sample")
    ax_bottom.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=150)
    plt.close(fig)
