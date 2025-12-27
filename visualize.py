from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


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

    metric_defs = [
        ("acc", "accuracy"),
        ("precision", "precision"),
        ("recall", "recall"),
        ("f1", "F1 score"),
        ("spk_per_sample", "spikes/sample"),
    ]

    for name, history in [("baseline", baseline_history), ("hybrid", hybrid_history)]:
        cols = 3
        rows = math.ceil(len(metric_defs) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.flatten()
        for idx, (key, label) in enumerate(metric_defs):
            _plot_metric(axes[idx], history, key, label)
        # hide any unused axes
        for j in range(len(metric_defs), len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"Training dynamics ({name})")
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_training.png", dpi=150)
        plt.close(fig)

    # Comparison charts with separated scales for readability
    b_val = baseline_history[-1]["val"]
    h_val = hybrid_history[-1]["val"]
    b_acc, b_prec, b_rec, b_f1, b_spk = (
        b_val["acc"],
        b_val["precision"],
        b_val["recall"],
        b_val["f1"],
        b_val["spk_per_sample"],
    )
    h_acc, h_prec, h_rec, h_f1, h_spk = (
        h_val["acc"],
        h_val["precision"],
        h_val["recall"],
        h_val["f1"],
        h_val["spk_per_sample"],
    )

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(7, 6))

    # Top: accuracy, precision, recall, and F1 on [0,1]
    x = list(range(4))
    width = 0.35
    ax_top.bar([i - width / 2 for i in x], [b_acc, b_prec, b_rec, b_f1], width, label="baseline")
    ax_top.bar([i + width / 2 for i in x], [h_acc, h_prec, h_rec, h_f1], width, label="hybrid")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(["accuracy", "precision", "recall", "F1"])
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


def plot_confusion_matrix(
    cm,
    labels: list[str] | tuple[str, ...] = ("bird", "drone"),
    title: str = "Confusion matrix",
    out_path: str | Path = "plots/confusion_matrix.png",
):
    """
    Save a confusion matrix heatmap. Rows = actual, cols = predicted.
    """
    cm_arr = np.asarray(cm, dtype=np.int32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # annotate cells
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, int(cm_arr[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
