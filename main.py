from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from aco import AntColonyOptimizer
from config import ExperimentConfig
from data import build_dataloaders
from train import evaluate_model, train_model
from visualize import plot_confusion_matrix, plot_training_histories


def print_confusion_matrix(cm, labels: tuple[str, str] = ("bird", "drone"), title: str = "Confusion matrix"):
    print(f"{title}:")
    print(f"            pred {labels[0]:>6} pred {labels[1]:>6}")
    for idx, label in enumerate(labels):
        row = cm[idx]
        print(f" actual {label:<5} {row[0]:>10} {row[1]:>10}")
    print()


def run_baseline(cfg: ExperimentConfig):
    print("=== Baseline training ===")
    train_loader, val_loader, test_loader = build_dataloaders(cfg.data, cfg.train.batch_size, cfg.seed)
    model, history = train_model(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        vth=cfg.snn.baseline_vth,
        tau=cfg.snn.baseline_tau,
    )
    train_metrics = history[-1]["train"]
    val_metrics = history[-1]["val"]
    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        device=cfg.train.device,
        timesteps=cfg.snn.timesteps,
    )
    cm = test_metrics.get("confusion_matrix")
    if cm is not None:
        print_confusion_matrix(cm, title="Baseline test confusion matrix")
        plot_confusion_matrix(cm, title="Baseline test confusion matrix", out_path="plots/baseline_confusion.png")
    baseline_summary = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "vth": cfg.snn.baseline_vth,
        "tau": cfg.snn.baseline_tau,
        "history": history,
    }
    print("Baseline metrics:", json.dumps(baseline_summary, indent=2))
    return baseline_summary, model, (train_loader, val_loader, test_loader)


def run_aco_search(cfg: ExperimentConfig, train_loader, val_loader):
    print("=== ACO search ===")
    optimizer = AntColonyOptimizer(cfg, train_loader, val_loader)
    result = optimizer.run()
    print(
        f"[ACO] best vth={result.best_vth:.2f} tau={result.best_tau:.2f} "
        f"fitness={result.best_fitness:.3f} val_metrics={result.best_val_metrics}"
    )
    return result


def run_hybrid(cfg: ExperimentConfig, train_loader, val_loader, test_loader, best_vth: float, best_tau: float):
    print("=== Hybrid training with ACO parameters ===")
    model, history = train_model(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        vth=best_vth,
        tau=best_tau,
    )
    train_metrics = history[-1]["train"]
    val_metrics = history[-1]["val"]
    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        device=cfg.train.device,
        timesteps=cfg.snn.timesteps,
    )
    cm = test_metrics.get("confusion_matrix")
    if cm is not None:
        print_confusion_matrix(cm, title="Hybrid test confusion matrix")
        plot_confusion_matrix(cm, title="Hybrid test confusion matrix", out_path="plots/hybrid_confusion.png")
    hybrid_summary = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "vth": best_vth,
        "tau": best_tau,
        "history": history,
    }
    print("Hybrid metrics:", json.dumps(hybrid_summary, indent=2))
    return hybrid_summary


def main():
    parser = argparse.ArgumentParser(description="SNN + ACO pipeline (bird vs drone)")
    parser.add_argument(
        "--mode",
        choices=["baseline", "aco", "hybrid"],
        default="hybrid",
        help="Run baseline only, ACO search only, or full hybrid (search + retrain)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Open generated plots after the hybrid run (only applies to --mode hybrid)",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig()

    if args.mode == "baseline":
        run_baseline(cfg)
        return

    baseline_summary, _, loaders = run_baseline(cfg)
    train_loader, val_loader, test_loader = loaders

    if args.mode == "aco":
        run_aco_search(cfg, train_loader, val_loader)
        return

    # hybrid
    aco_result = run_aco_search(cfg, train_loader, val_loader)
    hybrid_summary = run_hybrid(cfg, train_loader, val_loader, test_loader, aco_result.best_vth, aco_result.best_tau)

    comparison = {
        "baseline": baseline_summary,
        "hybrid": hybrid_summary,
    }
    print("\n=== Comparison (baseline vs hybrid) ===")
    print(json.dumps(comparison, indent=2))
    # Compact summary for quick read  
    b, h = baseline_summary["val"], hybrid_summary["val"]
    print(
        "\nSummary (val): "
        f"baseline acc {b['acc']:.3f}, prec {b['precision']:.3f}, rec {b['recall']:.3f}, "
        f"f1 {b['f1']:.3f}, spk/sample {b['spk_per_sample']:.3f} | "
        f"hybrid acc {h['acc']:.3f}, prec {h['precision']:.3f}, rec {h['recall']:.3f}, "
        f"f1 {h['f1']:.3f}, spk/sample {h['spk_per_sample']:.3f}"
    )
    # Percentage improvements (hybrid vs baseline); spikes reported as reduction
    def pct_change(new, old):
        return ((new - old) / old * 100.0) if old != 0 else float("inf")

    acc_impr = pct_change(h["acc"], b["acc"])
    f1_impr = pct_change(h["f1"], b["f1"])
    spk_reduct = ((b["spk_per_sample"] - h["spk_per_sample"]) / b["spk_per_sample"] * 100.0) if b["spk_per_sample"] != 0 else float("inf")
    print(
        "Percent change (hybrid vs baseline, val): "
        f"acc {acc_impr:+.2f}%, f1 {f1_impr:+.2f}%, spikes/sample {spk_reduct:+.2f}% (reduction positive)"
    )

    # Save plots comparing training dynamics and final metrics
    plot_training_histories(
        baseline_history=baseline_summary["history"],
        hybrid_history=hybrid_summary["history"],
        out_path="plots",
    )
    print("Saved plots to plots/ (baseline_training.png, hybrid_training.png, comparison.png)")
    if args.show_plots:
        for name in ["baseline_training.png", "hybrid_training.png", "comparison.png"]:
            path = Path("plots") / name
            if path.exists():
                try:
                    if sys.platform.startswith("win"):
                        os.startfile(path)  # type: ignore[attr-defined]
                    else:
                        import subprocess

                        subprocess.Popen(["xdg-open", str(path)])
                except Exception as exc:
                    print(f"Could not open {path}: {exc}")


if __name__ == "__main__":
    main()
