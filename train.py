from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ExperimentConfig
from metrics import compute_classification_metrics, compute_confusion_matrix
from models import SpikingConvNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: SpikingConvNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    timesteps: int,
    log_interval: int,
    train: bool = True,
    max_batches: int | None = None,
    collect_confusion: bool = False,
    num_classes: int = 2,
) -> Dict[str, float | list[list[int]]]:
    criterion = nn.CrossEntropyLoss()
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_spikes = 0.0
    total_samples = 0
    processed_batches = 0
    first_val_logged = False
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64) if collect_confusion else None
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        if hasattr(model, "reset_state"):
            model.reset_state()
            if not train and not first_val_logged:
                m1 = getattr(model, "mem1", None)
                m2 = getattr(model, "mem2", None)
                m3 = getattr(model, "mem3", None)
                mem_reset_max = max(
                    float(m.abs().max().item()) if m is not None else 0.0 for m in (m1, m2, m3)
                )
                print(f"[val debug] reset_state mem_max={mem_reset_max}")
        logits, extras = model(x, timesteps=timesteps, record_spikes=True)
        loss = criterion(logits, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            metrics = compute_classification_metrics(logits, y)
            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_acc += metrics["acc"] * batch_size
            total_precision += metrics["precision"] * batch_size
            total_recall += metrics["recall"] * batch_size
            total_f1 += metrics["f1"] * batch_size
            spk_metric = extras.get("spk_per_sample", 0.0)
            total_spikes += spk_metric * batch_size
            total_samples += batch_size
            processed_batches += 1
            if not train and not first_val_logged:
                # Debug info for the first validation batch only
                spk = getattr(model, "last_spk", None)
                mem = getattr(model, "last_mem", None)
                spk_shape = tuple(spk.shape) if spk is not None else None
                total_spk_batch = float(spk.sum().item()) if spk is not None else 0.0
                max_mem = float(mem.max().item()) if mem is not None else None
                vth = float(model.lif3.vth.item()) if hasattr(model, "lif3") else None
                print(
                    f"[val debug] spike tensor shape={spk_shape}, "
                    f"batch spikes={total_spk_batch:.1f}, "
                    f"max membrane={max_mem}, vth={vth}"
                )
                first_val_logged = True
        if train and (batch_idx + 1) % log_interval == 0:
            print(
                f"[train] batch {batch_idx+1}/{len(loader)} "
                f"loss={loss.item():.4f} acc={metrics['acc']:.3f} "
                f"prec={metrics['precision']:.3f} rec={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
                f"spk/sample={extras.get('spike_count', 0.0):.1f}"
            )
        if collect_confusion:
            preds = torch.argmax(logits, dim=1)
            confusion += compute_confusion_matrix(preds, y, num_classes=num_classes)
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
    if total_samples == 0:
        metrics: Dict[str, float | list[list[int]]] = {
            "loss": 0.0,
            "acc": 0.0,
            "f1": 0.0,
            "spk_per_sample": 0.0,
        }
        if confusion is not None:
            metrics["confusion_matrix"] = confusion.tolist()
        return metrics
    if not train:
        # Validation summary debug
        spk_per_sample = total_spikes / total_samples
        print(
            f"[val debug summary] batches={processed_batches}, "
            f"samples={total_samples}, total spikes={total_spikes:.1f}, "
            f"spikes/sample={spk_per_sample:.3f}"
        )
    metrics: Dict[str, float | list[list[int]]] = {
        "loss": total_loss / total_samples,
        "acc": total_acc / total_samples,
        "precision": total_precision / total_samples,
        "recall": total_recall / total_samples,
        "f1": total_f1 / total_samples,
        "spk_per_sample": total_spikes / total_samples,
    }
    if confusion is not None:
        metrics["confusion_matrix"] = confusion.tolist()
    return metrics


def train_model(
    cfg: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vth: float,
    tau: float,
    epochs: int | None = None,
    max_batches: int | None = None,
) -> Tuple[SpikingConvNet, list[Dict[str, Dict[str, float]]]]:
    set_seed(cfg.seed)
    device = cfg.train.device
    model = SpikingConvNet(
        img_size=cfg.data.img_size,
        vth=vth,
        tau=tau,
        reset_voltage=cfg.snn.reset_voltage,
        spike_input_scale=cfg.snn.spike_input_scale,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    epochs = epochs or cfg.train.epochs
    history: list[Dict[str, Dict[str, float]]] = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} (vth={vth}, tau={tau})")
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            timesteps=cfg.snn.timesteps,
            log_interval=cfg.train.log_interval,
            train=True,
            max_batches=max_batches,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            timesteps=cfg.snn.timesteps,
            log_interval=cfg.train.log_interval,
            train=False,
            max_batches=max_batches,
        )
        print(
            f" train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.3f} "
            f"prec={train_metrics['precision']:.3f} rec={train_metrics['recall']:.3f} "
            f"f1={train_metrics['f1']:.3f} spk={train_metrics['spk_per_sample']:.1f}"
        )
        print(
            f"  val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.3f} "
            f"prec={val_metrics['precision']:.3f} rec={val_metrics['recall']:.3f} "
            f"f1={val_metrics['f1']:.3f} spk={val_metrics['spk_per_sample']:.1f}"
        )
        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
    return model, history


def evaluate_model(
    model: SpikingConvNet,
    loader: DataLoader,
    device: str,
    timesteps: int,
) -> Dict[str, float | list[list[int]]]:
    return run_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        device=device,
        timesteps=timesteps,
        log_interval=10,
        train=False,
        collect_confusion=True,
    )
