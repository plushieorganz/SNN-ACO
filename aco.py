from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from config import ExperimentConfig
from train import train_model


@dataclass
class ACOResult:
    best_vth: float
    best_tau: float
    best_fitness: float
    best_val_metrics: Dict[str, float]


def _sample_with_pheromone(rng: np.random.Generator, pheromone: np.ndarray, candidates: tuple):
    probs = pheromone / pheromone.sum()
    idx = rng.choice(len(candidates), p=probs)
    return candidates[idx], idx


class AntColonyOptimizer:
    def __init__(self, cfg: ExperimentConfig, train_loader, val_loader):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vth_pheromone = np.ones(len(cfg.aco.vth_candidates))
        self.tau_pheromone = np.ones(len(cfg.aco.tau_candidates))
        self.rng = np.random.default_rng(cfg.seed)

    def fitness(self, val_metrics: Dict[str, float]) -> float:
        spk = val_metrics["spk_per_sample"]
        acc = val_metrics["acc"]
        if spk < self.cfg.aco.min_spike_per_sample:
            return -1e9
        if spk > self.cfg.aco.max_spike_per_sample:
            return -1e9
        spk_norm = spk / 1000.0
        score = acc / (1.0 + self.cfg.aco.spike_penalty * spk_norm)
        # debug for first candidate per iteration handled in run()
        return score

    def run(self) -> ACOResult:
        best: Tuple[float, float, int, int, float, Dict[str, float]] | None = None
        for iteration in range(self.cfg.aco.iterations):
            print(f"\n[ACO] iteration {iteration+1}/{self.cfg.aco.iterations}")
            iteration_best: Tuple[float, float, int, int, float, Dict[str, float]] | None = None
            for ant in range(self.cfg.aco.ants):
                vth, vth_idx = _sample_with_pheromone(self.rng, self.vth_pheromone, self.cfg.aco.vth_candidates)
                tau, tau_idx = _sample_with_pheromone(self.rng, self.tau_pheromone, self.cfg.aco.tau_candidates)
                print(f" ant {ant+1}/{self.cfg.aco.ants} testing vth={vth:.2f}, tau={tau:.2f}")
                _, history = train_model(
                    cfg=self.cfg,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    vth=vth,
                    tau=tau,
                    epochs=self.cfg.train.search_epochs,
                    max_batches=self.cfg.aco.quick_subset,
                )
                val_metrics = history[-1]["val"]
                fit = self.fitness(val_metrics)
                if ant == 0:
                    spk = val_metrics["spk_per_sample"]
                    acc = val_metrics["acc"]
                    spk_norm = spk / 1000.0
                    print(
                        f"[ACO debug] acc={acc:.3f} spk={spk:.1f} spk_norm={spk_norm:.3f} "
                        f"penalty={self.cfg.aco.spike_penalty} fitness={fit:.6f}"
                    )
                print(
                    f"  val f1={val_metrics['f1']:.3f} acc={val_metrics['acc']:.3f} "
                    f"spk={val_metrics['spk_per_sample']:.1f} fitness={fit:.3f}"
                )
                if iteration_best is None or fit > iteration_best[4]:
                    iteration_best = (vth, tau, vth_idx, tau_idx, fit, val_metrics)
                # keep best globally
                if best is None or fit > best[4]:
                    best = (vth, tau, vth_idx, tau_idx, fit, val_metrics)
            # pheromone evaporation
            self.vth_pheromone *= (1 - self.cfg.aco.evaporation)
            self.tau_pheromone *= (1 - self.cfg.aco.evaporation)
            # pheromone deposit using iteration best
            if iteration_best:
                vth_idx = iteration_best[2]
                tau_idx = iteration_best[3]
                deposit = max(iteration_best[4], 1e-6) * self.cfg.aco.q
                self.vth_pheromone[vth_idx] += deposit
                self.tau_pheromone[tau_idx] += deposit
                print(
                    f"  depositing pheromone={deposit:.3f} on vth idx {vth_idx}, tau idx {tau_idx} "
                    f"pheromone vth={self.vth_pheromone} tau={self.tau_pheromone}"
                )
        assert best is not None, "ACO search failed to produce any candidate"
        return ACOResult(
            best_vth=best[0],
            best_tau=best[1],
            best_fitness=best[2],
            best_val_metrics=best[3],
        )
