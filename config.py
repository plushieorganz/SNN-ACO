from dataclasses import dataclass, field
import torch


@dataclass
class DataConfig:
    data_root: str = "data/bird_drone"  # expected ImageFolder-style tree with train/val/test/bird, drone
    use_synthetic: bool = False  # use real dataset; set True for synthetic fallback
    synthetic_samples: int = 400  # per split when synthetic data is enabled
    img_size: int = 64  # images are converted to 1xHxW grayscale tensors
    train_split: float = 0.7
    val_split: float = 0.15


@dataclass
class SNNConfig:
    timesteps: int = 8  # number of simulation steps per sample
    spike_input_scale: float = 1.7  # slightly higher drive to keep neurons firing
    baseline_vth: float = 0.35  # higher threshold to stay selective while allowing spikes
    baseline_tau: float = 1.4  # short time constant for responsive membranes
    reset_voltage: float = 0.0  # reset level after spikes


@dataclass
class TrainConfig:
    epochs: int = 6
    search_epochs: int = 3  # epochs per candidate during ACO search
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 10


@dataclass
class ACOConfig:
    ants: int = 6
    iterations: int = 4
    evaporation: float = 0.4
    q: float = 1.0  # pheromone deposit magnitude
    vth_candidates: tuple = (0.05, 0.1, 0.2, 0.4, 0.8)
    tau_candidates: tuple = (0.8, 1.0, 1.2, 1.6, 2.0)
    spike_penalty: float = 0.10  # less aggressive spike penalty to protect accuracy
    quick_subset: int | None = None  # use full loader during search for clearer signal
    min_spike_per_sample: float = 25.0  # allow lower but non-zero spikes
    low_spike_penalty: float = 0.05  # mild penalty for overly silent models
    max_spike_per_sample: float = 5000.0  # reject extreme spike explosions


@dataclass
class ExperimentConfig:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    snn: SNNConfig = field(default_factory=SNNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    aco: ACOConfig = field(default_factory=ACOConfig)
