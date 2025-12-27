from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeavisideSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input_,) = ctx.saved_tensors
        surrogate_grad = 1.0 / (1.0 + torch.abs(input_))
        return grad_output * surrogate_grad


class LIFLayer(nn.Module):
    def __init__(self, vth: float, tau: float, reset_voltage: float = 0.0):
        super().__init__()
        self.vth = nn.Parameter(torch.tensor(float(vth)), requires_grad=False)
        self.tau = nn.Parameter(torch.tensor(float(tau)), requires_grad=False)
        self.reset_voltage = reset_voltage

    def forward(self, input_current: torch.Tensor, mem: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if mem is None:
            mem = torch.zeros_like(input_current)
        # Discrete LIF update with simple leak
        mem = mem + (input_current - mem) / self.tau
        spk = HeavisideSTE.apply(mem - self.vth)
        mem = mem * (1 - spk) + self.reset_voltage * spk
        return spk, mem

    def set_params(self, vth: float, tau: float):
        self.vth.data.fill_(float(vth))
        self.tau.data.fill_(float(tau))


class SpikingConvNet(nn.Module):
    def __init__(self, img_size: int, vth: float, tau: float, reset_voltage: float, spike_input_scale: float, input_gain: float = 1.1):
        super().__init__()
        self.spike_input_scale = spike_input_scale
        self.input_gain = input_gain
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        # compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size)
            dummy = self.conv2(self.conv1(dummy))
            self._flatten_dim = dummy.numel()
        self.fc1 = nn.Linear(self._flatten_dim, 128)
        self.fc2 = nn.Linear(128, 2)
        self.lif1 = LIFLayer(vth=vth, tau=tau, reset_voltage=reset_voltage)
        self.lif2 = LIFLayer(vth=vth, tau=tau, reset_voltage=reset_voltage)
        self.lif3 = LIFLayer(vth=vth, tau=tau, reset_voltage=reset_voltage)

    def lif_layers(self):
        return [self.lif1, self.lif2, self.lif3]

    def update_neuron_params(self, vth: float, tau: float):
        for lif in self.lif_layers():
            lif.set_params(vth, tau)

    def reset_state(self):
        # explicit state reset per batch for debug/consistency
        self.mem1 = None
        self.mem2 = None
        self.mem3 = None
        self.last_spk = None
        self.last_mem = None

    def forward(self, x: torch.Tensor, timesteps: int, record_spikes: bool = False):
        # x: (B, 1, H, W) in [0,1]
        batch_size = x.size(0)
        spike_count = 0.0
        logits_sum = torch.zeros(batch_size, 2, device=x.device)
        for _ in range(timesteps):
            # clamp probabilities to valid range to avoid Bernoulli errors
            encoded = torch.bernoulli((x * self.spike_input_scale).clamp(0.0, 1.0))
            cur1 = self.conv1(encoded) * self.input_gain
            mem1_prev = self.mem1 if self.mem1 is not None else torch.zeros_like(cur1)
            spk1, self.mem1 = self.lif1(cur1, mem1_prev)

            cur2 = self.conv2(spk1)
            mem2_prev = self.mem2 if self.mem2 is not None else torch.zeros_like(cur2)
            spk2, self.mem2 = self.lif2(cur2, mem2_prev)

            flat = spk2.flatten(1)
            cur3 = self.fc1(flat)
            mem3_prev = self.mem3 if self.mem3 is not None else torch.zeros_like(cur3)
            spk3, self.mem3 = self.lif3(cur3, mem3_prev)

            logits = self.fc2(spk3.float())
            logits_sum = logits_sum + logits

            spike_count += spk1.sum().item() + spk2.sum().item() + spk3.sum().item()

        logits_avg = logits_sum / timesteps
        spike_per_sample = spike_count / batch_size / timesteps
        extras = {"spk_per_sample": spike_per_sample} if record_spikes else {}
        # stash last spike/membrane for debug logging without changing the API
        self.last_spk = spk3.detach()
        self.last_mem = self.mem3.detach() if self.mem3 is not None else None
        return logits_avg, extras
