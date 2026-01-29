from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.stats import norm
import torch.distributed as dist

def get_schedule_sampler(config, diffusion) -> "ScheduleSampler":
    name = config.name

    if name == "continuous-uniform":
        return ContinuousUniformSampler(diffusion, getattr(config, 'min_t', 0.01))
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ContinuousUniformSampler:
    def __init__(self, diffusion, min_t=0.01, max_t=0.99):
        self.diffusion = diffusion
        self.min_t = min_t
        self.max_t = max_t

    def sample(self, batch_size, device):
        ts = torch.rand(batch_size).to(device) * (self.max_t - self.min_t) + self.min_t
        return ts, torch.ones_like(ts)

