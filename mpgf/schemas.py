from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import torch
CONTRACT_VERSION = 'MPGF-v1'
OPERATOR_VARIANTS = ('mpgf',)

@dataclass(frozen=True)
class FusionConfig:
    atom_slots: int
    role_dim: int
    output_dim: int
    variant: str = 'mpgf'
    epsilon: float = 0.0001
    distance_epsilon: float = 0.0001
    length_scale: float = 1.0
    node_chunk_size: int = 256
    edge_chunk_size: int = 512
    alpha: float = 1.0
    role_scale_mode: str = 'embedding'

    def __post_init__(self):
        for name in ('atom_slots', 'role_dim', 'output_dim', 'node_chunk_size', 'edge_chunk_size'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('%s must be a positive integer' % name)
        if self.variant not in OPERATOR_VARIANTS:
            raise ValueError('Unsupported standalone fusion variant: %s' % self.variant)
        for name in ('epsilon', 'distance_epsilon', 'length_scale'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) or (value <= 0):
                raise ValueError('%s must be a finite positive scalar' % name)
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError('alpha must be in [0,1]')
        if self.role_scale_mode != 'embedding':
            raise ValueError('Only embedding role scale is supported')

@dataclass
class PreparedMoments:
    mean: torch.Tensor
    row_covariance: torch.Tensor
    lift: torch.Tensor
    variant: str = 'mpgf'
    chemical_occupancy: Optional[torch.Tensor] = None

    def index_select(self, index):
        occupancy = getattr(self, 'chemical_occupancy', None)
        return PreparedMoments(self.mean[index], self.row_covariance[index], self.lift[index], self.variant, None if occupancy is None else occupancy[index])

@dataclass
class FusionOutput:
    edge_features: torch.Tensor
    raw_relation: Optional[torch.Tensor] = None
    normalized_relation: Optional[torch.Tensor] = None
    mean: Optional[torch.Tensor] = None
    row_covariance: Optional[torch.Tensor] = None
