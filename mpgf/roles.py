from __future__ import annotations
import hashlib
import json
from typing import Optional
import torch
from torch import nn
from .moments import check_floating

class AtomRoleTable(nn.Module):

    def __init__(self, attributes, chemical_occupancy, weights=None, *, learnable_attributes=False, learnable_weights=False, identity_symbols=None, model_token_indices=None, atom_names=None):
        super().__init__()
        check_floating(attributes, 'attributes')
        if attributes.ndim != 3 or min(attributes.shape) <= 0:
            raise ValueError('attributes must have shape [K,C,d]')
        k, c, _ = attributes.shape
        if chemical_occupancy.shape != (k, c) or chemical_occupancy.device != attributes.device:
            raise ValueError('chemical_occupancy shape/device mismatch')
        if not bool(((chemical_occupancy == 0) | (chemical_occupancy == 1)).all()):
            raise ValueError('chemical_occupancy must encode candidate existence as 0/1')
        weights = torch.ones((k, c), dtype=attributes.dtype, device=attributes.device) if weights is None else weights
        check_floating(weights, 'weights')
        if weights.shape != (k, c) or weights.device != attributes.device:
            raise ValueError('weights shape/device mismatch')
        if learnable_attributes:
            self.attributes = nn.Parameter(attributes.clone())
        else:
            self.register_buffer('attributes', attributes.clone())
        if learnable_weights:
            self.weights = nn.Parameter(weights.clone())
        else:
            self.register_buffer('weights', weights.clone())
        self.register_buffer('chemical_occupancy', chemical_occupancy.to(attributes.dtype).clone())
        if identity_symbols is None:
            identity_symbols = ['candidate_%d' % i for i in range(k)]
        if model_token_indices is None:
            model_token_indices = list(range(k))
        if len(identity_symbols) != k or len(set(identity_symbols)) != k:
            raise ValueError('identity_symbols must be unique, with K entries')
        if len(model_token_indices) != k or len(set(model_token_indices)) != k or any((isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in model_token_indices)):
            raise ValueError('model_token_indices must be unique nonnegative integers')
        if atom_names is not None and (len(atom_names) != k or any((len(row) != c for row in atom_names))):
            raise ValueError('atom_names must describe each [K,C] candidate slot')
        self.metadata = {'identity_symbols': list(identity_symbols), 'model_token_indices': list(model_token_indices), 'atom_names': atom_names, 'role_scale_mode': 'embedding'}

    def forward(self):
        return self.attributes * (self.chemical_occupancy * self.weights).unsqueeze(-1)

    def get_extra_state(self):
        return self.metadata

    def set_extra_state(self, state):
        if state != self.metadata:
            raise RuntimeError('Role-table vocabulary/atom-name metadata mismatch')

    def role_hash(self):
        hasher = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode())
        tensor = self().detach().cpu().contiguous()
        hasher.update(str(tensor.dtype).encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(tensor.view(torch.uint8).numpy().tobytes())
        return hasher.hexdigest()

    @torch.no_grad()
    def spectral_norms(self):
        return torch.linalg.matrix_norm(self().to(torch.float64), ord=2)
