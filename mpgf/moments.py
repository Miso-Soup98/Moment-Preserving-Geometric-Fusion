from __future__ import annotations
import torch
FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

def computation_dtype(*tensors):
    return torch.float64 if any((t.dtype == torch.float64 for t in tensors)) else torch.float32

def check_floating(tensor, name):
    if not isinstance(tensor, torch.Tensor) or tensor.dtype not in FLOAT_DTYPES:
        raise TypeError('%s must be a floating-point tensor' % name)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError('%s contains nonfinite values' % name)

def validate_probabilities(prob, tolerance=None):
    check_floating(prob, 'identity_prob')
    if prob.ndim != 2 or prob.shape[1] == 0:
        raise ValueError('identity_prob must have shape [N,K] with K > 0')
    if bool((prob < 0).any()):
        raise ValueError('identity_prob must be nonnegative')
    if tolerance is None:
        tolerance = {torch.float64: 1e-10, torch.float32: 2e-06, torch.float16: 0.002, torch.bfloat16: 0.01}[prob.dtype]
    if not bool(((prob.to(computation_dtype(prob)).sum(-1) - 1).abs() <= tolerance).all()):
        raise ValueError('identity_prob rows must sum to one')
    return prob

def validate_roles(prob, roles):
    validate_probabilities(prob)
    check_floating(roles, 'role_table')
    if roles.ndim != 3 or min(roles.shape) <= 0 or roles.shape[0] != prob.shape[1]:
        raise ValueError('role_table must have shape [K,C,d] compatible with probabilities')
    if roles.device != prob.device:
        raise ValueError('identity_prob and role_table must be on the same device')

def role_moments(prob, roles, node_chunk_size=256):
    validate_roles(prob, roles)
    if isinstance(node_chunk_size, bool) or not isinstance(node_chunk_size, int) or node_chunk_size <= 0:
        raise ValueError('node_chunk_size must be a positive integer')
    return _role_moments_unchecked(prob, roles, node_chunk_size)

def _role_moments_unchecked(prob, roles, node_chunk_size):
    dtype = computation_dtype(prob, roles)
    with torch.autocast(device_type=roles.device.type, enabled=False):
        p, b = (prob.to(dtype), roles.to(dtype))
        if len(p) == 0:
            mean = torch.einsum('nk,kcd->ncd', p, b)
            cov = mean @ mean.transpose(-1, -2)
            return (mean, cov)
        means, covs = ([], [])
        for start in range(0, len(p), node_chunk_size):
            pc = p[start:start + node_chunk_size]
            mu = torch.einsum('nk,kcd->ncd', pc, b)
            centered = b.unsqueeze(0) - mu.unsqueeze(1)
            cov = torch.einsum('nk,nkcd,nked->nce', pc, centered, centered)
            means.append(mu)
            covs.append((cov + cov.transpose(-1, -2)) * 0.5)
        return (torch.cat(means), torch.cat(covs))
