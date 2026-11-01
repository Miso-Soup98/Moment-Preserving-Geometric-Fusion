from __future__ import annotations
import math
import warnings
import torch
from torch.autograd.function import once_differentiable
from .moments import check_floating, computation_dtype

class PSDRoundoffWarning(RuntimeWarning):
    pass

class _NullPreservingSqrt(torch.autograd.Function):

    @staticmethod
    def forward(ctx, covariance, epsilon, psd_rtol, psd_atol):
        sym = (covariance + covariance.transpose(-1, -2)) * 0.5
        values, vectors = torch.linalg.eigh(sym)
        scale = values.abs().amax(-1)
        tolerance = psd_atol + psd_rtol * scale
        minimum = values.amin(-1)
        if bool((minimum < -tolerance).any()):
            bad = (minimum < -tolerance).nonzero().tolist()
            raise ValueError('covariance is not PSD within per-matrix tolerance; bad batch indices %s' % bad[:8])
        if bool((values < 0).any()):
            warnings.warn('Clipped %d roundoff-negative eigenvalues within per-matrix PSD tolerance' % int((values < 0).sum()), PSDRoundoffWarning, stacklevel=3)
        clipped = values.clamp_min(0)
        roots = torch.sqrt(clipped + epsilon)
        response = clipped / (roots + math.sqrt(epsilon))
        ctx.save_for_backward(vectors, roots)
        return vectors * response.unsqueeze(-2) @ vectors.transpose(-1, -2)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        vectors, roots = ctx.saved_tensors
        symmetric_grad = (grad_output + grad_output.transpose(-1, -2)) * 0.5
        local = vectors.transpose(-1, -2) @ symmetric_grad @ vectors
        local = local / (roots.unsqueeze(-1) + roots.unsqueeze(-2))
        grad = vectors @ local @ vectors.transpose(-1, -2)
        return ((grad + grad.transpose(-1, -2)) * 0.5, None, None, None)

def null_preserving_sqrt(cov, epsilon=0.0001, *, psd_rtol=None, psd_atol=0.0):
    check_floating(cov, 'covariance')
    if cov.ndim < 2 or cov.shape[-1] != cov.shape[-2] or cov.shape[-1] == 0:
        raise ValueError('covariance must have shape [...,C,C], C > 0')
    if isinstance(epsilon, (bool, torch.Tensor)) or not isinstance(epsilon, (int, float)) or (not math.isfinite(epsilon)) or (epsilon <= 0):
        raise ValueError('epsilon must be a fixed finite positive scalar')
    dtype = computation_dtype(cov)
    if not torch.finfo(dtype).tiny <= epsilon <= torch.finfo(dtype).max:
        raise ValueError('epsilon must be representable as a normal positive number in the computation dtype')
    if psd_rtol is None:
        psd_rtol = 16 * cov.shape[-1] * torch.finfo(dtype).eps
    if not math.isfinite(psd_rtol) or psd_rtol < 0 or (not math.isfinite(psd_atol)) or (psd_atol < 0):
        raise ValueError('PSD tolerances must be finite and nonnegative')
    with torch.autocast(device_type=cov.device.type, enabled=False):
        work = cov.to(dtype)
        if work.numel() == 0:
            return work.clone()
        scale = work.abs().amax((-2, -1))
        residual = (work - work.transpose(-1, -2)).abs().amax((-2, -1))
        if bool((residual > psd_atol + psd_rtol * scale).any()):
            raise ValueError('covariance must be symmetric within per-matrix tolerance')
        return _NullPreservingSqrt.apply(work, float(epsilon), float(psd_rtol), float(psd_atol))
