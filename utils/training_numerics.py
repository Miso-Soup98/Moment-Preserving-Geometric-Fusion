import math
import torch

def require_finite_loss(loss):
    if not torch.is_tensor(loss) or loss.numel() != 1:
        raise ValueError('Training loss must be one scalar tensor')
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError('Training loss contains nonfinite values')

def _gradient_values(gradient):
    return gradient.coalesce().values() if gradient.is_sparse else gradient

def _norm64(value):
    dtype = torch.complex128 if value.is_complex() else torch.float64
    return torch.linalg.vector_norm(value.detach(), ord=2, dtype=dtype)

@torch.no_grad()
def check_and_clip_gradients(parameters, max_norm):
    if max_norm is not None:
        max_norm = float(max_norm)
        if not math.isfinite(max_norm) or max_norm < 0:
            raise ValueError('grad_clip must be finite and nonnegative, or None')
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.tensor(0.0, dtype=torch.float64)
    device = gradients[0].device
    norms = [_norm64(_gradient_values(gradient)).to(device) for gradient in gradients]
    total_norm = torch.linalg.vector_norm(torch.stack(norms), ord=2)
    if bool(torch.isfinite(total_norm)):
        if max_norm is None:
            return total_norm
        coefficient = (max_norm / (total_norm + 1e-06)).clamp(max=1.0)
    else:
        values = [_gradient_values(gradient) for gradient in gradients]
        if not all((bool(torch.isfinite(value).all()) for value in values)):
            raise FloatingPointError('Training gradients contain nonfinite values')
        if max_norm is None:
            return total_norm

        def component_max(value):
            if value.is_complex():
                return torch.maximum(value.real.abs().max(), value.imag.abs().max())
            return value.abs().max()
        scale = torch.stack([component_max(value.detach()).to(device=device, dtype=torch.float64) if value.numel() else torch.tensor(0.0, device=device, dtype=torch.float64) for value in values]).max()

        def scaled_value(value):
            local_scale = scale.to(value.device)
            if value.is_complex():
                return torch.complex(value.real / local_scale, value.imag / local_scale)
            return value / local_scale
        scaled_norms = [_norm64(scaled_value(value)).to(device) for value in values]
        scaled_norm = torch.linalg.vector_norm(torch.stack(scaled_norms), ord=2)
        coefficient = (max_norm / scale / (scaled_norm + 1e-06 / scale)).clamp(max=1.0)
    buckets = {}
    for gradient in gradients:
        if gradient.is_sparse:
            gradient.mul_(coefficient.to(gradient.device))
        else:
            buckets.setdefault((gradient.device, gradient.dtype), []).append(gradient)
    for (gradient_device, _), group in buckets.items():
        torch._foreach_mul_(group, coefficient.to(gradient_device))
    return total_norm
