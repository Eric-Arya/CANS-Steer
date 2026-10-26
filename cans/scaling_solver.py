"""Closed-form CANS coefficients and the Grad-only comparison."""

import math

import torch


def cans_scales(gradient, fisher, epsilon, damping=None, cap=None):
    """Maximize the signed steering objective under a diagonal quadratic budget.

    Use F + damping in both the direction and the quadratic budget.
    Clipping follows normalization and is never followed by renormalization.
    """
    g, f = gradient.double(), fisher.double()
    if g.shape != f.shape or g.ndim != 2 or not g.numel():
        raise ValueError("Expected matching [layers, intermediate_size] tensors")
    if not g.isfinite().all() or not f.isfinite().all() or (f < 0).any():
        raise ValueError("Gradients must be finite and Fisher must be nonnegative")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    damping = float(f.median()) if damping is None else float(damping)
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be positive; specify it if the median is zero")
    if cap is not None and (not math.isfinite(cap) or cap <= 0):
        raise ValueError("cap must be positive or None")
    metric = f + damping
    direction = g / metric
    normalization = float((g * direction).sum())
    coefficient = math.sqrt(2 * epsilon / normalization) if normalization else 0.0
    raw_delta = coefficient * direction
    delta = raw_delta if cap is None else raw_delta.clamp(-cap, cap)
    saturation = 0.0 if cap is None else float((raw_delta.abs() > cap).double().mean())
    scales = 1 + delta
    if not scales.isfinite().all():
        raise FloatingPointError("Non-finite neuron scales")
    return scales, {
        "epsilon": epsilon, "damping": damping, "cap": cap,
        "coefficient": coefficient, "capped_fraction": saturation,
        "pre_cap_cost": float((metric * raw_delta.square()).sum() / 2),
        "post_cap_cost": float((metric * delta.square()).sum() / 2),
    }


def grad_only_scales(target_gradient, top_k, multiplier):
    """Rank positive mean target gradients, then scale the selected coordinates.

    Use a multiplier above one to promote the target, below one to suppress it.
    Ranking uses the original target gradient even for suppressive steering.
    """
    if target_gradient.ndim != 2 or not target_gradient.isfinite().all():
        raise ValueError("Expected finite [layers, intermediate_size] gradients")
    if not math.isfinite(multiplier):
        raise ValueError("multiplier must be finite")
    flat = target_gradient.float().flatten()
    if not isinstance(top_k, int) or not 0 < top_k <= int((flat > 0).sum()):
        raise ValueError("top_k exceeds the positive-gradient neuron pool")
    # Stable sorting makes tied coordinates follow layer-major order.
    selected = torch.argsort(flat, descending=True, stable=True)[:top_k]
    scales = torch.ones_like(flat, dtype=torch.float64)
    scales[selected] = multiplier
    return scales.reshape(target_gradient.shape)
