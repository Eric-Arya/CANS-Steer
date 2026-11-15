"""Closed-form CANS coefficients and the Grad-only comparison."""

import math

import torch


def cans_scales(gradient, fisher, epsilon, damping=None, cap=None):
    """Maximize the signed steering objective under a diagonal quadratic budget.

    Use F + damping in both the direction and the quadratic budget. Omitted
    damping is the median over every Fisher coordinate, including zeros.
    The current protocol deploys the unmodified solution (cap=None).
    Explicit historical clipping follows normalization without renormalization.
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


def grad_only_scales(target_gradient, coefficient, cap=None, *, objective_sign=1):
    """Identity-curvature CANS: retain every signed FP32 mean-gradient coordinate.

    The current protocol is unclipped; vary only ``coefficient``.
    ``objective_sign`` is +1 for safety/hallucination and -1 for toxicity.
    ``target_gradient`` is the target log-probability gradient before applying
    that task sign. No ranking, Fisher division, damping, or normalization is
    applied. Explicit ``cap`` remains available for historical comparisons.
    """
    if (target_gradient.ndim != 2 or not target_gradient.numel()
            or target_gradient.dtype != torch.float32
            or not target_gradient.isfinite().all()):
        raise ValueError("Expected finite, nonempty FP32 [layers, intermediate_size] gradients")
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("coefficient must be finite and nonnegative")
    if cap is not None and (not math.isfinite(cap) or cap <= 0):
        raise ValueError("cap must be finite and positive, or None for uncapped CANS")
    if objective_sign not in (-1, 1):
        raise ValueError("objective_sign must be +1 or -1")
    raw_delta = coefficient * target_gradient.double()
    if not raw_delta.isfinite().all():
        raise ValueError("coefficient times gradient is non-finite")
    delta = raw_delta if cap is None else raw_delta.clamp(-cap, cap)
    return 1 + objective_sign * delta
