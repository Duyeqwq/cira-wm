"""Experimental world-metre constraint operators, not an installed predictor."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RigidFit:
    rotation: torch.Tensor
    translation: torch.Tensor
    rms_m: torch.Tensor
    valid: torch.Tensor


def fit_rigid(source, target, exists):
    """Fit matched points with column-vector convention y = R x + t.

    Invalid fits return identity plus valid=False, not an accepted estimate.
    Intended for observed/planned robot geometry, never future object geometry.
    """
    if (source.shape != target.shape or source.ndim != 3 or source.shape[-1] != 3
            or exists.shape != source.shape[:2] or exists.dtype != torch.bool):
        raise ValueError("Expected matched [B,N,3] and boolean [B,N]")
    if not (torch.isfinite(source[exists]).all() and torch.isfinite(target[exists]).all()):
        raise ValueError("Nonfinite valid robot geometry")
    if source.shape[1] < 1:
        raise ValueError("Empty point bank")
    x = source.masked_fill(~exists[..., None], 0)
    y = target.masked_fill(~exists[..., None], 0)
    count = exists.sum(-1).clamp_min(1)
    cx = x.sum(1) / count[:, None]
    cy = y.sum(1) / count[:, None]
    x = (x - cx[:, None]).masked_fill(~exists[..., None], 0)
    y = (y - cy[:, None]).masked_fill(~exists[..., None], 0)
    covariance = x.transpose(-1, -2) @ y
    u, singular, vh = torch.linalg.svd(covariance)
    correction = torch.ones_like(singular)
    correction[:, -1] = torch.det(vh.transpose(-1, -2) @ u.transpose(-1, -2))
    rotation = (vh.transpose(-1, -2) * correction[:, None]) @ u.transpose(-1, -2)
    translation = cy - (rotation @ cx[..., None]).squeeze(-1)
    residual = source.masked_fill(~exists[..., None], 0) @ rotation.transpose(-1, -2)
    residual = (residual + translation[:, None] - target.masked_fill(
        ~exists[..., None], 0)).masked_fill(~exists[..., None], 0)
    rms = (residual.square().sum((1, 2)) / count).sqrt()
    # A planar palm is sufficient; a line cannot identify rotation around itself.
    valid = (exists.sum(-1) >= 3) & (
        singular[:, 1] > singular[:, 0].clamp_min(torch.finfo(source.dtype).tiny) * 1e-6)
    identity = torch.eye(3, dtype=source.dtype, device=source.device).expand_as(rotation)
    return RigidFit(
        torch.where(valid[:, None, None], rotation, identity),
        torch.where(valid[:, None], translation, torch.zeros_like(translation)),
        rms, valid)


def rigid_references(current, rotation, translation):
    """[B,N,3], [B,A,3,3], [B,A,3] -> [B,N,A,3], including rotation."""
    if (current.ndim != 3 or current.shape[-1] != 3
            or rotation.ndim != 4 or rotation.shape[-2:] != (3, 3)
            or translation.shape != rotation.shape[:2] + (3,)
            or rotation.shape[0] != current.shape[0]):
        raise ValueError("Incompatible rigid-reference shapes")
    return torch.einsum("baij,bnj->bnai", rotation, current) + translation[:, None]


def fuse_constraints(free, reference, factors, strength, free_precision=1.0):
    """Minimize ||y-free||^2*p + sum_a ||L_a^T(y-ref_a)||^2*s_a.

    factors [B,N,A,3,K] define PSD directional weights L L^T. A scalar
    proximity gate alone cannot represent free sliding in a tangential plane.
    The returned disagreement is a residual, NOT calibrated uncertainty.
    """
    if (free.ndim != 3 or free.shape[-1] != 3
            or reference.ndim != 4 or reference.shape[:2] != free.shape[:2]
            or reference.shape[-1] != 3
            or factors.ndim != 5 or factors.shape[:3] != reference.shape[:3]
            or factors.shape[-2] != 3 or factors.shape[-1] < 1
            or strength.shape != reference.shape[:-1]):
        raise ValueError("Invalid constraint shapes")
    if not isinstance(free_precision, (float, int)) or not 0 < free_precision < float("inf"):
        raise ValueError("Positive finite free precision required")
    if not all(torch.isfinite(x).all() for x in (free, reference, factors, strength)):
        raise ValueError("Nonfinite constraint input")
    if (strength < 0).any():
        raise ValueError("Constraint strengths must be nonnegative")
    precision = (factors @ factors.transpose(-1, -2)) * strength[..., None, None]
    identity = torch.eye(3, device=free.device, dtype=free.dtype)
    system = free_precision * identity + precision.sum(2)
    rhs = free_precision * free + (precision @ reference[..., None]).sum(2).squeeze(-1)
    value = torch.linalg.solve(system, rhs[..., None]).squeeze(-1)
    error = value[:, :, None] - reference
    disagreement = torch.einsum("bnai,bnaij,bnaj->bn", error, precision, error)
    return value, disagreement


def filter_attachment(previous, establish, release):
    """One probabilistic persistence update; rates must be inferred, not GT labels.

    Release is applied last and can erase prior attachment in one step.
    This is a per-arm marginal; simultaneous constraints are not added as flows.
    """
    if previous.shape != establish.shape or previous.shape != release.shape:
        raise ValueError("Attachment-state shapes differ")
    if not all(torch.isfinite(x).all() and ((x >= 0) & (x <= 1)).all()
               for x in (previous, establish, release)):
        raise ValueError("Finite probabilities required")
    return (previous + (1 - previous) * establish) * (1 - release)


def carry_reference(previous_reference, current, renewal, rotation, translation):
    """Transport a binding reference without re-anchoring it to every prediction.

    renewal is the probability mass of newly established binding, normalized
    against retained binding. A detached binding (mass zero) must renew at the
    next establishment. This operator does not infer any probabilities.
    """
    if (previous_reference.ndim != 4 or previous_reference.shape[-1] != 3
            or current.shape != previous_reference.shape[:2] + (3,)
            or renewal.shape != previous_reference.shape[:-1]
            or not torch.isfinite(renewal).all()
            or not ((renewal >= 0) & (renewal <= 1)).all()):
        raise ValueError("Invalid carrier-reference shapes/probabilities")
    source = previous_reference + renewal[..., None] * (
        current[:, :, None] - previous_reference)
    return torch.einsum("baij,bnaj->bnai", rotation, source) + translation[:, None]
