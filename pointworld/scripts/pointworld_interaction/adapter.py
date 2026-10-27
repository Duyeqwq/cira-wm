"""Residual interaction mechanisms on encoded PointWorld action points."""
from __future__ import annotations

from contextlib import contextmanager
import inspect
import math

import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ("baseline", "global", "source", "source_local", "global_local")


def integer_labels(value, shape, name):
    if value is None:
        raise ValueError(f"{name} is required")
    if value.ndim == len(shape) + 1 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if tuple(value.shape) != tuple(shape) or value.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
    ):
        raise ValueError(f"{name} must be integer labels with shape {tuple(shape)}")
    return value


def masked_pool(features, mask):
    count = mask.sum(-1, keepdim=True)
    clean = features.masked_fill(~mask[..., None], 0)
    mean = clean.sum(-2) / count.clamp_min(1)
    maximum = features.masked_fill(~mask[..., None], -torch.inf).amax(-2)
    maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), -1), count.squeeze(-1) > 0


class InteractionAdapter(nn.Module):
    """Shared point encoding, temporal/source tokens, scene-conditioned weights.

    Global and source have exactly the same trainable parameters. Global-local
    is an additional capacity-matched control for source-local.
    """

    def __init__(self, channels, variant="source", width=128, neighbors=8,
                 scene_chunk=128, geometry_scale_m=.1):
        super().__init__()
        if variant not in VARIANTS[1:]:
            raise ValueError(f"Unknown adapter variant: {variant}")
        if min(channels, width, neighbors, scene_chunk) <= 0 or geometry_scale_m <= 0:
            raise ValueError("Adapter dimensions and scale must be positive")
        self.channels, self.variant, self.width = channels, variant, width
        self.neighbors, self.scene_chunk = neighbors, scene_chunk
        self.geometry_scale_m = geometry_scale_m
        self.point_encoder = nn.Sequential(
            nn.Linear(channels, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.token_encoder = nn.Linear(2 * width, width)
        self.query = nn.Linear(channels, width)
        self.key = nn.Linear(width, width)
        self.effect = nn.Linear(width, width)
        self.local_encoder = None
        if variant.endswith("_local"):
            self.local_encoder = nn.Sequential(
                nn.Linear(channels + 9, width), nn.GELU(),
                nn.Linear(width, width), nn.GELU())
        self.output = nn.Linear(width, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _tokens(self, robot_features, robot_exists, source_ids):
        b, t, n, _ = robot_features.shape
        source_ids = integer_labels(source_ids, (b, t, n), "robot_source_ids")
        if not torch.all((source_ids[robot_exists] >= 0) & (source_ids[robot_exists] < 2)):
            raise ValueError("Valid robot points require source A=0 or B=1")
        encoded = self.point_encoder(robot_features)
        tokens, valid = [], []
        for arm in range(2):
            mask = robot_exists
            if self.variant.startswith("source"):
                mask = mask & (source_ids == arm)
            pooled, present = masked_pool(encoded, mask)
            tokens.append(self.token_encoder(pooled))
            valid.append(present)
        return torch.stack(tokens, 2), torch.stack(valid, 2)

    def _local_tokens(self, scene_xyz, robot_xyz, robot_features, robot_exists,
                      source_ids, roles):
        b, t, n, _ = robot_features.shape
        roles = integer_labels(roles, (b, t, n), "robot_role_ids")
        if not torch.all((roles[robot_exists] >= 0) & (roles[robot_exists] < 3)):
            raise ValueError("Roles must be certified palm=0/leftfinger=1/rightfinger=2")
        role_features = F.one_hot(roles.clamp(0, 2).long(), 3).to(robot_features.dtype)
        # Only known robot trajectories are differenced. No future scene states.
        motion = torch.cat((robot_xyz[:, 1:] - robot_xyz[:, :-1],
                            torch.zeros_like(robot_xyz[:, :1])), 1)
        outputs, validity = [], []
        for arm in range(2):
            mask = robot_exists
            if self.variant.startswith("source"):
                mask = mask & (source_ids == arm)
            distances = torch.cdist(
                scene_xyz[:, None].expand(-1, t, -1, -1).float(),
                robot_xyz.float()).masked_fill(~mask[:, :, None], torch.inf)
            values, indices = distances.topk(min(self.neighbors, n), dim=-1, largest=False)
            good = torch.isfinite(values)
            batch = torch.arange(b, device=indices.device)[:, None, None, None]
            time = torch.arange(t, device=indices.device)[None, :, None, None]
            neighbor_feat = robot_features[batch, time, indices]
            relative = robot_xyz[batch, time, indices] - scene_xyz[:, None, :, None]
            local = torch.cat((
                neighbor_feat, relative / self.geometry_scale_m,
                motion[batch, time, indices] / self.geometry_scale_m,
                role_features[batch, time, indices]), -1)
            encoded = self.local_encoder(local)
            pooled, present = masked_pool(encoded, good)
            outputs.append(self.token_encoder(pooled))
            validity.append(present)
        # B,Nscene,T,arm,H
        return torch.stack(outputs, 3).permute(0, 2, 1, 3, 4), torch.stack(
            validity, 3).permute(0, 2, 1, 3)

    def forward(self, scene_features, robot_features, scene_xyz, robot_xyz,
                scene_exists, robot_exists, source_ids, roles=None):
        source_ids = integer_labels(source_ids, robot_exists.shape, "robot_source_ids")
        dtype = self.output.weight.dtype
        scene_features, robot_features = scene_features.to(dtype), robot_features.to(dtype)
        tokens, valid = self._tokens(robot_features, robot_exists, source_ids)
        residuals = []
        for first in range(0, scene_features.shape[1], self.scene_chunk):
            scene = scene_features[:, first:first + self.scene_chunk]
            query = self.query(scene)
            if self.local_encoder is not None:
                local, local_valid = self._local_tokens(
                    scene_xyz[:, first:first + self.scene_chunk], robot_xyz,
                    robot_features, robot_exists, source_ids, roles)
                grouped = tokens[:, None] + local
                present = valid[:, None] & local_valid
            else:
                grouped = tokens[:, None].expand(-1, scene.shape[1], -1, -1, -1)
                present = valid[:, None].expand(-1, scene.shape[1], -1, -1)
            grouped = grouped.flatten(2, 3)
            present = present.flatten(2, 3)
            scores = (query[:, :, None] * self.key(grouped)).sum(-1) / math.sqrt(self.width)
            # Empty arms and entirely padded samples cannot contribute via biases.
            scores = scores.masked_fill(~present, torch.finfo(scores.dtype).min)
            weights = scores.softmax(-1) * present
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
            fused = (weights[..., None] * self.effect(grouped)).sum(-2)
            delta = self.output(fused)
            active = present.any(-1) & scene_exists[:, first:first + self.scene_chunk]
            residuals.append(delta.masked_fill(~active[..., None], 0))
        return torch.cat(residuals, 1)


class AdaptedDynamicsPredictor(nn.Module):
    """Run the original forward unchanged, injecting a residual at its head inputs.

    Hooks are scoped to one call and always removed. They are not process-global
    patches. Compilation, concurrent calls on one instance and DDP are deferred.
    """

    def __init__(self, base, adapter):
        super().__init__()
        for name in ("predictor_model", "dynamics_head", "log_var_head", "in_c"):
            if not hasattr(base, name):
                raise TypeError(f"Local DynamicsPredictor lacks {name}")
        if hasattr(base, "source_effect_mlp"):
            raise ValueError("Do not mix the legacy compositional summary into this ablation")
        self.base, self.adapter = base, adapter
        self.enabled = True
        self.roles = None
        self._active = False

    @contextmanager
    def disabled(self):
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous

    def forward(self, *args, **kwargs):
        if not self.enabled:
            return self.base(*args, **kwargs)
        if self._active:
            raise RuntimeError("Concurrent/reentrant forwards on one adapter are unsupported")
        values = inspect.signature(self.base.forward).bind(*args, **kwargs)
        values.apply_defaults()
        data = values.arguments
        scene_exists, robot_exists = data["scene_exists0"], data["robot_exists"]
        captured = {}

        def capture(_module, _inputs, point):
            features = point.feat
            exists = torch.cat((scene_exists, robot_exists.flatten(1)), 1)
            padded = features.new_zeros((*exists.shape, features.shape[-1]))
            padded[exists] = features
            ns = scene_exists.shape[1]
            captured["scene"] = padded[:, :ns]
            captured["robot"] = padded[:, ns:].reshape(*robot_exists.shape, -1)

        def inject(_module, inputs):
            if "delta" not in captured:
                captured["delta"] = self.adapter(
                    captured["scene"], captured["robot"], data["scene_coord0"],
                    data["robot_coord_seq"], scene_exists, robot_exists,
                    data.get("robot_source_ids"), self.roles)
            return (inputs[0] + captured["delta"].to(inputs[0].dtype), *inputs[1:])

        handles = []
        self._active = True
        try:
            handles.append(self.base.predictor_model.register_forward_hook(capture))
            for head in (self.base.dynamics_head, self.base.log_var_head):
                handles.append(head.register_forward_pre_hook(inject))
            return self.base(*args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
            self._active = False


def install_adapter(model, variant, **options):
    """Official weights must already be loaded strictly before this function."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if hasattr(model, "robot_source_emb"):
        raise ValueError("Source embeddings are a separate ablation")
    if isinstance(model.dynamics_predictor, AdaptedDynamicsPredictor):
        raise ValueError("Adapter is already installed")
    if variant == "baseline":
        if hasattr(model.dynamics_predictor, "source_effect_mlp"):
            raise ValueError("Baseline must not enable the legacy summary")
        return None
    base = model.dynamics_predictor
    parameter = next(base.parameters())
    adapter = InteractionAdapter(base.in_c, variant, **options).to(
        device=parameter.device, dtype=parameter.dtype)
    wrapped = AdaptedDynamicsPredictor(base, adapter)
    model.dynamics_predictor = wrapped
    return wrapped


def forward_model(model, batch, *, planning=False, **kwargs):
    kinds = batch.get("action_kind")
    kinds = [kinds] if isinstance(kinds, str) else kinds
    if not kinds or any(k not in ("recorded_offline", "planned_fk") for k in kinds):
        raise ValueError("Explicit action_kind is required")
    if planning and any(k != "planned_fk" for k in kinds):
        raise PermissionError("Recorded future robot states are offline-only")
    predictor = model.dynamics_predictor
    if isinstance(predictor, AdaptedDynamicsPredictor):
        previous = predictor.roles
        predictor.roles = batch.get("robot_role_ids")
        try:
            return model(batch, **kwargs)
        finally:
            predictor.roles = previous
    return model(batch, **kwargs)


def freeze_scene_encoder(model):
    for parameter in model.scene_feature_encoder.parameters():
        parameter.requires_grad_(False)
    model.scene_feature_encoder.eval()
    return [p for p in model.parameters() if p.requires_grad]
