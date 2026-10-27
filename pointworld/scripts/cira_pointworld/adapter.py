"""One PTv3 pass, recurrent local relations, original dense point-flow heads."""
from __future__ import annotations

import inspect

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from pointworld_interaction.adapter import (
    AdaptedDynamicsPredictor, integer_labels, masked_pool,
)

VARIANTS = ("baseline", "cira_no_feedback", "cira")


class RelationalFeedback(nn.Module):
    """Merge encoded per-arm edges before updating each scene receiver.

    Geometry is in world metres; motion means frame displacement, not velocity.
    Both variants retain the same recurrent feature state. Only the geometric
    receiver state/motion feedback is removed in the control.
    """

    def __init__(self, channels, width=64, neighbors=4, scene_chunk=128):
        super().__init__()
        if min(channels, width, neighbors, scene_chunk) < 1:
            raise ValueError("Positive relation dimensions required")
        self.channels, self.width = channels, width
        self.neighbors, self.scene_chunk = neighbors, scene_chunk
        self.initial = nn.Linear(channels, width)
        self.edge = nn.Sequential(
            nn.Linear(2 * channels + 11, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU())
        self.gate = nn.Linear(width, 1)
        self.receiver = nn.GRUCell(channels + 2 * width, width)
        self.output = nn.Linear(width, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _chunk(self, scene_feat, robot_feat, scene_xyz, scene_motion,
               robot_xyz, robot_motion, scene_exists, robot_exists, sources,
               roles, hidden, time):
        b, n, _ = scene_feat.shape
        banks, masks = [], []
        batch = torch.arange(b, device=scene_feat.device)[:, None, None]
        for arm in range(2):
            valid_robot = robot_exists & (sources == arm)
            # Discrete neighborhood selection; gradients use gathered geometry.
            with torch.no_grad():
                distances = torch.cdist(scene_xyz.float(), robot_xyz.float()).masked_fill(
                    ~valid_robot[:, None], torch.inf)
                values, indices = distances.topk(
                    min(self.neighbors, robot_xyz.shape[1]), dim=-1, largest=False)
                valid = torch.isfinite(values) & scene_exists[..., None]
            relative = robot_xyz[batch, indices] - scene_xyz[:, :, None]
            relative_motion = robot_motion[batch, indices] - scene_motion[:, :, None]
            features = torch.cat((
                scene_feat[:, :, None].expand(-1, -1, indices.shape[-1], -1),
                robot_feat[batch, indices], relative,
                torch.linalg.vector_norm(relative, dim=-1, keepdim=True),
                relative_motion,
                F.one_hot(roles[batch, indices].clamp(0, 2).long(), 3).to(scene_feat.dtype),
                scene_feat.new_full((*indices.shape, 1), time),
            ), -1).masked_fill(~valid[..., None], 0)
            message = self.edge(features)
            banks.append(message * self.gate(message).sigmoid())
            masks.append(valid)
        pooled, active = masked_pool(torch.cat(banks, 2), torch.cat(masks, 2))
        updated = self.receiver(
            torch.cat((scene_feat, pooled), -1).reshape(b * n, -1),
            hidden.reshape(b * n, -1)).reshape(b, n, self.width)
        updated = torch.where(active[..., None], updated, hidden)
        delta = self.output(updated).masked_fill(~active[..., None], 0)
        return delta, updated

    def step(self, scene_feat, robot_feat, scene_xyz, scene_motion, robot_xyz,
             robot_motion, scene_exists, robot_exists, sources, roles, hidden, time):
        deltas, states = [], []
        for first in range(0, scene_feat.shape[1], self.scene_chunk):
            sl = slice(first, first + self.scene_chunk)
            inputs = (scene_feat[:, sl], robot_feat, scene_xyz[:, sl], scene_motion[:, sl],
                      robot_xyz, robot_motion, scene_exists[:, sl], robot_exists,
                      sources, roles, hidden[:, sl], time)
            if self.training and torch.is_grad_enabled():
                delta, state = checkpoint(self._chunk, *inputs, use_reentrant=False)
            else:
                delta, state = self._chunk(*inputs)
            deltas.append(delta)
            states.append(state)
        return torch.cat(deltas, 1), torch.cat(states, 1)


class CIRADynamicsPredictor(AdaptedDynamicsPredictor):
    def __init__(self, base, adapter, feedback=True):
        super().__init__(base, adapter)
        self.feedback = feedback

    def forward(self, *args, **kwargs):
        if not self.enabled:
            return self.base(*args, **kwargs)
        if self._active:
            raise RuntimeError("Concurrent/reentrant CIRA calls are unsupported")
        bound = inspect.signature(self.base.forward).bind(*args, **kwargs)
        bound.apply_defaults()
        data = bound.arguments
        scene_mask, robot_mask = data["scene_exists0"], data["robot_exists"]
        xyz0, robot_xyz = data["scene_coord0"], data["robot_coord_seq"]
        b, t, nr = robot_mask.shape
        if t != self.base.T or t < 2 or nr < 1 or xyz0.shape[1] < 1:
            raise ValueError("CIRA requires nonempty points and the original horizon")
        sources = integer_labels(data.get("robot_source_ids"), (b, t, nr), "robot_source_ids")
        roles = integer_labels(self.roles, (b, t, nr), "robot_role_ids")
        if not ((sources[robot_mask] >= 0) & (sources[robot_mask] < 2)).all():
            raise ValueError("Source labels must be 0/1")
        if not ((roles[robot_mask] >= 0) & (roles[robot_mask] < 3)).all():
            raise ValueError("Role labels must be palm/leftfinger/rightfinger")
        paired = robot_mask[:, 1:] & robot_mask[:, :-1]
        if ((sources[:, 1:] != sources[:, :-1]) & paired).any() or (
                (roles[:, 1:] != roles[:, :-1]) & paired).any():
            raise ValueError("Robot point identities must correspond across frames")
        for value, mask in ((xyz0, scene_mask), (robot_xyz, robot_mask)):
            if not torch.isfinite(value[mask]).all():
                raise ValueError("Nonfinite valid coordinates")
        unnormalize = data.get("unnormalize_fn")
        if unnormalize is None:
            raise ValueError("Full per-time output unnormalization required for feedback")
        captured, handles = {}, []

        def capture_backbone(_module, _inputs, point):
            exists = torch.cat((scene_mask, robot_mask.flatten(1)), 1)
            padded = point.feat.new_zeros((*exists.shape, point.feat.shape[-1]))
            padded[exists] = point.feat
            captured["robot"] = padded[:, scene_mask.shape[1]:].reshape(b, t, nr, -1)

        def capture_head(_module, inputs):
            captured["fused"] = inputs[0]

        self._active = True
        try:
            handles.append(self.base.predictor_model.register_forward_hook(capture_backbone))
            handles.append(self.base.dynamics_head.register_forward_pre_hook(capture_head))
            original = self.base(*args, **kwargs)
            for handle in handles:
                handle.remove()
            handles.clear()
            with torch.autocast(device_type=xyz0.device.type, enabled=False):
                return self._rollout(original, captured, data, sources, roles, unnormalize)
        finally:
            for handle in handles:
                handle.remove()
            self._active = False

    def _rollout(self, original, captured, data, sources, roles, unnormalize):
        mask, robot_mask = data["scene_exists0"], data["robot_exists"]
        xyz0 = data["scene_coord0"].float().masked_fill(~mask[..., None], 0)
        robot_xyz = data["robot_coord_seq"].float().masked_fill(~robot_mask[..., None], 0)
        fused = captured["fused"].float().masked_fill(~mask[..., None], 0)
        robot_feat = captured["robot"].float().masked_fill(~robot_mask[..., None], 0)
        hidden = self.adapter.initial(fused).tanh().masked_fill(~mask[..., None], 0)
        current, previous = xyz0, xyz0
        predictions, variances = [original["pred"][:, 0]], [original["log_var"][:, 0]]
        b, ns, _ = xyz0.shape
        for step in range(self.base.T - 1):
            pair = robot_mask[:, step] & robot_mask[:, step + 1]
            robot_motion = (robot_xyz[:, step + 1] - robot_xyz[:, step]).masked_fill(
                ~pair[..., None], 0)
            delta, hidden = self.adapter.step(
                fused, robot_feat[:, step], current if self.feedback else xyz0,
                current - previous if self.feedback else torch.zeros_like(xyz0),
                robot_xyz[:, step], robot_motion, mask, robot_mask[:, step],
                sources[:, step], roles[:, step], hidden, step / (self.base.T - 1))
            # The heads retain absolute-to-initial readouts, NOT step increments.
            features = captured["fused"].float() + delta
            prediction = self.base.dynamics_head(features).view(
                b, ns, self.base.T - 1, 3)[:, :, step]
            variance = self.base.log_var_head(features).view(
                b, ns, self.base.T - 1, self.base.u_dim)[:, :, step]
            predictions.append(prediction)
            variances.append(variance)
            if self.feedback and step < self.base.T - 2:
                # The normalization API requires the full time axis. No future
                # object observation is used to fill the unread time slots.
                partial = torch.stack(predictions + [torch.zeros_like(prediction)]
                                      * (self.base.T - len(predictions)), 1)
                displacement = unnormalize(partial)[:, step + 1]
                if not torch.isfinite(displacement[mask]).all():
                    raise ValueError("Nonfinite predicted receiver state")
                previous, current = current, (xyz0 + displacement).masked_fill(~mask[..., None], 0)
        return {**original, "pred": torch.stack(predictions, 1),
                "log_var": torch.stack(variances, 1)}


def install_cira(model, variant, **options):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown CIRA variant: {variant}")
    base = model.dynamics_predictor
    if isinstance(base, AdaptedDynamicsPredictor) or hasattr(model, "robot_source_emb"):
        raise ValueError("Do not mix CIRA with an existing adapter/source embedding")
    if hasattr(base, "source_effect_mlp"):
        raise ValueError("Legacy summary must be disabled in every group")
    if variant == "baseline":
        return None
    parameter = next(base.parameters())
    adapter = RelationalFeedback(base.in_c, **options).to(
        device=parameter.device, dtype=parameter.dtype)
    model.dynamics_predictor = CIRADynamicsPredictor(
        base, adapter, feedback=variant == "cira")
    return model.dynamics_predictor
