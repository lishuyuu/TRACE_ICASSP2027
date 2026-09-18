from __future__ import annotations
import torch
from torch.nn import functional as F
from . import trace_networks as networks

class UnorderedRelationalPlanner(networks.RelationalPlanner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.b_head.requires_grad_(False)

    def _forward_roles(self, native, speaker, valid, counts, target_mask):
        batch, words, _ = native.shape
        role = target_mask.long()
        hidden = self.native_projection(native) + self.target_role(role) + self.speaker_projection(speaker)[:, None, :] + networks.sinusoidal_positions(words, self.width, base=self.position_base, reference=native)[None]
        hidden = hidden.masked_fill(~valid[:, :, None], 0)
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=~valid)
            hidden = hidden.masked_fill(~valid[:, :, None], 0)
        hidden = self.final_norm(hidden).masked_fill(~valid[:, :, None], 0)
        b = self.b_head(hidden).squeeze(-1).masked_fill(~valid, 0).detach()
        u = self.u_head(hidden).masked_fill(~valid[:, :, None], 0)
        plans = []
        for index, count in enumerate(counts.tolist()):
            row = u[index, :count]
            plan = row - row.mean(dim=0, keepdim=True)
            plans.append(F.pad(plan, (0, 0, 0, words - count)))
        coordinates = torch.stack(plans)
        return networks.PlannerOutput(coordinates, b, u, valid.detach().clone())
