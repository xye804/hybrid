import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
import torch.nn.functional as F
from einops import repeat, pack, unpack, rearrange
from torch.cuda.amp import autocast
from functools import partial
from models.mf import (
    TimestepEmbedder,
    RMSNorm,
    CausalAttention,
    ExpertBlock,
    FinalLayer,
    get_2d_sincos_pos_embed,
    FusionModule,
)


class HybridSign(nn.Module):
    def __init__(
        self,
        channels=2,
        dim=512,
        hidden_dim=256,
        depth=4,
        num_heads=4,
    ):
        super().__init__()
        self.x_embedder = nn.Linear(channels, dim)
        self.t_embedder = TimestepEmbedder(dim)
        self.r_embedder = TimestepEmbedder(dim)
        self.y_proj = nn.Linear(768, dim)
        self.f_expert = ExpertBlock(dim, hidden_dim, num_heads)
        self.b_expert = ExpertBlock(dim, hidden_dim, num_heads)
        self.hs_expert = ExpertBlock(dim, hidden_dim, num_heads)
        self.fusion = FusionModule(dim, hidden_dim)
        self.final_layer = FinalLayer(dim, channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.kernel_size ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, r, c, y):
        x = self.x_embedder(x)

        T, J, D = x.shape[1:]
        pos_embed = get_2d_sincos_pos_embed(T, J, D)

        x = x + pos_embed.unsqueeze(0).to("cuda")

        t = self.t_embedder(t)
        r = self.r_embedder(r)
        t = t + r

        y = self.y_proj(y)

        b, bc = x[:, :, :25, :], c[:, :, :25, :]
        b = self.b_expert(b, t, y, bc)
        f, fc = x[:, :, 25:95, :], c[:, :, 25:95, :]
        f = self.f_expert(f, t, y, fc)
        hs, hsc = x[:, :, 95:137, :], c[:, :, 95:137, :]
        hs = self.hs_expert(hs, t, y, hsc)

        x = self.fusion(b, f, hs)

        x = self.final_layer(x, y)

        return x


def build_model(cfg):
    return HybridSign(
        channels=cfg["channels"],
        dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
    )
