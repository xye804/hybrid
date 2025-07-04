import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
import torch.nn.functional as F
from einops import repeat, pack, unpack, rearrange
from torch.cuda.amp import autocast
from functools import partial
from eval import dtw
from torch.func import jvp
from soft_dtw_cuda import SoftDTW
from loss import bone_length_loss, adaptive_l2_loss, DMDLoss
import logging


def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g


class CausalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.norm = RMSNorm(embed_dim)
        self.q_norm = RMSNorm(embed_dim)
        self.k_norm = RMSNorm(embed_dim)
        self.v_norm = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.out_norm = RMSNorm(embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, c, kv_cache=None):
        x = self.norm(x)
        B, T, J, E = x.shape
        assert T == 1, "This module expects T=1 for incremental caching"

        q = self.q_norm(self.q_proj(x))  # [B,1,J,E]
        k = self.k_norm(self.k_proj(x))
        v = self.v_norm(self.v_proj(x))

        q = q.view(B, T, J, self.num_heads, self.head_dim)  # [B,1,J,nh,hd]
        k = k.view(B, T, J, self.num_heads, self.head_dim)
        v = v.view(B, T, J, self.num_heads, self.head_dim)

        if kv_cache is None:
            kv_cache = {"k": k, "v": v}
            cached_T = 1
        else:
            kv_cache["k"] = torch.cat([kv_cache["k"], k], dim=1)
            kv_cache["v"] = torch.cat([kv_cache["v"], v], dim=1)
            cached_T = kv_cache["k"].shape[1]

        q = q.squeeze(1).permute(0, 1, 2, 3)  # [B,J,nh,hd]
        k = kv_cache["k"].permute(0, 2, 3, 1, 4)  # [B,J,nh,cached_T,hd]
        v = kv_cache["v"].permute(0, 2, 3, 1, 4)  # [B,J,nh,cached_T,hd]

        attn_logits = torch.einsum("bjnc,bjnmc->bjnm", q, k) / math.sqrt(
            self.head_dim
        )  # [B,J,nh,cached_T]

        mask = (
            torch.tril(torch.ones(cached_T, cached_T, device=x.device))
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # [1,1,1,cached_T,cached_T]

        causal_mask = torch.tril(
            torch.ones(cached_T, cached_T, device=x.device)
        )  # [cached_T,cached_T]
        causal_mask = (
            causal_mask[-1, :].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )  # [1,1,1,cached_T]

        attn_logits = attn_logits.masked_fill(causal_mask == 0, float("-inf"))

        c_ = c.squeeze(1).squeeze(-1)  # [B,J]
        c_expand = (
            c_.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.num_heads, cached_T)
        )  # [B,J,nh,cached_T]

        attn_logits = attn_logits * c_expand

        attn_weights = F.softmax(attn_logits, dim=-1)  # [B,J,nh,cached_T]

        # v: [B,J,nh,cached_T,hd]
        out = torch.einsum("bjnm,bjnmd->bjn d", attn_weights, v)  # [B,J,nh,hd]

        out = out.permute(0, 2, 1, 3).contiguous()  # [B,nh,J,hd]
        out = out.view(B, self.num_heads, J, self.head_dim)
        out = out.permute(0, 2, 1, 3).contiguous()  # [B,J,nh,hd]
        out = out.view(B, 1, J, self.embed_dim)  # [B,1,J,E]

        out = self.out_norm(self.out_proj(out))

        return out, kv_cache


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key_value):
        B, *extra_dims, D = query.shape
        query = query.view(B, -1, D)  # flatten extra dims
        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)

        T_q = Q.size(1)
        T_kv = K.size(1)

        Q = Q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        )
        output = self.out_proj(attn_output)

        # reshape back to original query shape (if needed)
        output = output.view(B, *extra_dims, self.embed_dim)

        return output, attn_weights


class ExpertBlock(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.causal_attn = CausalAttention(embed_dim=dim, num_heads=num_heads)
        self.norm2 = RMSNorm(dim)
        self.cross_attn = CrossAttention(embed_dim=dim, num_heads=num_heads)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim, hidden_features=hidden_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, t, y, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(t).chunk(6, dim=-1)
        )

        x = modulate(self.norm1(x), scale_msa, shift_msa)

        t = x.shape[1]
        x = x[:, 0:1, :, :]
        x_hats = []
        kv_cache = None
        for i in range(t):
            x, kv_cache = self.causal_attn(x, c[:, i : i + 1, :, :], kv_cache)
            x_hats.append(x)

        x = torch.cat(x_hats, dim=1)

        x = x + gate_msa.unsqueeze(1).unsqueeze(1) * x
        x, _ = self.cross_attn(x, y)

        x = x + gate_mlp.unsqueeze(1).unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), scale_mlp, shift_mlp)
        )

        return x


class FusionModule(nn.Module):

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)

    def forward(self, b, f, hs):
        x = torch.cat([b, f, hs], dim=-2)
        x = self.norm1(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm_final = RMSNorm(dim)
        self.linear = nn.Linear(dim, out_dim)
        # self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, y):
        # y = torch.mean(y, dim=1)
        # shift, scale = self.adaLN_modulation(y).chunk(2, dim=-1)
        # x = modulate(self.norm_final(x), shift, scale)
        x = self.norm_final(x)
        x = self.linear(x)
        return x


def get_2d_sincos_pos_embed(T, J, D):
    assert D % 2 == 0, "Embedding dimension must be even."

    d_half = D // 2

    pos_t = torch.arange(T).unsqueeze(1)  # [T, 1]
    dim_t = torch.arange(d_half).unsqueeze(0)  # [1, d_half]
    freq_t = 1 / (10000 ** (2 * (dim_t // 2) / d_half))
    emb_t = pos_t * freq_t  # [T, d_half]
    emb_t = torch.stack(
        [torch.sin(emb_t[:, 0::2]), torch.cos(emb_t[:, 1::2])], dim=-1
    ).flatten(
        1
    )  # [T, d_half]

    pos_j = torch.arange(J).unsqueeze(1)  # [J, 1]
    dim_j = torch.arange(d_half).unsqueeze(0)  # [1, d_half]
    freq_j = 1 / (10000 ** (2 * (dim_j // 2) / d_half))
    emb_j = pos_j * freq_j  # [J, d_half]
    emb_j = torch.stack(
        [torch.sin(emb_j[:, 0::2]), torch.cos(emb_j[:, 1::2])], dim=-1
    ).flatten(
        1
    )  # [J, d_half]

    emb_t = emb_t.unsqueeze(1).expand(-1, J, -1)  # [T, J, d_half]
    emb_j = emb_j.unsqueeze(0).expand(T, -1, -1)  # [T, J, d_half]
    emb = torch.cat([emb_t, emb_j], dim=-1)  # [T, J, D]

    return emb  # [T, J, D]


class Normalizer:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 1, 1, -1)
        self.std = torch.tensor(std).view(1, 1, 1, -1)

    def norm(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)


def stopgrad(x):
    return x.detach()


class MeanFlow:

    def __init__(
        self,
        normalizer,
        channels=3,
        flow_ratio=0.50,
        time_dist=[-0.4, 1.0],
    ):
        super().__init__()
        self.channels = channels
        self.normer = normalizer
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist

    def sample_t_r(self, batch_size, device):

        mu, sigma = self.time_dist[0], self.time_dist[1]
        normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
        samples = 1 / (1 + np.exp(-normal_samples))

        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def loss(self, model, x, y):

        c = x[:, :, :, 2:3]
        x = x[:, :, :, :2]
        B, T, J, C = x.shape

        device = x.device

        t, r = self.sample_t_r(B, device)

        t_ = rearrange(t, "b -> b 1 1 1")
        r_ = rearrange(r, "b -> b 1 1 1")

        x = self.normer.norm(x)

        e = torch.randn_like(x)

        z = (1 - t_) * x + t_ * e
        v = e - x

        model_partial = partial(model, c=c, y=y)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v, torch.ones_like(t), torch.zeros_like(r)),
        )

        u, dudt = jvp(*jvp_args)

        u_tgt = v - (t_ - r_) * dudt

        sdtw = SoftDTW(use_cuda=True, gamma=0.1)
        sdtw_loss = sdtw(x.view(B, T, -1), u.view(B, T, -1)).mean() / T
        dtw_score = dtw(x, u)

        bone_loss = bone_length_loss(x, u)

        joint_loss = F.mse_loss(x, u, reduction="mean")

        # error = u - stopgrad(u_tgt)
        # ada_loss = adaptive_l2_loss(error)

        loss = 0.2 * sdtw_loss + 1.0 * joint_loss + 0.1 * bone_loss

        if torch.isnan(u).any() or torch.isnan(loss).any():
            logging.info(
                f"loss: {loss}, sdtw_loss: {sdtw_loss}, mse_loss: {mse_loss}, bone_loss: {bone_loss} u: {u}"
            )

        return loss, dtw_score


def build_meanflow(cfg, normalizer):
    return MeanFlow(
        normalizer=normalizer,
        channels=cfg["channels"],
        flow_ratio=cfg["flow_ratio"],
        time_dist=[-0.4, 1.0],
    )
