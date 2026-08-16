"""
SAM-style mask head that produces a (T_v, H_v, W_v) binary-mask logit
volume from frozen Qwen3-VL visual tokens.

Forward contract:
    visual_tokens : [B, N_v, D_vlm]    raw last-hidden of Qwen3-VL at
                                       <|video_pad|> positions (D_vlm = 4096
                                       for Qwen3-VL-8B).
    T_v, H_v, W_v : ints, with T_v * H_v * W_v == N_v
    -> mask_logits: [B, T_v, H_v, W_v]

Design notes:
  * The head learns its OWN mask query (no dependency on mllm.video_queries),
    so loading a stage-2 mllm checkpoint stays bit-identical and the head
    can be plugged in / out without affecting any inference path.
  * Two-way transformer blocks (à la SAM): query attends to vision and vice
    versa, then a final cross-attn refines the mask query.
  * Mask is produced by dot-product between the mask token and the
    (vision-projected) tokens, kept at patch-grid resolution.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_xavier(m: nn.Module) -> None:
    for p in m.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention. Q/K/V can come from different tensors."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must divide num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, Lq, _ = q.shape
        Lk = k.shape[1]
        q = self.q_proj(q).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale     # [B, h, Lq, Lk]
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)                                   # [B, h, Lq, d]
        out = out.transpose(1, 2).reshape(B, Lq, self.dim)
        return self.o_proj(out)


class FFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoWayBlock(nn.Module):
    """One two-way attention block (SAM-style):
        q ← q + self-attn(q)
        q ← q + cross-attn(q -> v)
        v ← v + cross-attn(v -> q)
        q, v ← FFN
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)

        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_v_kv1 = nn.LayerNorm(dim)
        self.q_to_v = MultiHeadAttention(dim, num_heads, dropout)

        self.norm_v1 = nn.LayerNorm(dim)
        self.norm_q_kv1 = nn.LayerNorm(dim)
        self.v_to_q = MultiHeadAttention(dim, num_heads, dropout)

        self.norm_q3 = nn.LayerNorm(dim)
        self.q_ffn = FFN(dim, mult=4, dropout=dropout)

        self.norm_v2 = nn.LayerNorm(dim)
        self.v_ffn = FFN(dim, mult=4, dropout=dropout)

    def forward(self, q: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. self-attn on q
        q = q + self.self_attn(self.norm_q1(q), self.norm_q1(q), self.norm_q1(q))
        # 2. q attends to v
        q = q + self.q_to_v(self.norm_q2(q), self.norm_v_kv1(v), self.norm_v_kv1(v))
        # 3. v attends to q
        v = v + self.v_to_q(self.norm_v1(v), self.norm_q_kv1(q), self.norm_q_kv1(q))
        # 4. FFNs
        q = q + self.q_ffn(self.norm_q3(q))
        v = v + self.v_ffn(self.norm_v2(v))
        return q, v


class MaskHead(nn.Module):
    """
    Inputs
    ------
    visual_tokens : [B, N_v, D_vlm]
    grid          : (T_v, H_v, W_v) with T_v*H_v*W_v == N_v
    ctx_features  : [B, L_ctx, D_ctx]   (optional, only used when use_prompt_ctx=True)
                    Connector-output instruction queries (post-mllm).
                    Required if use_prompt_ctx=True.

    Output
    ------
    mask_logits : [B, T_v, H_v, W_v]    (raw logits, apply sigmoid for mask)

    Architecture
    ------------
    - `use_prompt_ctx=False` (legacy): only visual_tokens enter; mask token
      attends to vision through TwoWayBlock + final cross-attn.
    - `use_prompt_ctx=True`: ctx_features get projected into the same
      `hidden` dim, given a separate type embedding (so the model can tell
      visual vs prompt apart from each other in attention), and CONCAT'd
      onto the visual token sequence as additional context for q ↔ v
      attention. The final per-token mask logits are read out from only
      the visual prefix (length N_v); prompt tokens condition attention
      but don't get masks of their own.
    """

    def __init__(
        self,
        vlm_dim: int = 4096,
        hidden: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        num_query: int = 4,
        mask_token_index: int = 0,
        dropout: float = 0.0,
        use_grid_pe: bool = True,
        max_t_v: int = 64,
        max_h_v: int = 64,
        max_w_v: int = 64,
        use_prompt_ctx: bool = False,
        ctx_dim: int = 5120,
    ) -> None:
        super().__init__()
        self.vlm_dim = vlm_dim
        self.hidden = hidden
        self.num_query = num_query
        self.mask_token_index = mask_token_index
        self.use_grid_pe = use_grid_pe
        self.use_prompt_ctx = use_prompt_ctx
        self.ctx_dim = ctx_dim

        # vision-side projection from VLM hidden into the head's working dim
        self.proj_v = nn.Linear(vlm_dim, hidden)
        self.norm_v_in = nn.LayerNorm(hidden)

        # learnable queries (no dependency on mllm)
        self.query_embed = nn.Parameter(torch.zeros(num_query, hidden))
        nn.init.normal_(self.query_embed, std=0.02)

        # optional 3-axis additive positional encoding for the visual grid
        if use_grid_pe:
            self.pe_t = nn.Parameter(torch.zeros(max_t_v, hidden))
            self.pe_h = nn.Parameter(torch.zeros(max_h_v, hidden))
            self.pe_w = nn.Parameter(torch.zeros(max_w_v, hidden))
            nn.init.normal_(self.pe_t, std=0.02)
            nn.init.normal_(self.pe_h, std=0.02)
            nn.init.normal_(self.pe_w, std=0.02)
            self.max_t_v, self.max_h_v, self.max_w_v = max_t_v, max_h_v, max_w_v

        # Prompt-context (B-plan) sub-modules. Only constructed when
        # use_prompt_ctx=True; otherwise unused -> ckpt key set unchanged
        # for legacy ckpts.
        if use_prompt_ctx:
            self.proj_c = nn.Linear(ctx_dim, hidden)
            self.norm_c_in = nn.LayerNorm(hidden)
            # Type embeddings to disambiguate visual vs prompt segments
            # (so attention can learn segment-aware behavior).
            self.type_emb_visual = nn.Parameter(torch.zeros(hidden))
            self.type_emb_prompt = nn.Parameter(torch.zeros(hidden))
            nn.init.normal_(self.type_emb_visual, std=0.02)
            nn.init.normal_(self.type_emb_prompt, std=0.02)

        self.blocks = nn.ModuleList(
            [TwoWayBlock(hidden, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )

        # final cross-attn from query to vision (a la SAM mask decoder)
        self.norm_q_final = nn.LayerNorm(hidden)
        self.norm_v_final = nn.LayerNorm(hidden)
        self.final_q_to_v = MultiHeadAttention(hidden, num_heads, dropout)

        # produce per-query mask-token embedding
        self.mask_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def _add_grid_pe(self, v: torch.Tensor, T_v: int, H_v: int, W_v: int) -> torch.Tensor:
        """Add a separable 3-D positional encoding to a vision sequence.
        v : [B, N_v, D] in (T, H, W) row-major order.
        """
        if not self.use_grid_pe:
            return v
        if T_v > self.max_t_v or H_v > self.max_h_v or W_v > self.max_w_v:
            raise ValueError(
                f"Grid {T_v}x{H_v}x{W_v} exceeds MaskHead PE capacity "
                f"({self.max_t_v}x{self.max_h_v}x{self.max_w_v}). "
                "Increase max_t_v / max_h_v / max_w_v at construction."
            )
        pe = (
            self.pe_t[:T_v, None, None, :]
            + self.pe_h[None, :H_v, None, :]
            + self.pe_w[None, None, :W_v, :]
        )                                              # [T_v, H_v, W_v, D]
        pe = pe.reshape(1, T_v * H_v * W_v, -1)
        return v + pe

    def forward(
        self,
        visual_tokens: torch.Tensor,
        T_v: int,
        H_v: int,
        W_v: int,
        ctx_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N_v, D = visual_tokens.shape
        assert N_v == T_v * H_v * W_v, f"N_v={N_v}, T_v*H_v*W_v={T_v*H_v*W_v}"
        assert D == self.vlm_dim, f"visual dim {D} != configured vlm_dim {self.vlm_dim}"

        v = self.proj_v(visual_tokens)                 # [B, N_v, hidden]
        v = self.norm_v_in(v)
        v = self._add_grid_pe(v, T_v, H_v, W_v)

        # Prompt-context branch: project ctx and concat to the visual sequence.
        if self.use_prompt_ctx:
            if ctx_features is None:
                raise ValueError(
                    "MaskHead was built with use_prompt_ctx=True but "
                    "ctx_features is None. Pass the connector's instruction "
                    "queries (post-mllm) of shape [B, L_ctx, ctx_dim]."
                )
            assert ctx_features.shape[0] == B, \
                f"ctx_features batch {ctx_features.shape[0]} != visual_tokens batch {B}"
            assert ctx_features.shape[-1] == self.ctx_dim, \
                f"ctx_features dim {ctx_features.shape[-1]} != configured ctx_dim {self.ctx_dim}"
            c = self.proj_c(ctx_features)              # [B, L_ctx, hidden]
            c = self.norm_c_in(c)
            v = v + self.type_emb_visual               # mark visual segment
            c = c + self.type_emb_prompt               # mark prompt segment
            v = torch.cat([v, c], dim=1)               # [B, N_v + L_ctx, hidden]

        q = self.query_embed.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, K, hidden]

        for blk in self.blocks:
            q, v = blk(q, v)

        # final cross-attn: refine query embedding with vision
        q = q + self.final_q_to_v(self.norm_q_final(q), self.norm_v_final(v), self.norm_v_final(v))

        # take the mask token from the query bank
        mask_token = self.mask_mlp(q[:, self.mask_token_index])           # [B, hidden]

        # dot product with vision tokens → per-token logits.
        # When use_prompt_ctx=True, v is [B, N_v + L_ctx, hidden]; we only
        # want logits over the N_v visual prefix (prompt segment is context).
        v_visual = v[:, :N_v, :] if self.use_prompt_ctx else v
        logits = torch.einsum("bd,bnd->bn", mask_token, v_visual)         # [B, N_v]
        return logits.view(B, T_v, H_v, W_v)


# -----------------------------------------------------------------------------
# Quick shape self-test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_dim", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_query", type=int, default=4)
    parser.add_argument("--T_v", type=int, default=5)
    parser.add_argument("--H_v", type=int, default=14)
    parser.add_argument("--W_v", type=int, default=26)
    parser.add_argument("--bs", type=int, default=1)
    args = parser.parse_args()

    head = MaskHead(
        vlm_dim=args.vlm_dim,
        hidden=args.hidden,
        num_layers=args.num_layers,
        num_query=args.num_query,
    )
    n_param = sum(p.numel() for p in head.parameters())
    print(f"MaskHead param count = {n_param/1e6:.2f}M")

    x = torch.randn(args.bs, args.T_v * args.H_v * args.W_v, args.vlm_dim)
    out = head(x, args.T_v, args.H_v, args.W_v)
    print(f"input  shape = {tuple(x.shape)}")
    print(f"output shape = {tuple(out.shape)}")
    assert out.shape == (args.bs, args.T_v, args.H_v, args.W_v)
    # quick autograd check
    out.sum().backward()
    print("backward OK")
