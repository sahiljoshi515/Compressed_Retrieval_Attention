import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
import os, sys
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    flex_attention,
)

# Keep this repo root on sys.path so local kernels import cleanly.
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kernels.sparse import rw_decode_sparse_attention
from kernels.soft_hash_score import soft_hash_score_ext


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    block_size: int = 276480
    max_seq_length: int = 276480
    vocab_size: int = 32000
    n_layer: int = 1
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 40000
    norm_eps: float = 1e-5

    # hashing / routing
    L: int = 40
    R: int = 1024
    K: int = 10
    sparse_block_size: int = 8
    heavy_const: int = 1160   # token budget
    max_batch_size: int = 1
    cache_dtype: torch.dtype = torch.bfloat16
    sink_size: int = 60
    window_size: int = 60

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        config = [c for c in transformer_configs if c in str(name).upper() or c in str(name)]
        assert len(config) == 1, name
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=1, dim=4096, rope_base=1000000),
    "7B": dict(n_layer=1, n_head=32, dim=4096),
    "13B": dict(n_layer=1, n_head=40, dim=5120),
    "30B": dict(n_layer=1, n_head=52, dim=6656),
    "34B": dict(n_layer=1, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000),
    "70B": dict(n_layer=1, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
}


# ============================================================
# Utilities
# ============================================================

def block_reduce_any_last(x_bool: torch.Tensor, sparse_block_size: int) -> torch.Tensor:
    """
    x_bool: [..., T] bool
    returns: [..., Tb] bool where Tb=ceil(T/sparse_block_size), using any() within each block.
    """
    T = x_bool.shape[-1]
    Tb = (T + sparse_block_size - 1) // sparse_block_size
    pad = Tb * sparse_block_size - T
    if pad > 0:
        x_bool = F.pad(x_bool, (0, pad), value=False)
    x_view = x_bool.view(*x_bool.shape[:-1], Tb, sparse_block_size)
    return x_view.any(dim=-1)


def build_block_index(
    scores_bhtb: torch.Tensor,         # [B,H,Tb] float32
    T: int,
    bs: int,
    sink_tokens: int,
    window_tokens: int,
    Mb: int,
) -> torch.Tensor:
    """
    Build block list [B,H,KLIST] with:
      - sink blocks from start
      - window blocks at end
      - top Mb blocks by score

    Note: this returns a fully-filled list for typical inputs (no -1),
    but the decode kernel should still safely handle -1.
    """
    device = scores_bhtb.device
    B, H, Tb = scores_bhtb.shape

    sink_tokens = max(0, min(sink_tokens, T))
    window_tokens = max(0, min(window_tokens, T))

    sink_b = (sink_tokens + bs - 1) // bs
    window_b = (window_tokens + bs - 1) // bs

    sink_b = min(sink_b, Tb)
    window_b = min(window_b, Tb)

    cur_block = (T - 1) // bs
    win_start = max(0, cur_block - window_b + 1)
    win_end = cur_block + 1

    sink_ids = torch.arange(0, sink_b, device=device, dtype=torch.int32)              # [sink_b]
    window_ids = torch.arange(win_start, win_end, device=device, dtype=torch.int32)  # [<=window_b]

    # Avoid sink/window overlap (important when T small)
    if sink_b > 0 and window_ids.numel() > 0:
        window_ids = window_ids[window_ids >= sink_b]

    scores = scores_bhtb.clone()
    if sink_b > 0:
        scores[..., :sink_b] = float("-inf")
    if window_ids.numel() > 0:
        scores[..., window_ids.long()] = float("-inf")

    Mb = max(0, min(Mb, Tb))
    if Mb > 0:
        _, top_ids = torch.topk(scores, k=Mb, dim=-1)  # [B,H,Mb]
        top_ids = top_ids.to(torch.int32)
    else:
        top_ids = torch.empty((B, H, 0), device=device, dtype=torch.int32)

    KLIST = sink_ids.numel() + window_ids.numel() + top_ids.shape[-1]
    KLIST = max(1, KLIST)

    block_index = torch.full((B, H, KLIST), -1, device=device, dtype=torch.int32)

    pos = 0
    if sink_ids.numel() > 0:
        block_index[..., pos:pos + sink_ids.numel()] = sink_ids.view(1, 1, -1)
        pos += sink_ids.numel()
    if window_ids.numel() > 0:
        block_index[..., pos:pos + window_ids.numel()] = window_ids.view(1, 1, -1)
        pos += window_ids.numel()
    if top_ids.numel() > 0:
        block_index[..., pos:pos + top_ids.shape[-1]] = top_ids

    return block_index


# ============================================================
# KVCache: store KV as (B, H, T, D) to avoid decode-time permutes/copies
# ============================================================

class KVCache(nn.Module):
    """
    Stores:
      - token-level K/V as (B, H, T, D) for direct decode kernel consumption
      - block-level routing metadata in full-head layout:
          v_norm_blk:  (B, H, Tb)
          k_hard_blk:  (B, H, L, Tb)
      - block accumulators to build block representative keys:
          k_sum_blk:   (B, H, Tb, D)
          k_cnt_blk:   (B, Tb)
    """
    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_head: int,
        head_dim: int,
        L: int,
        R: int,
        sparse_block_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        B, T, H, D = max_batch_size, max_seq_length, n_head, head_dim
        Tb = (T + sparse_block_size - 1) // sparse_block_size

        self.max_batch_size = B
        self.max_seq_length = T
        self.n_head = H
        self.head_dim = D
        self.L = L
        self.R = R
        self.sparse_block_size = sparse_block_size
        self.Tb = Tb

        # Token payload (kernel-friendly)
        self.register_buffer("k_cache", torch.zeros((B, H, T, D), dtype=dtype))
        self.register_buffer("v_cache", torch.zeros((B, H, T, D), dtype=dtype))

        # Block routing metadata (full-head layout)
        self.register_buffer("v_norm_blk", torch.zeros((B, H, Tb), dtype=torch.float16))
        self.register_buffer("k_hard_blk", torch.zeros((B, H, L, Tb), dtype=torch.int16))

        # Block accumulators (for representative keys)
        self.register_buffer("k_sum_blk", torch.zeros((B, H, Tb, D), dtype=torch.float32))
        self.register_buffer("k_cnt_blk", torch.zeros((B, Tb), dtype=torch.int32))

        self.register_buffer("prefill_len", torch.zeros((), dtype=torch.int32))


    # def update(self, input_pos, k_val, v_val):
    #     """
    #     input_pos: [S]
    #     k_val: [B, S, H, D]
    #     v_val: [B, S, H, D]
    #     """
    #     assert input_pos.shape[0] == k_val.shape[1]
    #     self.k_cache[:, input_pos] = k_val
    #     self.v_cache[:, input_pos] = v_val

    #     # track length (assumes monotonic positions)
    #     # works for both prefill (S>1) and decode (S=1)
    #     max_pos = int(input_pos.max().item()) + 1
    #     self.prefill_len.fill_(max(self.prefill_len.item(), max_pos))

    #     return self.k_cache, self.v_cache
    @torch.no_grad()
    def update(
        self,
        input_pos: torch.Tensor,     # [S]
        k_bhsd: torch.Tensor,        # [B,H,S,D]
        v_bhsd: torch.Tensor,        # [B,H,S,D]
    ):
        assert input_pos.ndim == 1
        B, H, S, D = k_bhsd.shape
        assert v_bhsd.shape == (B, H, S, D)
        assert H == self.n_head and D == self.head_dim
        assert input_pos.numel() == S

        # write into time dimension
        self.k_cache[:, :, input_pos, :] = k_bhsd
        self.v_cache[:, :, input_pos, :] = v_bhsd

        max_pos = int(input_pos.max().item()) + 1
        self.prefill_len.fill_(max(int(self.prefill_len.item()), max_pos))

        T = int(self.prefill_len.item())
        return self.k_cache[:, :, :T, :], self.v_cache[:, :, :T, :]


    @torch.no_grad()
    def update_tokens(
        self,
        input_pos_s: torch.Tensor,       # [S]
        k_bshd: torch.Tensor,            # [B,S,H,D]  (FULL heads)
        v_bshd: torch.Tensor,            # [B,S,H,D]
        v_norm_bsh: torch.Tensor,        # [B,S,H] fp16
    ) -> None:
        assert input_pos_s.ndim == 1
        B, S, H, D = k_bshd.shape
        assert H == self.n_head and D == self.head_dim
        assert v_bshd.shape == (B, S, H, D)
        assert v_norm_bsh.shape == (B, S, H)

        # --- write token KV payloads into (B,H,T,D) caches ---
        # (B,S,H,D) -> (B,H,S,D)
        k_bhsd = k_bshd.permute(0, 2, 1, 3).contiguous()
        v_bhsd = v_bshd.permute(0, 2, 1, 3).contiguous()
        self.k_cache[:, :, input_pos_s, :] = k_bhsd
        self.v_cache[:, :, input_pos_s, :] = v_bhsd

        # --- update prefill_len ---
        max_pos = int(input_pos_s.max().item()) + 1
        self.prefill_len.fill_(max(int(self.prefill_len.item()), max_pos))

        bs = self.sparse_block_size

        # --- compute block ids for each token position ---
        # blk_ids: [S] int64 for indexing ops
        blk_ids = (input_pos_s.to(torch.int64) // bs)  # [S]

        # ------------------------------------------------------------
        # 1) k_sum_blk update: (B,H,Tb,D) += sum over tokens in each blk
        # ------------------------------------------------------------
        # src for index_add along dim=2 must be (B,H,S,D)
        src_sum = k_bhsd.to(torch.float32)  # (B,H,S,D)
        self.k_sum_blk.index_add_(dim=2, index=blk_ids, source=src_sum)

        # ------------------------------------------------------------
        # 2) k_cnt_blk update: (B,Tb) += counts per blk
        # ------------------------------------------------------------
        ones = torch.ones((B, S), device=self.k_cnt_blk.device, dtype=torch.int32)
        self.k_cnt_blk.index_add_(dim=1, index=blk_ids, source=ones)

        # ------------------------------------------------------------
        # 3) v_norm_blk update: (B,H,Tb) = max(current, max v_norm in blk)
        # ------------------------------------------------------------
        # Need per-(B,H,blk) max over tokens mapping to that blk.
        # Use scatter_reduce_ if available (PyTorch 2.0+ generally).
        # v_norm_bsh: (B,S,H) -> (B,H,S)
        vnorm_bhs = v_norm_bsh.permute(0, 2, 1).contiguous()  # (B,H,S)

        # Build an updates tensor shaped (B,H,Tb) containing max over tokens per blk.
        # Start with -inf so max-reduce works.
        Tb = self.Tb
        upd = torch.full((B, H, Tb), float("-inf"), device=vnorm_bhs.device, dtype=torch.float32)

        # Expand indices to match upd's shape along the scattered dimension
        # We scatter along dim=2 (Tb), so index shape must match source shape
        # source is (B,H,S), index should be (B,H,S)
        idx = blk_ids.view(1, 1, S).expand(B, H, S)

        # scatter_reduce to compute max into upd at block positions
        # (include_self=False => ignores the initial -inf values)
        if hasattr(upd, "scatter_reduce_"):
            upd.scatter_reduce_(dim=2, index=idx, src=vnorm_bhs.to(torch.float32), reduce="amax", include_self=False)
        else:
            # Fallback (still vectorized) for older PyTorch:
            # Use index_put_ accumulate=True on a max-friendly transform is not exact for max.
            # If you hit this path, upgrade torch; exact max segment-reduce without scatter_reduce is painful.
            raise RuntimeError("Your PyTorch lacks scatter_reduce_. Please upgrade for exact v_norm_blk max update.")

        # Now merge into cache v_norm_blk: take max with existing
        self.v_norm_blk[:, :, :Tb] = torch.maximum(self.v_norm_blk[:, :, :Tb], upd.to(self.v_norm_blk.dtype))


# ============================================================
# Model blocks
# ============================================================

class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        assert config.head_dim == config.dim // config.n_head

        self.config = config
        self.n_head = config.n_head
        self.n_local_heads = config.n_local_heads
        self.head_dim = config.head_dim
        self.dim = config.dim

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)

        # Hash params
        self.L = config.L
        self.R = config.R
        self.K = config.K

        # Hash buffers
        self.register_buffer("planes", torch.randn(self.L, self.K, self.head_dim) * 0.02)
        self.register_buffer("protos_T", torch.randn(self.K, self.R) * 0.02)

        self.kv_cache: Optional[KVCache] = None

    def pack_bits(self, bits: torch.Tensor) -> torch.Tensor:
        K = bits.shape[-1]
        weights = (1 << torch.arange(K - 1, -1, -1, device=bits.device, dtype=torch.int16))
        view_shape = (1,) * (bits.ndim - 1) + (K,)
        return (bits.to(torch.int16) * weights.view(view_shape)).sum(dim=-1)

    @torch.no_grad()
    def hard_hash_anyheads(self, keys_bthd: torch.Tensor) -> torch.Tensor:
        """
        keys_bthd: [B,T,H,D]
        returns:   [B,T,H,L] int16
        """
        proj = torch.einsum("bthd,lkd->bthlk", keys_bthd.float(), self.planes.float())  # [B,T,H,L,K]
        bits = proj >= 0
        codes = self.pack_bits(bits).to(torch.int32)  # [B,T,H,L]
        if self.R != (1 << self.K):
            codes = (codes % self.R)
        return codes.to(torch.int16)

    def soft_hash(self, q_bhd: torch.Tensor) -> torch.Tensor:
        """
        q_bhd: [B,H,D]
        returns: [B,H,L,R] fp16
        """
        q = q_bhd.unsqueeze(2)  # [B,H,1,D]
        q_proj = torch.einsum("bhqd,lkd->bhqlk", q.float(), self.planes.float())  # [B,H,1,L,K]
        temp = math.sqrt(q_bhd.size(-1))
        logits = torch.einsum(
            "bhqlk,kr->bhqlr",
            (torch.tanh(q_proj) / max(temp, 1e-6)).float(),
            self.protos_T.float(),
        )
        scores = F.softmax(logits, dim=-1).squeeze(2)  # [B,H,L,R]
        return scores.to(torch.float16)

    def sparse_forward(
        self,
        x: torch.Tensor,          # [B,S,dim]
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor,  # [1,1,S,Tmax] (causal in your usage)
        input_pos: Optional[torch.Tensor] = None,  # [S]
    ) -> torch.Tensor:
        assert input_pos is not None, "Need input_pos for cache update"
        assert self.kv_cache is not None, "Call setup_caches first"

        B, S, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(B, S, self.n_head, self.head_dim)              # [B,S,H,D]
        k = k.view(B, S, self.n_local_heads, self.head_dim)       # [B,S,Hl,D]
        v = v.view(B, S, self.n_local_heads, self.head_dim)       # [B,S,Hl,D]

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # v_norm (local heads)
        with torch.no_grad():
            v_norm = torch.linalg.vector_norm(v.float(), ord=2, dim=-1).to(torch.float16)  # [B,S,Hl]

        # Expand local KV -> full heads (so cache is (B,H,T,D))
        assert self.n_head % self.n_local_heads == 0
        rep = self.n_head // self.n_local_heads
        if rep == 1:
            k_full = k
            v_full = v
            v_norm_full = v_norm
        else:
            k_full = k.repeat_interleave(rep, dim=2)          # [B,S,H,D]
            v_full = v.repeat_interleave(rep, dim=2)          # [B,S,H,D]
            v_norm_full = v_norm.repeat_interleave(rep, dim=2)  # [B,S,H]

        # Update caches and block accumulators
        self.kv_cache.update_tokens(input_pos, k_full, v_full, v_norm_full)

        # ------------------------------------------------------------------
        # Incremental block hashing (NO LOOPS): update only blocks touched by input_pos
        # ------------------------------------------------------------------
        bs = self.config.sparse_block_size
        with torch.no_grad():
            # touched block ids
            blk_ids = (input_pos.to(torch.int64) // bs)            # [S]
            blk_ids = torch.unique(blk_ids)                        # [U]
            blk_ids = blk_ids.clamp_(0, self.kv_cache.Tb - 1)      # in range

            # Gather sums / counts for these blocks
            # k_sum_blk: (B,H,Tb,D) -> (B,H,U,D)
            k_sum = self.kv_cache.k_sum_blk.index_select(2, blk_ids)

            # k_cnt_blk: (B,Tb) -> (B,U)
            cnt = self.kv_cache.k_cnt_blk.index_select(1, blk_ids).clamp_min(1).to(torch.float32)

            # Representative keys for blocks: (B,H,U,D)
            k_rep = k_sum / cnt[:, None, :, None]

            # Hash expects [B,T,H,D], T=U. So permute -> (B,U,H,D)
            codes = self.hard_hash_anyheads(k_rep.permute(0, 2, 1, 3).contiguous())  # (B,U,H,L) int16

            # Store into k_hard_blk: (B,H,L,Tb) at blocks blk_ids along dim=3
            # codes: (B,U,H,L) -> (B,H,L,U)
            self.kv_cache.k_hard_blk.index_copy_(
                dim=3,
                index=blk_ids,
                source=codes.permute(0, 2, 3, 1).contiguous()
            )

        # ------------------------------------------------------------------
        # Prefill: dense SDPA for correctness
        # ------------------------------------------------------------------
        if S != 1:
            T = int(self.kv_cache.prefill_len.item())
            assert T > 0

            # SDPA expects [B,H,S,D] and [B,H,T,D]
            q_sdpa = q.transpose(1, 2).contiguous()                      # [B,H,S,D]
            k_sdpa = self.kv_cache.k_cache[:, :, :T, :].contiguous()     # [B,H,T,D]
            v_sdpa = self.kv_cache.v_cache[:, :, :T, :].contiguous()     # [B,H,T,D]

            y = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=attn_mask[..., :T],
                dropout_p=0.0
            )
            y = y.transpose(1, 2).contiguous().view(B, S, self.dim)
            return self.wo(y)

        # ------------------------------------------------------------------
        # Decode: routing + sparse decode
        # ------------------------------------------------------------------
        T = int(self.kv_cache.prefill_len.item())
        assert T > 0

        Tb = (T + bs - 1) // bs

        # Q: [B,H,1,D]
        q_bh1d = q.transpose(1, 2).contiguous()  # [B,H,1,D]

        # KV already in (B,H,T,D)
        k_bhtd = self.kv_cache.k_cache[:, :, :T, :]
        v_bhtd = self.kv_cache.v_cache[:, :, :T, :]

        # Routing metadata (full-head)
        v_norm_blk_bhtb = self.kv_cache.v_norm_blk[:, :, :Tb]       # (B,H,Tb)
        k_hard_blk_bhltb = self.kv_cache.k_hard_blk[:, :, :, :Tb]   # (B,H,L,Tb)

        # ------------------------------------------------------------------
        # Causal-only: remove mask -> block reduction entirely
        # kernel enforces cols < seqlen and cols <= q_pos
        # ------------------------------------------------------------------
        allowed_blk_bhtb = torch.ones((B, self.n_head, Tb), device=x.device, dtype=torch.bool)

        # Soft query probs
        q_bhd = q_bh1d[:, :, 0, :].contiguous()  # (B,H,D)
        q_probs = self.soft_hash(q_bhd)          # (B,H,L,R)

        # Block scores
        scores = soft_hash_score_ext.soft_hash_score(
            q_probs.to(torch.float16).contiguous(),
            k_hard_blk_bhltb.to(torch.int16).contiguous(),
            v_norm_blk_bhtb.to(torch.float16).contiguous(),
            allowed_blk_bhtb.contiguous(),
        )  # (B,H,Tb)

        Mb = max(1, min(self.config.heavy_const // bs, Tb))

        block_index = build_block_index(
            scores_bhtb=scores,
            T=T,
            bs=bs,
            sink_tokens=self.config.sink_size,
            window_tokens=self.config.window_size,
            Mb=Mb,
        )  # (B,H,KLIST) int32

        seqlens_b = torch.full((B,), T, device=x.device, dtype=torch.int32)

        out_bh1d = rw_decode_sparse_attention(
            q=q_bh1d.to(self.config.cache_dtype),
            k=k_bhtd.to(self.config.cache_dtype),
            v=v_bhtd.to(self.config.cache_dtype),
            seqlens=seqlens_b,
            block_index=block_index,
            block_n=bs,
        )  # (B,H,1,D)

        y = out_bh1d.transpose(1, 2).contiguous().view(B, 1, self.dim)
        return self.wo(y)


    def forward(self, x: Tensor, freqs_cis: Tensor, mask: BlockMask, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,          # or a proper bool mask if needed
            dropout_p=0.0,
            is_causal=(seqlen > 1),  # PyTorch 2.1+ supports is_causal
        )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size=max_batch_size,
                max_seq_length=max_seq_length,
                n_head=self.config.n_head,  # IMPORTANT: full heads
                head_dim=head_dim,
                L=self.config.L,
                R=self.config.R,
                sparse_block_size=self.config.sparse_block_size,
                dtype=self.config.cache_dtype,
            )

        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.dim // self.config.n_head,
            self.config.rope_base
        )
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        # mask = self.causal_mask[None, None, input_pos]
        # input_pos is [S] positions; K length is S for dense forward here
        mask = self.causal_mask[input_pos][:, input_pos]          # [S,S]
        mask = mask[None, None, :, :]                             # [1,1,S,S]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for layer in self.layers:
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    def sparse_forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer.sparse_forward(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        return self.output(x)

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def sparse_forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention.sparse_forward(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
