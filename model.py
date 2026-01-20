import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
import os, sys
# Keep this repo root on sys.path so local kernels import cleanly.
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    
from kernels.sparse import decode_sparse_attention
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
    L: int = 40       
    R: int = 1024      
    K: int = 10
    sparse_block_size: int = 8
    heavy_const: int = 2400   # budget
    max_batch_size: int = 1
    cache_dtype: torch.dtype = torch.bfloat16
    sink_size: int = 30
    window_size: int = 30

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
        # fuzzy search
        config = [config for config in transformer_configs if config in str(name).upper() or config in str(name)]
        assert len(config) == 1, name
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=1, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=1, n_head=32, dim=4096),
    "13B": dict(n_layer=1, n_head=40, dim=5120),
    "30B": dict(n_layer=1, n_head=52, dim=6656),
    "34B": dict(n_layer=1, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=1, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
}


# ============================================================
# Utilities (minimal placeholders where your project already has these)
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

def block_reduce_max_last(x: torch.Tensor, sparse_block_size: int) -> torch.Tensor:
    """
    x: [..., T]
    returns: [..., Tb] max within each block
    """
    T = x.shape[-1]
    Tb = (T + sparse_block_size - 1) // sparse_block_size
    pad = Tb * sparse_block_size - T
    if pad > 0:
        x = F.pad(x, (0, pad), value=float("-inf"))
    x_view = x.view(*x.shape[:-1], Tb, sparse_block_size)
    return x_view.max(dim=-1).values

def gqa_repeat_kv(x_bthd: torch.Tensor, rep: int) -> torch.Tensor:
    """
    Repeat-interleave heads dim for GQA:
      input:  [B,T,Hl,D]
      output: [B,T,H,D] where H = Hl*rep
    """
    if rep == 1:
        return x_bthd
    return x_bthd.repeat_interleave(rep, dim=2)

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
      - top Mb blocks by score (excluding duplicates is optional; we keep it simple)

    Pads with -1.
    """
    device = scores_bhtb.device
    B, H, Tb = scores_bhtb.shape

    sink_tokens = max(0, min(sink_tokens, T))
    window_tokens = max(0, min(window_tokens, T))

    sink_b = (sink_tokens + bs - 1) // bs
    window_b = (window_tokens + bs - 1) // bs

    sink_b = min(sink_b, Tb)
    window_b = min(window_b, Tb)

    # Current block index (last token is T-1)
    cur_block = (T - 1) // bs
    win_start = max(0, cur_block - window_b + 1)
    win_end = cur_block + 1  # exclusive

    sink_ids = torch.arange(0, sink_b, device=device, dtype=torch.int32)              # [sink_b]
    window_ids = torch.arange(win_start, win_end, device=device, dtype=torch.int32)  # [<=window_b]

    # Mask out mandatory blocks from top-k selection (optional but avoids wasted work)
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
# KVCache: token payload + block routing metadata + incremental update
# ============================================================

class KVCache(nn.Module):
    """
    Stores:
      - token-level K/V: needed for correct attention
      - token-level v_norm: for routing feature
      - block-level k_hard_blk and v_norm_blk: routing metadata

    Incremental update supports:
      - prefill (S>1)
      - decode (S==1)
    """
    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_local_heads: int,
        head_dim: int,
        L: int,
        R: int,
        sparse_block_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        B, T, Hl, D = max_batch_size, max_seq_length, n_local_heads, head_dim
        Tb = (T + sparse_block_size - 1) // sparse_block_size

        self.max_batch_size = B
        self.max_seq_length = T
        self.n_local_heads = Hl
        self.head_dim = D
        self.L = L
        self.R = R
        self.sparse_block_size = sparse_block_size
        self.Tb = Tb

        # Token payload
        self.register_buffer("k_cache", torch.zeros((B, T, Hl, D), dtype=dtype))
        self.register_buffer("v_cache", torch.zeros((B, T, Hl, D), dtype=dtype))

        # Token routing feature
        self.register_buffer("v_norm_tok", torch.zeros((B, T, Hl), dtype=torch.float16))

        # Incremental block accumulators (routing-only)
        # Use fp32 sum for stability; you can switch to fp16 for speed later.
        self.register_buffer("k_sum_blk", torch.zeros((B, Tb, Hl, D), dtype=torch.float32))
        self.register_buffer("k_cnt_blk", torch.zeros((B, Tb), dtype=torch.int32))
        self.register_buffer("v_norm_blk", torch.zeros((B, Tb, Hl), dtype=torch.float16))

        # Block hashes (per local head, per table)
        # layout: [B, Tb, Hl, L] int16
        self.register_buffer("k_hard_blk", torch.zeros((B, Tb, Hl, L), dtype=torch.int16))

        # Track current filled length
        self.register_buffer("prefill_len", torch.zeros((), dtype=torch.int32))

    @torch.no_grad()
    def update_tokens(
        self,
        input_pos_s: torch.Tensor,     # [S] int64/int32
        k_bshd: torch.Tensor,          # [B,S,Hl,D]
        v_bshd: torch.Tensor,          # [B,S,Hl,D]
        v_norm_bsh: torch.Tensor,      # [B,S,Hl] fp16
    ) -> None:
        """
        Writes token payload and updates block accumulators.
        """
        assert input_pos_s.ndim == 1
        B, S, Hl, D = k_bshd.shape
        assert Hl == self.n_local_heads and D == self.head_dim
        assert v_bshd.shape == (B, S, Hl, D)
        assert v_norm_bsh.shape == (B, S, Hl)

        # Write token caches
        self.k_cache[:, input_pos_s] = k_bshd
        self.v_cache[:, input_pos_s] = v_bshd
        self.v_norm_tok[:, input_pos_s] = v_norm_bsh

        # Update length (monotonic positions assumed)
        max_pos = int(input_pos_s.max().item()) + 1
        self.prefill_len.fill_(max(int(self.prefill_len.item()), max_pos))

        # Update block accumulators per position
        bs = self.sparse_block_size
        for si in range(S):
            t = int(input_pos_s[si].item())
            blk = t // bs
            # accumulate sums/counts
            self.k_sum_blk[:, blk] += k_bshd[:, si].to(torch.float32)
            self.k_cnt_blk[:, blk] += 1
            # vnorm max within block
            self.v_norm_blk[:, blk] = torch.maximum(self.v_norm_blk[:, blk], v_norm_bsh[:, si])

    @torch.no_grad()
    def finalize_blocks(
        self,
        hard_hash_fn,  # callable(keys_bthd)->codes_bthl int16 in [0,R-1]
        upto_T: Optional[int] = None,
    ) -> None:
        """
        (Re)compute k_hard_blk for blocks whose counts > 0, up to time T.
        This is cheap enough for prefill; for decode you can call this only for the current block.
        """
        T = int(self.prefill_len.item()) if upto_T is None else int(upto_T)
        bs = self.sparse_block_size
        Tb = (T + bs - 1) // bs
        Tb = max(0, min(Tb, self.Tb))

        # Compute representative key = sum / count for each block
        cnt = self.k_cnt_blk[:, :Tb].clamp_min(1).to(torch.float32)  # [B,Tb]
        k_rep = self.k_sum_blk[:, :Tb] / cnt[:, :, None, None]       # [B,Tb,Hl,D]

        # Hash reps: hard_hash expects [B,T,H,D]
        codes = hard_hash_fn(k_rep)  # [B,Tb,Hl,L] int16
        self.k_hard_blk[:, :Tb] = codes


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

        # Projections (GQA style: q has n_head, kv has n_local_heads)
        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)

        # Hash params
        self.L = config.L
        self.R = config.R
        self.K = config.K

        # Buffers for hashing (routing)
        self.register_buffer("planes", torch.randn(self.L, self.K, self.head_dim) * 0.02)
        self.register_buffer("protos_T", torch.randn(self.K, self.R) * 0.02)

        self.kv_cache: Optional[KVCache] = None

    # -------------------------
    # Cache setup
    # -------------------------
    def setup_caches(self, device: torch.device):
        self.kv_cache = KVCache(
            max_batch_size=self.config.max_batch_size,
            max_seq_length=self.config.max_seq_length,
            n_local_heads=self.n_local_heads,
            head_dim=self.head_dim,
            L=self.L,
            R=self.R,
            sparse_block_size=self.config.sparse_block_size,
            dtype=self.config.cache_dtype,
        ).to(device=device)

    # -------------------------
    # Hashing helpers
    # -------------------------
    def pack_bits(self, bits: torch.Tensor) -> torch.Tensor:
        """
        bits: [..., K] bool
        returns: [...] int16 (big-endian)
        """
        K = bits.shape[-1]
        weights = (1 << torch.arange(K - 1, -1, -1, device=bits.device, dtype=torch.int16))
        view_shape = (1,) * (bits.ndim - 1) + (K,)
        return (bits.to(torch.int16) * weights.view(view_shape)).sum(dim=-1)

    @torch.no_grad()
    def hard_hash_anyheads(self, keys_bthd: torch.Tensor) -> torch.Tensor:
        """
        keys_bthd: [B,T,H,D] (H can be local or global)
        returns codes: [B,T,H,L] int16 in [0,R-1] (by mod if needed)
        """
        proj = torch.einsum("bthd,lkd->bthlk", keys_bthd.float(), self.planes.float())  # [B,T,H,L,K]
        bits = proj >= 0
        codes = self.pack_bits(bits).to(torch.int32)  # [B,T,H,L]

        # Ensure within [0,R-1] if R != 2^K
        if self.R != (1 << self.K):
            codes = (codes % self.R)

        return codes.to(torch.int16)

    def soft_hash(self, q_bhd: torch.Tensor) -> torch.Tensor:
        """
        q_bhd: [B,H,D]
        returns scores: [B,H,L,R] in fp16/fp8
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

    # -------------------------
    # Forward (prefill dense, decode sparse)
    # -------------------------
    def sparse_forward(
        self,
        x: torch.Tensor,                 # [B,S,dim]
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor,         # [1,1,S,Tmax] bool-ish for SDPA; decode uses prefix slice
        input_pos: Optional[torch.Tensor] = None,  # [S]
    ) -> torch.Tensor:
        assert input_pos is not None, "Need input_pos for cache update"
        assert self.kv_cache is not None, "Call setup_caches(device) first"

        B, S, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(B, S, self.n_head, self.head_dim)              # [B,S,H,D]
        k = k.view(B, S, self.n_local_heads, self.head_dim)       # [B,S,Hl,D]
        v = v.view(B, S, self.n_local_heads, self.head_dim)       # [B,S,Hl,D]

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Token v_norm routing feature
        with torch.no_grad():
            v_norm = torch.linalg.vector_norm(v.float(), ord=2, dim=-1).to(torch.float16)  # [B,S,Hl]

        # Update caches (token payload + block accumulators)
        self.kv_cache.update_tokens(input_pos, k, v, v_norm)

        # If prefill (S>1): do standard dense attention for correctness
        # Also finalize all block hashes up to current T for later decode routing.
        if S != 1:
            with torch.no_grad():
                self.kv_cache.finalize_blocks(self.hard_hash_anyheads)

            # Build full K/V for dense SDPA (expand local->global if needed)
            assert self.n_head % self.n_local_heads == 0
            rep = self.n_head // self.n_local_heads

            T = int(self.kv_cache.prefill_len.item())
            k_tok = self.kv_cache.k_cache[:, :T]   # [B,T,Hl,D]
            v_tok = self.kv_cache.v_cache[:, :T]   # [B,T,Hl,D]
            k_full = gqa_repeat_kv(k_tok, rep)     # [B,T,H,D]
            v_full = gqa_repeat_kv(v_tok, rep)     # [B,T,H,D]

            # SDPA expects [B,H,S,D] and [B,H,T,D]
            q_sdpa = q.transpose(1, 2)             # [B,H,S,D]
            k_sdpa = k_full.transpose(1, 2)        # [B,H,T,D]
            v_sdpa = v_full.transpose(1, 2)        # [B,H,T,D]

            y = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=attn_mask[..., :T], dropout_p=0.0
            )
            y = y.transpose(1, 2).contiguous().view(B, S, self.dim)
            return self.wo(y)

        # -------------------------
        # Decode (S==1): routing + sparse Triton decode
        # -------------------------
        T = int(self.kv_cache.prefill_len.item())
        assert T > 0

        bs = self.config.sparse_block_size
        Tb = (T + bs - 1) // bs

        # Make sure block hashes exist for the current (possibly partial) block
        # For decode, just finalize up to T (cheap enough); later you can optimize to only current block.
        with torch.no_grad():
            self.kv_cache.finalize_blocks(self.hard_hash_anyheads, upto_T=T)

        # Prepare per-head tensors (expand Hl -> H for routing + kernel, for simplicity)
        assert self.n_head % self.n_local_heads == 0
        rep = self.n_head // self.n_local_heads

        # Q: [B,H,1,D]
        q_bh1d = q.transpose(1, 2)  # [B,H,1,D]

        # Prefix K/V token payload for kernel: [B,H,T,D]
        k_tok = self.kv_cache.k_cache[:, :T]         # [B,T,Hl,D]
        v_tok = self.kv_cache.v_cache[:, :T]         # [B,T,Hl,D]
        k_full = gqa_repeat_kv(k_tok, rep).permute(0, 2, 1, 3).contiguous()  # [B,H,T,D]
        v_full = gqa_repeat_kv(v_tok, rep).permute(0, 2, 1, 3).contiguous()  # [B,H,T,D]

        # Routing metadata:
        # block v_norm max: [B,Tb,Hl] -> [B,H,Tb]
        vnb = self.kv_cache.v_norm_blk[:, :Tb]  # [B,Tb,Hl]
        vnb = gqa_repeat_kv(vnb, rep)           # [B,Tb,H]
        v_norm_blk_bhtb = vnb.permute(0, 2, 1).contiguous()  # [B,H,Tb]

        # block hard hashes: [B,Tb,Hl,L] -> [B,H,L,Tb]
        khb = self.kv_cache.k_hard_blk[:, :Tb]  # [B,Tb,Hl,L]
        khb = gqa_repeat_kv(khb, rep)           # [B,Tb,H,L]
        k_hard_blk_bhltb = khb.permute(0, 2, 3, 1).contiguous()  # [B,H,L,Tb]

        # Allowed mask at block granularity (from attn_mask prefix)
        # attn_mask is [1,1,S,Tmax]; for decode S==1, we slice prefix [1,1,1,T]
        # Expand to [B,H,T] then reduce blocks -> [B,H,Tb]
        allowed_bht = attn_mask[..., :T].expand(B, self.n_head, 1, T).squeeze(2).contiguous().to(torch.bool)  # [B,H,T]
        allowed_blk_bhtb = block_reduce_any_last(allowed_bht, sparse_block_size=bs)                                  # [B,H,Tb]

        # Soft query probs: [B,H,L,R]
        q_bhd = q_bh1d[:, :, 0, :].contiguous()
        q_probs = self.soft_hash(q_bhd)

        # Block scores: [B,H,Tb] float32
        scores = soft_hash_score_ext.soft_hash_score(
            q_probs.to(torch.float16).contiguous(),
            k_hard_blk_bhltb.to(torch.int16).contiguous(),
            v_norm_blk_bhtb.to(torch.float16).contiguous(),
            allowed_blk_bhtb.contiguous(),
        )

        # Routing budget in blocks:
        Mb = max(1, min(self.config.heavy_const // bs, Tb))

        # Build block index list [B,H,KLIST]
        block_index = build_block_index(
            scores_bhtb=scores,
            T=T,
            bs=bs,
            sink_tokens=self.config.sink_size,
            window_tokens=self.config.window_size,
            Mb=Mb,
        )

        # seqlens (B,) int32: current prefix lengths
        seqlens_b = torch.full((B,), T, device=x.device, dtype=torch.int32)

        # Run Triton sparse decode
        out_bh1d = decode_sparse_attention(
            q_bh1d=q_bh1d.to(self.config.cache_dtype),
            k_bhtd=k_full.to(self.config.cache_dtype),
            v_bhtd=v_full.to(self.config.cache_dtype),
            seqlens_b=seqlens_b,
            block_index=block_index,
            block_n=self.config.sparse_block_size,
        )  # [B,H,1,D]

        # Project back
        y = out_bh1d.transpose(1, 2).contiguous().view(B, 1, self.dim)
        return self.wo(y)

    
    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)             # [B,S,H,D]
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)      # [B,S,Hl,D]
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)      # [B,S,Hl,D]

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # ---- IMPORTANT: update cache BEFORE transpose ----
        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache.update(input_pos, k, v)    # [B,T,Hl,D]
            k = k_cache
            v = v_cache

        # now transpose for attention math
        q = q.transpose(1, 2)                                           # [B,H,S,D]
        k = k.transpose(1, 2)                                           # [B,Hl,T,D]
        v = v.transpose(1, 2)                                           # [B,Hl,T,D]

        # expand local heads to global heads
        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)  # [B,H,T,D]
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)  # [B,H,T,D]

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.wo(y)


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
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
                n_local_heads=self.config.n_local_heads,
                head_dim=head_dim,
                L=self.config.L,
                R=self.config.R,
                sparse_block_size=self.config.sparse_block_size,
                dtype=self.config.cache_dtype,
            )

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits
    
    def sparse_forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]        # keep same mask type/shape
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


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000
) -> Tensor:
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
