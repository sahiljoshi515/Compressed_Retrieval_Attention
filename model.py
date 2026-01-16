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
    
from kernels.sparse import build_sparse_list_decode, sparse_attention_fwd

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 16384
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 40000
    norm_eps: float = 1e-5
    L: int = 60       
    R: int = 1024      
    K: int = 10
    heavy_const: int = 64   # budget

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
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
}

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim,
                 L, R, dtype=torch.bfloat16):
        super().__init__()
        B, T, H, D = max_batch_size, max_seq_length, n_heads, head_dim

        # Store as [B, T, H, D] for easier CSR lookup by token position
        self.register_buffer("k_cache", torch.zeros((B, T, H, D), dtype=dtype))
        self.register_buffer("v_cache", torch.zeros((B, T, H, D), dtype=dtype))

        self.L = L
        self.R = R
        self.register_buffer("k_hard", torch.zeros((max_batch_size, max_seq_length, n_heads, L), dtype=torch.int16))
        self.register_buffer("v_norm", torch.zeros((max_batch_size, max_seq_length, n_heads), dtype=torch.float16))
        self.register_buffer('attn_out', torch.zeros((max_batch_size, n_heads, head_dim), dtype=dtype))
        self.register_buffer("prefill_len", torch.zeros((), dtype=torch.int32))


    def append_decode_pos(self, b: int, h: int, l: int, r: int, t: int):
        w = int(self.dec_counts[b, h, l, r].item())
        self.dec_indices[b, h, l, r, w] = t
        self.dec_counts[b, h, l, r] = w + 1

    def update(self, input_pos, k_val, v_val, v_norm=None, k_hard=None):
        """
        input_pos: [S]
        k_val: [B, S, H, D]
        v_val: [B, S, H, D]
        """
        assert input_pos.shape[0] == k_val.shape[1]
        self.k_cache[:, input_pos] = k_val
        self.v_cache[:, input_pos] = v_val

        if v_norm is not None:
            self.v_norm[:, input_pos] = v_norm
        if k_hard is not None:
            self.k_hard[:, input_pos] = k_hard

        # track length (assumes monotonic positions)
        # works for both prefill (S>1) and decode (S=1)
        max_pos = int(input_pos.max().item()) + 1
        self.prefill_len.fill_(max(self.prefill_len.item(), max_pos))

        return self.k_cache, self.v_cache

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
                max_batch_size,
                max_seq_length,
                self.config.n_local_heads,
                head_dim,
                L=self.config.L,
                R=self.config.R,
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
        h = x + self.attention.sparse_forward(self.attention_norm(x), freqs_cis, mask, None, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

        # --- NEW: hash params ---
        self.config = config
        self.L = config.L
        self.R = config.R
        self.K = config.K
        self.heavy_const = config.heavy_const
        

        # planes: [L,K,D] and protos_T: [K,R]
        self.register_buffer("planes", torch.randn(self.L, self.K, self.head_dim) * 0.02)
        self.register_buffer("protos_T", torch.randn(self.K, self.R) * 0.02)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])
    
    def soft_hash(self, queries_bhd: Tensor) -> Tensor:
        """
        queries_bhd: [B,H,D]
        returns probs: [B,H,L,R]
        """
        queries = queries_bhd.unsqueeze(2)  # [B,H,1,D]
        q_proj = torch.einsum("bhqd,lkd->bhqlk", queries, self.planes)  # [B,H,1,L,K]
        temp = math.sqrt(queries.size(-1))
        logits = torch.einsum(
            "bhqlk,kr->bhqlr",
            torch.tanh(q_proj) / max(temp, 1e-6),
            self.protos_T,
        )  # [B,H,1,L,R]
        return torch.softmax(logits, dim=-1).squeeze(2)  # [B,H,L,R]


    def pack_bits(self, bits: Tensor) -> Tensor:
        """
        bits: [..., K] bool
        returns: [...] int16 (big-endian)
        """
        K = bits.shape[-1]
        weights = (1 << torch.arange(K - 1, -1, -1, device=bits.device, dtype=torch.int16))
        # broadcast weights to bits shape
        view_shape = (1,) * (bits.ndim - 1) + (K,)
        return (bits.to(torch.int16) * weights.view(view_shape)).sum(dim=-1)

    @torch.no_grad()
    def hard_hash_keys(self, keys_bthd: Tensor) -> Tensor:
        """
        keys_bthd: [B, T, H_local, D]
        returns bucket ids: [B, T, H_local, L] int16 in [0, R-1]
        """
        # project: [B, T, H, L, K]
        proj = torch.einsum("bthd,lkd->bthlk", keys_bthd, self.planes)  # (L,K,D)
        bits = proj >= 0
        codes = self.pack_bits(bits)  # [B,T,H,L]
        return codes.to(torch.int16)

    # def sparse_forward(
    #     self,
    #     x: torch.Tensor,
    #     freqs_cis: torch.Tensor,
    #     mask1: torch.Tensor,      # [1,1,S,Tmax] bool
    #     mask2: torch.Tensor,      # unused
    #     input_pos: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     assert input_pos is not None, "sparse_forward expects input_pos"
    #     bsz, seqlen, _ = x.shape
    #     assert self.kv_cache is not None, "Call setup_caches() first so kv_cache exists"

    #     kv_size = self.n_local_heads * self.head_dim
    #     q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

    #     q = q.view(bsz, seqlen, self.n_head, self.head_dim)             # [B,S,H,D]
    #     k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)      # [B,S,Hl,D]
    #     v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)      # [B,S,Hl,D]

    #     q = apply_rotary_emb(q, freqs_cis)
    #     k = apply_rotary_emb(k, freqs_cis)

    #     # # Cache updates: write k/v plus k_hard + v_norm for these positions
    #     # with torch.no_grad():
    #     #     k_hard = self.hard_hash_keys(k)  # [B,S,Hl,L] int16
    #     #     v_norm = torch.linalg.vector_norm(v.float(), ord=2, dim=-1).to(torch.float16)  # [B,S,Hl] fp16

    #     # k_cache, v_cache = self.kv_cache.update(input_pos, k, v, v_norm=v_norm, k_hard=k_hard)  # [B,Tmax,Hl,D]

    #     # # Expand local heads -> global heads for dense prefill path only
    #     # assert self.n_head % self.n_local_heads == 0
    #     # rep = self.n_head // self.n_local_heads

    #     # SDPA layout tensors for prefill
    #     q_sdpa = q.transpose(1, 2)  # [B,H,S,D]

    #     if rep == 1:
    #         k_full = k_cache
    #         v_full = v_cache
    #     else:
    #         k_full = k_cache.repeat_interleave(rep, dim=2)
    #         v_full = v_cache.repeat_interleave(rep, dim=2)

    #     k_sdpa = k_full.transpose(1, 2)  # [B,H,Tmax,D]
    #     v_sdpa = v_full.transpose(1, 2)  # [B,H,Tmax,D]

    #     # Prefill/chunked decode: dense attention
    #     if seqlen != 1:
    #         y = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=mask1, dropout_p=0.0)
    #         y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
    #         return self.wo(y)

    #     # -------------------------
    #     # True decode (S == 1)
    #     # Backend-only timing path: create a dummy sparse_list.
    #     # -------------------------
    #     T = int(self.kv_cache.prefill_len.item())
    #     assert T > 0
    #     q_bhd = q_sdpa[:, :, 0, :].contiguous()  # [B,H,D]

    #     # Use the last K positions as the sparse set (cheap + deterministic).
    #     Ktotal = min(256, T)
    #     start = max(0, T - Ktotal)
    #     idx = torch.arange(start, T, device=x.device, dtype=torch.int32)
    #     sparse_list = idx.view(1, 1, -1).expand(bsz, self.n_head, -1).contiguous()  # [B,H,Ktotal]
    #     sparse_len = torch.full((bsz, self.n_head), Ktotal, device=x.device, dtype=torch.int32)

    #     # # -------------------------
    #     # # True decode (S == 1)
    #     # # -------------------------
    #     # T = int(self.kv_cache.prefill_len.item())
    #     # assert T > 0

    #     # # allowed mask for prefix: [B,H,T]
    #     # allowed_bht = mask1[..., :T].expand(bsz, self.n_head, 1, T).squeeze(2).contiguous()  # bool

    #     # # k_hard/v_norm for prefix:
    #     # # kv_cache.k_hard: [B,T,Hl,L] -> expand Hl->H -> [B,H,L,T]
    #     # k_hard_pref = self.kv_cache.k_hard[:, :T]  # [B,T,Hl,L]
    #     # if rep != 1:
    #     #     k_hard_pref = k_hard_pref.repeat_interleave(rep, dim=2)  # [B,T,H,L]
    #     # k_hard_bhlt = k_hard_pref.permute(0, 2, 3, 1).contiguous()   # [B,H,L,T]

    #     # v_norm_pref = self.kv_cache.v_norm[:, :T]  # [B,T,Hl]
    #     # if rep != 1:
    #     #     v_norm_pref = v_norm_pref.repeat_interleave(rep, dim=2)  # [B,T,H]
    #     # v_norm_bht = v_norm_pref.permute(0, 2, 1).contiguous()       # [B,H,T]

    #     # # Query for indexer/backend: [B,H,D]
    #     # q_bhd = q_sdpa[:, :, 0, :].contiguous()

    #     # # sink/window + heavy M
    #     # sink = int(getattr(self.config, "sink_size", 128))
    #     # window = int(getattr(self.config, "window_size", 128))
    #     # M = int(self.heavy_const)
    #     # sink = max(0, min(sink, T))
    #     # window = max(0, min(window, T))
    #     # M = max(0, min(M, T))
    #     # q_probs = self.soft_hash(q_bhd)  # [B,H,L,R] fp16
    #     # # -------------------------
    #     # # Stage 0: indexer -> sparse_list / sparse_len
    #     # # -------------------------
    #     # sparse_list, sparse_len = build_sparse_list_decode(
    #     #     q_probs,
    #     #     k_hard_bhlt,
    #     #     v_norm_bht,
    #     #     allowed_bht,
    #     #     sink=sink,
    #     #     window=window,
    #     #     M=M,
    #     #     KC=8,
    #     #     BLOCK_N=512,
    #     #     num_warps=8,   # try 8 first for L=60
    #     #     num_stages=2,
    #     # )

    #     # # Backend expects int32 indices
    #     # if sparse_list.dtype != torch.int32:
    #     #     sparse_list = sparse_list.to(torch.int32)
    #     # if sparse_len.dtype != torch.int32:
    #     #     sparse_len = sparse_len.to(torch.int32)


    #     k_backend = k_cache[:, :T].permute(0, 2, 1, 3).contiguous()  # [B,Hl,T,D]
    #     v_backend = v_cache[:, :T].permute(0, 2, 1, 3).contiguous()  # [B,Hl,T,D]
        
    #     out_bhd = sparse_attention_fwd(
    #         q_bhd,            # [B,H,D]
    #         k_backend,        # [B,Hl,T,D]
    #         v_backend,        # [B,Hl,T,D]
    #         sparse_list,      # [B,H,Ktotal]
    #         sparse_len,       # [B,H]
    #         block_seq=256,
    #     )  # [B,H,D]

    #     # Project output
    #     y = out_bhd.unsqueeze(2)  # [B,H,1,D]
    #     y = y.transpose(1, 2).contiguous().view(bsz, 1, self.dim)
    #     return self.wo(y)

    def sparse_forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask1: torch.Tensor,      # [1,1,S,Tmax] bool (for dense prefill)
        mask2: torch.Tensor,      # unused
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert input_pos is not None, "sparse_forward expects input_pos"
        bsz, seqlen, _ = x.shape
        assert self.kv_cache is not None, "Call setup_caches() first so kv_cache exists"

        # ---- QKV ----
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)             # [B,S,H,D]
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)      # [B,S,Hl,D]
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)      # [B,S,Hl,D]

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # ---- KV cache update (NO hashing) ----
        # Assumes kv_cache.update can be called without v_norm/k_hard.
        # If your update() requires those args, make them optional in kv_cache.update.
        k_cache, v_cache = self.kv_cache.update(input_pos, k, v)  # [B,Tmax,Hl,D]

        # ---- Dense prefill / chunked decode ----
        if seqlen != 1:
            # SDPA expects [B,H,S,D] and [B,H,T,D]
            assert self.n_head % self.n_local_heads == 0
            rep = self.n_head // self.n_local_heads

            q_sdpa = q.transpose(1, 2)  # [B,H,S,D]

            # Expand local-head cache to global heads for dense path only
            if rep == 1:
                k_full = k_cache
                v_full = v_cache
            else:
                k_full = k_cache.repeat_interleave(rep, dim=2)
                v_full = v_cache.repeat_interleave(rep, dim=2)

            k_sdpa = k_full.transpose(1, 2)  # [B,H,Tmax,D]
            v_sdpa = v_full.transpose(1, 2)  # [B,H,Tmax,D]

            y = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=mask1,
                dropout_p=0.0,
            )  # [B,H,S,D]

            y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
            return self.wo(y)

        # ---- True decode (S == 1): sparse backend with dummy last-128 indices ----
        T = int(self.kv_cache.prefill_len.item())
        assert T > 0

        # Query: [B,H,D]
        q_bhd = q.transpose(1, 2)[:, :, 0, :].contiguous()

        # Backend expects K/V as local-head layout [B,Hl,T,D]
        k_backend = k_cache[:, :T].permute(0, 2, 1, 3).contiguous()  # [B,Hl,T,D]
        v_backend = v_cache[:, :T].permute(0, 2, 1, 3).contiguous()  # [B,Hl,T,D]

        # Dummy sparse list: last 128 tokens
        Ktotal = min(128, T)
        start = T - Ktotal

        # Optional: reuse buffers to avoid per-token allocations
        if getattr(self, "_dummy_sparse_buf", None) is None or self._dummy_sparse_buf.shape[-1] < Ktotal:
            cap = max(Ktotal, 128)
            self._dummy_idx_buf = torch.empty((cap,), device=x.device, dtype=torch.int32)
            self._dummy_sparse_buf = torch.empty((1, self.n_head, cap), device=x.device, dtype=torch.int32)
            self._dummy_len_buf = torch.empty((1, self.n_head), device=x.device, dtype=torch.int32)

        idx = self._dummy_idx_buf[:Ktotal]
        idx.copy_(torch.arange(start, T, device=x.device, dtype=torch.int32))

        sparse_list = self._dummy_sparse_buf[:, :, :Ktotal]
        sparse_list.copy_(idx.view(1, 1, Ktotal).expand(1, self.n_head, Ktotal))

        sparse_len = self._dummy_len_buf
        sparse_len.fill_(Ktotal)

        out_bhd = sparse_attention_fwd(
            q_bhd, k_backend, v_backend,
            sparse_list, sparse_len,
            block_seq=256,
        )  # [B,H,D]

        y = out_bhd.unsqueeze(2).transpose(1, 2).contiguous().view(bsz, 1, self.dim)
        return self.wo(y)


    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        print("DENSE")
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
