from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import flashinfer

from .utils import apply_rotary_pos_emb, layer_norm, topp_temperature_decode
from .attnserver import LSHSparseAttnServer, AttnServer


@dataclass
class GenerationStats:
    prefill_tokens: int
    generated_tokens: int
    decode_ms_per_token: float


class LLMLayer:
    """
    Holds raw tensors for a single Llama decoder layer.
    """
    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx

        # Attention weights
        self.wq: Optional[torch.Tensor] = None
        self.wk: Optional[torch.Tensor] = None
        self.wv: Optional[torch.Tensor] = None
        self.wo: Optional[torch.Tensor] = None

        # MLP weights
        self.gate_proj: Optional[torch.Tensor] = None
        self.up_proj: Optional[torch.Tensor] = None
        self.down_proj: Optional[torch.Tensor] = None

        # RMSNorm weights + eps
        self.input_layernorm_weight: Optional[torch.Tensor] = None
        self.input_layernorm_variance_epsilon: float = 0.0
        self.post_attention_layernorm_weight: Optional[torch.Tensor] = None
        self.post_attention_layernorm_variance_epsilon: float = 0.0

    def init_parameters(self, hf_layer: LlamaDecoderLayer) -> None:
        # Attention
        self.wq = hf_layer.self_attn.q_proj.weight.detach()
        self.wk = hf_layer.self_attn.k_proj.weight.detach()
        self.wv = hf_layer.self_attn.v_proj.weight.detach()
        self.wo = hf_layer.self_attn.o_proj.weight.detach()

        # MLP
        self.gate_proj = hf_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_layer.mlp.up_proj.weight.detach()
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()

        # Norms
        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach()
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon
        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach()
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon

    def to_device(self, device: str) -> None:
        assert self.wq is not None
        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        self.wq = self.wq.to(device, non_blocking=True)
        self.wk = self.wk.to(device, non_blocking=True)
        self.wv = self.wv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        self.gate_proj = self.gate_proj.to(device, non_blocking=True)
        self.up_proj = self.up_proj.to(device, non_blocking=True)
        self.down_proj = self.down_proj.to(device, non_blocking=True)


class LLM:
    """
    Custom Llama inference engine:
      - Extract HF weights into raw tensors
      - Dense FlashInfer prefill (prompt)
      - Decode via attention_server.decode(...) (dense or sparse depending on server)
    """

    def __init__(
        self,
        model_name: str,
        K: int = 10,
        L: int = 150,
        batch_size: int = 1,
        max_length: int = 8192,
        generation_buffer: int = 256,
        device: str = "cuda:0",
        dtype=torch.bfloat16,
        chunk_size: int = 8192,
        use_lsh_server: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_length = max_length
        self.chunk_size = chunk_size

        self.config = LlamaConfig.from_pretrained(model_name)

        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        self.eos_tokens = self.config.eos_token_id if isinstance(self.config.eos_token_id, list) else [self.config.eos_token_id]

        # Init weights + RoPE caches
        self._init_parameters()

        # Attention server
        if use_lsh_server and K > 0:
            self.attention_server = LSHSparseAttnServer(
                config=self.config,
                K=K,
                L=L,
                batch_size=batch_size,
                max_length=max_length,
                generation_buffer=generation_buffer,
                device=device,
                dtype=dtype,
            )
        else:
            self.attention_server = AttnServer(
                config=self.config,
                K=K,
                L=L,
                batch_size=batch_size,
                max_length=max_length,
                generation_buffer=generation_buffer,
                device=device,
                dtype=dtype,
            )

        # Global KV caches for dense prefill (FlashInfer prefill kernel uses these)
        self.k_cache = torch.zeros((max_length, self.num_key_value_heads, self.head_dim), dtype=dtype, device=device)
        self.v_cache = torch.zeros((max_length, self.num_key_value_heads, self.head_dim), dtype=dtype, device=device)

        # Stream for prefill overlap
        self.wrt_stream = torch.cuda.Stream()

        # Chunk metadata (set per request in prefill)
        self.num_chunk: int = 0
        self.chunk_start: List[int] = []
        self.chunk_end: List[int] = []

    def _init_parameters(self) -> None:
        hf_model = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)

        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)

        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon

        # RoPE caches
        self.inv_freq = hf_model.model.rotary_emb.inv_freq.detach().to(self.device)
        self.attention_scaling = hf_model.model.rotary_emb.attention_scaling

        position_ids = torch.arange(0, self.max_length, device=self.device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cache = emb.cos()[0] * self.attention_scaling
        sin_cache = emb.sin()[0] * self.attention_scaling
        self.cos_cache = cos_cache.to(self.dtype)
        self.sin_cache = sin_cache.to(self.dtype)

        # Extract layer tensors
        self.layers: List[LLMLayer] = []
        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = LLMLayer(idx)
            layer.init_parameters(hf_layer)
            layer.to_device(self.device)
            self.layers.append(layer)

            # Free the HF layer module to save memory
            hf_model.model.layers[idx] = None
            gc.collect()

        self.num_layers = len(self.layers)

    # ---------------------------
    # Core math helpers
    # ---------------------------

    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,  # [B, T, H]
        eps: float,
        ln_w: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
    ):
        """
        Returns:
          query_states: [B, num_heads, T, head_dim]
          key_states:   [B, num_kv_heads, T, head_dim]
          value_states: [B, num_kv_heads, T, head_dim]
        """
        hidden_states = layer_norm(hidden_states, eps, ln_w)
        bsz, q_len, _ = hidden_states.size()

        q = F.linear(hidden_states, wq)
        k = F.linear(hidden_states, wk)
        v = F.linear(hidden_states, wv)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def post_attention_compute(
        self,
        attn_out: torch.Tensor,       # [B, T, H]
        residual: torch.Tensor,       # [B, T, H]
        eps: float,
        ln_w: torch.Tensor,
        wo: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
    ) -> torch.Tensor:
        # output proj
        hidden = F.linear(attn_out, wo)
        hidden = residual + hidden

        # MLP
        residual = hidden
        hidden = layer_norm(hidden, eps, ln_w)
        up = F.linear(hidden, up_proj)
        gate = F.linear(hidden, gate_proj)
        gate = F.silu(gate)
        hidden = gate * up
        hidden = F.linear(hidden, down_proj)
        hidden = residual + hidden
        return hidden

    # ---------------------------
    # Decode step (1 token usually)
    # ---------------------------

    @torch.inference_mode()
    def layer_decode_step(
        self,
        layer: LLMLayer,
        layer_idx: int,
        hidden_states: torch.Tensor,   # [B, 1, H]
        position_ids: torch.Tensor,    # [B, 1] or [1] depending on your apply_rotary_pos_emb
    ) -> torch.Tensor:
        residual = hidden_states
        q, k, v = self.pre_attention_compute(
            hidden_states,
            layer.input_layernorm_variance_epsilon,
            layer.input_layernorm_weight,
            layer.wq, layer.wk, layer.wv,
        )

        # RoPE
        k = apply_rotary_pos_emb(k, self.cos_cache, self.sin_cache, position_ids)
        q = apply_rotary_pos_emb(q, self.cos_cache, self.sin_cache, position_ids)

        # ============================================================
        # INTEGRATE YOUR SPARSE ATTENTION METHOD HERE (decode)
        #
        # Today this calls your server, which can do dense/sparse per-layer:
        #   - dense_layers -> FlashInfer decode
        #   - sparse layers -> (paged GPU small-window) + (CPU retrieved) + merge
        #
        # If you want to replace LSH retrieval / sparse compute:
        #   - do it inside attention_server.decode(...)
        # ============================================================
        attn_out = self.attention_server.decode(q, k, v, layer_idx)

        hidden_states = self.post_attention_compute(
            attn_out,
            residual,
            layer.post_attention_layernorm_variance_epsilon,
            layer.post_attention_layernorm_weight,
            layer.wo,
            layer.gate_proj,
            layer.up_proj,
            layer.down_proj,
        )
        return hidden_states

    @torch.inference_mode()
    def inference_step(self, input_id: torch.LongTensor, position_id: torch.LongTensor) -> torch.Tensor:
        """
        Single-step inference:
          input_id:    [B, 1]
          position_id: [B, 1] (absolute position)
        Returns logits: [B, 1, vocab]
        """
        self.attention_server.plan()

        hidden = F.embedding(input_id, self.embed_tokens)  # [B, 1, H]
        for i in range(self.num_layers):
            hidden = self.layer_decode_step(self.layers[i], i, hidden, position_id)

        hidden = layer_norm(hidden, self.norm_variance_epsilon, self.norm_weight)
        logits = F.linear(hidden, self.lm_head).float()
        return logits

    # ---------------------------
    # Dense prefill (FlashInfer) over a prefix [0:seq_len)
    # ---------------------------

    @torch.inference_mode()
    def layer_dense_prefill(
        self,
        layer: LLMLayer,
        layer_idx: int,
        hidden_states: torch.Tensor,    # [B, T, H] (T = seq_len)
        position_ids_1d: torch.Tensor,  # [T] (int32 is fine)
        request_id: int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Dense prefill for first seq_len tokens using FlashInfer prefill kernel.

        IMPORTANT:
          This function is where "dense context only" is enforced via seq_len.
          Later, you can ingest question tokens via sparse decode (teacher forcing).
        """
        with torch.cuda.stream(self.wrt_stream):
            residual = hidden_states

            # build chunk boundaries for seq_len (not full prompt)
            num_chunk = ((seq_len // self.chunk_size) if (seq_len % self.chunk_size == 0) else (seq_len // self.chunk_size + 1))
            chunk_start = [i * self.chunk_size for i in range(num_chunk)]
            chunk_end = [min((i + 1) * self.chunk_size, seq_len) for i in range(num_chunk)]

            for start, end in zip(chunk_start, chunk_end):
                h = layer_norm(
                    hidden_states[:, start:end, :],
                    layer.input_layernorm_variance_epsilon,
                    layer.input_layernorm_weight,
                )
                bsz, q_len, _ = h.size()

                q = F.linear(h, layer.wq)
                k = F.linear(h, layer.wk)
                v = F.linear(h, layer.wv)

                # NOTE: prefill kernel below expects "NHD" layout (T, H, D)
                q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)      # [B,H,T,D]
                k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [B,kvH,T,D]
                v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                # You used a variant in your original code that flattens to [T,H,D].
                # Keep consistent with your apply_rotary_pos_emb utility.
                k = apply_rotary_pos_emb(k, self.cos_cache, self.sin_cache, position_ids_1d[start:end])
                q = apply_rotary_pos_emb(q, self.cos_cache, self.sin_cache, position_ids_1d[start:end])

                # Store KV for dense prefill
                # original code stores [T, kvH, D]
                # Here: k is [B, kvH, T, D] -> transpose and squeeze B
                self.k_cache[start:end].copy_(k[0].transpose(0, 1))
                self.v_cache[start:end].copy_(v[0].transpose(0, 1))

                # Dense causal attention over [0:end)
                h2 = flashinfer.prefill.single_prefill_with_kv_cache(
                    q=q[0].transpose(0, 1),           # [T, H, D]
                    k=self.k_cache[:end],             # [end, kvH, D]
                    v=self.v_cache[:end],
                    causal=True,
                    allow_fp16_qk_reduction=True,
                    kv_layout="NHD",
                )

                h2 = h2.reshape(bsz, q_len, self.hidden_size)
                h2 = F.linear(h2, layer.wo)
                residual[:, start:end, :].add_(h2)

        # Build sparse index for PREVIOUS layer while current computes (optional pipelining)
        if layer_idx >= 1:
            # ============================================================
            # INTEGRATE HERE:
            # Build sparse tables for layer_idx-1 from the DENSE CONTEXT KV.
            # Pass seq_len (context_len), not full prompt length.
            # ============================================================
            self.attention_server.build_table(layer_idx - 1, request_id, seq_len)

        self.wrt_stream.synchronize()

        # MLP chunkwise
        with torch.cuda.stream(self.wrt_stream):
            hidden_states = residual
            num_chunk = ((seq_len // self.chunk_size) if (seq_len % self.chunk_size == 0) else (seq_len // self.chunk_size + 1))
            chunk_start = [i * self.chunk_size for i in range(num_chunk)]
            chunk_end = [min((i + 1) * self.chunk_size, seq_len) for i in range(num_chunk)]

            for start, end in zip(chunk_start, chunk_end):
                h = layer_norm(
                    hidden_states[:, start:end, :],
                    layer.post_attention_layernorm_variance_epsilon,
                    layer.post_attention_layernorm_weight,
                )
                up = F.linear(h, layer.up_proj)
                gate = F.linear(h, layer.gate_proj)
                gate = F.silu(gate)
                h = gate * up
                h = F.linear(h, layer.down_proj)
                residual[:, start:end, :].add_(h)

        # ============================================================
        # INTEGRATE HERE:
        # Feed DENSE CONTEXT KV to server so sparse decode can query it.
        # Critically, seq_len here is "context_len".
        # ============================================================
        self.attention_server.fill(layer_idx, request_id, self.k_cache, self.v_cache, seq_len)

        if layer_idx == self.num_layers - 1:
            # build final layer index
            self.attention_server.build_table(layer_idx, request_id, seq_len)

        self.wrt_stream.synchronize()
        return residual

    # ---------------------------
    # Public APIs: prefill & generate
    # ---------------------------

    @torch.inference_mode()
    def prefill_dense_context(
        self,
        context_ids: torch.LongTensor,  # [B, Tctx]
        request_id: int = 0,
    ) -> torch.Tensor:
        """
        Dense prefill ONLY for context tokens. Returns logits for next token after context.
        """
        assert context_ids.dim() == 2 and context_ids.shape[0] == self.batch_size
        ctx_len = context_ids.shape[1]
        assert ctx_len <= self.max_length

        hidden = F.embedding(context_ids, self.embed_tokens)  # [B, Tctx, H]

        # Allocate buffers sized to ctx_len (server may allocate LSH sorting buffers etc.)
        self.attention_server.alloc_buffer(ctx_len)

        position_ids_1d = torch.arange(ctx_len, device=self.device, dtype=torch.int32)

        for i in range(self.num_layers):
            hidden = self.layer_dense_prefill(
                self.layers[i], i, hidden, position_ids_1d, request_id=request_id, seq_len=ctx_len
            )

        # logits from last token
        last = layer_norm(hidden[:, -1:, :], self.norm_variance_epsilon, self.norm_weight)
        logits = F.linear(last, self.lm_head).float()
        return logits

    @torch.inference_mode()
    def ingest_question_sparse(
        self,
        question_ids: torch.LongTensor,  # [B, Tq]
        start_pos: int,                  # absolute start position = context_len
    ) -> torch.Tensor:
        """
        Teacher-forced ingest of question tokens using the decode pipeline (sparse path).
        Returns logits after consuming the last question token.
        """
        assert question_ids.dim() == 2 and question_ids.shape[0] == self.batch_size
        Tq = question_ids.shape[1]

        logits = None
        for t in range(Tq):
            tok = question_ids[:, t:t+1]
            pos = torch.tensor([[start_pos + t]], device=self.device, dtype=torch.int64)
            logits = self.inference_step(tok, pos)  # uses attention_server.decode(...)
        assert logits is not None
        return logits

    @torch.inference_mode()
    def generate(
        self,
        context_ids: torch.LongTensor,        # [B, Tctx]
        question_ids: Optional[torch.LongTensor] = None,  # [B, Tq]
        max_tokens: int = 128,
        temperature: float = 0.6,
        topp: float = 0.9,
        verbose: bool = False,
    ):
        """
        Policy:
          - Dense prefill over context_ids
          - Sparse ingest over question_ids (optional)
          - Sparse decode for generated tokens

        Returns:
          generated token ids (python list) and optional stats.
        """
        generated: List[int] = []

        # 1) Dense context prefill
        ctx_len = context_ids.shape[1]
        logits = self.prefill_dense_context(context_ids=context_ids, request_id=0)

        # 2) Sparse ingest question (optional)
        q_len = 0
        if question_ids is not None and question_ids.numel() > 0:
            q_len = question_ids.shape[1]
            logits = self.ingest_question_sparse(question_ids=question_ids, start_pos=ctx_len)

        # 3) Generate with sparse decode
        if verbose:
            torch.cuda.synchronize()
            t1 = time.time()

        start_gen_pos = ctx_len + q_len
        for k in range(max_tokens):
            if temperature < 0.1:
                next_id = logits.argmax(dim=-1)
            else:
                next_id = topp_temperature_decode(logits, temperature, topp)

            pos = torch.tensor([[start_gen_pos + k]], device=self.device, dtype=torch.int64)
            logits = self.inference_step(next_id, pos)

            tok = int(next_id[0, 0].item())
            generated.append(tok)
            if tok in self.eos_tokens:
                break

        stats = None
        if verbose:
            torch.cuda.synchronize()
            t2 = time.time()
            ms_per_tok = 1000.0 * (t2 - t1) / max(1, len(generated))
            stats = GenerationStats(
                prefill_tokens=ctx_len,
                generated_tokens=len(generated),
                decode_ms_per_token=ms_per_tok,
            )
            print(f"\033[94m[INFO] Dense context prefill {ctx_len} tokens\033[0m")
            if q_len:
                print(f"\033[94m[INFO] Sparse ingest question {q_len} tokens\033[0m")
            print(f"\033[94m[INFO] Generate {len(generated)} tokens\033[0m")
            print(f"\033[94m[INFO] Decoding Latency {ms_per_tok:.2f} ms/token\033[0m")

        self.clear()
        return generated, stats

    def clear(self) -> None:
        self.attention_server.clear()
        self.k_cache.zero_()
        self.v_cache.zero_()
