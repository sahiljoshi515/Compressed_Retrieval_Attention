import math
import torch
import triton
import triton.language as tl

@torch.no_grad()
def build_sparse_list_topk(scores, sink, window, M):
    B, H, T = scores.shape
    device = scores.device

    M_eff = min(M, T)
    if M_eff > 0:
        top = torch.topk(scores, k=M_eff, dim=-1, sorted=False)
        heavy_idx = top.indices
        heavy_valid = torch.isfinite(top.values)
        heavy_count = heavy_valid.sum(dim=-1).to(torch.int32)
        heavy_idx = heavy_idx.masked_fill(~heavy_valid, 0)
    else:
        heavy_idx = torch.empty((B, H, 0), device=device, dtype=torch.int64)
        heavy_count = torch.zeros((B, H), device=device, dtype=torch.int32)

    parts = []
    if sink > 0:
        parts.append(torch.arange(sink, device=device))
    if window > 0:
        start = max(T - window, sink)
        if start < T:
            parts.append(torch.arange(start, T, device=device))

    base = torch.cat(parts).to(torch.int32) if parts else torch.empty((0,), device=device, dtype=torch.int32)
    base = base.view(1, 1, -1).expand(B, H, -1)

    sparse_list = torch.cat([base, heavy_idx.to(torch.int32)], dim=-1)
    sparse_len = (base.shape[-1] + heavy_count).to(torch.int32)
    return sparse_list, sparse_len



@torch.no_grad()
def score_all_tokens(q_scores_fp8, k_hard_bhlt, v_norm_bht, allowed_bht):
    B, H, L, R = q_scores_fp8.shape
    _, _, _, T = k_hard_bhlt.shape

    q_scores = q_scores_fp8.to(torch.float16)
    k_hard = k_hard_bhlt.clamp(0, R - 1).to(torch.int64)

    q2 = q_scores.reshape(B * H * L, R)
    k2 = k_hard.reshape(B * H * L, T)
    g2 = torch.gather(q2, dim=1, index=k2).reshape(B, H, L, T)

    scores = g2.float().sum(dim=2) * v_norm_bht.float()

    # mask ONCE here
    scores = scores.masked_fill(~allowed_bht, float("-inf"))
    return scores




# =========================================================
# BACKEND (Stage 1+2): your flash-decode style sparse attention
# =========================================================

@triton.jit
def _fwd_kernel_sparse_decode_stage1(
    Q, K, V, sm_scale,
    Sparse_List, Sparse_Len,
    Mid_O, Mid_O_LogExpSum,
    stride_sparse_b, stride_sparse_h,
    stride_qbs, stride_qh, stride_qd,
    stride_kbb, stride_kh, stride_ks,
    stride_vbb, stride_vh, stride_vs,
    stride_splen_b, stride_splen_h,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    gqa_group_size: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)
    cur_kv_head = cur_head // gqa_group_size

    offs_d = tl.arange(0, BLOCK_DMODEL)

    cur_seq_len_ptr = Sparse_Len + cur_batch * stride_splen_b + cur_head * stride_splen_h
    cur_seq_len = tl.load(cur_seq_len_ptr)

    cur_block_start = seq_start_block * BLOCK_SEQ
    cur_block_end = tl.minimum(cur_seq_len, cur_block_start + BLOCK_SEQ)

    sparse_ptr_base = Sparse_List + cur_batch * stride_sparse_b + cur_head * stride_sparse_h

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + off_q)

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    block_n_size = (
        tl.where(cur_block_end - cur_block_start <= 0, 0,
                 cur_block_end - cur_block_start + BLOCK_N - 1) // BLOCK_N
    )

    offs_n = cur_block_start + tl.arange(0, BLOCK_N)

    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n

        token_idx = tl.load(
            sparse_ptr_base + offs_n_new,
            mask=offs_n_new < cur_seq_len,
            other=0,
        )

        base_ptr = cur_batch * stride_kbb + cur_kv_head * stride_kh
        off_k = base_ptr + token_idx[:, None] * stride_ks + offs_d[None, :]
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_seq_len, other=0.0)
        v = tl.load(V + off_k, mask=offs_n_new[:, None] < cur_seq_len, other=0.0)

        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        att_value = tl.where(offs_n_new < cur_seq_len, att_value, float("-inf"))

        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)

        acc *= logic_scale
        acc += tl.sum(exp_logic[:, None] * v, axis=0)
        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=0)
        max_logic = new_max_logic

    need_store = tl.where(block_n_size == 0, 0, 1)
    for _ in range(0, need_store, 1):
        off_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + seq_start_block * stride_mid_os
            + offs_d
        )
        off_mid_o_logexpsum = (
            cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block
        )
        tl.store(Mid_O + off_mid_o, acc / sum_exp)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp))


@triton.jit
def _fwd_kernel_sparse_decode_stage2(
    Sparse_Len,
    Mid_O,
    Mid_O_LogExpSum,
    O,
    stride_splen_b, stride_splen_h,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    stride_obs, stride_oh, stride_od,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)

    cur_seq_len_ptr = Sparse_Len + cur_batch * stride_splen_b + cur_head * stride_splen_h
    cur_seq_len = tl.load(cur_seq_len_ptr)

    block_n_size = (tl.where(cur_seq_len <= 0, 0, cur_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ)

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh

    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)

        new_max_logic = tl.maximum(tlogic, max_logic)
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic

    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d
    tl.store(O + off_o, acc / sum_exp)


@torch.no_grad()
def sparse_decode_stage1(
    q: torch.Tensor,            # [B,H,D]
    k: torch.Tensor,            # [B,Kv,S,D]
    v: torch.Tensor,            # [B,Kv,S,D]
    sparse_list: torch.Tensor,  # [B,H,Ktotal]
    sparse_len: torch.Tensor,   # [B,H]
    max_len_in_batch: int,
    mid_out: torch.Tensor,         # [B,H,block_seq_num,D] fp32
    mid_out_logsumexp: torch.Tensor,# [B,H,block_seq_num] fp32
    block_seq: int,
):
    BLOCK_N = 16
    D = q.shape[-1]
    assert D in {16, 32, 64, 128}

    sm_scale = 1.0 / math.sqrt(D)
    B, H = q.shape[0], q.shape[1]
    grid = (B, H, triton.cdiv(max_len_in_batch, block_seq))
    gqa_group_size = H // k.shape[1]

    _fwd_kernel_sparse_decode_stage1[grid](
        q, k, v, sm_scale,
        sparse_list, sparse_len,
        mid_out, mid_out_logsumexp,
        sparse_list.stride(0), sparse_list.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        sparse_len.stride(0), sparse_len.stride(1),
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_out_logsumexp.stride(0), mid_out_logsumexp.stride(1), mid_out_logsumexp.stride(2),
        gqa_group_size,
        BLOCK_SEQ=block_seq,
        BLOCK_DMODEL=D,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )


@torch.no_grad()
def sparse_decode_stage2(
    mid_out: torch.Tensor,
    mid_out_logsumexp: torch.Tensor,
    sparse_len: torch.Tensor,
    out: torch.Tensor,       # [B,H,D] fp16/bf16
    block_seq: int,
):
    D = out.shape[-1]
    assert D in {16, 32, 64, 128}

    B, H = out.shape[0], out.shape[1]
    grid = (B, H)

    _fwd_kernel_sparse_decode_stage2[grid](
        sparse_len,
        mid_out,
        mid_out_logsumexp,
        out,
        sparse_len.stride(0), sparse_len.stride(1),
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_out_logsumexp.stride(0), mid_out_logsumexp.stride(1), mid_out_logsumexp.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_SEQ=block_seq,
        BLOCK_DMODEL=D,
        num_warps=8,
        num_stages=2,
    )


@torch.no_grad()
def sparse_attention_fwd(
    query: torch.Tensor,      # [B,H,D]
    key: torch.Tensor,        # [B,Kv,S,D]
    value: torch.Tensor,      # [B,Kv,S,D]
    sparse_list: torch.Tensor,# [B,H,Ktotal]
    sparse_len: torch.Tensor, # [B,H]
    block_seq: int = 256,
) -> torch.Tensor:
    assert query.is_cuda and key.is_cuda and value.is_cuda and sparse_list.is_cuda and sparse_len.is_cuda
    B, H, D = query.shape
    max_len_in_batch = int(sparse_len.max().item())

    block_seq_num = (max_len_in_batch + block_seq - 1) // block_seq
    mid_o = torch.empty((B, H, block_seq_num, D), dtype=torch.float32, device=query.device)
    mid_o_log = torch.empty((B, H, block_seq_num), dtype=torch.float32, device=query.device)
    out = torch.empty((B, H, D), dtype=query.dtype, device=query.device)

    sparse_decode_stage1(query, key, value, sparse_list, sparse_len, max_len_in_batch, mid_o, mid_o_log, block_seq)
    sparse_decode_stage2(mid_o, mid_o_log, sparse_len, out, block_seq)
    return out


@triton.jit
def rw_decode_fwd(
    Q, K, V, seqlens, block_index, Out,
    stride_qz, stride_qh, stride_qt, stride_qd,
    stride_kz, stride_kh, stride_kt, stride_kd,
    stride_vz, stride_vh, stride_vt, stride_vd,
    stride_oz, stride_oh, stride_od,
    H: tl.constexpr,
    KLIST: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    dtype: tl.constexpr,
):
    # program per (b,h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    seqlen = tl.load(seqlens + b)  # int32
    # last token index in absolute KV positions
    q_pos = seqlen - 1

    # offsets
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    # base pointers
    q_off = b * stride_qz + h * stride_qh
    k_off = b * stride_kz + h * stride_kh
    v_off = b * stride_vz + h * stride_vh
    o_off = b * stride_oz + h * stride_oh

    # Q points at time dimension (T_q==1). We load index 0.
    q_ptrs = Q + q_off + 0 * stride_qt + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=offs_d < BLOCK_D, other=0.0).to(tl.float32)

    # scale into log2 space for exp2
    qk_scale = sm_scale * 1.44269504
    q = (q * qk_scale).to(tl.float32)

    # output accumulator
    m_i = tl.full([], -float("inf"), tl.float32)    # scalar
    l_i = tl.zeros([], tl.float32)                  # scalar
    acc = tl.zeros([BLOCK_D], tl.float32)           # vector

    # block list pointer: (B,H,KLIST)
    blk_ptr = block_index + (b * H + h) * KLIST

    # loop over chosen blocks
    for j in range(0, KLIST):
        blk = tl.load(blk_ptr + j).to(tl.int32)
        do_blk = blk >= 0
        blk_safe = tl.maximum(blk, 0)     # <--- add this
        start_t = blk_safe * BLOCK_N      # <--- use blk_safe
        cols = start_t + offs_n

        kv_mask = (cols < seqlen) & do_blk


        # load K block: shape (BLOCK_N, D)
        k_ptrs = K + k_off + cols[:, None] * stride_kt + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=kv_mask[:, None] & (offs_d[None, :] < BLOCK_D), other=0.0).to(tl.float32)

        # logits: (BLOCK_N,)
        qk = tl.sum(k * q[None, :], axis=1)  # fp32

        # causal for decode: cols <= q_pos
        causal = cols <= q_pos
        qk = tl.where(kv_mask, qk, -float("inf"))

        # online softmax update (scalar m_i/l_i, vector acc)
        block_max = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, block_max)

        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new)

        # load V block: shape (BLOCK_N, D)
        v_ptrs = V + v_off + cols[:, None] * stride_vt + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=kv_mask[:, None] & (offs_d[None, :] < BLOCK_D), other=0.0).to(tl.float32)

        acc = acc * alpha + tl.sum(v * p[:, None], axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    acc = acc / (l_i + 1e-9)

    o_ptrs = Out + o_off + offs_d * stride_od
    tl.store(o_ptrs, acc.to(dtype), mask=offs_d < BLOCK_D)


def rw_decode_sparse_attention(
    q: torch.Tensor,          # (B,H,1,D)
    k: torch.Tensor,          # (B,H,T_k,D)
    v: torch.Tensor,          # (B,H,T_k,D)
    seqlens: torch.Tensor,    # (B,) int32
    block_index: torch.Tensor,# (B,H,KLIST) int32
    block_n: int = 64,
) -> torch.Tensor:
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.shape[0] == k.shape[0] == v.shape[0]
    assert q.shape[1] == k.shape[1] == v.shape[1]
    B, H, Tq, D = q.shape
    assert Tq == 1
    assert D in (16, 32, 64, 128)
    assert block_index.dtype == torch.int32
    KLIST = block_index.shape[-1]

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    block_index = block_index.contiguous()

    out = torch.empty((B, H, D), device=q.device, dtype=q.dtype)

    grid = (B * H,)

    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    sm_scale = (D ** -0.5)

    rw_decode_fwd[grid](
        q, k, v, seqlens, block_index, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        H=H,
        KLIST=KLIST,
        sm_scale=sm_scale,
        BLOCK_N=block_n,
        BLOCK_D=D,
        dtype=dtype,
        num_warps=4,
        num_stages=2,
    )
    return out.unsqueeze(2)  # (B,H,1,D)