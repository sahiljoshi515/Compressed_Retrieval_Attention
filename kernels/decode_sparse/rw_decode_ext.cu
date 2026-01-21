#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#include <cuda_bf16.h>
#endif

// -------------------------
// type utils
// -------------------------
template <typename T>
__device__ __forceinline__ float to_f32(T x);

template <>
__device__ __forceinline__ float to_f32<half>(half x) { return __half2float(x); }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
template <>
__device__ __forceinline__ float to_f32<nv_bfloat16>(nv_bfloat16 x) { return __bfloat162float(x); }
#endif

template <typename T>
__device__ __forceinline__ T from_f32(float x);

template <>
__device__ __forceinline__ half from_f32<half>(float x) { return __float2half_rn(x); }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
template <>
__device__ __forceinline__ nv_bfloat16 from_f32<nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }
#endif

__device__ __forceinline__ float exp2_approx(float x) {
  return exp2f(x);
}

// warp reduce max
__device__ __forceinline__ float warp_max(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
  return v;
}

// warp reduce sum
__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

// block reduce max using warp reductions
template<int WARPS>
__device__ __forceinline__ float block_max(float v) {
  __shared__ float smem[WARPS];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  v = warp_max(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  float out = -INFINITY;
  if (warp == 0) {
    out = (lane < WARPS) ? smem[lane] : -INFINITY;
    out = warp_max(out);
  }
  return __shfl_sync(0xffffffff, out, 0);
}

// block reduce sum
template<int WARPS>
__device__ __forceinline__ float block_sum(float v) {
  __shared__ float smem[WARPS];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  float out = 0.f;
  if (warp == 0) {
    out = (lane < WARPS) ? smem[lane] : 0.f;
    out = warp_sum(out);
  }
  return __shfl_sync(0xffffffff, out, 0);
}

// -------------------------
// kernel
// One block per (b,h)
// Assumes D=128
// -------------------------
template <typename scalar_t, int THREADS, int WARPS>
__global__ void rw_decode_fwd_128_kernel(
    const scalar_t* __restrict__ Q,           // (B,H,1,128)
    const scalar_t* __restrict__ K,           // (B,H,T,128)
    const scalar_t* __restrict__ V,           // (B,H,T,128)
    const int32_t* __restrict__ seqlens,      // (B,)
    const int32_t* __restrict__ block_index,  // (B,H,KLIST)
    scalar_t* __restrict__ Out,               // (B,H,128)
    int B, int H, int T,
    int KLIST,
    int BLOCK_N,
    float sm_scale
) {
  int pid = blockIdx.x; // 0..B*H-1
  int b = pid / H;
  int h = pid % H;

  int seqlen = seqlens[b];
  int q_pos  = seqlen - 1;

  // scale for exp2 softmax
  float qk_scale = sm_scale * 1.4426950408889634f; // log2(e)

  // load q into shared (128 floats)
  __shared__ float q_sh[128];
  for (int d = threadIdx.x; d < 128; d += THREADS) {
    int qidx = ((b * H + h) * 1 + 0) * 128 + d;
    q_sh[d] = to_f32(Q[qidx]) * qk_scale;
  }
  __syncthreads();

  // online softmax state
  float m_i = -INFINITY;
  float l_i = 0.0f;

  // output accumulator in registers: 128 floats
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  float acc4 = 0.f, acc5 = 0.f, acc6 = 0.f, acc7 = 0.f;
  float acc8 = 0.f, acc9 = 0.f, acc10 = 0.f, acc11 = 0.f;
  float acc12 = 0.f, acc13 = 0.f, acc14 = 0.f, acc15 = 0.f;
  // 16 * 8 = 128 elements if each thread owns 8? We'll do strided per-thread store below instead.
  // Instead, each thread accumulates only the d it owns (d = tid + k*THREADS).
  // We'll write to shared at end.

  __shared__ float out_sh[128];
  for (int d = threadIdx.x; d < 128; d += THREADS) out_sh[d] = 0.f;
  __syncthreads();

  const int32_t* blk_ptr = block_index + (b * H + h) * KLIST;

  // Iterate selected blocks
  for (int j = 0; j < KLIST; ++j) {
    int blk = blk_ptr[j];
    if (blk < 0) continue;

    int start_t = blk * BLOCK_N;

    // pass 1: compute max logit in this block
    float local_max = -INFINITY;

    for (int n = threadIdx.x; n < BLOCK_N; n += THREADS) {
      int col = start_t + n;
      if (col < seqlen && col <= q_pos) {
        const scalar_t* kptr = K + ((b * H + h) * T + col) * 128;
        float sum = 0.f;
        #pragma unroll
        for (int d = 0; d < 128; ++d) {
          sum += to_f32(kptr[d]) * q_sh[d];
        }
        local_max = fmaxf(local_max, sum);
      }
    }

    float block_m = block_max<WARPS>(local_max);

    // merge with running max
    float m_new = fmaxf(m_i, block_m);
    float alpha = exp2_approx(m_i - m_new);   // scale old
    float beta  = exp2_approx(block_m - m_new); // for numerical reasoning (optional)

    // scale previous l and out
    l_i *= alpha;
    for (int d = threadIdx.x; d < 128; d += THREADS) {
      out_sh[d] *= alpha;
    }
    __syncthreads();

    // pass 2: accumulate exp and V
    float l_add = 0.f;

    for (int n = threadIdx.x; n < BLOCK_N; n += THREADS) {
      int col = start_t + n;
      if (col < seqlen && col <= q_pos) {
        const scalar_t* kptr = K + ((b * H + h) * T + col) * 128;
        float sum = 0.f;
        #pragma unroll
        for (int d = 0; d < 128; ++d) sum += to_f32(kptr[d]) * q_sh[d];

        float p = exp2_approx(sum - m_new); // exp2(logit - m_new)
        l_add += p;

        const scalar_t* vptr = V + ((b * H + h) * T + col) * 128;
        for (int d = 0; d < 128; ++d) {
          // each thread updates only its owned d to avoid atomics
          // but we are in the "n loop", so we can't do d loop for all threads efficiently
        }
        // Instead: update out_sh in a second loop over d by threads
        // We'll do that outside token loop to keep it simple.
      }
    }

    float l_block = block_sum<WARPS>(l_add);
    if (threadIdx.x == 0) l_i += l_block;
    __syncthreads();

    // Now accumulate V with p; do it in a second pass where threads iterate d
    // This is 2-pass over tokens (still OK for sparse KLIST; remove later with better mapping)
    for (int n = 0; n < BLOCK_N; ++n) {
      int col = start_t + n;
      if (col < seqlen && col <= q_pos) {
        const scalar_t* kptr = K + ((b * H + h) * T + col) * 128;
        float sum = 0.f;
        #pragma unroll
        for (int d = 0; d < 128; ++d) sum += to_f32(kptr[d]) * q_sh[d];
        float p = exp2_approx(sum - m_new);

        const scalar_t* vptr = V + ((b * H + h) * T + col) * 128;
        for (int d = threadIdx.x; d < 128; d += THREADS) {
          out_sh[d] += to_f32(vptr[d]) * p;
        }
      }
    }
    __syncthreads();

    m_i = m_new;
  }

  // normalize and store
  float inv_l = 1.0f / (l_i + 1e-9f);
  for (int d = threadIdx.x; d < 128; d += THREADS) {
    float out = out_sh[d] * inv_l;
    Out[(b * H + h) * 128 + d] = from_f32<scalar_t>(out);
  }
}

// -------------------------
// C++ interface (PyTorch extension entrypoint)
// -------------------------
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(__CUDA_BF16_TYPES_EXIST__)
#include <cuda_bf16.h>
#define HAS_BF16 1
#else
#define HAS_BF16 0
#endif

template <typename scalar_t, int THREADS, int WARPS>
__global__ void rw_decode_fwd_128_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const int32_t* __restrict__ seqlens,
    const int32_t* __restrict__ block_index,
    scalar_t* __restrict__ Out,
    int B, int H, int T,
    int KLIST,
    int BLOCK_N,
    float sm_scale);

torch::Tensor rw_decode_sparse_attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor seqlens,
    torch::Tensor block_index,
    int64_t block_n
) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA");
  TORCH_CHECK(seqlens.is_cuda() && block_index.is_cuda(), "seqlens/block_index must be CUDA");
  TORCH_CHECK(seqlens.scalar_type() == torch::kInt32, "seqlens must be int32");
  TORCH_CHECK(block_index.scalar_type() == torch::kInt32, "block_index must be int32");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "q/k/v dtype mismatch");

  TORCH_CHECK(q.dim() == 4 && q.size(2) == 1 && q.size(3) == 128, "q must be [B,H,1,128]");
  TORCH_CHECK(k.dim() == 4 && k.size(3) == 128, "k must be [B,H,T,128]");
  TORCH_CHECK(v.dim() == 4 && v.size(3) == 128, "v must be [B,H,T,128]");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
  TORCH_CHECK(seqlens.is_contiguous(), "seqlens must be contiguous");
  TORCH_CHECK(block_index.is_contiguous(), "block_index must be contiguous");

  const int B = (int)q.size(0);
  const int H = (int)q.size(1);
  const int T = (int)k.size(2);
  const int KLIST = (int)block_index.size(2);
  const int BLOCK_N = (int)block_n;

  auto out = torch::empty({B, H, 128}, q.options());
  const float sm_scale = 1.0f / std::sqrt(128.0f);

  constexpr int THREADS = 256;
  constexpr int WARPS   = THREADS / 32;

  dim3 grid(B * H);
  dim3 block(THREADS);

  at::cuda::CUDAGuard device_guard(q.device());
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  if (q.scalar_type() == torch::kFloat16) {
    rw_decode_fwd_128_kernel<half, THREADS, WARPS><<<grid, block, 0, stream>>>(
        (const half*)q.data_ptr(),
        (const half*)k.data_ptr(),
        (const half*)v.data_ptr(),
        (const int32_t*)seqlens.data_ptr(),
        (const int32_t*)block_index.data_ptr(),
        (half*)out.data_ptr(),
        B, H, T, KLIST, BLOCK_N, sm_scale);
  } else if (q.scalar_type() == torch::kBFloat16) {
#if HAS_BF16
    rw_decode_fwd_128_kernel<nv_bfloat16, THREADS, WARPS><<<grid, block, 0, stream>>>(
        (const nv_bfloat16*)q.data_ptr(),
        (const nv_bfloat16*)k.data_ptr(),
        (const nv_bfloat16*)v.data_ptr(),
        (const int32_t*)seqlens.data_ptr(),
        (const int32_t*)block_index.data_ptr(),
        (nv_bfloat16*)out.data_ptr(),
        B, H, T, KLIST, BLOCK_N, sm_scale);
#else
    TORCH_CHECK(false, "BF16 not supported by this CUDA toolchain");
#endif
  } else {
    TORCH_CHECK(false, "Unsupported dtype. Use float16 or bfloat16.");
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.unsqueeze(2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rw_decode_sparse_attention", &rw_decode_sparse_attention_cuda,
        "RW sparse decode attention (CUDA)");
}

