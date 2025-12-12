class DenseAttnServer:
    """
    Dense-only decode server using FlashInfer paged KV cache.
    Keeps the same interface as your future sparse server.
    """
    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_length: int,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_size = num_attention_heads * head_dim
        self.max_length = max_length
        self.device = device
        self.dtype = dtype

        # Paged KV cache wrapper
        self.max_num_pages = batch_size
        self.page_size = max_length  # dense uses full length
        self.kv_page_indices = torch.arange(self.max_num_pages, device=device).int()
        self.kv_page_indptr = torch.arange(batch_size + 1, device=device).int()
        self.kv_last_page_len = torch.zeros(batch_size, device=device).int()

        self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, "HND", use_tensor_cores=True
        )

        # flashinfer_kv_cache[layer] has shape:
        #   [max_num_pages, 2, kv_heads, page_size, head_dim]
        self.flashinfer_kv_cache = [
            torch.zeros(
                self.max_num_pages,
                2,
                self.num_key_value_heads,
                self.page_size,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(num_layers)
        ]

    def alloc_buffer(self, seq_len: int):
        # Dense-only: nothing to allocate.
        pass

    def fill(self, layer_idx: int, request_id: int, key_cache: torch.Tensor, value_cache: torch.Tensor, seq_len: int):
        """
        Called after dense prefill. We copy full KV into the paged cache for dense decode.
        key_cache/value_cache are [T, kvH, D].
        """
        self.flashinfer_kv_cache[layer_idx][request_id][0, :, :seq_len, :].copy_(key_cache[:seq_len].transpose(0, 1))
        self.flashinfer_kv_cache[layer_idx][request_id][1, :, :seq_len, :].copy_(value_cache[:seq_len].transpose(0, 1))
        self.kv_last_page_len[request_id] = seq_len

    def build_table(self, layer_idx: int, request_id: int, seq_len: int):
        # Dense-only: no index/table.
        pass

    def plan(self):
        # Prepare wrapper for next decode step; FlashInfer expects "last_page_len" to include the appended token,
        # your original code increments before planning.
        self.kv_last_page_len += 1
        self.decode_wrapper.plan(
            self.kv_page_indptr,
            self.kv_page_indices,
            self.kv_last_page_len,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.page_size,
            pos_encoding_mode="NONE",
            q_data_type=self.dtype,
            data_type=self.dtype,
        )

    def decode(self, query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        """
        query_states: [B, H, 1, D]
        key_states:   [B, kvH, 1, D]
        value_states: [B, kvH, 1, D]
        Returns:      [B, 1, hidden]
        """
        # Append new KV for this step
        k = key_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
        v = value_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)

        flashinfer.append_paged_kv_cache(
            k,
            v,
            self.kv_page_indptr,
            self.flashinfer_kv_cache[layer_idx],
            self.kv_page_indices,
            self.kv_page_indptr,
            self.kv_last_page_len,
            kv_layout="HND",
        )

        q = query_states.reshape(self.batch_size, self.num_attention_heads, self.head_dim)
        out = self.decode_wrapper.run(q, self.flashinfer_kv_cache[layer_idx])  # [B, H, D]
        out = out.reshape(self.batch_size, 1, self.hidden_size)
        return out

    def clear(self):
        self.kv_last_page_len.zero_()
        for i in range(self.num_layers):
            self.flashinfer_kv_cache[i].zero_()
