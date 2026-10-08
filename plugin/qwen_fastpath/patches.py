import logging

import torch

logger = logging.getLogger("vllm.qwen_fastpath")

MAX_SPEC_QUERY_LEN = 8


def patch_flash_attention():
    """Route uniform speculative-decode batches (every request has Q = 1 + k query tokens, 1 < Q <= 8) through the
    split-KV GQA-packed kernel. Everything else (prefill, mixed batches, Q == 1) stays on FlashAttention-2."""
    from vllm.v1.attention.backends import flash_attn as fa

    from .spec_attn import gqa_packed_spec_attention

    orig_forward = fa.FlashAttentionImpl.forward
    if getattr(orig_forward, "_qwen_fastpath", False):
        return

    def eligible(impl) -> bool:
        ok = getattr(impl, "_qf_eligible", None)
        if ok is None:
            ok = (
                impl.attn_type == fa.AttentionType.DECODER
                and getattr(impl, "dcp_world_size", 1) == 1
                and impl.vllm_flash_attn_version == 2
                and not getattr(impl, "fa4_hd256", False)
                and tuple(impl.sliding_window) == (-1, -1)
                and impl.alibi_slopes is None
                and not impl.logits_soft_cap
                and getattr(impl, "sinks", None) is None
                and not fa.is_quantized_kv_cache(impl.kv_cache_dtype)
                and impl.num_heads % impl.num_kv_heads == 0
                and impl.head_size in (64, 128, 256)
                and impl.kv_sharing_target_layer_name is None
            )
            impl._qf_eligible = ok
        return ok

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        md = attn_metadata
        if (
            md is not None
            and output_scale is None
            and output_block_scale is None
            and 1 < md.max_query_len <= MAX_SPEC_QUERY_LEN
            and not md.use_cascade
            and md.causal is True
            and md.mm_prefix_query_range_tensor is None
            and md.rswa_prefix_lens is None
            and eligible(self)
        ):
            num_reqs = md.query_start_loc.shape[0] - 1
            q_len = md.max_query_len
            if md.num_actual_tokens == num_reqs * q_len:      # uniform: every request has exactly q_len tokens
                key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
                q3 = query.view(query.shape[0], self.num_heads, self.head_size)
                o3 = output.view(output.shape[0], self.num_heads, self.head_size)
                gqa_packed_spec_attention(q3, key_cache, value_cache, o3, md.seq_lens, md.block_table,
                                          num_reqs, q_len, self.num_kv_heads, self.scale)
                return output
        return orig_forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                            output_scale, output_block_scale)

    forward._qwen_fastpath = True
    fa.FlashAttentionImpl.forward = forward
    logger.info("qwen_fastpath: split-KV GQA-packed attention enabled for speculative-decode verification")


def patch_int8_linear():
    """Small-M (decode) W8A8 linears -> W8A16 Triton kernel on the same int8 weights; large M keeps CUTLASS."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassInt8ScaledMMLinearKernel
    from vllm.utils.torch_utils import direct_register_custom_op

    from .w8a16 import SMALL_M, pick_config, w8a16_mm

    orig_apply = CutlassInt8ScaledMMLinearKernel.apply_weights
    if getattr(orig_apply, "_qwen_fastpath", False):
        return

    def qf_int8_linear(x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
                       bias: torch.Tensor | None) -> torch.Tensor:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        M, K = x2.shape
        N = w_q.shape[1]
        if M <= SMALL_M and x2.dtype == torch.bfloat16 and w_q.stride(0) == 1 and w_s.numel() == N:
            cfg = pick_config(N, K)
            if cfg is not None:
                if x2.stride(-1) != 1:
                    x2 = x2.contiguous()
                y = w8a16_mm(x2, w_q, w_s, cfg, x.dtype)
                if bias is not None:
                    y = y + bias
                return y.view(*lead, N)
        x_q, x_s, _ = ops.scaled_int8_quant(x2.contiguous(), None, None, symmetric=True)
        y = ops.cutlass_scaled_mm(x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=bias)
        return y.view(*lead, N)

    def qf_int8_linear_fake(x, w_q, w_s, bias):
        return x.new_empty((*x.shape[:-1], w_q.shape[1]))

    direct_register_custom_op("qf_int8_linear", qf_int8_linear, mutates_args=[], fake_impl=qf_int8_linear_fake)

    def apply_weights(self, layer, x, bias=None):
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)
        if i_s is None and i_zp is None and azp_adj is None:     # dynamic symmetric per-token (this checkpoint)
            return torch.ops.vllm.qf_int8_linear(x, w_q, w_s, bias)
        return orig_apply(self, layer, x, bias)

    apply_weights._qwen_fastpath = True
    CutlassInt8ScaledMMLinearKernel.apply_weights = apply_weights
    logger.info("qwen_fastpath: W8A16 decode path for W8A8-int8 linears (M <= %d), CUTLASS above", SMALL_M)


def patch_draft_lm_head():
    """MTP drafting takes the argmax of a full 248K-vocab lm_head GEMV per draft token (2.5 GB of bf16 weights each).
    Give the drafter an int8 per-row copy of the (shared) lm_head and run it through the W8A16 kernel: half the bytes.
    Only *proposals* change; every token is still accepted or resampled against the target model's bf16 logits by the
    rejection sampler, so the output distribution is unchanged. Costs ~1.3 GB of GPU memory."""
    from vllm.model_executor.models import qwen3_5_mtp
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

    from .w8a16 import SMALL_M, pick_config, w8a16_mm

    orig_load = MTPSpeculator.load_draft_model
    if getattr(orig_load, "_qwen_fastpath", False):
        return

    def load_draft_model(self, target_model, target_attn_layer_names):
        draft_model = orig_load(self, target_model, target_attn_layer_names)
        lm_head = getattr(draft_model, "lm_head", None)
        weight = getattr(lm_head, "weight", None)
        if isinstance(draft_model, qwen3_5_mtp.Qwen3_5MTP) and weight is not None and weight.dim() == 2 \
                and getattr(lm_head, "tp_size", 1) == 1:
            V, Hd = weight.shape
            w8 = torch.empty((V, Hd), dtype=torch.int8, device=weight.device)
            scale = torch.empty((V,), dtype=torch.float32, device=weight.device)
            with torch.no_grad():
                for i in range(0, V, 16384):
                    blk = weight[i:i + 16384].float()
                    s = blk.abs().amax(dim=1).clamp_(min=1e-8) / 127.0
                    w8[i:i + 16384] = torch.round(blk / s[:, None]).clamp_(-127, 127).to(torch.int8)
                    scale[i:i + 16384] = s
            num_pad = lm_head.shard_indices.num_org_vocab_padding
            draft_model._qf_head_i8 = (w8.t(), scale, num_pad, pick_config(V, Hd))
            logger.info("qwen_fastpath: int8 draft lm_head (%d x %d, %.2f GB) for MTP proposals", V, Hd,
                        w8.numel() / 1e9)
        return draft_model

    load_draft_model._qwen_fastpath = True
    MTPSpeculator.load_draft_model = load_draft_model

    def int8_logits(model, hidden_states):
        q = model.__dict__.get("_qf_head_i8")
        lp = model.logits_processor
        if (q is None or hidden_states.dim() != 2 or hidden_states.shape[0] > SMALL_M
                or hidden_states.dtype != torch.bfloat16 or lp.logits_as_input or lp.soft_cap is not None
                or lp.scale != 1.0 or getattr(model, "draft_id_to_target_id", None) is not None):
            return None
        w_t, scale, num_pad, cfg = q
        x = hidden_states if hidden_states.stride(-1) == 1 else hidden_states.contiguous()
        return w8a16_mm(x, w_t, scale, cfg, torch.bfloat16)

    orig_logits = qwen3_5_mtp.Qwen3_5MTP.compute_logits

    def compute_logits(self, hidden_states, spec_step_idx=0):
        logits = int8_logits(self, hidden_states)
        if logits is None:
            return orig_logits(self, hidden_states, spec_step_idx)
        return logits[..., : self.logits_processor.org_vocab_size]

    orig_top = qwen3_5_mtp.Qwen3_5MTP.get_top_tokens

    def get_top_tokens(self, hidden_states):
        logits = int8_logits(self, hidden_states)
        if logits is None:
            return orig_top(self, hidden_states)
        num_pad = self.lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        return logits.argmax(dim=-1) + self.lm_head.shard_indices.org_vocab_start_index

    qwen3_5_mtp.Qwen3_5MTP.compute_logits = compute_logits
    qwen3_5_mtp.Qwen3_5MTP.get_top_tokens = get_top_tokens
    logger.info("qwen_fastpath: int8 draft lm_head enabled")
