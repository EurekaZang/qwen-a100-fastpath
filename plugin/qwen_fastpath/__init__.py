"""vLLM general plugin: single-user decode fast paths for Qwen3.8-27B-INT8 on an A100.

Loaded by vLLM in every process through the `vllm.general_plugins` entry point. Each patch is exact or more precise
than the code it replaces and can be switched off with an environment variable:

  QWEN_FAST_ATTN=0    keep FlashAttention-2 for speculative-decode verification batches
  QWEN_FAST_LINEAR=0  keep CUTLASS W8A8 for small-batch (decode) linear layers
  QWEN_FAST_DRAFT_HEAD=0  keep the bf16 lm_head for MTP draft proposals
"""
import logging
import os

logger = logging.getLogger("vllm.qwen_fastpath")
_registered = False


def register():
    global _registered
    if _registered:
        return
    _registered = True
    from . import patches
    if os.environ.get("QWEN_FAST_ATTN", "1") == "1":
        patches.patch_flash_attention()
    if os.environ.get("QWEN_FAST_LINEAR", "1") == "1":
        patches.patch_int8_linear()
    if os.environ.get("QWEN_FAST_DRAFT_HEAD", "1") == "1":
        patches.patch_draft_lm_head()
