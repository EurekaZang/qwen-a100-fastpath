W=${QWEN_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}   # deployment directory (this file lives in it)
set -a; . $W/serve.env; set +a
export PYTHONPATH=$W/auth
export HOME=$W                       # real $HOME is read-only AFS
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT=$W/vllm-cache VLLM_CONFIG_ROOT=$W/vllm-config
export XDG_CACHE_HOME=$W/cache XDG_CONFIG_HOME=$W/config
export TRITON_CACHE_DIR=$W/triton-cache FLASHINFER_WORKSPACE_BASE=$W
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 OMP_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$W/venv/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin     # venv/bin first: FlashInfer JIT needs `ninja`
[ -f $W/tuning.env ] && { set -a; . $W/tuning.env; set +a; }   # exported: the fastpath plugin reads its switches from the environment
