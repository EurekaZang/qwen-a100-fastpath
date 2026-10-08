set -euo pipefail
W=${QWEN_HOME:-$(cd "$(dirname "$0")" && pwd)}; cd $W
export UV_CACHE_DIR=$W/uv-cache UV_PYTHON_INSTALL_DIR=$W/uv-python
export XDG_CACHE_HOME=$W/cache RUSTUP_HOME=$W/rust/rustup CARGO_HOME=$W/rust/cargo   # llguidance 1.7 has no glibc-2.28 wheel: built from source, Rust must not touch the read-only AFS home
mkdir -p $UV_CACHE_DIR $UV_PYTHON_INSTALL_DIR $XDG_CACHE_HOME $RUSTUP_HOME $CARGO_HOME bin
if [ ! -x bin/uv ]; then
  F=uv-x86_64-unknown-linux-gnu.tar.gz
  curl -fsSL -o $F https://github.com/astral-sh/uv/releases/latest/download/$F
  curl -fsSL -o $F.sha256 https://github.com/astral-sh/uv/releases/latest/download/$F.sha256
  sha256sum -c $F.sha256
  tar xzf $F --strip-components=1 -C bin && rm -f $F $F.sha256
fi
bin/uv --version
bin/uv python install 3.12                                  # standalone build: ships Python.h (Triton JIT needs it; system python has no headers)
bin/uv venv --managed-python --python 3.12 venv
venv/bin/python -c "import sysconfig,os; i=sysconfig.get_paths()['include']; print('Python.h present:', os.path.exists(os.path.join(i,'Python.h')))"
bin/uv pip install --python venv/bin/python "vllm==0.31.0"
venv/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| gpu ok:', torch.cuda.is_available())"
echo INSTALL_DONE
