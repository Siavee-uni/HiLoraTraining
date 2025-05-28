#!/usr/bin/env bash
set -euxo pipefail

echo "🚀 HiDream-LoRA Training Setup Starting..."

# ─── 1. Hugging Face Token für gated Repos ────────────────────────────────────
export HUGGINGFACE_HUB_TOKEN=""

# ─── 2. Variablen ──────────────────────────────────────────────────────────────
WORKSPACE="/workspace"
PIPE_DIR="$WORKSPACE/diffusion-pipe"
EXAMPLE_DIR="$PIPE_DIR/examples"
CONFIG_FILE="$EXAMPLE_DIR/hidream.toml"
DATA_DIR="$WORKSPACE/dataset/test"
OUTPUT_DIR="$WORKSPACE/output"
VENV_DIR="$WORKSPACE/venv"
PYTHON_BIN="python3"

# ─── 3. Repo klonen (inkl. HiDream-Submodule) ──────────────────────────────────
if [ -d "$PIPE_DIR/.git" ]; then
  cd "$PIPE_DIR" && git pull --ff-only
else
  git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe.git "$PIPE_DIR"
fi

cd "$PIPE_DIR"
git submodule sync
git submodule update --init --recursive submodules/HiDream

# ─── 4. Virtuelle Umgebung mit venv ────────────────────────────────────────────
[ -d "$VENV_DIR" ] || $PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# ─── 5. PyTorch installieren (CUDA 12.4) ───────────────────────────────────────
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

# ─── 6. Restliche Abhängigkeiten ───────────────────────────────────────────────
pip install -r requirements.txt
pip install git+https://github.com/ollanoinc/hyvideo.git
pip install --upgrade git+https://github.com/huggingface/diffusers.git
pip install deepspeed accelerate flash-attn huggingface_hub

# ─── 7. HF-CLI Login ───────────────────────────────────────────────────────────
# nutzt den bereits installierten huggingface-cli
huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN"

# ─── 8. examples-Verzeichnis & TOMLs erstellen ─────────────────────────────────
mkdir -p "$EXAMPLE_DIR"

cat > "$EXAMPLE_DIR/hidream_dataset.toml" <<EOF
resolutions = [1024]
enable_ar_bucket = true
min_ar = 0.5
max_ar = 2.0
num_ar_buckets = 7

[[directory]]
path = "$DATA_DIR"
num_repeats = 5
EOF

cat > "$EXAMPLE_DIR/hidream.toml" <<EOF
output_dir = "$OUTPUT_DIR"
dataset = "examples/hidream_dataset.toml"

epochs = 10
micro_batch_size_per_gpu = 1
gradient_accumulation_steps = 4
gradient_clipping = 1
warmup_steps = 100
blocks_to_swap = 20
eval_every_n_epochs = 1
eval_before_first_step = true
eval_micro_batch_size_per_gpu = 1
eval_gradient_accumulation_steps = 1
save_every_n_epochs = 1
checkpoint_every_n_minutes = 120
activation_checkpointing = true
partition_method = "parameters"
save_dtype = "bfloat16"
caching_batch_size = 1
steps_per_print = 1
video_clip_mode = "single_middle"
pipeline_stages = 1

[model]
type                        = "hidream"
diffusers_path              = "HiDream-ai/HiDream-I1-Full"
llama3_path                 = "meta-llama/Llama-3.1-8B-Instruct"
dtype                       = "bfloat16"
transformer_dtype           = "float8"
llama3_4bit                 = true
max_llama3_sequence_length  = 128

[adapter]
type               = "lora"
rank               = 32
lora_dtype         = "bfloat16"
only_double_blocks = false

[optimizer]
type = "adamw_optimi"
lr   = 2e-5
betas = [0.9, 0.99]
weight_decay = 0.01
eps = 1e-8
EOF

# ─── 9. Training starten ───────────────────────────────────────────────────────
[ -d "$DATA_DIR" -a -n "$(ls -A "$DATA_DIR")" ] \
  || { echo "✖ Keine Bilder im $DATA_DIR"; exit 1; }

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

deepspeed --num_gpus=1 train.py --deepspeed --config "$CONFIG_FILE"
