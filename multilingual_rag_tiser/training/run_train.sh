#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${1:-}"              # qwen
LANG="${2:-}"               # en, it, de, fa, en_it_de_mixed_15000, mixed
PROMPT_NAME="${3:-tiser_full}"
MAX_SAMPLES="${4:-2000}"        # optional

if [[ -z "$MODEL" || -z "$LANG" ]]; then
  echo "Usage:"
  echo "  bash multilingual_tiser/training/run_train.sh qwen en tiser_full 15000"
  echo "  bash multilingual_tiser/training/run_train.sh qwen it tiser_full 15000"
  echo "  bash multilingual_tiser/training/run_train.sh qwen de tiser_full 15000"
  echo "  bash multilingual_tiser/training/run_train.sh qwen en_it_de_mixed_15000 tiser_full"
  echo "  bash multilingual_tiser/training/run_train.sh qwen mixed tiser_full 15000"
  exit 1
fi

# -------------------------
# Base model
# -------------------------
if [[ "$MODEL" == "qwen" ]]; then
  MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
else
  echo "ERROR: unsupported model type: $MODEL"
  echo "Currently supported: qwen"
  exit 1
fi

# -------------------------
# Select train file
# -------------------------
TRAIN_FILE=""

if [[ "$LANG" == "en" ]]; then
  TRAIN_FILE="data/splits/train/TISER_train_en.json"

elif [[ "$LANG" == "it" ]]; then
  # Use passed file if it exists, otherwise fallback to raw translated file
  if [[ -f "data/splits/train/it/TISER_train_it_passed.json" ]]; then
    TRAIN_FILE="data/splits/train/it/TISER_train_it_passed.json"
  else
    TRAIN_FILE="data/splits/train/TISER_train_it.json"
  fi

elif [[ "$LANG" == "de" ]]; then
  if [[ -f "data/splits/train/de/TISER_train_de_passed.json" ]]; then
    TRAIN_FILE="data/splits/train/de/TISER_train_de_passed.json"
  else
    TRAIN_FILE="data/splits/train/TISER_train_de.json"
  fi

elif [[ "$LANG" == "fa" ]]; then
  if [[ -f "data/splits/train/fa/TISER_train_fa_passed.json" ]]; then
    TRAIN_FILE="data/splits/train/fa/TISER_train_fa_passed.json"
  else
    TRAIN_FILE="data/splits/train/TISER_train_fa.json"
  fi

elif [[ "$LANG" == "mixed" ]]; then
  # Default mixed file based on MAX_SAMPLES
  if [[ -z "$MAX_SAMPLES" ]]; then
    echo "ERROR: for LANG=mixed you should pass MAX_SAMPLES."
    echo "Example:"
    echo "  bash multilingual_tiser/training/run_train.sh qwen mixed tiser_full 15000"
    exit 1
  fi

  TRAIN_FILE="data/splits/train/TISER_train_en_it_de_mixed_${MAX_SAMPLES}.json"

elif [[ "$LANG" == *"mixed"* ]]; then
  # Example:
  # LANG=en_it_mixed_2000
  # LANG=en_it_de_mixed_15000
  TRAIN_FILE="data/splits/train/TISER_train_${LANG}.json"

else
  echo "ERROR: unknown language/dataset name: $LANG"
  exit 1
fi

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "ERROR: train file not found:"
  echo "$TRAIN_FILE"
  echo
  echo "Available train files:"
  ls -lh data/splits/train/*.json 2>/dev/null || true
  echo
  echo "Available scored files:"
  ls -lh data/splits/train/*/*.json 2>/dev/null || true
  exit 1
fi

# -------------------------
# max samples arg
# -------------------------
MAX_SAMPLES_ARG=""
if [[ -n "$MAX_SAMPLES" ]]; then
  MAX_SAMPLES_ARG="--max_train_samples $MAX_SAMPLES"
fi

# -------------------------
# Output dir
# -------------------------
SAMPLE_NAME="${MAX_SAMPLES:-full}"
OUT_DIR="experiments/${MODEL}/${LANG}_${PROMPT_NAME}_${SAMPLE_NAME}_8gb_val_qlora"
mkdir -p "$OUT_DIR"

# -------------------------
# 8GB-safe defaults
# -------------------------
EPOCHS=2
LR=3e-4
MAX_LEN=1536

LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

PER_DEVICE_BS=1
GRAD_ACCUM=16

VAL_SPLIT=0.1

# Safe eval/save steps
if [[ -n "$MAX_SAMPLES" && "$MAX_SAMPLES" -gt 0 ]]; then
  EVAL_STEPS=$(( MAX_SAMPLES / 20 ))

  if [[ "$EVAL_STEPS" -lt 10 ]]; then
    EVAL_STEPS=10
  fi
else
  EVAL_STEPS=500
fi

echo "=============================="
echo " STARTING 8GB VRAM TRAINING"
echo "Model Type    : $MODEL"
echo "Model         : $MODEL_NAME"
echo "Prompt        : $PROMPT_NAME"
echo "Lang/Dataset  : $LANG"
echo "Train File    : $TRAIN_FILE"
echo "Output Dir    : $OUT_DIR"
echo "Max Samples   : ${MAX_SAMPLES:-all}"
echo "Max Length    : $MAX_LEN"
echo "LoRA Rank     : $LORA_R"
echo "LoRA Alpha    : $LORA_ALPHA"
echo "Batch Size    : $PER_DEVICE_BS"
echo "Grad Accum    : $GRAD_ACCUM"
echo "Validation    : $VAL_SPLIT"
echo "Eval Steps    : $EVAL_STEPS"
echo "Save Steps    : $EVAL_STEPS"
echo "=============================="

python multilingual_rag_tiser/training/train_qlora.py \
  --model_name "$MODEL_NAME" \
  --train_file "$TRAIN_FILE" \
  --output_dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --max_length "$MAX_LEN" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --per_device_batch_size "$PER_DEVICE_BS" \
  --grad_accum "$GRAD_ACCUM" \
  --save_steps "$EVAL_STEPS" \
  --logging_steps 10 \
  --eval_steps "$EVAL_STEPS" \
  --validation_split "$VAL_SPLIT" \
  --dataloader_num_workers 0 \
  --gradient_checkpointing 1 \
  --gpu_memory_gb 7 \
  --cpu_memory_gb 16 \
  $MAX_SAMPLES_ARG \
  --only_passed \
  --prompt_name "$PROMPT_NAME"