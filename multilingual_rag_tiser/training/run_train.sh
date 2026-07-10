#!/usr/bin/env bash
#
# QLoRA training launcher (8GB-VRAM-safe defaults).
# Base model is Qwen/Qwen2.5-3B-Instruct (override with BASE_MODEL=...).
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LANG="${1:-}"               # en, it, de, fr, mixed, en_it_de_mixed_15000, ...
PROMPT_NAME="${2:-tiser_full}"
MAX_SAMPLES="${3:-2000}"    # optional

if [[ -z "$LANG" ]]; then
  echo "Usage:"
  echo "  bash multilingual_rag_tiser/training/run_train.sh en tiser_full 15000"
  echo "  bash multilingual_rag_tiser/training/run_train.sh it tiser_full 15000"
  echo "  bash multilingual_rag_tiser/training/run_train.sh de tiser_full 15000"
  echo "  bash multilingual_rag_tiser/training/run_train.sh fr tiser_full 15000"
  echo "  bash multilingual_rag_tiser/training/run_train.sh mixed tiser_full 15000"
  echo "  bash multilingual_rag_tiser/training/run_train.sh en_it_de_mixed_15000 tiser_full"
  echo ""
  echo "Env overrides: BASE_MODEL=Qwen/Qwen2.5-3B-Instruct  BALANCE=dataset|lang_dataset"
  exit 1
fi

# -------------------------
# Base model
# -------------------------
MODEL_NAME="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

# -------------------------
# Subsample balancing (per (lang, dataset) cell for mixed-language files)
# -------------------------
if [[ -z "${BALANCE:-}" ]]; then
  if [[ "$LANG" == *"mixed"* ]]; then BALANCE="lang_dataset"; else BALANCE="dataset"; fi
fi

# -------------------------
# Select train file
# -------------------------
# For single languages: prefer the quality-filtered "_passed" file, fall back
# to the raw translated split.
TRAIN_FILE=""

if [[ "$LANG" == "en" ]]; then
  # Frozen 15k snapshot = the exact data behind the reported EN numbers (docs/DATA.md §3).
  # Passing MAX_SAMPLES=15000 on a pre-balanced 15k file is content-preserving (returns the
  # same set, just reshuffled by seed=42). Fall back to the full split only if it is missing.
  if [[ -f "data/splits/train/train_tiser_15000_en.json" ]]; then
    TRAIN_FILE="data/splits/train/train_tiser_15000_en.json"
  else
    TRAIN_FILE="data/splits/train/TISER_train_en.json"
  fi

elif [[ "$LANG" == "it" || "$LANG" == "de" || "$LANG" == "fr" ]]; then
  if [[ -f "data/splits/train/${LANG}/TISER_train_${LANG}_passed.json" ]]; then
    TRAIN_FILE="data/splits/train/${LANG}/TISER_train_${LANG}_passed.json"
  else
    TRAIN_FILE="data/splits/train/TISER_train_${LANG}.json"
  fi

elif [[ "$LANG" == "mixed" ]]; then
  # Default mixed file based on MAX_SAMPLES
  if [[ -z "$MAX_SAMPLES" ]]; then
    echo "ERROR: for LANG=mixed you should pass MAX_SAMPLES."
    echo "Example:"
    echo "  bash multilingual_rag_tiser/training/run_train.sh mixed tiser_full 15000"
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
MAX_SAMPLES_ARG=()
if [[ -n "$MAX_SAMPLES" ]]; then
  MAX_SAMPLES_ARG=(--max_train_samples "$MAX_SAMPLES")
fi

# -------------------------
# Output dir
# -------------------------
SAMPLE_NAME="${MAX_SAMPLES:-full}"
OUT_DIR="experiments/qwen/${LANG}_${PROMPT_NAME}_${SAMPLE_NAME}_8gb_val_qlora"
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
echo "Model         : $MODEL_NAME"
echo "Prompt        : $PROMPT_NAME"
echo "Lang/Dataset  : $LANG"
echo "Train File    : $TRAIN_FILE"
echo "Output Dir    : $OUT_DIR"
echo "Max Samples   : ${MAX_SAMPLES:-all}  (balance: $BALANCE)"
echo "Max Length    : $MAX_LEN"
echo "LoRA Rank     : $LORA_R"
echo "LoRA Alpha    : $LORA_ALPHA"
echo "Batch Size    : $PER_DEVICE_BS"
echo "Grad Accum    : $GRAD_ACCUM"
echo "Validation    : $VAL_SPLIT"
echo "Eval/Save Steps: $EVAL_STEPS"
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
  --logging_steps 10 \
  --eval_steps "$EVAL_STEPS" \
  --validation_split "$VAL_SPLIT" \
  --dataloader_num_workers 0 \
  --gradient_checkpointing 1 \
  --gpu_memory_gb 7 \
  --cpu_memory_gb 16 \
  --balance "$BALANCE" \
  "${MAX_SAMPLES_ARG[@]}" \
  --only_passed \
  --prompt_name "$PROMPT_NAME"