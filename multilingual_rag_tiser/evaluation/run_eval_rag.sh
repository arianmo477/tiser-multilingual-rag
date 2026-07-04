#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

MODEL_TYPE="${1:-}"
LANG="${2:-}"
ADAPTER_DIR="${3:-none}"
STRATEGY="${4:-iterative}"
PROMPT_NAME="${5:-tiser_full}"
MAX_SAMPLES="${6:-500}"

if [[ -z "$MODEL_TYPE" || -z "$LANG" ]]; then
    cat <<EOF
Usage:
  bash run_eval_rag.sh <qwen|mistral> <en|it|de|fr|fa|mixed> \\
    [adapter|none] [base|iterative] [prompt] [max_samples]

Env overrides:
  USE_RAG=0|1          Toggle RAG (default 1)
  RAG_MODE=few_shot|context_stuffing   (default few_shot)
  RAG_TOP_K=1          Exemplars (1–2)
  RAG_MIN_SCORE=0.60   Cosine threshold (NOT 0.9)
  RAG_INDEX_DIR=...    Override index dir
  RAG_TRAIN_FILE=...   Override train file for index build
EOF
    exit 1
fi

# ---- RAG defaults ----
USE_RAG="${USE_RAG:-1}"
RAG_MODE="${RAG_MODE:-few_shot}"
RAG_TOP_K="${RAG_TOP_K:-1}"
RAG_MIN_SCORE="${RAG_MIN_SCORE:-0.60}"
RAG_EMBED_MODEL="${RAG_EMBED_MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"

# ---- Base model ----
case "$MODEL_TYPE" in
    qwen)    BASE_MODEL="Qwen/Qwen2.5-3B-Instruct" ;;
    mistral) BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2" ;;
    *) echo "ERROR: unknown model type: $MODEL_TYPE"; exit 1 ;;
esac

# ---- Test file ----
find_first() {
    for f in "$@"; do [[ -f "$f" ]] && { echo "$f"; return 0; }; done
    return 1
}

TEST_FILE="${TEST_FILE:-$(find_first \
    "data/splits/test/${LANG}/TISER_test_${LANG}_passed.json" \
    "data/splits/test/${LANG}/TISER_test_${LANG}.json" \
    "data/splits/test/TISER_test_${LANG}.json")}" || {
        echo "ERROR: no test file found for lang=$LANG"; exit 1;
    }

# ---- RAG train file ----
if [[ -z "${RAG_TRAIN_FILE:-}" ]]; then
    if [[ "$LANG" == "mixed" ]]; then
        RAG_TRAIN_FILE=$(find_first \
            "data/splits/train/TISER_train_de_it_fr_en_mixed.json" \
            "data/splits/train/TISER_train_en_it_de_mixed.json" \
            "data/splits/train/TISER_train_mixed.json") || true
    elif [[ "$LANG" == "it" ]]; then
        RAG_TRAIN_FILE=$(find_first \
            "data/splits/train/it/TISER_train_it_passed.json" \
            "data/splits/train/TISER_train_it.json") || true
    elif [[ "$LANG" == "en" ]]; then
        # Frozen 15k snapshot so the full 174MB TISER_train_en.json can be pruned (docs/DATA.md §3).
        RAG_TRAIN_FILE=$(find_first \
            "data/splits/train/train_tiser_15000_en.json" \
            "data/splits/train/TISER_train_en.json") || true
    else
        RAG_TRAIN_FILE="data/splits/train/TISER_train_${LANG}.json"
    fi
fi

RAG_INDEX_DIR="${RAG_INDEX_DIR:-data/rag/train_${LANG}}"

# ---- Output ----
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
if [[ "$ADAPTER_DIR" == "none" ]]; then
    RESULTS_DIR="experiments/${MODEL_TYPE}_base/results"
    MODEL_NAME="base"
    ADAPTER_ARG=()
else
    ADAPTER_CLEAN="${ADAPTER_DIR%/}"
    RESULTS_DIR="${ADAPTER_CLEAN}/results"
    MODEL_NAME="$(basename "$ADAPTER_CLEAN")"
    ADAPTER_ARG=(--adapter_dir "$ADAPTER_DIR")
fi
mkdir -p "$RESULTS_DIR"

RAG_SUFFIX=""
[[ "$USE_RAG" == "1" ]] && RAG_SUFFIX="_rag_${RAG_MODE}_k${RAG_TOP_K}_s${RAG_MIN_SCORE}"

GEN_OUTPUT="${RESULTS_DIR}/gen_${MODEL_NAME}_${LANG}_${STRATEGY}_${PROMPT_NAME}${RAG_SUFFIX}_${TIMESTAMP}.json"

MAX_SAMPLES_ARG=()
[[ -n "$MAX_SAMPLES" && "$MAX_SAMPLES" != "0" ]] && \
    MAX_SAMPLES_ARG=(--max_eval_samples "$MAX_SAMPLES")

SCRIPTS_DIR="multilingual_rag_tiser"
INFERENCE_SCRIPT="${SCRIPTS_DIR}/evaluation/inference.py"
BUILD_INDEX_SCRIPT="${SCRIPTS_DIR}/rag/build_rag_index.py"

# ---- Print config ----
cat <<EOF
======================================================================
TISER RAG Evaluation
======================================================================
Model:       $BASE_MODEL
Adapter:     $ADAPTER_DIR
Language:    $LANG
Strategy:    $STRATEGY
Prompt:      $PROMPT_NAME
Max samples: $MAX_SAMPLES
Test file:   $TEST_FILE
Output:      $GEN_OUTPUT

RAG enabled: $USE_RAG
EOF
if [[ "$USE_RAG" == "1" ]]; then
    cat <<EOF
RAG mode:    $RAG_MODE
RAG train:   ${RAG_TRAIN_FILE:-<none>}
RAG index:   $RAG_INDEX_DIR
RAG top-k:   $RAG_TOP_K
RAG score:   $RAG_MIN_SCORE  (cosine; 0.9 is too high)
RAG model:   $RAG_EMBED_MODEL
EOF
fi
echo "======================================================================"

# ---- Safety checks ----
[[ -f "$INFERENCE_SCRIPT" ]] || { echo "ERROR: missing $INFERENCE_SCRIPT"; exit 1; }
[[ -f "$TEST_FILE" ]] || { echo "ERROR: missing $TEST_FILE"; exit 1; }

# ---- Build RAG index if needed ----
RAG_ARG=()
if [[ "$USE_RAG" == "1" ]]; then
    if [[ ! -f "${RAG_INDEX_DIR}/index.faiss" || ! -f "${RAG_INDEX_DIR}/documents.json" ]]; then
        [[ -f "${RAG_TRAIN_FILE:-}" ]] || {
            echo "ERROR: RAG index missing and RAG_TRAIN_FILE not set/found."
            exit 1
        }
        echo ""
        echo "Building RAG index from: $RAG_TRAIN_FILE"

        LANG_ARG=()
        [[ "$LANG" != "mixed" ]] && LANG_ARG=(--language "$LANG")

        python "$BUILD_INDEX_SCRIPT" \
            --input "$RAG_TRAIN_FILE" \
            --output_dir "$RAG_INDEX_DIR" \
            --model_name "$RAG_EMBED_MODEL" \
            "${LANG_ARG[@]}"
    else
        echo "RAG index exists: $RAG_INDEX_DIR"
    fi

    RAG_ARG=(
        --use_rag
        --rag_index_dir "$RAG_INDEX_DIR"
        --rag_top_k "$RAG_TOP_K"
        --rag_min_score "$RAG_MIN_SCORE"
        --rag_model_name "$RAG_EMBED_MODEL"
        --rag_mode "$RAG_MODE"
    )
fi

# ---- Run inference ----
python "$INFERENCE_SCRIPT" \
    --base_model "$BASE_MODEL" \
    --test_file "$TEST_FILE" \
    --output_file "$GEN_OUTPUT" \
    --max_new_tokens 768 \
    --batch_size 1 \
    --strategy "$STRATEGY" \
    --prompt_name "$PROMPT_NAME" \
    --only_passed \
    --max_extensions 2 \
    "${ADAPTER_ARG[@]}" \
    "${MAX_SAMPLES_ARG[@]}" \
    "${RAG_ARG[@]}"

echo ""
echo "Generation complete: $GEN_OUTPUT"