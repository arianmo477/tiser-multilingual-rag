#!/usr/bin/env bash
#
# Experiment 1 — RAG ablation: base vs fine-tuned, RAG on/off.
#
# Runs the four cells on the SAME test samples and aggregates them:
#     base_norag   base model,       RAG off
#     base_rag     base model,       RAG on  (few-shot exemplars)
#     ft_norag     fine-tuned model, RAG off
#     ft_rag       fine-tuned model, RAG on
#
# Hypothesis: RAG helps the base model (supplies the reasoning format it never
# learned) but adds little to the fine-tuned model — retrieval and fine-tuning
# are substitutes, not complements.
#
# Usage:
#   bash multilingual_rag_tiser/evaluation/run_experiment1.sh \
#       <qwen|mistral> <en|it|de|fr|mixed> <finetuned_adapter_dir> [max_samples]
#
# Env overrides:
#   PROMPT_NAME=tiser_full   STRATEGY=iterative
#   RAG_MODE=few_shot        RAG_TOP_K=1        RAG_MIN_SCORE=0.60
#   TEST_FILE=...            RAG_TRAIN_FILE=...  RAG_INDEX_DIR=...
#   RAG_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
#   OUT_DIR=experiments/experiment1_rag_ablation/<lang>
#   SKIP_RUN=1               Only re-aggregate existing outputs (no inference).
#   REBUILD_INDEX=1          Force-rebuild the RAG index from RAG_TRAIN_FILE even if one exists
#                            (use after rewiring the source, e.g. to the 15k EN snapshot).

set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

MODEL_TYPE="${1:-}"
LANG="${2:-}"
ADAPTER_DIR="${3:-}"
MAX_SAMPLES="${4:-500}"

if [[ -z "$MODEL_TYPE" || -z "$LANG" || -z "$ADAPTER_DIR" ]]; then
    cat <<EOF
Usage:
  bash run_experiment1.sh <qwen|mistral> <en|it|de|fr|mixed> \\
    <finetuned_adapter_dir> [max_samples]

Runs 4 cells (base/ft x rag/norag) on the same samples and prints a comparison.
EOF
    exit 1
fi

PROMPT_NAME="${PROMPT_NAME:-tiser_full}"
STRATEGY="${STRATEGY:-iterative}"
RAG_MODE="${RAG_MODE:-few_shot}"
RAG_TOP_K="${RAG_TOP_K:-1}"
RAG_MIN_SCORE="${RAG_MIN_SCORE:-0.60}"
RAG_EMBED_MODEL="${RAG_EMBED_MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"
SKIP_RUN="${SKIP_RUN:-0}"
REBUILD_INDEX="${REBUILD_INDEX:-0}"

case "$MODEL_TYPE" in
    qwen)    BASE_MODEL="Qwen/Qwen2.5-3B-Instruct" ;;
    mistral) BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2" ;;
    *) echo "ERROR: unknown model type: $MODEL_TYPE"; exit 1 ;;
esac

find_first() {
    for f in "$@"; do [[ -f "$f" ]] && { echo "$f"; return 0; }; done
    return 1
}

# ---- Test file (same file for all 4 cells → identical samples) ----
TEST_FILE="${TEST_FILE:-$(find_first \
    "data/splits/test/${LANG}/TISER_test_${LANG}_passed.json" \
    "data/splits/test/${LANG}/TISER_test_${LANG}.json" \
    "data/splits/test/TISER_test_${LANG}.json")}" || {
        echo "ERROR: no test file found for lang=$LANG"; exit 1;
    }

# ---- RAG train file + index dir ----
if [[ -z "${RAG_TRAIN_FILE:-}" ]]; then
    if [[ "$LANG" == "mixed" ]]; then
        RAG_TRAIN_FILE=$(find_first \
            "data/splits/train/TISER_train_de_it_fr_en_mixed.json" \
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

OUT_DIR="${OUT_DIR:-experiments/experiment1_rag_ablation/${LANG}}"
mkdir -p "$OUT_DIR"

INFERENCE_SCRIPT="multilingual_rag_tiser/evaluation/inference.py"
BUILD_INDEX_SCRIPT="multilingual_rag_tiser/rag/build_rag_index.py"
COMPARE_SCRIPT="multilingual_rag_tiser/evaluation/compare_rag_ablation.py"

BASE_NORAG="${OUT_DIR}/gen_base_norag.json"
BASE_RAG="${OUT_DIR}/gen_base_rag.json"
FT_NORAG="${OUT_DIR}/gen_ft_norag.json"
FT_RAG="${OUT_DIR}/gen_ft_rag.json"

cat <<EOF
======================================================================
Experiment 1 — RAG ablation
======================================================================
Base model:  $BASE_MODEL
Adapter:     $ADAPTER_DIR
Language:    $LANG
Test file:   $TEST_FILE
Max samples: $MAX_SAMPLES
Prompt:      $PROMPT_NAME     Strategy: $STRATEGY
RAG:         mode=$RAG_MODE  top_k=$RAG_TOP_K  min_score=$RAG_MIN_SCORE
RAG index:   $RAG_INDEX_DIR
Output dir:  $OUT_DIR
======================================================================
EOF

[[ -f "$INFERENCE_SCRIPT" ]] || { echo "ERROR: missing $INFERENCE_SCRIPT"; exit 1; }
[[ -d "$ADAPTER_DIR" ]] || { echo "ERROR: adapter dir not found: $ADAPTER_DIR"; exit 1; }

# ---- Build RAG index once (shared by both RAG cells) ----
if [[ "$SKIP_RUN" != "1" ]]; then
    # Rebuild when: forced, or either artifact is missing. Forcing removes any stale index so
    # the new one is built from the current RAG_TRAIN_FILE (e.g. the 15k EN snapshot).
    if [[ "$REBUILD_INDEX" == "1" ]]; then
        echo "REBUILD_INDEX=1 → removing any stale index at $RAG_INDEX_DIR"
        rm -f "${RAG_INDEX_DIR}/index.faiss" "${RAG_INDEX_DIR}/documents.json"
    fi

    if [[ ! -f "${RAG_INDEX_DIR}/index.faiss" || ! -f "${RAG_INDEX_DIR}/documents.json" ]]; then
        [[ -f "${RAG_TRAIN_FILE:-}" ]] || {
            echo "ERROR: RAG index missing and RAG_TRAIN_FILE not found: ${RAG_TRAIN_FILE:-<unset>}"
            exit 1
        }
        echo "Building RAG index from: $RAG_TRAIN_FILE"
        LANG_ARG=()
        [[ "$LANG" != "mixed" ]] && LANG_ARG=(--language "$LANG")
        python "$BUILD_INDEX_SCRIPT" \
            --input "$RAG_TRAIN_FILE" \
            --output_dir "$RAG_INDEX_DIR" \
            --model_name "$RAG_EMBED_MODEL" \
            "${LANG_ARG[@]}"
    else
        echo "RAG index exists: $RAG_INDEX_DIR (set REBUILD_INDEX=1 to force rebuild)"
    fi
fi

MAX_SAMPLES_ARG=()
[[ -n "$MAX_SAMPLES" && "$MAX_SAMPLES" != "0" ]] && \
    MAX_SAMPLES_ARG=(--max_eval_samples "$MAX_SAMPLES")

RAG_ARG=(
    --use_rag
    --rag_index_dir "$RAG_INDEX_DIR"
    --rag_top_k "$RAG_TOP_K"
    --rag_min_score "$RAG_MIN_SCORE"
    --rag_model_name "$RAG_EMBED_MODEL"
    --rag_mode "$RAG_MODE"
)

# run_cell <output_file> <adapter_flag: use|none> <rag_flag: use|none>
run_cell() {
    local out_file="$1" adapter_flag="$2" rag_flag="$3"
    local adapter_arg=() rag_arg=()
    [[ "$adapter_flag" == "use" ]] && adapter_arg=(--adapter_dir "$ADAPTER_DIR")
    [[ "$rag_flag" == "use" ]] && rag_arg=("${RAG_ARG[@]}")

    echo ""
    echo ">>> CELL -> $out_file  (adapter=$adapter_flag, rag=$rag_flag)"
    python "$INFERENCE_SCRIPT" \
        --base_model "$BASE_MODEL" \
        --test_file "$TEST_FILE" \
        --output_file "$out_file" \
        --max_new_tokens 768 \
        --batch_size 1 \
        --strategy "$STRATEGY" \
        --prompt_name "$PROMPT_NAME" \
        --only_passed \
        --max_extensions 2 \
        "${adapter_arg[@]}" \
        "${MAX_SAMPLES_ARG[@]}" \
        "${rag_arg[@]}"
}

if [[ "$SKIP_RUN" != "1" ]]; then
    run_cell "$BASE_NORAG" none none
    run_cell "$BASE_RAG"   none use
    run_cell "$FT_NORAG"   use  none
    run_cell "$FT_RAG"     use  use
else
    echo "SKIP_RUN=1 → skipping inference, aggregating existing outputs."
fi

# ---- Aggregate ----
echo ""
echo "Aggregating 4 cells..."
python "$COMPARE_SCRIPT" \
    "base_norag=$BASE_NORAG" \
    "base_rag=$BASE_RAG" \
    "ft_norag=$FT_NORAG" \
    "ft_rag=$FT_RAG" \
    --per_dataset \
    --out "${OUT_DIR}/summary.json"
