#!/bin/bash
set -e
export PYTHONPATH="$(pwd):$PYTHONPATH"

MODEL_TYPE=$1                  # qwen or mistral
LANG=$2                        # it, fa, de, en
ADAPTER_DIR=${3:-none}         # adapter path, or "none" for base model
STRATEGY=${4:-base}            # base or iterative
PROMPT_NAME=${5:-tiser_full}   # tiser_full, tiser_compact, standard, answer_recovery
MAX_SAMPLES=${6:-}

if [[ -z "$MODEL_TYPE" || -z "$LANG" ]]; then
    echo "Usage: bash run_eval_pipeline.sh <qwen|mistral> <lang> [adapter_path|none] [base|iterative] [prompt_name] [max_samples]"
    echo ""
    echo "  Pass 'none' as adapter_path to evaluate the base model with no fine-tuning."
    exit 1
fi

# Select base model
if [[ "$MODEL_TYPE" == "qwen" ]]; then
    BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
else
    BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
fi

TEST_FILE="data/splits/test/TISER_test_${LANG}.json"
if [[ $LANG == "it" ]]; then
    TEST_FILE="data/splits/test/it/TISER_test_it_passed.json"
elif [[ "$LANG" == "fa" ]]; then
    TEST_FILE="data/splits/test/fa/TISER_test_fa.json"
elif [[ "$LANG" == "de" ]]; then
    TEST_FILE="data/splits/test/de/TISER_test_de.json"
fi
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Decide where to write results.
if [[ "$ADAPTER_DIR" == "none" ]]; then
    RESULTS_DIR="experiments/${MODEL_TYPE}_base/results"
    MODEL_NAME="base"
    ADAPTER_ARG=""
else
    RESULTS_DIR="${ADAPTER_DIR}/results"
    MODEL_NAME=$(basename "$ADAPTER_DIR")
    ADAPTER_ARG="--adapter_dir $ADAPTER_DIR"
fi

mkdir -p "$RESULTS_DIR"
GEN_OUTPUT="${RESULTS_DIR}/gen_${PROMPT_NAME}_${MODEL_NAME}_${STRATEGY}_${PROMPT_NAME}_${TIMESTAMP}.json"

MAX_SAMPLES_ARG=""
if [[ -n "$MAX_SAMPLES" ]]; then
    MAX_SAMPLES_ARG="--max_eval_samples $MAX_SAMPLES"
fi

echo "Model:    $BASE_MODEL"
echo "Adapter:  $ADAPTER_DIR"
echo "Lang:     $LANG"
echo "Strategy: $STRATEGY"
echo "Prompt:   $PROMPT_NAME"
echo "Output:   $GEN_OUTPUT"
echo "Test file: $TEST_FILE"
echo ""

python tiser_lite/evaluation/inference.py \
    --base_model "$BASE_MODEL" \
    --test_file "$TEST_FILE" \
    --output_file "$GEN_OUTPUT" \
    --max_new_tokens 512 \
    --batch_size 1 \
    --strategy "$STRATEGY" \
    --prompt_name "$PROMPT_NAME" \
    --only_passed \
    --max_extensions 2 \
    $ADAPTER_ARG \
    $MAX_SAMPLES_ARG

echo ""
echo "Generation complete: $GEN_OUTPUT"