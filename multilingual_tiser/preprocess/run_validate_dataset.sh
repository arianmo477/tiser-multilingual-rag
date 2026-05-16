#!/bin/bash
set -e

export PYTHONPATH="$(pwd):$PYTHONPATH"

category=${1:-train}  # train, dev, test
prompt_name=${2:-tiser_full}  # tiser_full, tiser_compact, standard, answer_recovery
shift || true  # drop the first arg so $@ contains only the extra flags

INPUT="data/TISER_${category}.json"
OUTPUT="data/splits/${category}/TISER_${category}_en.json"
INVALID_OUT="data/invalid_samples_${category}.json"

echo " Validating TISER dataset..."
echo " Input : $INPUT"
echo " Invalid samples (if any): $INVALID_OUT"
echo " Prompt        : $prompt_name"
echo " Extra flags: $@"
echo ""

python multilingual_tiser/preprocess/validate_tiser_dataset.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --save_invalid "$INVALID_OUT" \
    --split "$category" \
    --prompt_file "data/prompts/${prompt_name}.txt" \
    "$@"

echo ""
echo "Dataset validation completed."