#!/usr/bin/env bash
set -e

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# -------------------------
# Mix TISER language datasets
# -------------------------
CATEGORY="${1:-train}"
LANGS="${2:-en,it,de}"
MAX_SAMPLES="${3:-15000}"
OUTPUT_NAME="${4:-}"

SCRIPT="multilingual_tiser/preprocess/mix_dataset.py"

if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: script not found: $SCRIPT"
  exit 1
fi

IFS=',' read -ra LANG_ARRAY <<< "$LANGS"

if [[ "${#LANG_ARRAY[@]}" -lt 2 ]]; then
  echo "ERROR: provide at least two languages."
  echo "Example: bash multilingual_tiser/preprocess/run_mix_dataset.sh train en,it,de 15000"
  exit 1
fi

if [[ "${#LANG_ARRAY[@]}" -gt 4 ]]; then
  echo "ERROR: mix_dataset.py supports up to 4 languages (--dataset1/2/3/4)."
  exit 1
fi

DATASET_ARGS=()
NAME_PARTS=()

for i in "${!LANG_ARRAY[@]}"; do
  LANG="${LANG_ARRAY[$i]}"

  DATASET_PATH="data/splits/${CATEGORY}/TISER_${CATEGORY}_${LANG}.json"

  if [[ "$LANG" != "en" ]]; then
    # Prefer the audited "passed" file if it exists.
    PASSED_PATH="data/splits/${CATEGORY}/${LANG}/TISER_${CATEGORY}_${LANG}_passed.json"
    if [[ -f "$PASSED_PATH" ]]; then
      DATASET_PATH="$PASSED_PATH"
    fi
  fi

  if [[ ! -f "$DATASET_PATH" ]]; then
    echo "ERROR: missing dataset: $DATASET_PATH"
    exit 1
  fi

  ARG_NUM=$((i + 1))
  DATASET_ARGS+=("--dataset${ARG_NUM}" "$DATASET_PATH")
  NAME_PARTS+=("$LANG")
done

if [[ -z "$OUTPUT_NAME" ]]; then
  LANG_JOINED=$(IFS=_; echo "${NAME_PARTS[*]}")
  OUTPUT="data/splits/${CATEGORY}/TISER_${CATEGORY}_${LANG_JOINED}_mixed.json"
else
  OUTPUT="$OUTPUT_NAME"
fi

echo "========================================"
echo "Mixing TISER datasets (aligned)"
echo "Category:     $CATEGORY"
echo "Languages:    $LANGS"
echo "Max samples:  $MAX_SAMPLES"
echo "Output:       $OUTPUT"
echo "Script:       $SCRIPT"
echo "========================================"

python "$SCRIPT" \
  --category "$CATEGORY" \
  "${DATASET_ARGS[@]}" \
  --max_samples "$MAX_SAMPLES" \
  --output "$OUTPUT"

echo "========================================"
echo "Done."
echo "Mixed dataset written to:"
echo "$OUTPUT"
echo "========================================"