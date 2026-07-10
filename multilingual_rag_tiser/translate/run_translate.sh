#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CATEGORY="${1:-}"
LANG="${2:-it}"
MAX_SAMPLES="${3:-0}"

if [[ -z "$CATEGORY" ]]; then
    echo "Usage: bash multilingual_rag_tiser/translate/run_translate.sh <category> <lang> [max_samples]"
    echo "  category: train | val | test"
    echo "  lang:     it | de | fr"
    exit 1
fi

case "$LANG" in
    it|de|fr) ;;
    *) echo "ERROR: unsupported lang '$LANG' (use it|de|fr)"; exit 1 ;;
esac

INPUT="data/splits/${CATEGORY}/TISER_${CATEGORY}_en.json"
OUTPUT="data/splits/${CATEGORY}/TISER_${CATEGORY}_${LANG}.json"
CACHE="data/splits/${CATEGORY}/event_translation_cache_${LANG}.json"
SCRIPT="multilingual_rag_tiser/translate/translate_dataset.py"

echo "========================================"
echo "TISER Translation EN -> ${LANG^^}"
echo "Category:    $CATEGORY"
echo "Lang:        $LANG"
echo "Input:       $INPUT"
echo "Output:      $OUTPUT"
echo "Max samples: $MAX_SAMPLES"
echo "Cache:       $CACHE"
echo "========================================"

[[ ! -f "$SCRIPT" ]] && { echo "ERROR: script not found: $SCRIPT"; exit 1; }
[[ ! -f "$INPUT" ]]  && { echo "ERROR: input not found: $INPUT"; exit 1; }

CMD=(python "$SCRIPT"
    --input "$INPUT"
    --output "$OUTPUT"
    --target_lang "$LANG"
    --cache "$CACHE"
    --category "$CATEGORY"
)

if [[ "$MAX_SAMPLES" != "0" ]]; then
    CMD+=(--max_samples "$MAX_SAMPLES")
fi

"${CMD[@]}"

echo "========================================"
echo "Done. Wrote: $OUTPUT"
echo "========================================"