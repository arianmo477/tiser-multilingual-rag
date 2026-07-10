#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
# -------------------------
# TISER translation quality scorer
# -------------------------
# Usage:
#   bash multilingual_rag_tiser/translate/run_score.sh <category> <lang> [threshold]
#
# Examples:
#   bash multilingual_rag_tiser/translate/run_score.sh train it
#   bash multilingual_rag_tiser/translate/run_score.sh train de 0.94
#   bash multilingual_rag_tiser/translate/run_score.sh val   fr 0.95
# -------------------------

CATEGORY="${1:-}"
LANG="${2:-it}"
THRESHOLD="${3:-0.94}"

if [[ -z "$CATEGORY" ]]; then
    echo "Usage: bash multilingual_rag_tiser/translate/run_score.sh <category> <lang> [threshold]"
    echo "  category:  train | val | test"
    echo "  lang:      it | de | fr  (default: it)"
    echo "  threshold: pass threshold (default: 0.94)"
    exit 1
fi

case "$LANG" in
    it|de|fr) ;;
    *)
        echo "ERROR: unsupported lang '$LANG' (must be it|de|fr)"
        exit 1
        ;;
esac

INPUT="data/splits/${CATEGORY}/TISER_${CATEGORY}_${LANG}.json"
REPORT="data/splits/${CATEGORY}/${LANG}/translation_quality_report.json"
PASSED_OUT="data/splits/${CATEGORY}/${LANG}/TISER_${CATEGORY}_${LANG}_passed.json"
FAILED_OUT="data/splits/${CATEGORY}/${LANG}/TISER_${CATEGORY}_${LANG}_failed.json"
SCRIPT="multilingual_rag_tiser/translate/score_translation.py"

echo "========================================"
echo "TISER Translation Quality Scoring"
echo "Category:   $CATEGORY"
echo "Lang:       $LANG"
echo "Input:      $INPUT"
echo "Report:     $REPORT"
echo "Passed out: $PASSED_OUT"
echo "Failed out: $FAILED_OUT"
echo "Threshold:  $THRESHOLD"
echo "Script:     $SCRIPT"
echo "========================================"

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: scorer script not found: $SCRIPT"
    exit 1
fi

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: input file not found: $INPUT"
    echo
    echo "Available translated files for category '$CATEGORY' (lang '$LANG'):"
    ls -lh "data/splits/${CATEGORY}"/*"${LANG}"*.json 2>/dev/null || true
    echo
    echo "Available translated files in train/val/test:"
    ls -lh data/splits/{train,val,test}/*"${LANG}"*.json 2>/dev/null || true
    exit 1
fi

mkdir -p "data/splits/${CATEGORY}/${LANG}"

python "$SCRIPT" \
    --input "$INPUT" \
    --report "$REPORT" \
    --passed_out "$PASSED_OUT" \
    --failed_out "$FAILED_OUT" \
    --threshold "$THRESHOLD"

echo "========================================"
echo "Done scoring."
echo "Report: $REPORT"
echo "Passed: $PASSED_OUT"
echo "Failed: $FAILED_OUT"
echo "========================================"