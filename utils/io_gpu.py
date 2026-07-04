import gc
import hashlib
import json
import logging
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from functools import lru_cache
from pathlib import Path
import torch

log = logging.getLogger(__name__)

# ==================================================
# GPU / MEMORY
# ==================================================

def verify_gpu():
    print("===== GPU CHECK =====")
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Training will fail unless on CPU mode.")
    else:
        print("GPU:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print("VRAM (GB):", round(props.total_memory / (1024**3), 2))
    print("=====================")


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================================================
# IO
# ==================================================

def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.stat().st_size == 0:
        return None

    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        # Normal JSON list or dict
        if first in ["[", "{"]:
            return json.load(f)

        # JSONL fallback
        return [json.loads(line) for line in f if line.strip()]

def _strip_invalid_unicode(obj):
    """Recursively removes invalid unicode surrogates from strings."""
    if isinstance(obj, str):
        return obj.encode("utf-8", "ignore").decode("utf-8")
    elif isinstance(obj, list):
        return [_strip_invalid_unicode(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _strip_invalid_unicode(v) for k, v in obj.items()}
    return obj

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    clean_data = _strip_invalid_unicode(data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2, ensure_ascii=False)

def save_json_atomic(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    tmp = p.with_suffix(p.suffix + ".tmp")
    clean_obj = _strip_invalid_unicode(obj)

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean_obj, f, ensure_ascii=False, indent=2)

    tmp.replace(p)


def load_txt_as_string(path: str, fallback: str = "") -> str:
    if not os.path.exists(path):
        log.warning("Prompt file not found at %s. Using fallback.", path)
        return fallback
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content if content else fallback
    except Exception as e:
        log.error("Error reading prompt file: %s. Using fallback.", e)
        return fallback



def balance_by_dataset_name(data, category, max_samples, seed=42):
    random.seed(seed)
    buckets = defaultdict(list)
    for x in data:
        if category == "test" and x["dataset_name"] == "tot_semantic_test":
            continue
        buckets[x["dataset_name"]].append(x)

    names = list(buckets.keys())
    base = max_samples // len(names)
    selected = []
    leftovers = []

    for name in names:
        bucket = buckets[name]
        if len(bucket) <= base:
            # Whole bucket is used; nothing is left over for the top-up pool.
            selected.extend(bucket)
        else:
            # Sample by index so the chosen items are excluded from leftovers
            # exactly once (avoids re-adding a selected sample -> duplicates).
            chosen = set(random.sample(range(len(bucket)), base))
            selected.extend(bucket[i] for i in chosen)
            leftovers.extend(bucket[i] for i in range(len(bucket)) if i not in chosen)

    # Top up toward max_samples from items NOT already selected (no overlap).
    remaining = max_samples - len(selected)
    if remaining > 0 and leftovers:
        selected.extend(random.sample(leftovers, min(remaining, len(leftovers))))

    random.shuffle(selected)
    return selected




@lru_cache(maxsize=16)
def load_prompt_for_lang(prompt_name, lang):
    """Per-language prompt template, falling back to base English if the
    language-specific file is missing. Cached so we don't reread on every
    sample."""
    specific = Path(f"data/prompts/{prompt_name}_{lang}.txt")
    fallback = Path(f"data/prompts/{prompt_name}.txt")
    path = specific if specific.exists() else fallback
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt template not found: tried {specific} and {fallback}"
        )
    return load_txt_as_string(str(path)).strip()


# ==================================================
# STATS / HASHING  (moved from utils.py)
# ==================================================

def save_stats(stats_path: str | Path, stats_dict: dict) -> None:
    p = Path(stats_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, ensure_ascii=False, indent=2)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def language_counts(data: list[dict]) -> dict[str, int]:
    return dict(Counter(s.get("language", "unknown") for s in data))