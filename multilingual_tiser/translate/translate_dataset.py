#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import logging as hf_logging

from utils.io_gpu import (
    balance_by_dataset_name,
    load_json,
    save_json_atomic,
)
from utils.translation_utils import *


hf_logging.set_verbosity_error()

LANG = {"it": "ita_Latn", "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn", "fa": "pes_Arab"}
SRC = "eng_Latn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# =============================================================================
# Resume / sample identity
# =============================================================================

def sample_key(s):
    qid = str(s.get("question_id", "") or "").strip()
    if qid:
        return qid
    return f"{s.get('dataset_name', '')}::{s.get('question', '')[:200]}"


def load_existing_translations(path):
    p = Path(path)
    if not p.exists():
        return [], set()
    with open(p, encoding="utf-8") as f:
        existing = json.load(f)
    done = {sample_key(s) for s in existing}
    print(f"Resume: found {len(existing)} existing translated samples.")
    return existing, done


def choose_new_samples(data, done_ids, max_samples, category):
    remaining = [s for s in data if sample_key(s) not in done_ids]
    print(f"Total input samples:    {len(data)}")
    print(f"Already translated:     {len(done_ids)}")
    print(f"Remaining untranslated: {len(remaining)}")
    if max_samples > 0:
        remaining = balance_by_dataset_name(
            data=remaining,
            category=category,
            max_samples=max_samples,
        )
    print(f"Selected this run:      {len(remaining)}")
    return remaining


def dedupe_preserve_order(samples):
    out, seen = [], set()
    for s in samples:
        k = sample_key(s)
        if k not in seen:
            out.append(s)
            seen.add(k)
    return out


# =============================================================================
# NLLB primitive
# =============================================================================

def nllb(texts, tok, mdl, lang, batch=8, max_new=256, desc="NLLB", show_progress=False):
    fid = tok.convert_tokens_to_ids(LANG[lang])
    out = [""] * len(texts)
    valid = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    steps = range(0, len(valid), batch)

    if show_progress:
        steps = tqdm(
            steps,
            total=(len(valid) + batch - 1) // batch,
            desc=desc, unit="batch", dynamic_ncols=True, leave=True,
        )

    for s in steps:
        idx, chunk = zip(*valid[s:s + batch])
        inp = tok(
            list(chunk), return_tensors="pt", padding=True,
            truncation=True, max_length=1024,
        ).to(DEVICE)

        with torch.inference_mode():
            ids = mdl.generate(
                **inp, forced_bos_token_id=fid,
                max_new_tokens=max_new, num_beams=2,
            )

        for i, d in zip(idx, tok.batch_decode(ids, skip_special_tokens=True)):
            out[i] = clean(d)

    return out


# =============================================================================
# Event cache
# =============================================================================

def all_events(data):
    evs = []
    for s in data:
        for f in ("question", "temporal_context"):
            text = s.get(f, "") or ""
            evs += [unparen(text[a:b]) for a, b in parens(text)]
        ans = str(s.get("answer", "") or "").strip()
        if ans and not ans.isdigit() and ans.lower() not in BOOLS:
            evs.append(ans)
    return evs


def translate_missing(items, cache, tok, mdl, lang, batch):
    """Translate any entities not already in the cache.

    Three layers of defense:
      1. Proper-noun shortcut (no NLLB call, identity-pass)
      2. Carrier-prefix stripping in clean()
      3. Hallucination detection — fall back to source string if NLLB
         elaborated a short entity into a sentence.
    """
    miss, seen = [], set()
    pass_through = 0

    for x in items:
        x = str(x or "").strip()
        if not x or x in cache or x in seen:
            continue
        seen.add(x)
        if is_proper_noun(x):
            cache[x] = x
            pass_through += 1
            continue
        miss.append(x)

    if pass_through:
        print(f"Event cache: passed through {pass_through} proper-noun entities unchanged.")

    if not miss:
        print("Event cache: no new events to translate.")
        return cache

    print(f"Event cache: translating {len(miss)} new unique events.")
    src = [f"Translate this text: {x}" for x in miss]
    trs = nllb(
        src, tok, mdl, lang,
        batch=batch, max_new=256,
        desc="Translating events", show_progress=True,
    )

    rejected = 0
    for src_text, tr_text in zip(miss, trs):
        cleaned = clean(tr_text) if tr_text else ""
        if not cleaned or looks_hallucinated(cleaned, src_text):
            cache[src_text] = src_text
            rejected += 1
        else:
            cache[src_text] = cleaned

    if rejected:
        print(f"Event cache: rejected {rejected} hallucinated outputs (kept source).")

    return cache


# =============================================================================
# Context translation
# =============================================================================

def translate_context_text(text, tok, mdl, lang, batch):
    text = (text or "").strip()
    if not text:
        return text

    res = LANG_RESOURCES[lang]
    text = fix_months(text, lang)
    text = re.sub(r"\bstarts at\b|\bstarts in\b", res["starts_at"], text, flags=re.I)
    text = re.sub(r"\bends at\b|\bends in\b",     res["ends_at"],   text, flags=re.I)
    text = re.sub(r"\bstarts\b", res["starts"], text, flags=re.I)
    text = re.sub(r"\bends\b",   res["ends"],   text, flags=re.I)

    for a, b in res["possessives"]:
        text = re.sub(a, b, text)

    for pat, tpl in res["templates"]:
        m = re.match(pat, text)
        if m:
            return tpl.format(*m.groups())

    if re.search(r"\b(is|are|from|to|plays for|education|position|team|works for)\b", text, re.I):
        return nllb([text], tok, mdl, lang, batch=batch, max_new=256)[0]

    return text


def translate_context(ctx, cache, tok, mdl, lang, batch):
    ctx = ctx or ""
    if not ctx.strip():
        return ""

    spans = parens(ctx)

    if not spans:
        sentences = re.split(r"(?<=\.)\s+", ctx)
        text = " ".join(
            translate_context_text(s, tok, mdl, lang, batch)
            for s in sentences if s.strip()
        )
        return re.sub(r"\s+", " ", re.sub(r"\.\.+", ".", text)).strip()

    parts, last = [], 0
    for a, b in spans:
        before = translate_context_text(ctx[last:a], tok, mdl, lang, batch)
        inside = unparen(ctx[a:b]).strip()
        if before:
            parts.append(before)
        parts.append(f"({cache.get(inside, inside)})")
        last = b

    tail = translate_context_text(ctx[last:], tok, mdl, lang, batch)
    if tail:
        parts.append(tail)

    text = " ".join(parts)
    text = re.sub(r"\s+([.,:;])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\.\.+", ".", text)
    return re.sub(r"\s+", " ", text).strip()


# =============================================================================
# Question / answer translation
# =============================================================================

def translate_question(q, cache, tok, mdl, lang):
    masked, evs = mask_events(q)
    tr = nllb([masked], tok, mdl, lang, batch=1, max_new=512)[0]

    if all(PH.format(i) in tr for i in range(len(evs))):
        return restore(tr, evs, cache)

    repl = LANG_RESOURCES[lang]["question_fallbacks"]
    tr = q
    for a, b in repl.items():
        tr = tr.replace(a, b)

    return re.sub(r"\s+", " ", restore(*mask_events(tr), cache)).strip()


def translate_answer(ans, cache, tok, mdl, lang, batch):
    ans = str(ans or "").strip()
    if not ans or ans.isdigit():
        return ans
    if ans.lower() in BOOLS:
        return localize_boolean(ans, lang)
    if ans in cache:
        return cache[ans]
    # Proper-noun shortcut: skip NLLB entirely for clear named entities.
    if is_proper_noun(ans):
        cache[ans] = ans
        return ans

    tmp = {}
    translate_missing([ans], tmp, tok, mdl, lang, batch)
    if ans in tmp:
        cache[ans] = tmp[ans]
    return tmp.get(ans, ans)


# =============================================================================
# Output / structured trace translation
# =============================================================================

def split_tags(text):
    parts, last = [], 0
    text = text or ""
    for m in TAG_RE.finditer(text):
        if m.start() > last:
            parts.append(("text", text[last:m.start()]))
        parts.append(("tag", m.group(0)))
        last = m.end()
    if last < len(text):
        parts.append(("text", text[last:]))
    return parts


def translate_output_line(line, cache, tok, mdl, lang, batch):
    if not line.strip():
        return line

    for pat in [
        r"^(\s*\d+\.\s*)(.*?)(\s*\(\d{3,4}\)\s*)$",
        r"^(\s*)(.*?)(\s*\(\d{3,4}\)\s*)$",
    ]:
        m = re.match(pat, line)
        if m:
            prefix, event_text, year = m.groups()
            event_text = event_text.strip()
            event_native = cache.get(event_text)
            if event_native is None:
                if is_proper_noun(event_text):
                    event_native = event_text
                else:
                    event_native = nllb(
                        [event_text], tok, mdl, lang,
                        batch=batch, max_new=256,
                    )[0]
                    if looks_hallucinated(event_native, event_text):
                        event_native = event_text
                cache[event_text] = event_native
            return f"{prefix}{event_native} {year.strip()}"

    spans = parens(line)
    if not spans:
        return nllb([line], tok, mdl, lang, batch=batch, max_new=768)[0]

    protected, text = [], line
    for i, (a, b) in enumerate(reversed(spans)):
        original = line[a:b]
        ph = f"PROT{i}PROT"
        protected.append((ph, original))
        text = text[:a] + ph + text[b:]

    tr = nllb([text], tok, mdl, lang, batch=batch, max_new=768)[0]

    for ph, original in protected:
        inside = unparen(original).strip()
        tr = tr.replace(
            ph,
            f"({cache[inside]})" if inside in cache else original,
        )

    return tr


def translate_output(output_en, answer_native, cache, tok, mdl, lang, batch):
    rebuilt = []
    for kind, text in split_tags(output_en):
        if kind == "tag" or not text.strip():
            rebuilt.append(text)
            continue

        lines = text.splitlines(keepends=True)
        translated_lines = []
        for line in lines:
            newline = "\n" if line.endswith("\n") else ""
            line = line[:-1] if newline else line
            translated_lines.append(
                translate_output_line(line, cache, tok, mdl, lang, batch) + newline
            )
        rebuilt.append("".join(translated_lines))

    text = "".join(rebuilt)

    text = re.sub(
        r"(<answer>\s*)(.*?)(\s*</answer>)",
        lambda m: f"{m.group(1)}{answer_native}{m.group(3)}",
        text, flags=re.I | re.S,
    )

    text = re.sub(r"\s+\((\d{3,4})\)", r" (\1)", text)
    text = re.sub(r"\)(?=\S)", ") ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# =============================================================================
# Per-sample driver
# =============================================================================

def translate_sample(s, cache, tok, mdl, lang, batch, category):
    q_en = s.get("question", "")
    ctx_en = s.get("temporal_context", "")
    ans_en = str(s.get("answer", "") or "")
    output_en = s.get("output", "")

    ctx_native = translate_context(ctx_en, cache, tok, mdl, lang, batch)
    ans_native = translate_answer(ans_en, cache, tok, mdl, lang, batch)

    t = dict(s)
    if category == "train":
        t.update({
            "question_en": q_en,
            "temporal_context_en": ctx_en,
            "answer_en": ans_en,
            "output_en": output_en,
            "language": lang,
            "question": translate_question(q_en, cache, tok, mdl, lang),
            "temporal_context": ctx_native,
            "answer": ans_native,
            "output": translate_output(
                output_en, ans_native, cache, tok, mdl, lang, batch,
            ),
        })
    else:
        t.update({
            "question_en": q_en,
            "temporal_context_en": ctx_en,
            "answer_en": ans_en,
            "language": lang,
            "question": translate_question(q_en, cache, tok, mdl, lang),
            "temporal_context": ctx_native,
            "answer": ans_native,
        })
    return t


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target_lang", default="it", choices=list(LANG))
    ap.add_argument("--model_name", default="facebook/nllb-200-distilled-1.3B")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument(
        "--cache", default=None,
        help="Path to event-translation cache. "
             "Defaults to data/splits/<category>/event_translation_cache_<lang>.json",
    )
    ap.add_argument(
        "--fresh", action="store_true",
        help="Ignore existing output and start from zero.",
    )
    ap.add_argument(
        "--category", default="train",
        help="Dataset category for balanced sampling.",
    )
    args = ap.parse_args()

    if args.target_lang not in LANG_RESOURCES:
        raise ValueError(
            f"target_lang '{args.target_lang}' has no LANG_RESOURCES entry. "
            f"Supported: {list(LANG_RESOURCES)}"
        )

    if args.cache is None:
        args.cache = (
            f"data/splits/{args.category}/"
            f"event_translation_cache_{args.target_lang}.json"
        )

    data = load_json(args.input)
    category = args.category or "train"

    if args.fresh:
        existing, done_ids = [], set()
        print("Fresh mode: ignoring existing output file.")
    else:
        existing, done_ids = load_existing_translations(args.output)

    data_to_translate = choose_new_samples(data, done_ids, args.max_samples, category)

    if not data_to_translate:
        print("Nothing new to translate.")
        return

    print(f"Device:      {DEVICE}")
    print(f"Model:       {args.model_name}")
    print(f"Target lang: {args.target_lang}")
    print(f"Cache:       {args.cache}")

    tok = AutoTokenizer.from_pretrained(args.model_name, src_lang=SRC)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name, torch_dtype=DTYPE,
    ).to(DEVICE)
    mdl.generation_config.max_length = None
    mdl.eval()

    cache_path = Path(args.cache)
    cache = load_json(cache_path) if cache_path.exists() else {}
    old_cache_size = len(cache)

    # Self-healing cache cleanup. Repairs any entries from prior buggy runs:
    #   - leaked carrier prefixes ("Traduire ce texte: X")
    #   - hallucinated descriptions ("Penn Central est une ville...")
    rescued = 0
    purged = 0
    for k, v in list(cache.items()):
        cleaned = clean(v)
        if cleaned != v:
            cache[k] = cleaned
            rescued += 1
        if looks_hallucinated(cache[k], k):
            cache[k] = k
            purged += 1
    if rescued:
        print(f"Cache cleanup: rescued {rescued} entries with leftover prefixes.")
    if purged:
        print(f"Cache cleanup: purged {purged} hallucinated entries (kept source).")

    translate_missing(
        all_events(data_to_translate), cache, tok, mdl,
        args.target_lang, args.batch_size,
    )
    save_json_atomic(cache, args.cache)

    print(f"Cache size: {old_cache_size} -> {len(cache)}")
    print("Starting sample translation...")

    new_translated = []
    bar = tqdm(
        data_to_translate,
        desc="Translating new samples",
        unit="sample", dynamic_ncols=True, leave=True,
    )

    for idx, s in enumerate(bar, start=1):
        qid = str(s.get("question_id", "unknown"))
        dname = str(s.get("dataset_name", "unknown"))
        bar.set_postfix_str(
            f"{idx}/{len(data_to_translate)} | {dname[:18]} | {qid[:35]}"
        )
        new_translated.append(
            translate_sample(
                s, cache, tok, mdl, args.target_lang,
                args.batch_size, category=category,
            )
        )

    final = dedupe_preserve_order(existing + new_translated)
    save_json_atomic(final, args.output)

    # Persist any cache updates that happened during sample translation
    save_json_atomic(cache, args.cache)

    print("=" * 80)
    print("Done.")
    print(f"Language:               {args.target_lang}")
    print(f"New samples translated: {len(new_translated)}")
    print(f"Total output samples:   {len(final)}")
    print(f"Output:                 {args.output}")
    print(f"Cache:                  {args.cache}")
    print("=" * 80)


if __name__ == "__main__":
    main()