"""
utils/metrics.py

Shared metric and answer-extraction utilities for TISER multilingual evaluation.

Returns a metric dict from calculate_metrics() so callers can report:
  em        - strict normalized exact match
  soft_em   - substring match either direction
  norm_em   - order-insensitive EM (sorted token bag)
  f1        - SQuAD-style token F1
  chrf      - character n-gram F-score (robust to morphology)
  english_leak - 1 if pred matches gold_en when gold_target was non-English

All metrics use the same normalize_text (punctuation → space, not deleted).
"""

import re
from collections import Counter


# ==================================================
# Compiled regexes
# ==================================================

TAG_REGEX = re.compile(r"<[^>]+>")

ANSWER_REGEX = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.DOTALL | re.IGNORECASE,
)

ANSWER_CUTOFF_RE = re.compile(
    r"<answer>\s*(.*)",
    re.DOTALL | re.IGNORECASE,
)

TAG_LOOP_RE = re.compile(
    r"(</(?:reasoning|reflection|timeline|answer)>)(\s*\1)+",
    re.IGNORECASE,
)

PROSE_LOOP_RE = re.compile(
    r"((?:[^.\n]{20,200}[.\n])\s*)(?:\1){2,}",
    re.IGNORECASE,
)

EMPTY_REASONING_RE = re.compile(
    r"<reasoning>\s*</reasoning>",
    re.IGNORECASE,
)

DATE_REGEX = re.compile(
    r"\b("
    r"\d{4}|"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"januar|februar|märz|maerz|april|mai|juni|juli|august|september|oktober|november|dezember"
    r")\b",
    re.IGNORECASE,
)

_REASONING_LINE_RE = re.compile(
    r"("
    r"inizia nel|termina nel|dobbiamo|possiamo|pertanto|quindi|"
    r"ragionamento|timeline|reflection|cronologia|contesto|"
    r"starts at|ends at|we need|we can|therefore|"
    r"the reasoning|the timeline|the context|"
    r"la conclusione|conclude|la risposta dovrebbe|"
    r"die antwort|daher|deshalb|zeitlinie|kontext"
    r")",
    re.IGNORECASE,
)

_ANSWER_INTRO_PATTERNS = [
    (re.compile(r"la risposta [eè]+[:\s]+[\"']?([^.\n\"']{2,150}?)[\"']?\s*[.\n]", re.IGNORECASE), 1),
    (re.compile(r"the answer is[:\s]+[\"']?([^.\n\"']{2,150}?)[\"']?\s*[.\n]", re.IGNORECASE), 1),
    (re.compile(r"die antwort ist[:\s]+[\"']?([^.\n\"']{2,150}?)[\"']?\s*[.\n]", re.IGNORECASE), 1),
    (re.compile(r"risposta[:\s]+[\"']?([^.\n\"']{2,150}?)[\"']?\s*[.\n]", re.IGNORECASE), 1),
    (re.compile(r"answer[:\s]+[\"']?([^.\n\"']{2,150}?)[\"']?\s*[.\n]", re.IGNORECASE), 1),
    (re.compile(r"pertanto[,\s]+[\"']?([^.\n\"']{2,100}?)[\"']?\s*[.\n]", re.IGNORECASE), 1),
    (re.compile(r"quindi[,\s]+[\"']?([^.\n\"']{2,100}?)[\"']?\s*[.\n]", re.IGNORECASE), 1),
]


# ==================================================
# Helpers
# ==================================================

def is_date_like(text):
    return bool(DATE_REGEX.search(text or ""))


def _dedupe_doubled(s):
    """Collapse duplicated translation artifacts: 'russo russo' -> 'russo'."""
    if not s:
        return s
    s = str(s).strip()
    parts = s.split()
    n = len(parts)

    if n >= 2 and n % 2 == 0 and parts[: n // 2] == parts[n // 2:]:
        return " ".join(parts[: n // 2])

    if len(s) % 2 == 0:
        half = len(s) // 2
        if s[:half].strip() == s[half:].strip():
            return s[:half].strip()

    m = re.match(r"^(.+?)\s+\1$", s)
    if m:
        return m.group(1).strip()

    return s


# ==================================================
# Normalization
# ==================================================

def normalize_text(text):
    """Lowercase, replace punctuation with spaces, normalize Persian chars
    and apostrophe variants. Preserves Unicode word characters (so accented
    Italian letters survive)."""
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    return " ".join(text.split())


_STRIP_WORDS = {
    "il", "la", "lo", "i", "gli", "le", "l",
    "un", "una", "uno",
    "del", "della", "dello", "dei", "degli", "delle",
    "di", "da", "dal", "dalla", "dallo", "dai", "dagli", "dalle",
    "a", "al", "alla", "allo", "ai", "agli", "alle",
    "the", "a", "an", "of", "for", "in", "at", "on",
    "der", "die", "das", "des", "dem", "den",
    "ein", "eine", "einer", "eines", "einem", "einen",
}

_ABBREV_MAP = {
    "under19": "u19", "under 19": "u19", "under-19": "u19", "u-19": "u19",
    "under21": "u21", "under 21": "u21", "under-21": "u21", "u-21": "u21",
    "under17": "u17", "under 17": "u17", "under-17": "u17", "u-17": "u17",
    "f c": "fc", "s c": "sc", "a c": "ac",
}


def normalize_for_comparison(text):
    """Drift-aware normalization: strips articles, expands abbrevs, sorts tokens.
    Used as a backup comparison."""
    t = normalize_text(text)
    for k, v in _ABBREV_MAP.items():
        t = t.replace(k, v)
    words = [w for w in t.split() if w not in _STRIP_WORDS and len(w) > 1]
    return " ".join(sorted(words))


_TRUE_TOKENS = {"true", "yes", "vero", "ja", "wahr", "si", "sì"}
_FALSE_TOKENS = {"false", "no", "falso", "nein", "falsch"}


def normalize_boolean(text):
    t = normalize_text(text)
    if not t:
        return ""
    if t in _TRUE_TOKENS:
        return "true"
    if t in _FALSE_TOKENS:
        return "false"
    return ""


_UNKNOWN_ALIASES_SUBSTR = (
    "sconosciut", "unknown", "incognit", "non conosc",
    "non e chiaro", "non è chiaro", "unklar", "unbekannt",
)


def normalize_unknown(text):
    t = normalize_text(text)
    if not t:
        return ""
    if any(sub in t for sub in _UNKNOWN_ALIASES_SUBSTR):
        return "unknown"
    return ""


# ==================================================
# Additional metric primitives
# ==================================================

def normalized_em(pred, gold):
    """Order-insensitive EM. Sorted token bag comparison.
    Catches 'Leibniz Università di Hannover' == 'Università Leibniz Hannover'."""
    p = sorted(normalize_text(pred).split())
    g = sorted(normalize_text(gold).split())
    return 1 if p and g and p == g else 0


def chrf_score(pred, gold, n=6, beta=2):
    """Character n-gram F-beta score on normalized strings.
    Robust to morphology (russi/russe/russo all near-match russi).
    Returns 0-100."""
    p = normalize_text(pred)
    g = normalize_text(gold)
    if not p and not g:
        return 100.0
    if not p or not g:
        return 0.0

    def ngrams(s, k):
        return [s[i:i + k] for i in range(len(s) - k + 1)] if len(s) >= k else []

    f_scores = []
    for k in range(1, n + 1):
        p_ng = ngrams(p, k)
        g_ng = ngrams(g, k)
        if not p_ng and not g_ng:
            f_scores.append(1.0)
            continue
        if not p_ng or not g_ng:
            f_scores.append(0.0)
            continue
        common = sum(min(p_ng.count(x), g_ng.count(x)) for x in set(p_ng))
        if common == 0:
            f_scores.append(0.0)
            continue
        prec = common / len(p_ng)
        rec = common / len(g_ng)
        b2 = beta * beta
        f_scores.append((1 + b2) * prec * rec / (b2 * prec + rec))

    return 100.0 * sum(f_scores) / len(f_scores)


def is_english_leak(pred, gold_target, gold_en):
    """1 if pred matches gold_en exactly when gold_target is non-English.
    Diagnostic only — measures whether the model dropped back to English
    on the final answer despite reasoning in the target language."""
    if not gold_en or str(gold_en).strip().lower() in {"none", ""}:
        return 0
    p = normalize_text(pred)
    gt = normalize_text(gold_target)
    ge = normalize_text(gold_en)
    if not gt or not ge or gt == ge:
        return 0
    return 1 if p == ge else 0


# ==================================================
# Main scoring entry point
# ==================================================

def calculate_metrics(pred, gold_candidates, gold_en=None):
    """
    Returns a dict:
      em, soft_em, norm_em, f1, chrf, english_leak

    Best score across gold candidates for each metric.
    """
    pred = _dedupe_doubled(pred) if pred else pred

    cleaned_golds = []
    for g in gold_candidates:
        if g is None:
            continue
        g = str(g).strip()
        if not g:
            continue
        cleaned_golds.append(_dedupe_doubled(g))

    if not cleaned_golds:
        return {
            "em": 0, "soft_em": 0, "norm_em": 0,
            "f1": 0.0, "chrf": 0.0, "english_leak": 0,
        }

    def get_scores(p, g):
        # 1. Boolean branch (exact-token only)
        pb = normalize_boolean(p)
        gb = normalize_boolean(g)
        if gb:
            match = int(pb == gb)
            return {"em": match, "soft_em": match, "norm_em": match,
                    "f1": float(match), "chrf": 100.0 * match}

        # 2. Unknown branch
        pu = normalize_unknown(p)
        gu = normalize_unknown(g)
        if gu:
            match = int(pu == gu)
            return {"em": match, "soft_em": match, "norm_em": match,
                    "f1": float(match), "chrf": 100.0 * match}

        # 3. Standard EM/F1
        p_norm = normalize_text(TAG_REGEX.sub(" ", str(p or "")))
        g_norm = normalize_text(TAG_REGEX.sub(" ", str(g or "")))

        em = int(p_norm == g_norm)
        soft = int(
            bool(p_norm) and bool(g_norm)
            and (p_norm in g_norm or g_norm in p_norm)
        )
        nem = normalized_em(p, g)

        pt = p_norm.split()
        gt = g_norm.split()
        if not pt or not gt:
            f1 = 1.0 if pt == gt else 0.0
        else:
            common = Counter(pt) & Counter(gt)
            overlap = sum(common.values())
            f1 = 2 * overlap / (len(pt) + len(gt)) if overlap else 0.0

        # 4. Drift-aware backup
        if not em:
            p_drift = normalize_for_comparison(str(p or ""))
            g_drift = normalize_for_comparison(str(g or ""))
            if p_drift and g_drift and p_drift == g_drift:
                soft = 1
                f1 = max(f1, 1.0)
                nem = 1

        chrf = chrf_score(p, g)

        return {
            "em": em, "soft_em": soft, "norm_em": nem,
            "f1": f1, "chrf": chrf,
        }

    # Best per-metric across gold candidates (use F1 as the tiebreaker for "best")
    best = None
    for gold in cleaned_golds:
        scores = get_scores(pred, gold)
        if best is None:
            best = scores
        else:
            for k in scores:
                if scores[k] > best[k]:
                    best[k] = scores[k]

    # English-leak is computed once against gold_en
    leak = 0
    if gold_en is not None:
        for gold in cleaned_golds:
            leak = max(leak, is_english_leak(pred, gold, gold_en))
    best["english_leak"] = leak

    return best


# ==================================================
# Output cleaning
# ==================================================

def clean_output(text):
    """Sanitize model output: collapse duplicate opening tags, cut tag/prose
    loops, remove empty reasoning blocks."""
    if not text:
        return text
    text = str(text)

    text = re.sub(r"(<reasoning>)\s*<reasoning>", r"\1", text, flags=re.IGNORECASE)

    m = TAG_LOOP_RE.search(text)
    if m:
        text = text[: m.start() + len(m.group(1))]

    m = PROSE_LOOP_RE.search(text)
    if m:
        text = text[: m.start() + len(m.group(1))]

    text = EMPTY_REASONING_RE.sub("", text).strip()
    return text


def clean_extracted(text):
    """Final cleanup for extracted answer."""
    if not text:
        return ""
    text = str(text)
    text = TAG_REGEX.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()

    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()

    return text


# ==================================================
# Answer extraction
# ==================================================

def extract_answer(full_text):
    """Extract final answer with priority:
       1. Well-formed <answer>...</answer>
       2. Truncated <answer>...
       3. Intro-phrase patterns ('the answer is X', 'la risposta è X', ...)
       4. Last short non-reasoning clause
       5. First 100 chars of plain text"""
    if not full_text:
        return ""

    cleaned = clean_output(full_text)

    m = ANSWER_REGEX.search(cleaned)
    if m:
        ans = clean_extracted(m.group(1))
        if ans and len(ans) < 200:
            return ans

    m = ANSWER_CUTOFF_RE.search(cleaned)
    if m:
        raw = m.group(1).strip()
        ans = re.split(r"\n|<|\.\s", raw)[0].strip()
        ans = clean_extracted(ans)
        if ans and len(ans) < 200:
            return ans

    plain = TAG_REGEX.sub(" ", cleaned)
    plain = re.sub(r"\s+", " ", plain).strip()

    for pat, group_id in _ANSWER_INTRO_PATTERNS:
        matches = list(pat.finditer(plain))
        if matches:
            ans = matches[-1].group(group_id).strip().strip('"\'')
            ans = clean_extracted(ans)
            if ans and len(ans) < 150 and ans.count(" ") < 15:
                return ans

    candidate_lines = []
    for line in re.split(r"[.\n]", plain):
        line = line.strip()
        if not line or len(line) < 2 or len(line) > 120:
            continue
        if _REASONING_LINE_RE.search(line):
            continue
        candidate_lines.append(line)

    if candidate_lines:
        return clean_extracted(candidate_lines[-1])

    return clean_extracted(plain[:100])