"""
utils/metrics.py

Shared metric and answer-extraction utilities for TISER multilingual eval.

calculate_metrics() returns a dict with:
  em           - strict normalized exact match
  soft_em      - substring match either direction
  norm_em      - order-insensitive EM (sorted token bag)
  f1           - SQuAD-style token F1
  chrf         - character n-gram F-score (robust to morphology)
  english_leak - 1 if pred matches gold_en when gold_target was non-English

Supported languages: en, it, de, fr.

extract_answer() handles multiple output formats:
  1. <answer>...</answer> XML tag (fine-tuned format)
  2. <answer>... truncated tag
  3. Markdown "### Answer:" block (base-model fallback style)
  4. Intro phrases ("the answer is X", "die Antwort ist X", ...)
  5. Last short non-reasoning clause
  6. First 100 chars of plain text
"""

import re
from collections import Counter


# =============================================================================
# Compiled regexes
# =============================================================================

TAG_REGEX = re.compile(r"<[^>]+>")

ANSWER_REGEX = re.compile(
    r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE,
)

ANSWER_CUTOFF_RE = re.compile(
    r"<answer>\s*(.*)", re.DOTALL | re.IGNORECASE,
)

# Markdown-style answer blocks: base Qwen falls back to these when it doesn't
# see enough tag-format training data for a language (e.g. EN-only model
# evaluated on German).
MARKDOWN_ANSWER_BLOCK_RE = re.compile(
    r"###\s*Answer\s*:?\s*\n+\s*(.+?)(?:\n\s*\n|\n###|\Z)",
    re.IGNORECASE | re.DOTALL,
)
MARKDOWN_ANSWER_INLINE_RE = re.compile(
    r"###\s*Answer\s*:?\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)

# Answer-label prefixes to strip from extracted content (multilingual).
# Anchored: only strips when at the start of the extracted string.
ANSWER_PREFIX_RE = re.compile(
    r"^(?:"
    r"###\s*Answer\s*:?\s*|"      # markdown header
    r"Answer\s*:\s*|"              # English
    r"Antwort\s*:\s*|"             # German
    r"Risposta\s*:\s*|"            # Italian
    r"Réponse\s*:\s*|Reponse\s*:\s*"  # French (with/without accent)
    r")",
    re.IGNORECASE,
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
    r"<reasoning>\s*</reasoning>", re.IGNORECASE,
)

DATE_REGEX = re.compile(
    r"\b("
    r"\d{4}|"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"januar|februar|märz|maerz|mai|juni|juli|oktober|dezember|"
    r"janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre"
    r")\b",
    re.IGNORECASE,
)

# Reasoning-scaffolding cues to skip when picking last-line fallback
_REASONING_LINE_RE = re.compile(
    r"("
    # Italian
    r"inizia nel|termina nel|dobbiamo|possiamo|pertanto|quindi|"
    r"ragionamento|cronologia|contesto|la conclusione|conclude|la risposta dovrebbe|"
    # English
    r"starts at|ends at|we need|we can|therefore|"
    r"the reasoning|the timeline|the context|reflection|"
    # German
    r"die antwort|daher|deshalb|zeitlinie|kontext|"
    r"überlegung|ueberlegung|schlussfolgerung|"
    # French
    r"commence à|commence a|se termine|nous devons|nous pouvons|"
    r"donc|par conséquent|par consequent|"
    r"le raisonnement|la chronologie|la réflexion|la reflexion|"
    r"timeline"
    r")",
    re.IGNORECASE,
)

_ANSWER_INTRO_PATTERNS = [
    # Italian
    (re.compile(r"la risposta [eè]+[:\s]+[\"']?([^.\n\"']{2,150}?)[\"']?\s*[.\n]", re.IGNORECASE), 1),
    (re.compile(r"risposta[:\s]+[\"']?([^.\n\"']{2,150}?)[\"']?\s*[.\n]", re.IGNORECASE), 1),
    (re.compile(r"pertanto[,\s]+[\"']?([^.\n\"']{2,100}?)[\"']?\s*[.\n]", re.IGNORECASE), 1),
    (re.compile(r"quindi[,\s]+[\"']?([^.\n\"']{2,100}?)[\"']?\s*[.\n]", re.IGNORECASE), 1),
    # English
    (re.compile(r"the answer is[:\s]+[\"']?([^.\n\"']{2,150}?)[\"']?\s*[.\n]", re.IGNORECASE), 1),
    (re.compile(r"answer[:\s]+[\"']?([^.\n\"']{2,150}?)[\"']?\s*[.\n]", re.IGNORECASE), 1),
    # German
    (re.compile(r"die antwort ist[:\s]+[\"']?([^.\n\"']{2,150}?)[\"']?\s*[.\n]", re.IGNORECASE), 1),
    (re.compile(r"antwort[:\s]+[\"']?([^.\n\"']{2,150}?)[\"']?\s*[.\n]", re.IGNORECASE), 1),
    # French
    (re.compile(r"la r[ée]ponse est[:\s]+[\"']?([^.\n\"']{2,150}?)[\"']?\s*[.\n]", re.IGNORECASE), 1),
    (re.compile(r"r[ée]ponse[:\s]+[\"']?([^.\n\"']{2,150}?)[\"']?\s*[.\n]", re.IGNORECASE), 1),
]


# =============================================================================
# Helpers
# =============================================================================

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


# =============================================================================
# Normalization
# =============================================================================

def normalize_text(text):
    """
    Lowercase, replace punctuation with spaces, normalize apostrophe variants.
    Preserves Unicode word characters so accented letters survive.
    """
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    return " ".join(text.split())


_STRIP_WORDS = {
    # Italian
    "il", "la", "lo", "i", "gli", "le", "l",
    "un", "una", "uno",
    "del", "della", "dello", "dei", "degli", "delle",
    "di", "da", "dal", "dalla", "dallo", "dai", "dagli", "dalle",
    "a", "al", "alla", "allo", "ai", "agli", "alle",
    # English
    "the", "an", "of", "for", "in", "at", "on",
    # German
    "der", "die", "das", "des", "dem", "den",
    "ein", "eine", "einer", "eines", "einem", "einen",
    # French
    "le", "les", "un", "une", "des", "du", "de",
    "au", "aux",
}

_ABBREV_MAP = {
    "under19": "u19", "under 19": "u19", "under-19": "u19", "u-19": "u19",
    "under21": "u21", "under 21": "u21", "under-21": "u21", "u-21": "u21",
    "under17": "u17", "under 17": "u17", "under-17": "u17", "u-17": "u17",
    "f c": "fc", "s c": "sc", "a c": "ac",
}


def normalize_for_comparison(text):
    """Drift-aware normalization: strips articles, expands abbrevs, sorts tokens."""
    t = normalize_text(text)
    for k, v in _ABBREV_MAP.items():
        t = t.replace(k, v)
    words = [w for w in t.split() if w not in _STRIP_WORDS and len(w) > 1]
    return " ".join(sorted(words))


_TRUE_TOKENS = {"true", "yes", "vero", "ja", "wahr", "si", "sì", "oui", "vrai"}
_FALSE_TOKENS = {"false", "no", "falso", "nein", "falsch", "non", "faux"}


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
    "non e chiaro", "non è chiaro",
    "unklar", "unbekannt",
    "inconnu", "on ne sait pas", "pas connu",
)


def normalize_unknown(text):
    t = normalize_text(text)
    if not t:
        return ""
    if any(sub in t for sub in _UNKNOWN_ALIASES_SUBSTR):
        return "unknown"
    return ""


# =============================================================================
# Metric primitives
# =============================================================================

def normalized_em(pred, gold):
    """Order-insensitive EM. 'Leibniz Università Hannover' == 'Università Leibniz Hannover'."""
    p = sorted(normalize_text(pred).split())
    g = sorted(normalize_text(gold).split())
    return 1 if p and g and p == g else 0


def chrf_score(pred, gold, n=6, beta=2):
    """
    Character n-gram F-beta score on normalized strings.
    Robust to morphology (russi/russe/russo all near-match). Returns 0-100.
    """
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
    """
    1 if pred matches gold_en exactly when gold_target is non-English.

    Diagnostic only — measures whether the model dropped back to English
    on the final answer despite reasoning in the target language.
    """
    if not gold_en or str(gold_en).strip().lower() in {"none", ""}:
        return 0
    p = normalize_text(pred)
    gt = normalize_text(gold_target)
    ge = normalize_text(gold_en)
    if not gt or not ge or gt == ge:
        return 0
    return 1 if p == ge else 0


# =============================================================================
# Main scoring entry point
# =============================================================================

def _score_pair(p, g):
    """Score a single (prediction, gold) pair. Boolean/unknown branches first."""
    # Boolean
    pb = normalize_boolean(p)
    gb = normalize_boolean(g)
    if gb:
        match = int(pb == gb)
        return {"em": match, "soft_em": match, "norm_em": match,
                "f1": float(match), "chrf": 100.0 * match}

    # Unknown
    pu = normalize_unknown(p)
    gu = normalize_unknown(g)
    if gu:
        match = int(pu == gu)
        return {"em": match, "soft_em": match, "norm_em": match,
                "f1": float(match), "chrf": 100.0 * match}

    # Standard EM / soft / F1
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

    # Drift-aware backup
    if not em:
        p_drift = normalize_for_comparison(str(p or ""))
        g_drift = normalize_for_comparison(str(g or ""))
        if p_drift and g_drift and p_drift == g_drift:
            soft = 1
            f1 = max(f1, 1.0)
            nem = 1

    return {
        "em": em, "soft_em": soft, "norm_em": nem,
        "f1": f1, "chrf": chrf_score(p, g),
    }


def calculate_metrics(pred, gold_candidates, gold_en=None):
    """
    Returns a dict: em, soft_em, norm_em, f1, chrf, english_leak.

    Best score across gold candidates is taken independently for each metric.
    """
    pred = _dedupe_doubled(pred) if pred else pred

    cleaned_golds = []
    for g in gold_candidates:
        if g is None:
            continue
        g = str(g).strip()
        if g:
            cleaned_golds.append(_dedupe_doubled(g))

    if not cleaned_golds:
        return {"em": 0, "soft_em": 0, "norm_em": 0,
                "f1": 0.0, "chrf": 0.0, "english_leak": 0}

    # Best per-metric across candidates
    best = None
    for gold in cleaned_golds:
        scores = _score_pair(pred, gold)
        if best is None:
            best = scores
        else:
            for k in scores:
                if scores[k] > best[k]:
                    best[k] = scores[k]

    # English-leak diagnostic
    leak = 0
    if gold_en is not None:
        for gold in cleaned_golds:
            leak = max(leak, is_english_leak(pred, gold, gold_en))
    best["english_leak"] = leak

    return best


# =============================================================================
# Output cleaning
# =============================================================================

def clean_output(text):
    """Sanitize model output: collapse duplicate opening tags, cut loops."""
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


def _strip_answer_prefix(text):
    """Strip '### Answer:', 'Answer:', 'Antwort:', 'Risposta:', 'Réponse:'."""
    if not text:
        return ""
    text = text.strip()
    # May be nested: "### Answer: Antwort: Foo" — strip repeatedly
    for _ in range(3):
        new = ANSWER_PREFIX_RE.sub("", text).strip()
        if new == text:
            break
        text = new
    return text


def clean_extracted(text):
    """Final cleanup for extracted answer content."""
    if not text:
        return ""
    text = str(text)
    text = TAG_REGEX.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Strip label prefixes like "### Answer:" that leaked into the content
    text = _strip_answer_prefix(text)

    # Unwrap surrounding parens
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()

    # Unwrap surrounding quotes
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()

    # Strip trailing punctuation
    text = re.sub(r"[.,;:!?]+$", "", text).strip()

    return text


# =============================================================================
# Answer extraction
# =============================================================================

def extract_answer(full_text):
    """
    Extract final answer with priority:
      1. Well-formed <answer>...</answer>
      2. Truncated <answer>...
      3. Markdown '### Answer:' block (fallback for base-model style output)
      4. Intro-phrase patterns ('the answer is X', 'la risposta è X', ...)
      5. Last short non-reasoning clause
      6. First 100 chars of plain text
    """
    if not full_text:
        return ""

    cleaned = clean_output(full_text)

    # 1. Well-formed XML answer tag
    m = ANSWER_REGEX.search(cleaned)
    if m:
        ans = clean_extracted(m.group(1))
        if ans and len(ans) < 200:
            return ans

    # 2. Truncated XML answer tag
    m = ANSWER_CUTOFF_RE.search(cleaned)
    if m:
        raw = m.group(1).strip()
        ans = re.split(r"\n|<|\.\s", raw)[0].strip()
        ans = clean_extracted(ans)
        if ans and len(ans) < 200:
            return ans

    # 3. Markdown "### Answer:\n<content>" block
    m = MARKDOWN_ANSWER_BLOCK_RE.search(cleaned)
    if m:
        ans = clean_extracted(m.group(1))
        if ans and len(ans) < 200:
            return ans

    # 4. Markdown "### Answer: <content>" inline
    m = MARKDOWN_ANSWER_INLINE_RE.search(cleaned)
    if m:
        ans = clean_extracted(m.group(1))
        if ans and len(ans) < 200:
            return ans

    # Strip tags for the phrase-based fallbacks
    plain = TAG_REGEX.sub(" ", cleaned)
    plain = re.sub(r"\s+", " ", plain).strip()

    # 5. Intro-phrase patterns
    for pat, group_id in _ANSWER_INTRO_PATTERNS:
        matches = list(pat.finditer(plain))
        if matches:
            ans = matches[-1].group(group_id).strip().strip('"\'')
            ans = clean_extracted(ans)
            if ans and len(ans) < 150 and ans.count(" ") < 15:
                return ans

    # 6. Last short non-reasoning clause
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