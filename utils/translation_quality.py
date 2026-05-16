import re


BASE_FIELDS = ["question", "temporal_context", "answer"]
OPTIONAL_FIELDS = ["output"]
TAG_NAMES = ["reasoning", "timeline", "reflection", "answer"]


MONTH_NAMES = [
    # English
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",

    # Italian
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",

    # German
    "Januar", "Februar", "März", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Dezember",

    # French
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",

    
]


MONTH_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in MONTH_NAMES) + r")\s*,?\s*\d{4}\b",
    flags=re.IGNORECASE,
)


def fields_for_sample(sample):
    fields = list(BASE_FIELDS)

    if sample.get("output") and sample.get("output_en"):
        fields.append("output")

    return fields


def normalize_text(text):
    text = str(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_number_or_bool(text):
    x = str(text or "").strip().lower()
    return bool(re.fullmatch(r"\d+(\.\d+)?", x)) or x in {"true", "false"}


def exact_score(src, tgt):
    src = str(src or "").strip().lower()
    tgt = str(tgt or "").strip().lower()
    return 1.0 if src == tgt else 0.0


def extract_dates(text):
    text = str(text or "")
    years = re.findall(r"\b\d{4}\b", text)
    month_years = MONTH_RE.findall(text)
    return set(years), len(month_years)


def structure_errors(sample, field):
    errors = []

    src = str(sample.get(f"{field}_en", "") or "")
    tgt = str(sample.get(field, "") or "")

    if src.strip() and not tgt.strip():
        errors.append("empty_translation")

    if "()" in tgt:
        errors.append("empty_parentheses")

    src_years, _ = extract_dates(src)
    tgt_years, _ = extract_dates(tgt)

    missing_years = sorted(src_years - tgt_years)
    if missing_years:
        errors.append(f"missing_years:{','.join(missing_years[:10])}")

    if field == "output":
        for tag in TAG_NAMES:
            if f"<{tag}>" not in tgt or f"</{tag}>" not in tgt:
                errors.append(f"missing_tag:{tag}")

        answer = str(sample.get("answer", "") or "").strip()
        if answer and answer not in tgt:
            errors.append("answer_not_in_output")

    return errors


def english_leftover_penalty(text):
    text = str(text or "")

    patterns = [
        r"\bWhich event\b",
        r"\bTrue or false\b",
        r"\bstarted at the same year\b",
        r"\bin chronological order\b",
        r"\bstarts at\b",
        r"\bends at\b",
        r"\bwas born\b",
        r"\bwas married\b",
        r"\bwon prize\b",
        r"\bplayed for\b",
        r"\beducation/school is\b",
        r"\bposition is\b",
        r"\bteam is\b",
        r"\bplays for\b",
        r"\bfrom Jan\b",
        r"\bfrom Feb\b",
        r"\bfrom Mar\b",
        r"\bfrom Apr\b",
        r"\bfrom May\b",
        r"\bfrom Jun\b",
        r"\bfrom Jul\b",
        r"\bfrom Aug\b",
        r"\bfrom Sep\b",
        r"\bfrom Oct\b",
        r"\bfrom Nov\b",
        r"\bfrom Dec\b",
    ]

    hits = sum(bool(re.search(p, text, flags=re.I)) for p in patterns)

    if hits == 0:
        return 0.0
    if hits == 1:
        return 0.03
    if hits == 2:
        return 0.06

    return 0.10