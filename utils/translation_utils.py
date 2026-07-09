import re


BOOLS = {"true", "false"}
PH = "⟦EV{}⟧"
TAG_RE = re.compile(r"</?(reasoning|timeline|reflection|answer)>", re.I)


# =============================================================================
# Multilingual stripping of the "Translate this text:" wrapper that NLLB
# faithfully translates into the target language. Without this, you get
# pollution like "Übersetzen Sie diesen Text: Milan" baked into the cache.
# =============================================================================
TRANSLATE_PREFIX_RE = re.compile(
    r"^("
    r"Translate this text"             # English
    r"|Traduci questo testo"           # Italian
    r"|Traduzione di questo testo"
    r"|Übersetzen Sie diesen Text"     # German (formal)
    r"|Übersetze diesen Text"          # German (informal)
    r"|Übersetzung dieses Textes"
    r"|Traduisez ce texte"             # French (formal)
    r"|Traduis ce texte"               # French (informal)
    r"|Traduire ce texte"
    r"|Traduction de ce texte"
    r")\s*[:\-—\uFF1A]?\s*",
    flags=re.IGNORECASE,
)


# =============================================================================
# Hallucination detection — NLLB sometimes elaborates a short entity into
# a Wikipedia-style sentence ("Penn Central est une ville d'Angleterre",
# "Čedomir Janevski est né à Čedomir Janevski."). We catch and reject those.
# =============================================================================
_HALLUCINATION_PATTERNS = [
    re.compile(r"\b(est|ist|sono|è)\s+(une?|un|una|uno|eine?|einer|einem|a|an)\b\s+\w+", re.I),
    re.compile(r"\b(situé|située|located|gelegen|situato|situata)\b", re.I),
    re.compile(r"\b(né|née)\s+[aà]\b", re.I),
    re.compile(r"\bgeboren\s+in\b", re.I),
    re.compile(r"\bnato\s+(a|in)\b", re.I),
    re.compile(r"\b(en train d['e]|is currently|attualmente|gerade dabei)", re.I),
    re.compile(
        r"\b(groupe de rock|football team|équipe de football|squadra di calcio|"
        r"fußballmannschaft|équipe de hockey|hockey team)\b",
        re.I,
    ),
    re.compile(
        r"\bune?\s+(ville|équipe|entreprise|société|club|compagnie)\s+(de|du|d['e])\b",
        re.I,
    ),
    re.compile(r"\b(eine?|une?|un[oa]?)\s+(stadt|città|city|ville)\b", re.I),
]


def looks_hallucinated(translated, source):
    """Return True if NLLB elaborated a short entity into a description."""
    if not translated or not source:
        return False
    s_len = len(source)
    t_len = len(translated)
    # Length-based: short entity → translation can't grow much.
    if s_len < 60 and t_len > s_len * 2 + 15:
        for pat in _HALLUCINATION_PATTERNS:
            if pat.search(translated):
                return True
        # Entities don't end with periods. If a period appeared from nowhere,
        # NLLB probably elaborated.
        if "." in translated and "." not in source:
            return True
    return False


# =============================================================================
# Proper-noun shortcut. NLLB tends to hallucinate worst on isolated capitalized
# entities (person names, short team names). For these we skip NLLB entirely.
#
# Conservative: at most 3 words, all capitalized, no connecting small words.
# Catches: "Lou Boudreau", "Čedomir Janevski", "CSKA Sofia", "Northwood",
#          "Real Madrid", "FC Bayern".
# Skips:  "Italian Socialist Party" (3 words but worth trying), "the United
#          States", "New York Central Railroad" (4 words; sent to NLLB).
# =============================================================================
_PROPER_NOUN_SMALL_WORDS = {
    "of", "the", "and", "de", "di", "del", "della", "dello",
    "von", "der", "das", "die", "y", "e", "et", "&",
    "for", "in", "on", "at", "to", "from", "le", "la",
}


def is_proper_noun(text):
    """Heuristic: the entity is a proper noun we should pass through unchanged.
    True only for 1-3 capitalized words with no connecting small words."""
    if not text:
        return False
    text = text.strip()
    if not text:
        return False
    words = text.split()
    if not (1 <= len(words) <= 3):
        return False
    for w in words:
        if w.lower() in _PROPER_NOUN_SMALL_WORDS:
            return False
        # First char must be uppercase. Works for unicode (Č, Š, Ž all .isupper()).
        if not w[0].isupper():
            return False
    return True


# =============================================================================
# Per-language resources: months, structural phrases, sentence templates,
# question fallbacks, and boolean translations.
# =============================================================================
LANG_RESOURCES = {
    "it": {
        "months": {
            "Jan": "gennaio", "Feb": "febbraio", "Mar": "marzo", "Apr": "aprile",
            "May": "maggio", "Jun": "giugno", "Jul": "luglio", "Aug": "agosto",
            "Sep": "settembre", "Oct": "ottobre", "Nov": "novembre", "Dec": "dicembre",
            "January": "gennaio", "February": "febbraio", "March": "marzo",
            "April": "aprile", "June": "giugno", "July": "luglio",
            "August": "agosto", "September": "settembre", "October": "ottobre",
            "November": "novembre", "December": "dicembre",
        },
        "starts_at": "inizia nel",
        "ends_at": "termina nel",
        "starts": "inizia",
        "ends": "termina",
        "booleans": {"true": "Vero", "false": "Falso"},
        "possessives": [
            (r"(.+?)'s education/school is", r"l'istruzione/scuola di \1 è"),
            (r"(.+?)'s position is",         r"la posizione di \1 è"),
            (r"(.+?)'s position are",        r"le posizioni di \1 sono"),
            (r"(.+?)'s team is",             r"la squadra di \1 è"),
            (r"(.+?)'s occupation is",       r"l'occupazione di \1 è"),
            (r"(.+?)'s employer is",         r"il datore di lavoro di \1 è"),
            (r"(.+?)'s member of sports team is", r"la squadra sportiva di \1 è"),
        ],
        "templates": [
            (r"^(.+?) plays for (.+?) from (.+?) to (.+?)\.?$",      "{0} gioca per {1} da {2} a {3}."),
            (r"^(.+?) is a member of (.+?) from (.+?) to (.+?)\.?$", "{0} è membro di {1} da {2} a {3}."),
            (r"^(.+?) works for (.+?) from (.+?) to (.+?)\.?$",      "{0} lavora per {1} da {2} a {3}."),
        ],
        "question_fallbacks": {
            "Given the following five events:": "Dati i seguenti cinque eventi:",
            "Which event started first,": "Quale evento è iniziato prima,",
            "When did the event": "Quando è iniziato l'evento",
            "How long did the event": "Quanto è durato l'evento",
            "What happened right before the event": "Cosa è successo subito prima dell'inizio dell'evento",
            "What happened right after the event": "Cosa è successo subito dopo l'inizio dell'evento",
            "How much time passed between the start of event": "Quanto tempo è passato tra l'inizio dell'evento",
            "and the start of event": "e l'inizio dell'evento",
            "True or false:": "Vero o falso:",
            "in chronological order": "in ordine cronologico",
            "started at the same year": "sono iniziati nello stesso anno",
            "was longer in duration than": "è durato più a lungo di",
            "Which event is the first one in chronological order?":  "Quale evento è il primo in ordine cronologico?",
            "Which event is the second one in chronological order?": "Quale evento è il secondo in ordine cronologico?",
            "Which event is the third one in chronological order?":  "Quale evento è il terzo in ordine cronologico?",
            "Which event is the fourth one in chronological order?": "Quale evento è il quarto in ordine cronologico?",
            "Which event is the fifth one in chronological order?":  "Quale evento è il quinto in ordine cronologico?",
        },
    },

    "de": {
        "months": {
            "Jan": "Januar", "Feb": "Februar", "Mar": "März", "Apr": "April",
            "May": "Mai", "Jun": "Juni", "Jul": "Juli", "Aug": "August",
            "Sep": "September", "Oct": "Oktober", "Nov": "November", "Dec": "Dezember",
            "January": "Januar", "February": "Februar", "March": "März",
            "April": "April", "June": "Juni", "July": "Juli",
            "August": "August", "September": "September", "October": "Oktober",
            "November": "November", "December": "Dezember",
        },
        "starts_at": "beginnt im",
        "ends_at": "endet im",
        "starts": "beginnt",
        "ends": "endet",
        "booleans": {"true": "Wahr", "false": "Falsch"},
        "possessives": [
            (r"(.+?)'s education/school is", r"die Ausbildung/Schule von \1 ist"),
            (r"(.+?)'s position is",         r"die Position von \1 ist"),
            (r"(.+?)'s position are",        r"die Positionen von \1 sind"),
            (r"(.+?)'s team is",             r"das Team von \1 ist"),
            (r"(.+?)'s occupation is",       r"der Beruf von \1 ist"),
            (r"(.+?)'s employer is",         r"der Arbeitgeber von \1 ist"),
            (r"(.+?)'s member of sports team is", r"das Sportteam von \1 ist"),
        ],
        "templates": [
            (r"^(.+?) plays for (.+?) from (.+?) to (.+?)\.?$",      "{0} spielt für {1} von {2} bis {3}."),
            (r"^(.+?) is a member of (.+?) from (.+?) to (.+?)\.?$", "{0} ist Mitglied von {1} von {2} bis {3}."),
            (r"^(.+?) works for (.+?) from (.+?) to (.+?)\.?$",      "{0} arbeitet für {1} von {2} bis {3}."),
        ],
        "question_fallbacks": {
            "Given the following five events:": "Gegeben sind die folgenden fünf Ereignisse:",
            "Which event started first,": "Welches Ereignis begann zuerst,",
            "When did the event": "Wann begann das Ereignis",
            "How long did the event": "Wie lange dauerte das Ereignis",
            "What happened right before the event": "Was geschah unmittelbar vor dem Beginn des Ereignisses",
            "What happened right after the event": "Was geschah unmittelbar nach dem Beginn des Ereignisses",
            "How much time passed between the start of event": "Wie viel Zeit verging zwischen dem Beginn des Ereignisses",
            "and the start of event": "und dem Beginn des Ereignisses",
            "True or false:": "Richtig oder falsch:",
            "in chronological order": "in chronologischer Reihenfolge",
            "started at the same year": "begannen im selben Jahr",
            "was longer in duration than": "dauerte länger als",
            "Which event is the first one in chronological order?":  "Welches Ereignis ist das erste in chronologischer Reihenfolge?",
            "Which event is the second one in chronological order?": "Welches Ereignis ist das zweite in chronologischer Reihenfolge?",
            "Which event is the third one in chronological order?":  "Welches Ereignis ist das dritte in chronologischer Reihenfolge?",
            "Which event is the fourth one in chronological order?": "Welches Ereignis ist das vierte in chronologischer Reihenfolge?",
            "Which event is the fifth one in chronological order?":  "Welches Ereignis ist das fünfte in chronologischer Reihenfolge?",
        },
    },

    "fr": {
        "months": {
            "Jan": "janvier", "Feb": "février", "Mar": "mars", "Apr": "avril",
            "May": "mai", "Jun": "juin", "Jul": "juillet", "Aug": "août",
            "Sep": "septembre", "Oct": "octobre", "Nov": "novembre", "Dec": "décembre",
            "January": "janvier", "February": "février", "March": "mars",
            "April": "avril", "June": "juin", "July": "juillet",
            "August": "août", "September": "septembre", "October": "octobre",
            "November": "novembre", "December": "décembre",
        },
        "starts_at": "commence en",
        "ends_at": "se termine en",
        "starts": "commence",
        "ends": "se termine",
        "booleans": {"true": "Vrai", "false": "Faux"},
        "possessives": [
            (r"(.+?)'s education/school is", r"l'éducation/école de \1 est"),
            (r"(.+?)'s position is",         r"le poste de \1 est"),
            (r"(.+?)'s position are",        r"les postes de \1 sont"),
            (r"(.+?)'s team is",             r"l'équipe de \1 est"),
            (r"(.+?)'s occupation is",       r"la profession de \1 est"),
            (r"(.+?)'s employer is",         r"l'employeur de \1 est"),
            (r"(.+?)'s member of sports team is", r"l'équipe sportive de \1 est"),
        ],
        "templates": [
            (r"^(.+?) plays for (.+?) from (.+?) to (.+?)\.?$",      "{0} joue pour {1} de {2} à {3}."),
            (r"^(.+?) is a member of (.+?) from (.+?) to (.+?)\.?$", "{0} est membre de {1} de {2} à {3}."),
            (r"^(.+?) works for (.+?) from (.+?) to (.+?)\.?$",      "{0} travaille pour {1} de {2} à {3}."),
        ],
        "question_fallbacks": {
            "Given the following five events:": "Étant donné les cinq événements suivants :",
            "Which event started first,": "Quel événement a commencé en premier,",
            "When did the event": "Quand l'événement",
            "How long did the event": "Combien de temps a duré l'événement",
            "What happened right before the event": "Que s'est-il passé juste avant le début de l'événement",
            "What happened right after the event": "Que s'est-il passé juste après le début de l'événement",
            "How much time passed between the start of event": "Combien de temps s'est écoulé entre le début de l'événement",
            "and the start of event": "et le début de l'événement",
            "True or false:": "Vrai ou faux :",
            "in chronological order": "dans l'ordre chronologique",
            "started at the same year": "ont commencé la même année",
            "was longer in duration than": "a duré plus longtemps que",
            "Which event is the first one in chronological order?":  "Quel est le premier événement dans l'ordre chronologique ?",
            "Which event is the second one in chronological order?": "Quel est le deuxième événement dans l'ordre chronologique ?",
            "Which event is the third one in chronological order?":  "Quel est le troisième événement dans l'ordre chronologique ?",
            "Which event is the fourth one in chronological order?": "Quel est le quatrième événement dans l'ordre chronologique ?",
            "Which event is the fifth one in chronological order?":  "Quel est le cinquième événement dans l'ordre chronologique ?",
        },
    },
}


# =============================================================================
# Parenthesis / placeholder helpers
# =============================================================================

def parens(text):
    spans, depth, start = [], 0, None
    for i, c in enumerate(text or ""):
        if c == "(":
            if depth == 0:
                start = i
            depth += 1
        elif c == ")" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append((start, i + 1))
                start = None
    return spans


def unparen(x):
    x = x.strip()
    return x[1:-1].strip() if x.startswith("(") and x.endswith(")") else x


def mask_events(text):
    spans, out, evs, last = parens(text), [], [], 0
    for i, (a, b) in enumerate(spans):
        out += [text[last:a], PH.format(i)]
        evs.append(unparen(text[a:b]))
        last = b
    out.append((text or "")[last:])
    return "".join(out), evs


def restore(text, evs, cache):
    for i, ev in enumerate(evs):
        text = text.replace(PH.format(i), f"({cache.get(ev, ev)})")
    return text


# =============================================================================
# Output cleaning
# =============================================================================

def clean(text):
    """Strip the 'Translate this text:' wrapper that NLLB faithfully translates
    into the target language. Applied iteratively because some target outputs
    nest the prefix."""
    text = (text or "").strip()
    for _ in range(3):
        new_text = TRANSLATE_PREFIX_RE.sub("", text).strip()
        if new_text == text:
            break
        text = new_text
    return text


def fix_months(text, lang):
    months = LANG_RESOURCES[lang]["months"]
    for en, tgt in months.items():
        text = re.sub(rf"\b{re.escape(en)},\s*", f"{tgt} ", text)
        text = re.sub(rf"\b{re.escape(en)}\b", tgt, text)
    return text


def localize_boolean(value, lang):
    """Return the localized 'true'/'false' string, or capitalize as fallback."""
    bool_map = LANG_RESOURCES.get(lang, {}).get("booleans", {})
    return bool_map.get(str(value).lower(), str(value).capitalize())