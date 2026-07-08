"""
utils/months.py

Single source of month-name tokens, shared for date detection. Previously these
lists were duplicated in metrics.py (DATE_REGEX) and translation_quality.py
(MONTH_NAMES). The English → native month *mapping* used for translation lives
in translation_utils.LANG_RESOURCES — that is a different (directional) structure
and is intentionally kept separate.
"""

MONTHS_BY_LANG = {
    "en": [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ],
    "it": [
        "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
        "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
    ],
    "de": [
        "Januar", "Februar", "März", "April", "Mai", "Juni",
        "Juli", "August", "September", "Oktober", "November", "Dezember",
    ],
    "fr": [
        "janvier", "février", "mars", "avril", "mai", "juin",
        "juillet", "août", "septembre", "octobre", "novembre", "décembre",
    ],
}

# Flat, de-duplicated (order-preserving) list of every month token — used to
# build the date-detection regexes in metrics.py and translation_quality.py.
ALL_MONTHS = list(dict.fromkeys(
    month for months in MONTHS_BY_LANG.values() for month in months
))
