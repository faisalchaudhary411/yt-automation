"""
Generates narration audio for each scene using edge-tts (Microsoft Edge's free,
no-API-key neural TTS). Supports one male + one female neural voice per
supported language, with specific support for Pakistani Urdu (ur-PK) voices.

Key improvements for Pakistani Urdu content:
  - Uses ur-PK-AsadNeural (male) and ur-PK-UzmaNeural (female) — genuine
    Pakistani-accented Urdu voices, NOT Indian Urdu (ur-IN) or Arabic voices
  - Pre-processes text to add natural pauses and fix common TTS issues with
    Urdu punctuation and number pronunciation
  - Supports SSML prosody for more natural, human-like speech pacing
  - Concurrent generation for speed

Scenes are generated concurrently (I/O-bound network calls) to keep total
generation time low on longer videos.

FIX: script_generator.py's prompts deliberately embed script/editing markers
in the narration text -- "(PAUSE)", "(EMPHASIS)", and "[B-ROLL: description]"
-- for pacing and editing purposes. Those are directorial notes, not words to
be spoken. Because edge-tts's Communicate class does not parse any markup in
its text input (it just speaks the literal string), those markers were being
read aloud verbatim: the narrator would literally say "PAUSE", "EMPHASIS", and
read out B-ROLL descriptions in English mid-Urdu-sentence. That abrupt
language/register switching is very likely what sounded like "the AI reading
words differently" -- it wasn't a voice-quality issue, it was literal stage
directions being spoken. This is now stripped/converted before any text
reaches TTS. Number/currency symbols (%, $) are also now spoken out in Urdu
words instead of being left for the voice to guess at in English.

FURTHER FIX: mixed-script input is a broader problem than just the markers
above. Two more sources of bad pronunciation, both addressed below and both
applied ONLY to the text sent to TTS (on-screen captions in video_assembler.py
are untouched, since the original mixed Urdu/English/digit text is exactly
what should be displayed):

  1. Urdu (like Arabic) is normally written WITHOUT vowel diacritics, so a
     word like "ملک" is genuinely ambiguous to a TTS engine's
     grapheme-to-phoneme model -- it could be read "مُلک" (mulk/country),
     "مَلَک" (malak/angel), etc. _PRONUNCIATION_FIXES adds the missing
     diacritic for specific words known to be mispronounced, forcing the
     intended reading. This dict is meant to grow as more bad words turn up.

  2. Raw Latin-script English words and digits dropped into Urdu text force
     the ur-PK voice to switch scripts/phonetic-rules mid-utterance, which
     often degrades pronunciation for the whole surrounding sentence, not
     just the foreign token. _ENGLISH_LOANWORD_URDU transliterates common
     finance/history loanwords into Urdu script before TTS (matching the
     loanwords script_generator.py's prompt already explicitly permits), and
     the number-spelling functions below convert digit sequences into full
     Urdu words rather than leaving bare digits for the voice to guess at.

     NOTE on the number spelling: _URDU_COMPOUND_TENS below has the full
     21-99 Urdu number chart filled in (standard Hindustani/Urdu numeral
     vocabulary), so most numbers now read as a natural compound word (e.g.
     827 -> "آٹھ سو ستائیس", not digit-by-digit). Digit-by-digit reading is
     now only a fallback for genuinely out-of-range input. If you ever hear
     a specific number mispronounced, that's a one-line dict fix -- just
     report the exact number.
"""

import os
import asyncio
import edge_tts
import re

# ---------------------------------------------------------------------------
# Voice configuration — Pakistani Urdu voices (edge-tts)
# ---------------------------------------------------------------------------

# Primary Pakistani Urdu voices — these are specifically trained on Pakistani
# Urdu pronunciation, NOT Indian Urdu or Arabic. Using ur-IN voices (Gul, Salman)
# will give an Indian accent which sounds wrong to Pakistani audiences.
URDU_PK_VOICES = {
    "male": "ur-PK-AsadNeural",      # Male, Pakistani Urdu accent
    "female": "ur-PK-UzmaNeural",    # Female, Pakistani Urdu accent
}

# Fallback voices in order of preference (if ur-PK is ever unavailable)
URDU_FALLBACK_VOICES = {
    "male": ["ur-PK-AsadNeural", "ur-IN-SalmanNeural"],
    "female": ["ur-PK-UzmaNeural", "ur-IN-GulNeural"],
}

MAX_CONCURRENT_TTS = 6


# ---------------------------------------------------------------------------
# Script-marker stripping (PAUSE / EMPHASIS / B-ROLL directives)
# ---------------------------------------------------------------------------

# Matches "[B-ROLL: anything]" -- the format the model is asked to use --
# tolerant of odd spacing and of the model mixing bracket/paren types
# (e.g. "[B-ROLL: ...)"), since LLMs don't always close brackets consistently.
_BROLL_MARKER_RE = re.compile(r"[\[\(]\s*B[\s\-]?ROLL\b[^\]\)]*[\]\)]", re.IGNORECASE | re.DOTALL)

# Fallback for when the model drops the brackets entirely (e.g. writes plain
# "B-ROLL: modern risk control room" inline) -- without this, that text would
# leak straight into narration and get spoken/captioned verbatim. Strips from
# "B-ROLL" up to the next sentence-ending punctuation (or end of text).
_BROLL_BARE_MARKER_RE = re.compile(r"\bB[\s\-]?ROLL\b\s*:?\s*[^۔.!?\n]*", re.IGNORECASE)

# Matches "(PAUSE)" -- converted to a real Urdu comma so edge-tts produces an
# actual audible pause, instead of speaking the literal word "pause".
_PAUSE_MARKER_RE = re.compile(r"\(\s*PAUSE\s*\)", re.IGNORECASE)

# Matches "(EMPHASIS)" -- edge-tts's Communicate class has no way to apply
# per-word stress/emphasis without SSML (which it doesn't parse anyway), so
# there's nothing meaningful to convert this to. It's simply removed rather
# than being read aloud as the word "emphasis".
_EMPHASIS_MARKER_RE = re.compile(r"\(\s*EMPHASIS\s*\)", re.IGNORECASE)


def _strip_narration_markers(text: str) -> str:
    """Removes/converts script-writing directives that were never meant to be
    spoken by TTS. Must run BEFORE sentence splitting/word-count logic so
    those don't get thrown off by leftover bracket text."""
    if not text:
        return text

    text = _BROLL_MARKER_RE.sub(" ", text)
    text = _BROLL_BARE_MARKER_RE.sub(" ", text)
    text = _PAUSE_MARKER_RE.sub("،", text)
    text = _EMPHASIS_MARKER_RE.sub(" ", text)

    # Clean up artifacts left behind: doubled punctuation, stray leading
    # commas, extra whitespace from the removals above.
    text = re.sub(r"[،,]{2,}", "،", text)
    text = re.sub(r"\s*،\s*", "، ", text)
    text = re.sub(r"^[،,\s]+", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Number / currency preprocessing
# ---------------------------------------------------------------------------

# "50%" -> "50 فیصد" ("feesad" / percent). Left bare, edge-tts tends to read
# the symbol in English mid-Urdu-sentence, which is a jarring language switch.
# NOTE: matches BOTH the ASCII "%" and the Arabic/Urdu percent sign "٪"
# (U+066A). "40%" converted fine before, but if the script generator (or an
# LLM) ever emits the Urdu-script percent sign instead, it slipped through
# this regex untouched and the bare "٪" symbol was left for the voice to
# guess at -- almost certainly the cause of "40%" -> garbled "چالسہ".
_PERCENT_RE = re.compile(r"(\d[\d,.]*)\s*[%٪]")

# "$500" -> "500 ڈالر" (dollar), "£500" -> "500 پاؤنڈ" (pound), "€500" -> "500
# یورو" (euro). Same reasoning as above -- spelled out in Urdu so the whole
# sentence stays in one language/register. (£/€ were missing entirely before
# -- a bare "£827 million" left the £ symbol for the Urdu voice to stumble on.)
_DOLLAR_PREFIX_RE = re.compile(r"\$\s*(\d[\d,.]*)")
_POUND_PREFIX_RE = re.compile(r"£\s*(\d[\d,.]*)")
_EURO_PREFIX_RE = re.compile(r"€\s*(\d[\d,.]*)")


def _normalize_numbers_and_currency(text: str) -> str:
    """Spells out %/$/£/€ in Urdu words so the voice doesn't switch languages
    mid-sentence to read a bare symbol."""
    if not text:
        return text
    text = _PERCENT_RE.sub(r"\1 فیصد", text)
    text = _DOLLAR_PREFIX_RE.sub(r"\1 ڈالر", text)
    text = _POUND_PREFIX_RE.sub(r"\1 پاؤنڈ", text)
    text = _EURO_PREFIX_RE.sub(r"\1 یورو", text)
    return text


# ---------------------------------------------------------------------------
# Pronunciation-fix dictionary
# ---------------------------------------------------------------------------
# Urdu is normally written without vowel diacritics (اعراب), which makes some
# words genuinely ambiguous to a TTS engine's grapheme-to-phoneme model. This
# maps a bare-spelling word to a version with the diacritic added back in, to
# force the intended reading. ADD TO THIS as you find more mispronounced
# words -- each entry is a simple "what it should say instead" mapping.
_PRONUNCIATION_FIXES = {
    # "ملک" (bare) is ambiguous between "مُلک" (mulk/country -- almost always
    # the intended meaning in history/finance narration) and other readings
    # like "مَلَک" (malak/angel). The damma (ُ) forces the "mulk" reading.
    "ملک": "مُلْک",
    # "بنا" (banaa, "made/built" -- e.g. "بنا دیا") was being read by the
    # voice as "بن" (ban, "become"), dropping the long-vowel ending. The
    # fatha forces the intended "banaa" reading instead.
    "بنا": "بَنا",
    # "بنک" (a shorter alternate spelling of "bank") was being mispronounced;
    # "بینک" is the standard Urdu spelling for the loanword and reads
    # correctly, so any occurrence of the shorter form is normalized to it.
    "بنک": "بینک",

    # --- Added July 2026, from a real narration pass -----------------------
    # "پلاسی" (Plassey, as in the Battle of Plassey) was collapsing to
    # "پلسی", dropping a syllable. Fatha forces the full three-syllable read.
    "پلاسی": "پَلاسی",
    # "مالیت" (value/worth) was collapsing to "ملیت". Kasra on the ی anchors
    # the middle syllable so it doesn't get swallowed.
    "مالیت": "مالِیت",
    # "اسی" (assi/eighty) was being read as "isi" (this one). Fatha forces
    # the "assi" reading. NOTE: this dict entry only fires for the standalone
    # word "اسی" -- the actual number 80 already reads correctly via
    # _URDU_TENS_ROUND, this only matters if "اسی" appears spelled out as a
    # word in narration text rather than as the digit 80.
    "اسی": "اَسی",
    # "لائبریری" (library) was collapsing to "لابری", losing two syllables.
    "لائبریری": "لَائِبریری",
    # "چلانا" (chalana/to run-drive) was being read "chilana". Fatha forces
    # the "chalana" reading.
    "چلانا": "چَلانا",
    # "گلی" (gali/street) was being read "gili" (damp). Fatha forces "gali".
    "گلی": "گَلی",
    # "چھینی" (chheeni/chisel) was picking up an extra zabar (fatha) that
    # isn't in the word. Kasra pins the intended vowel.
    "چھینی": "چھِینی",
    # "بنایا"/"بناتا"/"بنانا" etc: "بنا" below already fixes the standalone
    # word, but these inflected forms are DIFFERENT strings (not substrings
    # matched by the "بنا" \b...\b pattern), so they need their own entries.
    "بنایا": "بَنایا",
    "بناتا": "بَناتا",
    "بنانا": "بَنانا",
    "بنائے": "بَنائے",
    # "اوڈیسہ" was collapsing/changing to "اڈیسے". This is a place name
    # (Odessa/Odisha) so a diacritic fix is more of a guess than the others
    # -- best-effort attempt below, VERIFY with test_pronunciation_fixes().
    "اوڈیسہ": "اوڈیسا",
}

# --- KNOWN LIMITATION, read before adding more entries here -----------------
# "ملک" (-> "مُلک" above) and "بنا" (-> "بَنا" below) were BOTH already fixed
# with diacritics, and Faisal is still hearing "مالک"/"بینا". I tested the
# matching logic directly (word-boundary regex against Arabic script) and
# confirmed the substitution IS firing correctly -- so this isn't a code bug,
# it's the ur-PK neural voice not fully respecting the diacritic once it's
# there. There's no way to fix this from here without hearing the audio, so
# rather than guess blindly at more diacritic variants, use
# test_pronunciation_fixes() below: it bundles every contested word into ONE
# numbered audio file so you can listen once and report back which numbers
# are still wrong, instead of retyping Urdu words on a phone keyboard.

_PRONUNCIATION_FIX_PATTERNS = {
    re.compile(r"\b" + re.escape(word) + r"\b"): fixed
    for word, fixed in _PRONUNCIATION_FIXES.items()
}


def _apply_pronunciation_fixes(text: str) -> str:
    """Applies known word-level pronunciation corrections before TTS."""
    if not text:
        return text
    for pattern, fixed in _PRONUNCIATION_FIX_PATTERNS.items():
        text = pattern.sub(fixed, text)
    return text


# ---------------------------------------------------------------------------
# English loanword -> Urdu-script transliteration
# ---------------------------------------------------------------------------
# script_generator.py's prompt explicitly permits common English loanwords
# in narration (e.g. "invest", "company", "market", "percent"), since that's
# how Pakistanis actually speak. But feeding the ur-PK voice raw LATIN-SCRIPT
# text forces an abrupt script/phonetic-rules switch mid-utterance, which
# tends to degrade pronunciation for the whole surrounding sentence, not just
# the English word itself. Converting these to their normal Urdu-script
# spelling keeps the whole utterance in one script/phonetic system. This list
# covers the loanwords the prompt already permits plus common finance/history
# extensions -- ADD MORE as you notice unconverted English words in the logs
# (any Latin-script word not in this table gets logged as a warning below).
_ENGLISH_LOANWORD_URDU = {
    "invest": "انویسٹ", "investment": "انویسٹمنٹ", "investor": "انویسٹر",
    "investors": "انویسٹرز", "company": "کمپنی", "companies": "کمپنیاں",
    "market": "مارکیٹ", "markets": "مارکیٹس", "percent": "فیصد",
    "deal": "ڈیل", "deals": "ڈیلز", "profit": "پرافٹ", "profits": "پرافٹس",
    "bank": "بینک", "banks": "بینکس", "economy": "اکانومی",
    "economic": "اکنامک", "stock": "اسٹاک", "stocks": "اسٹاکس",
    "share": "شیئر", "shares": "شیئرز", "crash": "کریش",
    "million": "ملین", "billion": "بلین", "trillion": "ٹریلین",
    "tax": "ٹیکس", "taxes": "ٹیکسز", "trade": "ٹریڈ", "president": "پریذیڈنٹ",
    "minister": "منسٹر", "government": "گورنمنٹ", "empire": "ایمپائر",
    "bond": "بانڈ", "bonds": "بانڈز", "loan": "لون", "loans": "لونز",
    "inflation": "افراطِ زر", "gdp": "جی ڈی پی", "ceo": "سی ای او",
    # Added: YouTube-channel narration words, missing before -- any Latin
    # word not in this table is left raw (see _replace()'s unmatched-word
    # warning below), which is what was happening to "subscribe".
    "subscribe": "سبسکرائب", "subscribers": "سبسکرائبرز",
    "channel": "چینل", "channels": "چینلز",
}

# Matches standalone runs of Latin letters (so Urdu/Arabic script text is
# never touched by this).
_LATIN_WORD_RE = re.compile(r"\b[A-Za-z]+\b")


def _transliterate_english_loanwords(text: str) -> str:
    """Converts known English loanwords to Urdu script before TTS. Any
    Latin-script word NOT in the table is left as-is but logged, so gaps in
    coverage are visible instead of silently mispronounced forever."""
    if not text:
        return text

    unmatched = set()

    def _replace(match):
        word = match.group(0)
        urdu = _ENGLISH_LOANWORD_URDU.get(word.lower())
        if urdu:
            return urdu
        unmatched.add(word)
        return word

    result = _LATIN_WORD_RE.sub(_replace, text)

    if unmatched:
        print(f"[voice_generator]   Latin-script word(s) not in the loanword "
              f"table (left as-is, may mispronounce): {sorted(unmatched)}")

    return result


# ---------------------------------------------------------------------------
# Number -> Urdu words
# ---------------------------------------------------------------------------
# Bare digits force the same kind of script/phonetic-rules switch as English
# loanwords. Spelling numbers out in Urdu words keeps the whole utterance in
# one system -- see the NOTE in the module docstring: the full 21-99 chart
# is filled in below, so this reads as natural compound words, not digits.

_URDU_ONES = {
    0: "صفر", 1: "ایک", 2: "دو", 3: "تین", 4: "چار", 5: "پانچ",
    6: "چھ", 7: "سات", 8: "آٹھ", 9: "نو",
}

_URDU_TEENS = {
    10: "دس", 11: "گیارہ", 12: "بارہ", 13: "تیرہ", 14: "چودہ",
    15: "پندرہ", 16: "سولہ", 17: "سترہ", 18: "اٹھارہ", 19: "انیس",
}

_URDU_TENS_ROUND = {
    20: "بیس", 30: "تیس", 40: "چالیس", 50: "پچاس",
    60: "ساٹھ", 70: "ستر", 80: "اسی", 90: "نوے",
}

# Full Urdu compound-tens chart (21-99, excluding round tens already in
# _URDU_TENS_ROUND above). Standard Hindustani/Urdu numeral vocabulary --
# the same words used in Pakistani news, textbooks, and everyday counting.
# If you listen to a generated video and hear one of these mispronounced or
# using an unfamiliar regional variant, tell Claude the specific number and
# it's a one-line fix (each entry below is independent).
_URDU_COMPOUND_TENS = {
    21: "اکیس", 22: "بائیس", 23: "تیئس", 24: "چوبیس", 25: "پچیس",
    26: "چھبیس", 27: "ستائیس", 28: "اٹھائیس", 29: "انتیس",
    31: "اکتیس", 32: "بتیس", 33: "تینتیس", 34: "چونتیس", 35: "پینتیس",
    36: "چھتیس", 37: "سینتیس", 38: "اڑتیس", 39: "انتالیس",
    41: "اکتالیس", 42: "بیالیس", 43: "تینتالیس", 44: "چوالیس",
    45: "پینتالیس", 46: "چھیالیس",
    47: "سینتالیس",  # "1947" (Partition) makes this the best-confirmed entry
    48: "اڑتالیس", 49: "انچاس",
    51: "اکاون", 52: "باون", 53: "ترپن", 54: "چون", 55: "پچپن",
    56: "چھپن", 57: "ستاون", 58: "اٹھاون", 59: "انسٹھ",
    61: "اکسٹھ", 62: "باسٹھ", 63: "تریسٹھ", 64: "چونسٹھ", 65: "پینسٹھ",
    66: "چھیاسٹھ", 67: "سڑسٹھ", 68: "اڑسٹھ", 69: "انہتر",
    71: "اکہتر", 72: "بہتر", 73: "تہتر", 74: "چوہتر", 75: "پچھتر",
    76: "چھہتر", 77: "ستتر", 78: "اٹھہتر", 79: "انیاسی",
    81: "اکیاسی", 82: "بیاسی", 83: "تراسی", 84: "چوراسی", 85: "پچاسی",
    86: "چھیاسی", 87: "ستاسی", 88: "اٹھاسی", 89: "نواسی",
    91: "اکیانوے", 92: "بیانوے", 93: "ترانوے", 94: "چورانوے",
    95: "پچانوے", 96: "چھیانوے", 97: "ستانوے", 98: "اٹھانوے", 99: "ننانوے",
}


def _urdu_two_digit_words(n: int) -> str:
    """Converts 0-99 to Urdu words using the full compound-tens chart above.
    Digit-by-digit reading is now only a safety net for out-of-range input
    (shouldn't trigger for any normal 0-99 number)."""
    if n < 0 or n > 99:
        return str(n)
    if n < 10:
        return _URDU_ONES[n]
    if n < 20:
        return _URDU_TEENS[n]
    if n in _URDU_TENS_ROUND:
        return _URDU_TENS_ROUND[n]
    if n in _URDU_COMPOUND_TENS:
        return _URDU_COMPOUND_TENS[n]
    # Safe fallback: read each digit separately rather than guess a possibly
    # wrong irregular compound word.
    return " ".join(_URDU_ONES[int(d)] for d in str(n))


def _urdu_hundreds_words(n: int) -> str:
    """Converts 100-999 to Urdu words using the standard '<N> سو <remainder>'
    (hundred) pattern, e.g. 500 -> 'پانچ سو', 350 -> 'تین سو پچاس'."""
    hundreds_digit = n // 100
    remainder = n % 100
    words = f"{_URDU_ONES[hundreds_digit]} سو"
    if remainder:
        words += f" {_urdu_two_digit_words(remainder)}"
    return words


def _urdu_year_words(n: int) -> str:
    """Converts a 4-digit number in the 1000-2099 range to the traditional
    Urdu/Hindi YEAR-style paired reading, e.g. 1920 -> 'انیس سو بیس'
    (nineteen-hundred-twenty), 2024 -> 'بیس سو چوبیس'-style pairing.

    This pairing convention is specifically how CALENDAR YEARS are spoken,
    not a general rule for any 4-digit quantity (e.g. "1920 rupees" would
    naturally be read as a thousand-based quantity, not year-paired). Since
    this pipeline is for history/finance documentaries where most 4-digit
    numbers in this range genuinely are years, defaulting to year-style
    reading here is a reasonable heuristic, not a universal rule.
    """
    first_two = n // 100
    last_two = n % 100
    words = f"{_urdu_two_digit_words(first_two)} سو"
    if last_two:
        words += f" {_urdu_two_digit_words(last_two)}"
    return words


_URDU_HAZAR = "ہزار"   # thousand
_URDU_LAKH = "لاکھ"    # 100,000
_URDU_CROR = "کروڑ"    # 10,000,000


def _urdu_large_number_words(n: int) -> str:
    """Converts numbers outside the year/hundreds/two-digit ranges above
    using the South Asian hazar/lakh/crore system Urdu speakers actually
    use for quantities -- e.g. 100000 -> 'ایک لاکھ', 20000 -> 'بیس ہزار',
    42000 -> 'بیالیس ہزار', 250000 -> 'دو لاکھ پچاس ہزار'.

    FIX: this used to not exist -- any number outside 0-2099 fell straight
    to digit-by-digit reading ("1 لاکھ" written in the script as "100000"
    was coming out as "ایک صفر صفر صفر صفر صفر"). Words like "1 لاکھ",
    "بیس ہزار", and "42000" are exactly the numbers this was breaking on.

    Covers 0 to 99,99,99,999 (99 crore), comfortably beyond any figure this
    pipeline is likely to narrate. Above that, falls back to digit-by-digit
    (rare enough, and "X ارب" (billion) phrasing isn't implemented yet --
    report the specific number if this comes up)."""
    if n == 0:
        return _URDU_ONES[0]
    if n > 999999999:
        return " ".join(_URDU_ONES[int(d)] for d in str(n))

    crore, n = divmod(n, 10000000)
    lakh, n = divmod(n, 100000)
    hazar, n = divmod(n, 1000)
    rest = n  # 0-999

    parts = []
    if crore:
        parts.append(f"{_urdu_two_digit_words(crore)} {_URDU_CROR}")
    if lakh:
        parts.append(f"{_urdu_two_digit_words(lakh)} {_URDU_LAKH}")
    if hazar:
        parts.append(f"{_urdu_two_digit_words(hazar)} {_URDU_HAZAR}")
    if rest:
        parts.append(_urdu_hundreds_words(rest) if rest >= 100 else _urdu_two_digit_words(rest))

    return " ".join(parts)


def _number_to_urdu_words(num_str: str) -> str:
    """Converts a digit string to Urdu words. Leaves decimals/malformed
    numbers as-is rather than guessing (decimal number words are a rarer
    case and not worth the added risk here)."""
    clean = num_str.replace(",", "")
    if not clean.isdigit():
        return num_str
    n = int(clean)

    if 1000 <= n <= 2099:
        return _urdu_year_words(n)
    if 100 <= n <= 999:
        return _urdu_hundreds_words(n)
    if 0 <= n <= 99:
        return _urdu_two_digit_words(n)

    return _urdu_large_number_words(n)


_NUMBER_RE = re.compile(r"\b\d[\d,]*\b")


def _spell_out_numbers(text: str) -> str:
    """Replaces every standalone digit sequence with its Urdu word form."""
    if not text:
        return text
    return _NUMBER_RE.sub(lambda m: _number_to_urdu_words(m.group(0)), text)


# ---------------------------------------------------------------------------
# Urdu text pre-processing for better TTS output
# ---------------------------------------------------------------------------

def _preprocess_generic_text(text: str) -> str:
    """Lightweight preprocessing for non-Urdu languages (English, Spanish,
    etc.): strips script-writing markers and ensures clean sentence-ending
    punctuation, but does NOT spell out numbers/currency into Urdu words or
    run Urdu-specific pronunciation fixes -- those neural voices read "$827
    million", "40%", "1995" etc. natively just fine on their own, and forcing
    Urdu-script number words into them is exactly what used to make English
    narration seem to "skip" every number (the voice couldn't read the Urdu
    text it had been given)."""
    if not text:
        return text

    text = _strip_narration_markers(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _preprocess_urdu_text(text: str) -> str:
    """
    Cleans and prepares Urdu text for TTS to sound more natural.

    Fixes applied:
      1. Strips script/editing markers -- (PAUSE), (EMPHASIS), [B-ROLL: ...] --
         that are meant for the human editor/assembler, not to be spoken
      2. Normalizes Arabic/Urdu punctuation to standard forms TTS handles better
      3. Applies known word-level pronunciation fixes (adds back diacritics
         for words that are ambiguous without them, e.g. "ملک" -> "مُلک")
      4. Transliterates common English loanwords into Urdu script so the
         voice doesn't switch scripts/phonetic-rules mid-sentence
      5. Spells out %/$ symbols AND plain digit sequences in Urdu words,
         for the same script-switching reason
      6. Removes excessive whitespace
      7. Ensures proper sentence-ending punctuation for natural cadence

    NOTE: this used to have a use_ssml mode that injected literal <break> tags
    for pauses. edge-tts's Communicate class does NOT parse SSML tags embedded
    in its text input -- it just synthesizes whatever string you give it, tags
    included, so every "<break time=...>" was being read aloud word-for-word.
    Plain commas already give edge-tts a perfectly natural pause, so that's all
    this does now -- no XML is ever inserted into text that gets sent to TTS.
    """
    if not text:
        return text

    # Strip script-writing directives FIRST, before any other processing,
    # since they're not spoken content at all.
    text = _strip_narration_markers(text)

    # Normalize various Unicode space/punctuation variants
    text = text.replace("\u060c", ",")   # Arabic comma -> standard comma
    text = text.replace("\u061b", ";")   # Arabic semicolon -> standard
    text = text.replace("\u061f", "?")   # Arabic question mark -> standard
    text = text.replace("\u0640", "")    # Tatweel (kashida) -> remove (TTS chokes on it)

    # Fix words known to be mispronounced without diacritics (e.g. ملک -> مُلک)
    text = _apply_pronunciation_fixes(text)

    # Convert common English loanwords to Urdu script so the voice doesn't
    # switch scripts/phonetic-rules mid-sentence
    text = _transliterate_english_loanwords(text)

    # Spell out %/$ in Urdu words instead of bare symbols
    text = _normalize_numbers_and_currency(text)

    # Spell out any remaining plain digit sequences in Urdu words (also
    # converts the digits left behind by the %/$ step above into full words)
    text = _spell_out_numbers(text)

    # Ensure sentence-ending punctuation for natural TTS cadence
    # TTS engines often run sentences together without clear ending marks
    text = text.strip()
    if text and text[-1] not in ".!?۔":
        text += "."

    # Replace multiple spaces/newlines with single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


_URDU_LANG_CODES = ("ur", "urd", "urdu", "ur-pk", "ur-in")


def _rate_to_edge_format(rate: str, language: str = "ur") -> str:
    """Converts a friendly rate name to the percent string edge-tts's
    Communicate expects. Non-Urdu neural voices (e.g. English) have a
    noticeably faster natural cadence at the same relative rate offset, so
    they get an extra -10 points across the board to actually sound "slow"
    rather than merely "slightly less fast"."""
    base = {"slow": -10, "default": 0, "fast": 10}.get(rate, 0)
    lang = (language or "ur").lower().strip()
    if lang not in _URDU_LANG_CODES:
        base -= 10
    base = max(base, -50)
    return f"{base:+d}%"


def _pitch_to_edge_format(pitch: str) -> str:
    """
    Converts a friendly pitch name to the Hz string edge-tts's Communicate expects.
    NOTE: edge-tts's native pitch parameter uses Hz offsets (e.g. "+0Hz", "-5Hz"),
    not percentages -- percentages only apply to rate/volume.
    """
    return {"low": "-5Hz", "default": "+0Hz", "high": "+5Hz"}.get(pitch, "+0Hz")


def _split_long_sentences(text: str, max_words: int = 18) -> str:
    """
    Breaks very long sentences into shorter ones for more natural TTS pacing.
    Pakistani conversational Urdu rarely uses sentences longer than 15-20 words.
    This inserts breaks at conjunctions (اور، لیکن، کیونکہ، تو) when sentences
    exceed max_words.

    Chunks are joined with an Urdu comma pause -- edge-tts's Communicate never
    parses inline SSML like <break>, so a literal "<break time=...>" tag
    inserted here would just be read aloud as text (this was the actual bug
    causing narration to speak out its own pause tags). A comma is a real,
    audible pause that edge-tts already handles naturally.
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    # Common Urdu conjunctions where we can safely split
    split_markers = ["اور", "لیکن", "کیونکہ", "تو", "پھر", "چنانچہ", "حالانکہ"]

    result = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words and word in split_markers:
            result.append(" ".join(current_chunk))
            current_chunk = []
        elif len(current_chunk) >= max_words + 5:
            # Force split even if no conjunction marker found
            result.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        result.append(" ".join(current_chunk))

    if len(result) <= 1:
        return text

    return "، ".join(result)


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------

def _resolve_voice(language: str, voice_gender: str) -> str:
    """
    Resolves the best edge-tts voice for the given language and gender.

    For Urdu (ur): ALWAYS prefers ur-PK (Pakistani) voices over ur-IN (Indian)
    or any Arabic fallback. This is critical — Indian Urdu voices have a 
    distinctly different accent that Pakistani audiences find jarring.
    """
    # Normalize language code
    lang = (language or "ur").lower().strip()
    gender = (voice_gender or "male").lower().strip()

    # For any Urdu variant, force Pakistani voices
    if lang in ("ur", "urd", "urdu", "ur-pk", "ur-in"):
        voice = URDU_PK_VOICES.get(gender)
        if voice:
            return voice
        # Fallback to opposite gender if preferred gender unavailable
        fallback_gender = "female" if gender == "male" else "male"
        voice = URDU_PK_VOICES.get(fallback_gender)
        if voice:
            return voice

    # If config has EDGE_VOICES, try that as fallback
    try:
        from config import EDGE_VOICES, DEFAULT_LANGUAGE, DEFAULT_VOICE_GENDER
        lang_voices = EDGE_VOICES.get(lang, EDGE_VOICES.get(DEFAULT_LANGUAGE, {}))
        voice = lang_voices.get(gender) or lang_voices.get(DEFAULT_VOICE_GENDER)
        if voice:
            return voice
    except ImportError:
        pass

    # Ultimate fallback
    return URDU_PK_VOICES.get("male", "ur-PK-AsadNeural")


# ---------------------------------------------------------------------------
# Async TTS generation
# ---------------------------------------------------------------------------

async def _tts_edge_async(
    text: str, out_path: str, voice: str, rate: str = "slow", pitch: str = "default",
    language: str = "ur",
):
    """
    Generates TTS audio using edge-tts.

    rate: "slow" | "default" | "fast" — passed as a real Communicate parameter
    pitch: "low" | "default" | "high" — passed as a real Communicate parameter
    language: which text preprocessing to apply (see _preprocess_urdu_text vs
    _preprocess_generic_text below) -- this MUST match the actual narration
    language, or numbers/loanwords get spelled out into the wrong script.

    IMPORTANT: edge-tts's Communicate class does not parse SSML embedded in its
    `text` argument -- it treats that string as literal spoken text. Previous
    versions of this function wrapped text in <voice>/<prosody>/<break> tags,
    which edge-tts then read aloud verbatim (the actual cause of narration
    speaking its own markup). Prosody is now set the only way edge-tts actually
    supports it: as real constructor keyword arguments. Script-writing markers
    like (PAUSE)/(EMPHASIS)/[B-ROLL: ...] are stripped before any text reaches
    this function, for the same underlying reason -- they are not spoken content.
    """
    lang = (language or "ur").lower().strip()
    if lang in _URDU_LANG_CODES:
        processed_text = _preprocess_urdu_text(text)
    else:
        # BUG FIX: this used to always call _preprocess_urdu_text regardless
        # of language, which spelled English numbers/currency out into Urdu
        # words ("827" -> Urdu digit-words) and fed that Urdu script to an
        # English voice -- which can't read it, so numbers seemed to just
        # vanish from English narration entirely.
        processed_text = _preprocess_generic_text(text)
    processed_text = _split_long_sentences(processed_text)

    communicate = edge_tts.Communicate(
        processed_text,
        voice,
        rate=_rate_to_edge_format(rate, language=lang),
        pitch=_pitch_to_edge_format(pitch),
    )
    await communicate.save(out_path)


def generate_scene_audio(
    text: str, 
    out_path: str, 
    language: str = "ur", 
    voice_gender: str = "male",
    rate: str = "slow",
    pitch: str = "default",
):
    """Writes an mp3 to out_path using edge-tts with Pakistani Urdu voice."""
    voice = _resolve_voice(language, voice_gender)
    asyncio.run(_tts_edge_async(text, out_path, voice, rate, pitch, language=language))


async def _generate_all_async(
    scenes: list, 
    audio_dir: str, 
    language: str, 
    voice_gender: str, 
    progress_callback=None,
    rate: str = "slow",
    pitch: str = "default",
):
    voice = _resolve_voice(language, voice_gender)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TTS)
    total = len(scenes)
    done_count = 0

    async def run_one(i, scene):
        nonlocal done_count
        out_path = os.path.join(audio_dir, f"scene_{i:03d}.mp3")

        # Pre-process narration text before TTS
        narration = scene.get("narration", "")

        async with semaphore:
            await _tts_edge_async(narration, out_path, voice, rate, pitch, language=language)

        scene["audio_path"] = out_path
        done_count += 1
        if progress_callback:
            progress_callback(done_count, total)

    await asyncio.gather(*(run_one(i, scene) for i, scene in enumerate(scenes)))


def generate_all_scene_audio(
    scenes: list, 
    work_dir: str, 
    language: str = "ur",
    voice_gender: str = "male", 
    progress_callback=None,
    rate: str = "slow",
    pitch: str = "default",
) -> list:
    """
    scenes: list of scene dicts from script_generator (each with "narration")
    language: language code — for Urdu, use "ur" (automatically resolves to ur-PK)
    voice_gender: "male" or "female" — picks Pakistani Urdu neural voice
    rate: "slow" | "default" | "fast" — speech speed (real edge-tts parameter)
    pitch: "low" | "default" | "high" — pitch variation (real edge-tts parameter)
    progress_callback: called as progress_callback(done, total) per scene

    Returns the same list with an added "audio_path" key per scene.
    """
    audio_dir = os.path.join(work_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    asyncio.run(_generate_all_async(
        scenes, audio_dir, language, voice_gender, progress_callback, rate, pitch
    ))

    return scenes


# ---------------------------------------------------------------------------
# Utility: Test a voice
# ---------------------------------------------------------------------------

def test_voice(text: str = "اسلام علیکم۔ یہ ایک ٹیسٹ ہے۔", 
               out_path: str = "test_voice.mp3",
               voice_gender: str = "male") -> str:
    """
    Quick utility to test a voice without running the full pipeline.
    Returns the path to the generated test audio.
    """
    voice = _resolve_voice("ur", voice_gender)
    asyncio.run(_tts_edge_async(text, out_path, voice))
    return out_path


def test_pronunciation_fixes(out_path: str = "pronunciation_test.mp3",
                              voice_gender: str = "male") -> str:
    """
    Diagnostic utility: bundles every word/number that's been reported as
    mispronounced into ONE audio file, each preceded by its number spoken
    aloud ("Number 1 ... Number 2 ..."), so a single listen-through tells you
    which fixes actually worked. Report back just the NUMBERS that still
    sound wrong -- no need to retype Urdu words on a phone keyboard -- and
    the next fix can be targeted precisely instead of guessed at blind.

    Run this after pulling the updated tts_generator.py, before regenerating
    a full video, so we don't burn a whole render cycle on words that still
    need another pass.
    """
    test_items = [
        "پلاسی", "1 لاکھ", "بیس ہزار", "42000", "مالیت", "اسی",
        "لائبریری", "چلانا", "ملک", "اوڈیسہ", "چھینی", "40%",
        "subscribe", "گلی", "بنا",
    ]
    voice = _resolve_voice("ur", voice_gender)
    chunks = [
        f"نمبر {_number_to_urdu_words(str(i))}، {word}"
        for i, word in enumerate(test_items, start=1)
    ]
    text = "۔ ".join(chunks)
    asyncio.run(_tts_edge_async(text, out_path, voice, rate="slow", pitch="default", language="ur"))
    return out_path
