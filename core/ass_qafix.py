"""
ASS QA & Auto-Fixer for subtitle workflows.

Ported from subs-tools/ass-qafix.py — business logic only (no CLI, no Rich tables).

Key behaviors:
- De-duplicates Dialogue lines by default (same Start/End/Style/Text).
- Merges consecutive dialogue lines with identical text and timestamps within 500ms gap.
- Fixes overlapping timestamps between consecutive dialogues (within 0.02s threshold).
- Validates and normalizes Style names against defined styles.
- Sanitizes Layer to valid integer.
- Trims Text; treats "-", "–", "—", "/" as empty OCR artifacts.
- Removes fake text (OCR artifacts like single digits, single letters, underscores).
- Preserves commas/spaces inside Text (safe maxsplit).
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

from rapidfuzz.distance import Levenshtein

# Optional jieba3 for CJK character validation
try:
    from jieba3.tok import BASE_MODEL_FREQ
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

DIALOGUE_PREFIX = "Dialogue:"
FORMAT_PREFIX = "Format:"
STYLES_SECTION_HDR = "[V4+ Styles]"
EVENTS_SECTION_HDR = "[Events]"

CANON_EVENTS_FIELDS = [
    "Layer",
    "Start",
    "End",
    "Style",
    "Name",
    "MarginL",
    "MarginR",
    "MarginV",
    "Effect",
    "Text",
]

FAKE_TEXT_PATTERNS = [
    re.compile(
        r"^\s*(Format:|Layer\s*,\s*Start\s*,\s*End\s*,\s*Style\s*,\s*Name\s*,"
        r"\s*MarginL\s*,\s*MarginR\s*,\s*MarginV\s*,\s*Effect\s*,\s*Text)\s*$",
        re.I,
    ),
]

# Single-glyph OCR artifacts considered "empty" after trimming
ARTIFACT_EMPTY_TEXT = {"-", "\u2013", "\u2014", "\u4e00", "/", "\u53e3"}

# Timestamp regex: H:MM:SS.cc (H = 1+ digits)
TIME_RE = re.compile(r"^\d+:\d{2}:\d{2}\.\d{2}$")

# ASS override tag removal pattern (matches {anything except closing brace})
ASS_TAG_RE = re.compile(r"\{[^}]*\}")

# Minimum duration threshold for dialogue lines (250ms = 25 centiseconds)
# Lines shorter than this are likely OCR merge errors
MIN_DURATION_CS = 25

# Threshold for short duration removal (50ms = 5 centiseconds)
SHORT_DURATION_THRESHOLD_CS = 5


@dataclass
class QAStats:
    total_lines: int = 0
    dialogue_lines: int = 0
    fixed_lines: int = 0
    style_fixes: int = 0
    empty_text_removed: int = 0
    fake_text_removed: int = 0
    duplicates_removed: int = 0
    consecutive_merges: int = 0
    alternating_merges: int = 0
    overlap_fixes: int = 0
    short_duration_removed: int = 0
    newline_garbage_cleaned: int = 0
    label_short_removed: int = 0
    label_dupes_removed: int = 0


@dataclass
class ASSDocument:
    lines: List[str]
    styles: Set[str] = field(default_factory=set)
    events_format: List[str] = field(default_factory=lambda: copy.deepcopy(CANON_EVENTS_FIELDS))
    events_format_line_index: Optional[int] = None
    events_section_start: Optional[int] = None
    events_section_end: Optional[int] = None
    styles_section_start: Optional[int] = None
    styles_section_end: Optional[int] = None
    styles_block_lines: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def strip_ass_tags(text: str) -> str:
    """Remove ASS override tags from text, returning only visible content."""
    clean = ASS_TAG_RE.sub("", text)
    clean = clean.replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def timestamp_to_seconds(ts: str) -> float:
    """Convert MM:SS.cc or H:MM:SS.cc timestamp to total seconds."""
    parts = ts.split(":")
    if len(parts) == 2:
        minutes, sec_cs = parts
        hours = 0
    else:
        hours, minutes, sec_cs = parts
    sec, cs = sec_cs.split(".")
    return int(hours) * 3600 + int(minutes) * 60 + int(sec) + int(cs) / 100


def is_valid_time(s: str) -> bool:
    return bool(TIME_RE.match(s))


def extract_minute(timestamp: str) -> Optional[int]:
    """Extract minute (MM) from H:MM:SS.cc timestamp."""
    if not is_valid_time(timestamp):
        return None
    parts = timestamp.split(":")
    return int(parts[1])


def time_to_cs(s: str) -> Optional[int]:
    if not is_valid_time(s):
        return None
    h, m, rest = s.split(":")
    sec, cs = rest.split(".")
    return (int(h) * 3600 + int(m) * 60 + int(sec)) * 100 + int(cs)


def cs_to_time(cs: int) -> str:
    """Convert centiseconds back to H:MM:SS.cc timestamp format."""
    total_seconds = cs // 100
    centiseconds = cs % 100
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


# ---------------------------------------------------------------------------
# Dictionary-based OCR artifact detection
# ---------------------------------------------------------------------------

_enchant_dict = None


def _get_enchant_dict():
    """Get the enchant English dictionary, lazy-loaded."""
    global _enchant_dict
    if _enchant_dict is None:
        try:
            try:
                stderr_fd = os.dup(2)
            except OSError:
                # No usable fd 2 (a GUI build without a console): nothing to
                # silence, and the dictionary must still load -- without it
                # the QA pass keeps words it would drop.
                stderr_fd = None
            if stderr_fd is None:
                import enchant
                _enchant_dict = enchant.Dict("en_US")
            else:
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, 2)
                try:
                    import enchant
                    _enchant_dict = enchant.Dict("en_US")
                finally:
                    os.dup2(stderr_fd, 2)
                    os.close(devnull)
                    os.close(stderr_fd)
        except Exception:
            _enchant_dict = False  # Mark as unavailable
    return _enchant_dict if _enchant_dict else None


def is_valid_english_word(word: str) -> bool:
    """Check if a short English word is valid using enchant dictionary."""
    d = _get_enchant_dict()
    if d is None:
        return True
    return d.check(word) or d.check(word.lower())


def is_valid_cjk_char(char: str) -> bool:
    """Check if a single CJK character is a valid word using jieba3 dictionary."""
    if not _HAS_JIEBA:
        return True  # If jieba3 unavailable, don't filter
    return BASE_MODEL_FREQ.get(char, 0) > 0


def is_fake_text(text: str) -> bool:
    t = text.strip()
    if t == "":
        return False
    for pat in FAKE_TEXT_PATTERNS:
        if pat.search(t):
            return True
    lower = t.lower().replace(" ", "")
    if "layer,start,end,style,name,marginl,marginr,marginv,effect,text" in lower:
        return True

    # Multi-dash patterns (OCR artifacts)
    if re.fullmatch(r"-[\s-]*-+", t):
        return True

    # Backticks (OCR artifacts)
    if re.fullmatch(r"`+", t):
        return True

    # Contains replacement character (garbled OCR)
    if "\u25a1" in t:
        return True

    # OCR artifacts: single digits/numbers, single punctuation, underscores
    if re.fullmatch(r"\d+", t):
        return True
    if re.fullmatch(r"[^\w\s]", t):
        return True
    if re.fullmatch(r"_+", t):
        return True

    # Single ASCII letter: only "I" is valid
    if re.fullmatch(r"[A-Za-z]", t):
        if t != "I":
            return True

    # Two ASCII letters: check against English dictionary
    if re.fullmatch(r"[A-Za-z]{2}", t):
        if not is_valid_english_word(t):
            return True

    # Single letter with trailing punctuation/garbage
    single_letter_match = re.fullmatch(r"([A-Za-z])[\s.\-]+", t)
    if single_letter_match:
        if single_letter_match.group(1) != "I":
            return True

    # Two letters with trailing punctuation
    two_letter_match = re.fullmatch(r"([A-Za-z]{2})[\s.\-]+", t)
    if two_letter_match:
        if not is_valid_english_word(two_letter_match.group(1)):
            return True

    # Single CJK character: check against jieba3 dictionary
    if len(t) == 1 and "\u4e00" <= t <= "\u9fff":
        if not is_valid_cjk_char(t):
            return True

    return False


def is_newline_segment_garbage(segment: str, conservative: bool = False) -> bool:
    """Check if a segment around \\N is OCR garbage."""
    s = segment.strip()
    if not s:
        return True

    if re.fullmatch(r"\d+", s):
        return True
    if re.fullmatch(r"[^\w\s]", s):
        return True
    if re.fullmatch(r"`+", s):
        return True
    if re.fullmatch(r"_+", s):
        return True

    if conservative:
        return False

    if re.fullmatch(r"[A-Za-z]", s):
        return True

    if re.fullmatch(r"[A-Za-z]{2}", s):
        if not is_valid_english_word(s):
            return True

    if len(s) == 1 and "\u4e00" <= s <= "\u9fff":
        return True

    return False


def clean_newline_garbage(text: str, conservative: bool = False) -> Tuple[str, bool]:
    """Clean up OCR garbage segments around \\N line breaks."""
    if "\\N" not in text:
        return text, False

    segments = text.split("\\N")
    valid_segments = []

    for seg in segments:
        clean_seg = ASS_TAG_RE.sub("", seg).strip()
        if not is_newline_segment_garbage(clean_seg, conservative):
            valid_segments.append(seg)

    if len(valid_segments) == len(segments):
        return text, False

    if valid_segments:
        if len(valid_segments) == 1:
            return valid_segments[0], True
        return "\\N".join(valid_segments), True
    else:
        return "", True


# ---------------------------------------------------------------------------
# Text matching / merging helpers
# ---------------------------------------------------------------------------

def texts_match_for_merge(
    text1: str, text2: str,
    duration1_cs: Optional[int] = None,
    duration2_cs: Optional[int] = None,
) -> bool:
    """Check if two texts should be considered matching for merge purposes."""
    t1 = strip_ass_tags(text1)
    t2 = strip_ass_tags(text2)

    if t1 == t2:
        return True
    either_short = (
        (duration1_cs is not None and duration1_cs < MIN_DURATION_CS)
        or (duration2_cs is not None and duration2_cs < MIN_DURATION_CS)
    )
    if either_short:
        return Levenshtein.distance(t1, t2) == 1
    return False


def texts_are_ocr_variants(text1: str, text2: str, min_similarity: float = 0.7) -> bool:
    """Check if two texts are OCR variants of each other."""
    t1 = strip_ass_tags(text1)
    t2 = strip_ass_tags(text2)

    if t1 == t2:
        return True
    if not t1 or not t2:
        return False

    max_len = max(len(t1), len(t2))
    distance = Levenshtein.distance(t1, t2)
    similarity = 1.0 - (distance / max_len)

    return similarity >= min_similarity


# ---------------------------------------------------------------------------
# Parsing / loading / saving
# ---------------------------------------------------------------------------

def parse_sections(lines: List[str]) -> Dict[str, Tuple[int, int]]:
    sections: Dict[str, Tuple[int, int]] = {}
    order: List[Tuple[str, int]] = []
    for i, line in enumerate(lines):
        m = re.match(r"^\s*\[(.+?)\]\s*$", line)
        if m:
            order.append((m.group(1), i))
    for i, (name, start) in enumerate(order):
        end = order[i + 1][1] if i + 1 < len(order) else len(lines)
        sections[name] = (start, end)
    return sections


def parse_styles(doc: ASSDocument) -> None:
    sections = parse_sections(doc.lines)
    if "V4+ Styles" not in sections:
        return
    start, end = sections["V4+ Styles"]
    doc.styles_section_start, doc.styles_section_end = start, end
    doc.styles_block_lines = doc.lines[start:end]

    fmt_fields: Optional[List[str]] = None
    for i in range(start, end):
        line = doc.lines[i].rstrip("\n")
        if line.strip().startswith(FORMAT_PREFIX):
            fmt_fields = [f.strip() for f in line.split(":", 1)[1].split(",")]
            break
    if fmt_fields:
        try:
            name_idx = [s.lower() for s in fmt_fields].index("name")
        except ValueError:
            name_idx = None
        for i in range(start, end):
            raw = doc.lines[i].rstrip("\n")
            if raw.strip().lower().startswith("style:"):
                payload = raw.split(":", 1)[1].lstrip()
                parts = [p.strip() for p in payload.split(",")]
                if name_idx is not None and name_idx < len(parts):
                    doc.styles.add(parts[name_idx])
                elif parts:
                    doc.styles.add(parts[0])


def parse_events_format(doc: ASSDocument) -> None:
    sections = parse_sections(doc.lines)
    if "Events" not in sections:
        return
    start, end = sections["Events"]
    doc.events_section_start, doc.events_section_end = start, end
    for i in range(start, end):
        line = doc.lines[i].rstrip("\n")
        if line.strip().startswith(FORMAT_PREFIX):
            fields = [f.strip() for f in line.split(":", 1)[1].split(",")]
            normalized = [
                next((c for c in CANON_EVENTS_FIELDS if c.lower() == f.lower()), f)
                for f in fields
            ]
            doc.events_format = normalized
            doc.events_format_line_index = i
            break
    if doc.events_format is None:
        doc.events_format = copy.deepcopy(CANON_EVENTS_FIELDS)


def load_ass(path: str) -> ASSDocument:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.read().splitlines()
    doc = ASSDocument(lines=lines)
    parse_styles(doc)
    parse_events_format(doc)
    return doc


def save_ass(path: str, doc: ASSDocument) -> None:
    out_lines = doc.lines[:]
    fmt_line = f"{FORMAT_PREFIX} {', '.join(doc.events_format)}"
    if doc.events_format_line_index is not None:
        out_lines[doc.events_format_line_index] = fmt_line
    else:
        if doc.events_section_start is not None:
            idx = doc.events_section_start + 1
            out_lines.insert(idx, fmt_line)
        else:
            out_lines += ["", EVENTS_SECTION_HDR, fmt_line]
        doc.lines = out_lines
    # newline="\n": LF on every OS (Windows' text mode would write CRLF).
    with open(path, "w", encoding="utf-8", errors="replace", newline="\n") as f:
        f.write("\n".join(doc.lines) + "\n")


def split_dialogue_payload(payload: str, expected_fields: int) -> List[str]:
    """Split payload into exactly `expected_fields` items."""
    parts = payload.split(",", expected_fields - 1)
    parts = [p if i == len(parts) - 1 else p.strip() for i, p in enumerate(parts)]
    if len(parts) < expected_fields:
        if not parts:
            return [""] * (expected_fields - 1) + [""]
        head, text = parts[:-1], parts[-1]
        while len(head) < expected_fields - 1:
            head.append("")
        return head + [text]
    return parts


def sanitize_int(val: str) -> Tuple[str, bool]:
    v = val.strip()
    changed = False
    if v == "" or not re.fullmatch(r"-?\d+", v):
        v = "0"
        changed = True
    if v.startswith("-"):
        v = "0"
        changed = True
    return v, changed


def rebuild_dialogue_line(fields_order: List[str], values: Dict[str, str]) -> str:
    parts = [values.get(k, "") for k in fields_order]
    return f"{DIALOGUE_PREFIX} " + ",".join(parts)


# ---------------------------------------------------------------------------
# Styles canonicalization
# ---------------------------------------------------------------------------

def replace_styles_with_canonical(doc: ASSDocument, canonical_block: List[str]) -> None:
    sections = parse_sections(doc.lines)
    if "V4+ Styles" not in sections:
        insert_at = sections["Events"][0] if "Events" in sections else len(doc.lines)
        doc.lines = doc.lines[:insert_at] + canonical_block + doc.lines[insert_at:]
        parse_styles(doc)
        return
    start, end = sections["V4+ Styles"]
    doc.lines = doc.lines[:start] + canonical_block + doc.lines[end:]
    parse_styles(doc)


def normalize_styles_block(block: List[str]) -> List[str]:
    return [ln.strip() for ln in block]


# ---------------------------------------------------------------------------
# Dialogue-level processing
# ---------------------------------------------------------------------------

def process_dialogue_line(
    raw_line: str,
    fields_order: List[str],
    styles: Set[str],
    stats: QAStats,
    keep_empty_text: bool,
    double_dialogue: bool = False,
    label_min: int = 2,
) -> Optional[str]:
    stats.dialogue_lines += 1
    payload = raw_line.split(":", 1)[1].lstrip()
    expected = len(fields_order)

    parts = split_dialogue_payload(payload, expected)
    values = {fields_order[i]: parts[i] if i < len(parts) else "" for i in range(expected)}
    changed_any = False

    # Layer
    layer, ch = sanitize_int(values.get("Layer", "0"))
    if ch or layer != values.get("Layer", ""):
        changed_any = True
    values["Layer"] = layer

    # Start/End
    values["Start"] = (values.get("Start", "") or "").strip()
    values["End"] = (values.get("End", "") or "").strip()

    # Style
    original_style = values.get("Style", "")
    style = original_style.strip() or "Default"

    style_normalized = style.lower()
    valid_styles_lower = {s.lower() for s in styles} if styles else set()

    if (
        not styles
        or not original_style.strip()
        or style_normalized not in valid_styles_lower
    ):
        style = "Default"
        stats.style_fixes += 1
        changed_any = True
    else:
        for defined_style in styles:
            if defined_style.lower() == style_normalized:
                style = defined_style
                break

    values["Style"] = style

    # Text cleanup
    text = values.get("Text", "")
    text = re.sub(r"[\ufeff\u200b\u200e\u200f]", "", text)
    text = text.strip()

    text, newline_cleaned = clean_newline_garbage(text, conservative=double_dialogue)
    if newline_cleaned:
        stats.newline_garbage_cleaned += 1
        changed_any = True

    clean_text = strip_ass_tags(text)

    if clean_text in ARTIFACT_EMPTY_TEXT:
        clean_text = ""

    if is_fake_text(clean_text):
        stats.fake_text_removed += 1
        return None

    if text.startswith(r"{\pos") and len(clean_text) < label_min:
        stats.label_short_removed += 1
        return None

    if not keep_empty_text and clean_text == "":
        stats.empty_text_removed += 1
        return None

    values["Text"] = text
    if changed_any:
        stats.fixed_lines += 1
    return rebuild_dialogue_line(fields_order, values)


# ---------------------------------------------------------------------------
# Batch dialogue operations (dedup, merge, overlap fix, short removal)
# ---------------------------------------------------------------------------

def dedupe_dialogues(dialogue_lines: List[str], fields_order: List[str]) -> Tuple[List[str], int]:
    seen: Set[Tuple[str, str, str, str]] = set()
    out: List[str] = []
    removed = 0
    for line in dialogue_lines:
        payload = line.split(":", 1)[1].lstrip()
        parts = split_dialogue_payload(payload, len(fields_order))
        m = {fields_order[i]: parts[i] if i < len(fields_order) else "" for i in range(len(fields_order))}
        raw_text = (m.get("Text", "") or "").strip()
        clean_text = strip_ass_tags(raw_text)
        key = (m.get("Start", ""), m.get("End", ""), m.get("Style", ""), clean_text)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(line)
    return out, removed


def merge_consecutive_dialogues(
    dialogue_lines: List[str], fields_order: List[str],
) -> Tuple[List[str], int]:
    """Merge consecutive dialogue lines with identical text within 500ms gap."""
    MAX_GAP_CS = 50
    if not dialogue_lines:
        return dialogue_lines, 0

    parsed_dialogues = []
    for line in dialogue_lines:
        payload = line.split(":", 1)[1].lstrip()
        parts = split_dialogue_payload(payload, len(fields_order))
        m = {fields_order[i]: parts[i] if i < len(fields_order) else "" for i in range(len(fields_order))}
        parsed_dialogues.append({
            "original_line": line,
            "start": m.get("Start", "").strip(),
            "end": m.get("End", "").strip(),
            "style": m.get("Style", "").strip(),
            "name": m.get("Name", "").strip(),
            "marginl": m.get("MarginL", "").strip(),
            "marginr": m.get("MarginR", "").strip(),
            "marginv": m.get("MarginV", "").strip(),
            "effect": m.get("Effect", "").strip(),
            "text": (m.get("Text", "") or "").strip(),
        })

    if len(parsed_dialogues) < 2:
        return dialogue_lines, 0

    merged_dialogues = []
    merges_count = 0
    i = 0

    while i < len(parsed_dialogues):
        current = parsed_dialogues[i]
        merged_start = current["start"]
        merged_end = current["end"]
        merged_text = current["text"]

        j = i + 1
        while j < len(parsed_dialogues):
            next_dialogue = parsed_dialogues[j]

            gap_cs = None
            if merged_end and next_dialogue["start"]:
                merged_end_cs = time_to_cs(merged_end)
                next_start_cs = time_to_cs(next_dialogue["start"])
                if merged_end_cs is not None and next_start_cs is not None:
                    gap_cs = next_start_cs - merged_end_cs

            current_duration_cs = None
            next_duration_cs = None
            current_start_cs = time_to_cs(current["start"])
            current_end_cs = time_to_cs(current["end"])
            next_start_cs_val = time_to_cs(next_dialogue["start"])
            next_end_cs = time_to_cs(next_dialogue["end"])

            if current_start_cs is not None and current_end_cs is not None:
                current_duration_cs = current_end_cs - current_start_cs
            if next_start_cs_val is not None and next_end_cs is not None:
                next_duration_cs = next_end_cs - next_start_cs_val

            # Short echo detection
            if (
                gap_cs is not None
                and gap_cs == 0
                and current_duration_cs is not None
                and next_duration_cs is not None
                and texts_are_ocr_variants(merged_text, next_dialogue["text"])
                and current["style"] == next_dialogue["style"]
                and current["name"] == next_dialogue["name"]
                and current["marginl"] == next_dialogue["marginl"]
                and current["marginr"] == next_dialogue["marginr"]
                and current["marginv"] == next_dialogue["marginv"]
                and current["effect"] == next_dialogue["effect"]
            ):
                if next_duration_cs < 5:
                    merged_end = next_dialogue["end"]
                    j += 1
                    merges_count += 1
                    continue
                elif current_duration_cs < 5:
                    merged_text = next_dialogue["text"]
                    merged_end = next_dialogue["end"]
                    j += 1
                    merges_count += 1
                    continue

            # Standard consecutive merge
            if (
                gap_cs is not None
                and gap_cs <= MAX_GAP_CS
                and texts_match_for_merge(merged_text, next_dialogue["text"], current_duration_cs, next_duration_cs)
                and current["style"] == next_dialogue["style"]
                and current["name"] == next_dialogue["name"]
                and current["marginl"] == next_dialogue["marginl"]
                and current["marginr"] == next_dialogue["marginr"]
                and current["marginv"] == next_dialogue["marginv"]
                and current["effect"] == next_dialogue["effect"]
            ):
                merged_end = next_dialogue["end"]
                j += 1
                merges_count += 1
            else:
                break

        if j > i + 1:
            merged_values = {
                "Layer": "0",
                "Start": merged_start,
                "End": merged_end,
                "Style": current["style"],
                "Name": current["name"],
                "MarginL": current["marginl"],
                "MarginR": current["marginr"],
                "MarginV": current["marginv"],
                "Effect": current["effect"],
                "Text": merged_text,
            }
            merged_line = rebuild_dialogue_line(fields_order, merged_values)
            merged_dialogues.append(merged_line)
            i = j
        else:
            merged_dialogues.append(current["original_line"])
            i += 1

    return merged_dialogues, merges_count


def merge_alternating_ocr_variants(
    dialogue_lines: List[str], fields_order: List[str],
) -> Tuple[List[str], int]:
    """Merge alternating OCR variant patterns (e.g., traditional/simplified Chinese)."""
    if not dialogue_lines or len(dialogue_lines) < 3:
        return dialogue_lines, 0

    parsed_dialogues = []
    for line in dialogue_lines:
        payload = line.split(":", 1)[1].lstrip()
        parts = split_dialogue_payload(payload, len(fields_order))
        m = {fields_order[i]: parts[i] if i < len(fields_order) else "" for i in range(len(fields_order))}
        raw_text = (m.get("Text", "") or "").strip()
        parsed_dialogues.append({
            "original_line": line,
            "start": m.get("Start", "").strip(),
            "end": m.get("End", "").strip(),
            "style": m.get("Style", "").strip(),
            "name": m.get("Name", "").strip(),
            "marginl": m.get("MarginL", "").strip(),
            "marginr": m.get("MarginR", "").strip(),
            "marginv": m.get("MarginV", "").strip(),
            "effect": m.get("Effect", "").strip(),
            "text": raw_text,
            "clean_text": strip_ass_tags(raw_text),
        })

    merged_dialogues = []
    merges_count = 0
    i = 0

    while i < len(parsed_dialogues):
        current = parsed_dialogues[i]

        if i + 2 < len(parsed_dialogues):
            first = parsed_dialogues[i]
            second = parsed_dialogues[i + 1]
            third = parsed_dialogues[i + 2]

            first_end_cs = time_to_cs(first["end"])
            second_start_cs = time_to_cs(second["start"])
            second_end_cs = time_to_cs(second["end"])
            third_start_cs = time_to_cs(third["start"])

            is_consecutive = (
                first_end_cs is not None
                and second_start_cs is not None
                and second_end_cs is not None
                and third_start_cs is not None
                and first_end_cs == second_start_cs
                and second_end_cs == third_start_cs
            )

            clean_a = first["clean_text"]
            clean_b = second["clean_text"]
            clean_third = third["clean_text"]

            is_alternating = (
                clean_a != clean_b
                and clean_a == clean_third
                and texts_are_ocr_variants(clean_a, clean_b)
            )

            fields_match = (
                first["style"] == second["style"] == third["style"]
                and first["name"] == second["name"] == third["name"]
                and first["marginl"] == second["marginl"] == third["marginl"]
                and first["marginr"] == second["marginr"] == third["marginr"]
                and first["marginv"] == second["marginv"] == third["marginv"]
                and first["effect"] == second["effect"] == third["effect"]
            )

            if is_consecutive and is_alternating and fields_match:
                merged_start = first["start"]
                merged_end = third["end"]
                j = i + 3

                while j < len(parsed_dialogues):
                    next_d = parsed_dialogues[j]
                    prev_d = parsed_dialogues[j - 1]

                    prev_end_cs = time_to_cs(prev_d["end"])
                    next_start_cs = time_to_cs(next_d["start"])
                    if prev_end_cs is None or next_start_cs is None or prev_end_cs != next_start_cs:
                        break

                    expected_clean = clean_a if (j - i) % 2 == 0 else clean_b
                    if next_d["clean_text"] != expected_clean:
                        if not texts_are_ocr_variants(next_d["clean_text"], expected_clean, min_similarity=0.9):
                            break

                    if not (
                        next_d["style"] == first["style"]
                        and next_d["name"] == first["name"]
                        and next_d["marginl"] == first["marginl"]
                        and next_d["marginr"] == first["marginr"]
                        and next_d["marginv"] == first["marginv"]
                        and next_d["effect"] == first["effect"]
                    ):
                        break

                    merged_end = next_d["end"]
                    j += 1

                merged_values = {
                    "Layer": "0",
                    "Start": merged_start,
                    "End": merged_end,
                    "Style": first["style"],
                    "Name": first["name"],
                    "MarginL": first["marginl"],
                    "MarginR": first["marginr"],
                    "MarginV": first["marginv"],
                    "Effect": first["effect"],
                    "Text": first["text"],
                }
                merged_line = rebuild_dialogue_line(fields_order, merged_values)
                merged_dialogues.append(merged_line)
                merges_count += j - i - 1
                i = j
                continue

        merged_dialogues.append(current["original_line"])
        i += 1

    return merged_dialogues, merges_count


def fix_overlapping_timestamps(
    dialogue_lines: List[str], fields_order: List[str],
) -> Tuple[List[str], int]:
    """Fix overlapping timestamps between consecutive dialogue lines."""
    if not dialogue_lines or len(dialogue_lines) < 2:
        return dialogue_lines, 0

    parsed_dialogues = []
    for line in dialogue_lines:
        payload = line.split(":", 1)[1].lstrip()
        parts = split_dialogue_payload(payload, len(fields_order))
        m = {fields_order[i]: parts[i] if i < len(fields_order) else "" for i in range(len(fields_order))}
        parsed_dialogues.append({
            "original_line": line,
            "layer": m.get("Layer", "").strip(),
            "start": m.get("Start", "").strip(),
            "end": m.get("End", "").strip(),
            "style": m.get("Style", "").strip(),
            "name": m.get("Name", "").strip(),
            "marginl": m.get("MarginL", "").strip(),
            "marginr": m.get("MarginR", "").strip(),
            "marginv": m.get("MarginV", "").strip(),
            "effect": m.get("Effect", "").strip(),
            "text": (m.get("Text", "") or "").strip(),
        })

    fixes_count = 0
    overlap_threshold = 2

    for idx in range(len(parsed_dialogues) - 1):
        cur = parsed_dialogues[idx]
        nxt = parsed_dialogues[idx + 1]

        current_cs_end = time_to_cs(cur["end"])
        next_cs_start = time_to_cs(nxt["start"])

        if current_cs_end is not None and next_cs_start is not None:
            if current_cs_end > next_cs_start:
                overlap = current_cs_end - next_cs_start
                if overlap <= overlap_threshold:
                    new_end_time = cs_to_time(next_cs_start)
                    new_start_time = cs_to_time(current_cs_end)
                    if new_end_time and new_start_time:
                        cur["end"] = new_end_time
                        nxt["start"] = new_start_time
                        fixes_count += 1

    fixed_dialogues = []
    for dialogue in parsed_dialogues:
        values = {
            "Layer": dialogue["layer"],
            "Start": dialogue["start"],
            "End": dialogue["end"],
            "Style": dialogue["style"],
            "Name": dialogue["name"],
            "MarginL": dialogue["marginl"],
            "MarginR": dialogue["marginr"],
            "MarginV": dialogue["marginv"],
            "Effect": dialogue["effect"],
            "Text": dialogue["text"],
        }
        fixed_dialogues.append(rebuild_dialogue_line(fields_order, values))

    return fixed_dialogues, fixes_count


def remove_short_duration_lines(
    dialogue_lines: List[str], fields_order: List[str],
) -> Tuple[List[str], int]:
    """Remove dialogue lines with duration less than 50ms."""
    if not dialogue_lines:
        return dialogue_lines, 0

    filtered_lines: List[str] = []
    removed_count = 0

    for line in dialogue_lines:
        if not line.strip().startswith(DIALOGUE_PREFIX):
            filtered_lines.append(line)
            continue

        payload = line.split(":", 1)[1].lstrip()
        parts = split_dialogue_payload(payload, len(fields_order))
        values = {fields_order[i]: parts[i] if i < len(parts) else "" for i in range(len(fields_order))}

        start_str = values.get("Start", "").strip()
        end_str = values.get("End", "").strip()

        start_cs = time_to_cs(start_str)
        end_cs = time_to_cs(end_str)

        if start_cs is not None and end_cs is not None:
            duration_cs = end_cs - start_cs
            if duration_cs < SHORT_DURATION_THRESHOLD_CS:
                removed_count += 1
                continue

        filtered_lines.append(line)

    return filtered_lines, removed_count


def remove_label_duplicates(
    dialogue_lines: List[str], fields_order: List[str],
) -> Tuple[List[str], int]:
    """Remove \\pos-tagged lines whose clean text matches an adjacent non-\\pos line."""
    if not dialogue_lines:
        return dialogue_lines, 0

    parsed = []
    for line in dialogue_lines:
        stripped = line.strip()
        if not stripped.startswith(DIALOGUE_PREFIX):
            parsed.append(None)
            continue
        payload = line.split(":", 1)[1].lstrip()
        parts = split_dialogue_payload(payload, len(fields_order))
        values = {fields_order[i]: parts[i] if i < len(parts) else "" for i in range(len(fields_order))}
        text = values.get("Text", "")
        clean = strip_ass_tags(text)
        is_pos = text.lstrip().startswith("{\\pos")
        parsed.append({
            "clean": clean,
            "is_pos": is_pos,
        })

    remove_indices = set()
    for idx, p in enumerate(parsed):
        if p is None or not p["is_pos"] or not p["clean"]:
            continue
        for ni in (idx - 1, idx + 1):
            if ni < 0 or ni >= len(parsed):
                continue
            nb = parsed[ni]
            if nb is None or nb["is_pos"]:
                continue
            if nb["clean"] == p["clean"]:
                remove_indices.add(idx)
                break

    filtered = [line for idx, line in enumerate(dialogue_lines) if idx not in remove_indices]
    return filtered, len(remove_indices)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_ass(
    path: str,
    inplace: bool,
    keep_empty_text: bool,
    canonical_styles_block: Optional[List[str]],
    have_canonical: bool,
    double_dialogue: bool = False,
    label_min: int = 2,
) -> Tuple[str, QAStats, str, Optional[List[str]], bool]:
    doc = load_ass(path)
    stats = QAStats()

    # Establish or enforce canonical styles
    if not have_canonical and doc.styles_block_lines:
        canonical_styles_block = [ln.rstrip("\n") for ln in doc.styles_block_lines]
        have_canonical = True
    elif have_canonical and canonical_styles_block:
        current = normalize_styles_block(doc.styles_block_lines or [])
        canon_norm = normalize_styles_block(canonical_styles_block)
        if current != canon_norm:
            replace_styles_with_canonical(doc, canonical_styles_block)

    fields_order = doc.events_format or copy.deepcopy(CANON_EVENTS_FIELDS)

    # Walk lines and fix dialogues
    new_lines: List[str] = []
    dialogue_buffer: List[str] = []
    in_events = False
    for line in doc.lines:
        stats.total_lines += 1
        stripped = line.strip()
        if stripped.startswith("[") and stripped.lower() == EVENTS_SECTION_HDR.lower():
            in_events = True
            new_lines.append(line)
            continue
        if in_events and stripped.startswith(FORMAT_PREFIX):
            fields = [f.strip() for f in line.split(":", 1)[1].split(",")]
            normalized = [
                next((c for c in CANON_EVENTS_FIELDS if c.lower() == f.lower()), f)
                for f in fields
            ]
            fields_order = normalized
            new_lines.append(f"{FORMAT_PREFIX} {', '.join(fields_order)}")
            continue
        if in_events and stripped.startswith(DIALOGUE_PREFIX):
            fixed = process_dialogue_line(
                line, fields_order, doc.styles, stats, keep_empty_text,
                double_dialogue, label_min,
            )
            if fixed is not None:
                dialogue_buffer.append(fixed)
            continue
        new_lines.append(line)

    # Dedupe
    if dialogue_buffer:
        dialogue_buffer, dup_count = dedupe_dialogues(dialogue_buffer, fields_order)
        stats.duplicates_removed = dup_count

    # Remove label duplicates
    if dialogue_buffer:
        dialogue_buffer, label_dup_count = remove_label_duplicates(dialogue_buffer, fields_order)
        stats.label_dupes_removed = label_dup_count

    # Merge alternating OCR variants
    if dialogue_buffer:
        dialogue_buffer, alt_count = merge_alternating_ocr_variants(dialogue_buffer, fields_order)
        stats.alternating_merges = alt_count

    # Merge consecutive
    if dialogue_buffer:
        dialogue_buffer, cons_count = merge_consecutive_dialogues(dialogue_buffer, fields_order)
        stats.consecutive_merges = cons_count

    # Fix overlapping timestamps
    if dialogue_buffer:
        dialogue_buffer, overlap_count = fix_overlapping_timestamps(dialogue_buffer, fields_order)
        stats.overlap_fixes = overlap_count

    # Remove short duration lines as final step
    if dialogue_buffer:
        dialogue_buffer, short_count = remove_short_duration_lines(dialogue_buffer, fields_order)
        stats.short_duration_removed = short_count

    # Splice dialogues after Events Format line
    result_lines: List[str] = []
    events_started = False
    inserted = False
    for line in new_lines:
        if line.strip().lower() == EVENTS_SECTION_HDR.lower():
            events_started = True
            result_lines.append(line)
            continue
        if events_started and line.strip().startswith(FORMAT_PREFIX):
            result_lines.append(line)
            result_lines.extend(dialogue_buffer)
            inserted = True
            events_started = False
            continue
        result_lines.append(line)
    if not inserted and dialogue_buffer:
        result_lines.append(EVENTS_SECTION_HDR)
        result_lines.append(f"{FORMAT_PREFIX} {', '.join(fields_order)}")
        result_lines.extend(dialogue_buffer)

    doc.lines = result_lines

    if inplace:
        save_ass(path, doc)
        out_path = path
    else:
        base, ext = os.path.splitext(path)
        out_path = base + ".fixed" + ext
        save_ass(out_path, doc)

    report = (
        f"Processed '{os.path.basename(path)}': "
        f"dialogue={stats.dialogue_lines}, fixed={stats.fixed_lines}, "
        f"style_fixes={stats.style_fixes}, "
        f"empty_text_removed={stats.empty_text_removed}, fake_text_removed={stats.fake_text_removed}, "
        f"deduped={stats.duplicates_removed}, consecutive_merged={stats.consecutive_merges}, "
        f"alternating_merged={stats.alternating_merges}, "
        f"overlap_fixes={stats.overlap_fixes}, short_removed={stats.short_duration_removed}, "
        f"newline_cleaned={stats.newline_garbage_cleaned}, label_short_removed={stats.label_short_removed}, "
        f"label_dupes_removed={stats.label_dupes_removed}"
    )
    return out_path, stats, report, canonical_styles_block, have_canonical


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_file(
    path: str,
    passes: int = 2,
    keep_empty_text: bool = False,
    double_dialogue: bool = False,
    label_min: int = 2,
) -> QAStats:
    """Process a single ASS file in-place with multiple QA passes.

    Args:
        path: Path to the .ass file to process.
        passes: Number of QA passes to run (default 2).
        keep_empty_text: If True, keep dialogue lines with empty/artifact text.
        double_dialogue: If True, use conservative \\N cleanup.
        label_min: Remove \\pos-tagged lines with visible text shorter than this.

    Returns:
        Aggregate QAStats across all passes.
    """
    grand = QAStats()
    for _ in range(passes):
        _, stats, _, _, _ = process_ass(
            path,
            inplace=True,
            keep_empty_text=keep_empty_text,
            canonical_styles_block=None,
            have_canonical=False,
            double_dialogue=double_dialogue,
            label_min=label_min,
        )
        for attr in vars(grand):
            setattr(grand, attr, getattr(grand, attr) + getattr(stats, attr))
    return grand
