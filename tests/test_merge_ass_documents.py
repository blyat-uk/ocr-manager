"""Fast, media-free unit tests for videocr.utils.merge_ass_documents.

get_subtitles(time_ranges=...) is required to "reproduce exactly" the merge
core/ocr_worker.py used to do by hand before this task: header from the
first part, every Dialogue line from every part, sorted by start timestamp,
stable on ties, no cross-part dedup, and a single part returned verbatim.
tests/test_multirange.py covers this via a digest, which is media-gated and
blind to headers by construction (digest() only hashes Dialogue lines).
These tests pin the actual contract directly, fast and without media.
"""
from videocr.utils import merge_ass_documents

HEADER_A = (
    "[Script Info]\n"
    "Title: A\n\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
)
HEADER_B = (
    "[Script Info]\n"
    "Title: B\n\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
)


def _doc(header: str, *dialogue_lines: str) -> str:
    return header + "".join(dialogue_lines)


def _line(start: str, end: str, text: str) -> str:
    return f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"


def _dialogue_lines(merged: str) -> list[str]:
    return [line for line in merged.splitlines() if line.startswith("Dialogue:")]


def test_single_part_is_returned_verbatim():
    # Even malformed/non-ASS input must pass through completely untouched --
    # no header re-derivation, no parsing, no reformatting -- since there is
    # nothing to merge it against.
    malformed = "not even a real ass document, no trailing newline"
    assert merge_ass_documents([malformed]) == malformed


def test_header_comes_from_first_part_only():
    part1 = _doc(HEADER_A, _line("0:00:01.00", "0:00:02.00", "a"))
    part2 = _doc(HEADER_B, _line("0:00:03.00", "0:00:04.00", "b"))

    merged = merge_ass_documents([part1, part2])

    assert merged.startswith(HEADER_A)
    assert "Title: B" not in merged


def test_dialogue_lines_are_sorted_by_start_timestamp_across_parts():
    # part1's line starts later than part2's, so a naive concatenation
    # would emit them out of order; merge_ass_documents must not.
    part1 = _doc(HEADER_A, _line("0:00:10.00", "0:00:11.00", "later"))
    part2 = _doc(HEADER_A, _line("0:00:01.00", "0:00:02.00", "earlier"))

    merged = merge_ass_documents([part1, part2])

    assert _dialogue_lines(merged) == [
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,earlier",
        "Dialogue: 0,0:00:10.00,0:00:11.00,Default,,0,0,0,,later",
    ]


def test_sort_is_stable_on_equal_start_times():
    # Two lines with the identical start timestamp, one from each part.
    # A stable sort must keep them in the order they were encountered --
    # part order, then within-part document order -- since that is the
    # only thing distinguishing them.
    part1 = _doc(HEADER_A, _line("0:00:01.00", "0:00:02.00", "from-part1"))
    part2 = _doc(HEADER_A, _line("0:00:01.00", "0:00:02.00", "from-part2"))

    merged = merge_ass_documents([part1, part2])
    lines = _dialogue_lines(merged)

    assert lines[0].endswith(",from-part1")
    assert lines[1].endswith(",from-part2")


def test_every_dialogue_line_from_every_part_is_kept_including_duplicates():
    # No cross-part dedup: an identical line appearing in two parts (e.g.
    # two adjacent ranges both catching the tail/head of the same subtitle)
    # must appear twice in the merged output, not be collapsed.
    line = _line("0:00:01.00", "0:00:02.00", "same")
    part1 = _doc(HEADER_A, line)
    part2 = _doc(HEADER_A, line)

    merged = merge_ass_documents([part1, part2])

    assert _dialogue_lines(merged) == [
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,same",
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,same",
    ]


def test_three_parts_interleave_by_start_time():
    part1 = _doc(HEADER_A, _line("0:05:00.00", "0:05:01.00", "c"))
    part2 = _doc(HEADER_A, _line("0:01:00.00", "0:01:01.00", "a"))
    part3 = _doc(HEADER_A, _line("0:03:00.00", "0:03:01.00", "b"))

    merged = merge_ass_documents([part1, part2, part3])

    assert [line[-1] for line in _dialogue_lines(merged)] == ["a", "b", "c"]


def test_a_part_with_no_dialogue_lines_contributes_nothing_but_is_not_an_error():
    part1 = _doc(HEADER_A, _line("0:00:01.00", "0:00:02.00", "only"))
    empty_part = HEADER_A  # header only, no Dialogue lines

    merged = merge_ass_documents([part1, empty_part])

    assert _dialogue_lines(merged) == [
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,only",
    ]
