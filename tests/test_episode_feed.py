"""app/episode_feed.py: the Working screen's drip (LineFeed), its ETA
(EtaEstimator) and the text helpers. Pure Python, driven by a fake clock:
every `now` below is a number the test chooses."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.episode_feed import (
    EtaEstimator,
    Line,
    LineFeed,
    duration_words,
    estimate_words,
    eta_words,
    format_ts,
    kept_position,
    kept_total,
    mid_time,
    speed_words,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def lines(count: int, first: float = 0.0, step: float = 3.0) -> list[tuple[float, float, str]]:
    return [(first + i * step, first + i * step + 2.0, f"line {i}") for i in range(count)]


def ready_all(feed: LineFeed, batch) -> None:
    for line in batch:
        feed.frame_ready(mid_time(Line(*line)))


# --------------------------------------------------------------------------
# The module is pure
# --------------------------------------------------------------------------

def test_episode_feed_imports_neither_qt_nor_core():
    tree = ast.parse((REPO_ROOT / "app" / "episode_feed.py").read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    assert not [m for m in modules if m.split(".")[0] in ("PyQt6", "core", "videocr")]


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seconds, text", [
    (0.0, "0:00:00.000"),
    (1.5, "0:00:01.500"),
    (631.0404, "0:10:31.040"),
    (3725.9996, "1:02:06.000"),
    (-3.0, "0:00:00.000"),
])
def test_format_ts(seconds, text):
    assert format_ts(seconds) == text


def test_mid_time_is_the_middle_of_the_span():
    assert mid_time(Line(10.0, 14.0, "a")) == 12.0
    assert Line(10.0, 14.0, "a").mid == 12.0


@pytest.mark.parametrize("seconds, text", [
    (0, "0 s"), (44.2, "45 s"), (190, "3 min 10 s"), (180, "3 min"), (3900, "1 h 5 min"), (7200, "2 h"),
])
def test_duration_words(seconds, text):
    assert duration_words(seconds) == text


def test_eta_and_speed_words():
    assert eta_words(190) == "≈ 3 min 10 s"
    assert speed_words(4.06) == "4.1× real time"


@pytest.mark.parametrize("seconds, text", [(20, "under a minute"), (61, "about 2 min"), (360, "about 6 min")])
def test_estimate_words(seconds, text):
    assert estimate_words(seconds) == text


def test_kept_position_walks_the_keep_spans_in_order():
    spans = [(10.0, 20.0), (30.0, 50.0)]
    assert kept_total(spans) == 30.0
    assert kept_position(5.0, spans) == 0.0
    assert kept_position(15.0, spans) == 5.0
    assert kept_position(25.0, spans) == 10.0          # in a skipped gap: the end of the span before
    assert kept_position(40.0, spans) == 20.0
    assert kept_position(99.0, spans) == 30.0


# --------------------------------------------------------------------------
# LineFeed
# --------------------------------------------------------------------------

def test_first_line_is_released_as_soon_as_its_frame_is_ready():
    feed = LineFeed()
    batch = lines(4)
    feed.add_batch(batch, now=100.0)
    assert feed.pending == 4
    assert feed.due(100.0) == []                       # no frame yet, and the timeout has not passed
    feed.frame_ready(mid_time(Line(*batch[0])))
    assert feed.due(100.1) == [Line(*batch[0])]
    assert feed.pending == 3


def test_lines_are_released_one_at_a_time_at_the_pace():
    feed = LineFeed()
    batch = lines(5)
    feed.add_batch(batch, now=0.0)
    ready_all(feed, batch)
    interval = feed.interval
    released = feed.due(0.0)
    assert len(released) == 1
    assert feed.due(interval * 0.9) == []              # not due yet
    assert len(feed.due(interval)) == 1


def test_pace_drains_the_queue_by_the_next_expected_batch():
    feed = LineFeed()
    feed.add_batch(lines(32), now=0.0)
    feed.flush()                                       # the first batch is gone
    feed.add_batch(lines(20, first=200), now=20.0)     # a 20 s gap between batches, 20 lines queued
    assert feed.interval == pytest.approx(1.0)


def test_pace_uses_the_mean_gap_between_batches():
    feed = LineFeed()
    feed.add_batch(lines(1), now=0.0)
    feed.add_batch(lines(1, first=10), now=10.0)
    feed.add_batch(lines(1, first=20), now=40.0)       # gaps 10 and 30: mean 20
    assert feed.pending == 3
    assert feed.interval == LineFeed.MAX_INTERVAL      # 20 s over 3 lines, 6.7 s a line, is clamped to 3 s
    feed.add_batch(lines(17, first=30), now=60.0)      # gaps 10, 30, 20: mean 20 over 20 lines
    assert feed.interval == pytest.approx(1.0)


def test_pace_is_clamped_to_at_least_a_quarter_second():
    feed = LineFeed()
    feed.add_batch(lines(1), now=0.0)
    feed.add_batch(lines(200, first=10), now=2.0)      # 2 s gap, 201 lines: 0.01 s a line
    assert feed.interval == LineFeed.MIN_INTERVAL


def test_a_line_whose_frame_never_arrives_is_released_after_the_timeout():
    feed = LineFeed()
    feed.add_batch(lines(2), now=0.0)
    assert feed.due(LineFeed.FRAME_TIMEOUT - 0.01) == []
    assert feed.due(LineFeed.FRAME_TIMEOUT) == [Line(*lines(2)[0])]


def test_the_timeout_counts_from_when_the_line_became_due():
    feed = LineFeed()
    batch = lines(2)
    feed.add_batch(batch, now=0.0)
    feed.frame_ready(mid_time(Line(*batch[0])))
    assert feed.due(0.0) == [Line(*batch[0])]
    due_at = feed.interval
    assert feed.due(due_at + LineFeed.FRAME_TIMEOUT - 0.01) == []
    assert feed.due(due_at + LineFeed.FRAME_TIMEOUT) == [Line(*batch[1])]


def test_order_is_preserved_even_when_later_frames_arrive_first():
    feed = LineFeed()
    batch = lines(3)
    feed.add_batch(batch, now=0.0)
    feed.frame_ready(mid_time(Line(*batch[2])))
    feed.frame_ready(mid_time(Line(*batch[1])))
    released = []
    now = 0.0
    while feed.pending:
        released += feed.due(now)
        now += 0.1
    assert released == [Line(*line) for line in batch]


def test_flush_returns_everything_still_queued_in_order():
    feed = LineFeed()
    feed.add_batch(lines(3), now=0.0)
    feed.add_batch(lines(2, first=100), now=5.0)
    assert feed.flush() == [Line(*line) for line in lines(3) + lines(2, first=100)]
    assert feed.pending == 0
    assert feed.due(1e9) == []


def test_frame_ready_matches_times_that_went_through_a_float_round_trip():
    feed = LineFeed()
    feed.add_batch([(1.1, 2.2, "a")], now=0.0)
    feed.frame_ready(1.6500000000000001)
    assert feed.due(0.0) == [Line(1.1, 2.2, "a")]


def test_an_empty_batch_changes_nothing():
    feed = LineFeed()
    feed.add_batch([], now=0.0)
    assert feed.pending == 0 and feed.due(10.0) == []


def test_reset_forgets_queue_pace_and_frames():
    feed = LineFeed()
    feed.add_batch(lines(3), now=0.0)
    feed.add_batch(lines(3, first=50), now=10.0)
    feed.reset()
    assert feed.pending == 0
    feed.add_batch(lines(1), now=100.0)
    assert feed.interval == LineFeed.FIRST_INTERVAL
    assert feed.due(100.0) == []                       # the old frame knowledge is gone too


# --------------------------------------------------------------------------
# EtaEstimator
# --------------------------------------------------------------------------

def test_no_estimate_without_prior_or_progress():
    eta = EtaEstimator()
    assert eta.seconds_left(0.0) is None
    assert eta.speed(600.0) is None


def test_prior_speed_alone_gives_the_estimate_at_the_start():
    eta = EtaEstimator(prior_speed=4.0, kept_duration=1200.0)     # 1200 s of video at 4x: 300 s
    assert eta.seconds_left(0.0) == pytest.approx(300.0)
    assert eta.speed(1200.0) == pytest.approx(4.0)


def test_measured_rate_over_the_window():
    eta = EtaEstimator()
    eta.update(0.10, now=0.0)
    eta.update(0.20, now=10.0)                         # 1 % a second
    assert eta.seconds_left(10.0) == pytest.approx(80.0)
    assert eta.seconds_left(15.0) == pytest.approx(75.0)          # counts down between reports
    assert eta.speed(600.0) == pytest.approx(6.0)


def test_the_window_forgets_samples_older_than_thirty_seconds():
    eta = EtaEstimator()
    eta.update(0.0, now=0.0)
    eta.update(0.5, now=10.0)                          # a fast start ...
    eta.update(0.51, now=50.0)
    eta.update(0.52, now=60.0)                         # ... then slow
    # The window [30, 60] is measured from its anchor, the last report at or
    # before its start (10 s): the fast start before it no longer counts.
    assert eta.rate() == pytest.approx(0.02 / 50)


def test_prior_is_blended_in_until_five_percent():
    eta = EtaEstimator(prior_speed=1.0, kept_duration=100.0)      # prior: 1 % a second
    eta.update(0.00, now=0.0)
    eta.update(0.025, now=10.0)                        # measured 0.25 % a second, 2.5 % done: half each
    assert eta.rate() == pytest.approx(0.5 * 0.0025 + 0.5 * 0.01)
    eta.update(0.05, now=20.0)                         # 5 %: measured only
    assert eta.rate() == pytest.approx(0.0025)


def test_a_progress_that_goes_back_starts_a_new_phase():
    eta = EtaEstimator()
    eta.update(0.9, now=0.0)
    eta.update(1.0, now=10.0)
    eta.update(0.0, now=11.0)                          # dialogue done, labels start at 0
    assert eta.rate() is None
    eta.update(0.1, now=21.0)
    assert eta.rate() == pytest.approx(0.01)


def test_seconds_left_never_goes_negative():
    eta = EtaEstimator()
    eta.update(0.5, now=0.0)
    eta.update(0.6, now=10.0)
    assert eta.seconds_left(1000.0) == 0.0
