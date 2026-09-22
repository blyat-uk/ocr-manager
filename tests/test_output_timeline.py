"""Output times count from the player's zero, the container start.

A line's time is its first frame's PTS minus the container's start_time, so
it lands on the frame the player paints it on. That is the zero a player
rebases to (libavformat's format start_time, the earliest of all the
streams; mpv's --rebase-start-time=yes default) -- not the video stream's
start_time, which only says when the first frame arrives.

delayed_video_subtitle_clip has the layout of Jinwu Guard episodes 07, 08,
09, 12 and 15 and of "Tales of Demon and Gods - 173 [4K].mkv": the audio and
container start before the first frame. Counting from the first frame there
put every line two frames early -- rendered through mpv, the line was on
screen two frames before the bar it was read from. These run the real
Video.run_ocr and get_subtitles, on the real decoder, with an engine that
reads the synthetic bar (frames 25-49).
"""
import numpy as np
import pytest

from videocr.video import Video


class _BarReadingOCR:
    """Reads 字幕 wherever the bar is, nothing elsewhere."""

    def predict(self, frames):
        out = []
        for frame in frames:
            h, w = frame.shape[:2]
            if frame.max() > 200:
                box = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
                out.append({"rec_texts": ["字幕"], "rec_scores": [0.99], "rec_polys": [box]})
            else:
                out.append({"rec_texts": [], "rec_scores": [], "rec_polys": []})
        return out


def _ocr(path):
    video = Video(str(path), None, None)
    live = []
    video.run_ocr(False, "ch", "", "", 95, True, 200, 0.3, 25, 0, None, None, None, None,
                  subtitle_callback=lambda start, end, text: live.append((start, end, text)),
                  ocr_engine=_BarReadingOCR())
    dialogue = [line for line in video.get_subtitles(82).splitlines() if line.startswith("Dialogue:")]
    return dialogue, live


@pytest.mark.parametrize("clip, start, end", [
    # container start 0, first frame 0: the two conventions agree
    ("synthetic_subtitle_video", 1.00, 2.00),
    # container (and audio) start 0, first frame 0.080: the bar is painted at
    # 1.080, so that is when the line has to come up
    ("delayed_video_subtitle_clip", 1.08, 2.08),
])
def test_dialogue_times_count_from_the_container_start(request, clip, start, end):
    dialogue, live = _ocr(request.getfixturevalue(clip))

    stamp = lambda t: f"0:00:{t:05.2f}"
    assert dialogue == [f"Dialogue: 0,{stamp(start)},{stamp(end)},Default,,0,0,0,,字幕"]
    assert live == [(pytest.approx(start, abs=1e-3), pytest.approx(end, abs=1e-3), "字幕")]
