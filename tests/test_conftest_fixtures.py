import subprocess


def _probe_frames(path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def test_synthetic_video_has_ten_frames(synthetic_video):
    assert synthetic_video.exists()
    assert _probe_frames(synthetic_video) == 10


def test_synthetic_subtitle_video_has_75_frames(synthetic_subtitle_video):
    assert _probe_frames(synthetic_subtitle_video) == 75
