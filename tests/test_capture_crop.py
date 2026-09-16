import numpy as np

from videocr.pyav_adapter import PyAVCapture

CROP = (64, 120, 192, 64)  # x, y, w, h inside a 320x240 frame


def _read_all(cap, n):
    frames = []
    for _ in range(n):
        ok, frame = cap.read()
        assert ok
        frames.append(frame.copy())
    return frames


def test_crop_in_graph_matches_numpy_slice(synthetic_video):
    x, y, w, h = CROP
    with PyAVCapture(str(synthetic_video)) as cap:
        full = _read_all(cap, 10)
    with PyAVCapture(str(synthetic_video), crop_rect=CROP) as cap:
        cropped = _read_all(cap, 10)

    assert len(full) == len(cropped) == 10
    for i, (f, c) in enumerate(zip(full, cropped)):
        expected = f[y:y + h, x:x + w]
        assert c.shape == expected.shape, f"frame {i} shape {c.shape} != {expected.shape}"
        assert np.array_equal(c, expected), (
            f"frame {i} differs, max abs diff "
            f"{int(np.abs(c.astype(int) - expected.astype(int)).max())}"
        )


def test_crop_pts_unchanged(synthetic_video):
    with PyAVCapture(str(synthetic_video)) as cap:
        _read_all(cap, 5)
        full_pts = cap.get_last_pts()
    with PyAVCapture(str(synthetic_video), crop_rect=CROP) as cap:
        _read_all(cap, 5)
        crop_pts = cap.get_last_pts()
    assert full_pts == crop_pts


def test_crop_matches_slice_with_decode_downscale(tmp_path):
    import subprocess
    src = tmp_path / "uhd.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=3840x2160:rate=25:duration=0.2",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", str(src)],
        check=True, capture_output=True,
    )
    crop = (400, 900, 1024, 96)  # in 1080p output space
    with PyAVCapture(str(src), decode_target_height=1080) as cap:
        full = _read_all(cap, 5)
    with PyAVCapture(str(src), decode_target_height=1080, crop_rect=crop) as cap:
        cropped = _read_all(cap, 5)
    x, y, w, h = crop
    for i, (f, c) in enumerate(zip(full, cropped)):
        expected = f[y:y + h, x:x + w]
        assert np.array_equal(c, expected), (
            f"frame {i}: max abs diff "
            f"{int(np.abs(c.astype(int) - expected.astype(int)).max())}"
        )
