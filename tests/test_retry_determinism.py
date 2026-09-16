import hashlib

import pytest

from videocr import video as video_mod


def _ass_for_batch_size(path, batch_size, crop, brightness):
    original = video_mod.BATCH_SIZE
    video_mod.BATCH_SIZE = batch_size
    try:
        v = video_mod.Video(str(path), None, None)
        v.run_ocr(
            use_gpu=True, lang='ch', time_start='0:00', time_end='',
            conf_threshold=95, use_fullframe=False, brightness_threshold=brightness,
            similar_image_threshold=0.3, similar_pixel_threshold=25, frames_to_skip=0,
            crop_x=crop[0], crop_y=crop[1], crop_width=crop[2], crop_height=crop[3],
        )
        return v.get_subtitles(82)
    finally:
        video_mod.BATCH_SIZE = original


@pytest.mark.needs_media
@pytest.mark.slow
def test_output_is_identical_across_batch_sizes(reference_media):
    entry = reference_media["slay"]
    digests = set()
    for batch_size in (8, 32, 256):
        ass = _ass_for_batch_size(
            entry["video"], batch_size, entry["crop"], entry["brightness"]
        )
        digests.add(hashlib.sha256(ass.encode("utf-8")).hexdigest())
    assert len(digests) == 1, f"output differs across batch sizes: {digests}"
