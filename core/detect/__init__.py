"""Qt-free detectors: crop box (crop), brightness threshold (brightness), keep
ranges (ranges), and the frame sources they read (crop's fetch layer,
ocr_view, vad).

Reference for callers (Stage 3's job runner and review UI). Read it before
re-fetching a frame a detector reported, cancelling a detector, or applying
a result without review.

(a) Frame addressing -- re-fetch a time with the function of the producer
    that recorded it; the domains do not convert into each other.

| producer | times it records | re-fetch with | semantics |
|---|---|---|---|
| crop.detect_crop | CropResult.sample_pts, hit_pts, samples[i].time: the REQUESTED probe times (samples[i].boxes are full-frame native pixels, not grab_frames' band coordinates) | crop.grab_frames(video, times) (either fetch path) | first frame at or after the time rounded to whole ms, container-relative (crop._seek_seconds). May precede the frame's own PTS by up to one frame; never re-fetch with the PTS. |
| OCR pass (videocr.video.Video.run_ocr) | ASS times from each frame's PTS minus the container start | range starts/ends are "MM:SS" strings: videocr.utils.get_frame_index truncates int(t * fps), then Capture.set(CAP_PROP_POS_FRAMES) | index -> first frame whose round(PTS * fps) reaches it |
| ocr_view.grab_ocr_strips_at (brightness sampling and neighbours) | the times it was given | ocr_view.grab_ocr_strips_at(video, crop_box, times) | index round(t * fps) (rounds, where the OCR pass truncates), then the same Capture.set as the OCR pass; pixels identical to what the OCR pass masks |
| label scanner phase 1 | PTS from Capture.get_last_pts() | Capture.seek_to_pts(pts) | first frame whose PTS is at or after it: exactly that frame |
| label scanner phases 3-4 | decision times t | Capture.seek_to_display_time(t) | the frame on screen at t: last frame whose PTS <= t (the first frame if t precedes it) |
| ranges.analyse | keep ranges as "MM:SS" strings (None for an open end) | the OCR pass's time_ranges | as the OCR pass row |

    Crop pixels are NOT OCR pixels. crop grabs a full-width bottom band at
    probe resolution (crop -> scale -> bgr24, or the system ffmpeg CLI; HDR
    sources are tone-mapped with the OCR pass's chain first). Measured
    against the OCR pass's own frames: identical on Slay the Gods (1080p
    8-bit), but 87-90% of pixels off by up to 13 levels on XWZ (4K 10-bit)
    and a different frame altogether on Jinwu Guard (h264 MKV). Anything
    that previews or tunes a brightness threshold must use ocr_view, never
    crop.grab_frames.

(b) Cancellation, per detector.

| detector | how to cancel | what comes back |
|---|---|---|
| crop.detect_crop | cancel_check callable, polled during audio extraction (every 0.1 s), between probe batches and between rounds | a CropResult, not an exception: flagged gains "cancelled", and `box` is whatever the evidence so far gives -- possibly clipped, possibly None. auto_applicable is False. |
| brightness.detect_brightness | cancel_check callable, polled before each sampling round, before OCR verification, and before each OCR batch / neighbour grab of the dim-text check | a BrightnessResult with flagged == "cancelled", value DEFAULT_BRIGHTNESS (nothing measured), plateau None, curve []. auto_applicable is False. |
| ranges.pipeline.analyse | cancel callable, polled between files and between phases | raises ranges.pipeline.AnalysisCancelled (no partial result). core/audio_analysis.py turns it into finished({}). |

(c) Flags and auto_applicable. Both detectors join reasons with "+"
    (`flagged` is None when clean) and expose `auto_applicable`: True only
    when there is a result to apply and EVERY flag on it is informational.
    Any other flag -- or one a caller does not recognise -- means review
    before applying.

| detector | informational (still auto-applicable) | blocking |
|---|---|---|
| crop (CropResult; per-flag reasons in its auto_applicable docstring) | no-speech, speech-probes-exhausted | top-positioned?, low-agreement, static-content?, multiple-positions?, outlier-discarded?, cancelled; and, always without a box, static-content, ceiling-exceeded, unknown-rejection. A result without a box is never auto-applicable. |
| brightness (BrightnessResult) | no-clean-threshold | needs-crop, ranges-empty?, no-text, thin-evidence?, coloured-text?, no-plateau?, narrow-plateau?, dim-text?, escalate (cheap path: re-run full detection), cancelled |
| ranges | no flags. A file absent from analyse()'s result has no keep ranges (no repeated segment, or no gap of MIN_GAP_SEC): OCR it whole | nothing is flagged; decode errors (e.g. no audio stream) raise, and core/audio_analysis.py reports them through error() |

(d) Values detectors take from core.config.Config's DEFAULTS, not from the
    user's settings. Threading a user value through would touch several
    internal call sites in each case, so they are listed here instead.

| detector | value | used for | effect of a user setting that differs |
|---|---|---|---|
| crop | Config.label_max_duration (5.0 s) | WATERMARK_MIN_SPAN_SEC = it + 1.0 s: the span identical extents must cover before a box is rejected as a watermark (static-content) rather than kept as static-content? | a user who raised label_max_duration still gets the 6 s watermark span |
| brightness | Config.ocr_lang ("ch") | _reading() joins OCR words the way the OCR pass does for that language (no spaces for "ch") | for another OCR language, readings are joined without spaces, so modal agreement compares differently joined text than that OCR pass emits |
| brightness | Config.brightness (230) | DEFAULT_BRIGHTNESS: the value reported when nothing was measured | none: such results are flagged and never auto-applicable |
"""
