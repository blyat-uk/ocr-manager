from . import engine_registry, utils
from .video import Video
from .progress import ProgressTracker
from .pyav_adapter import assert_reference_backend


def _scaled_progress_callback(progress_callback, range_index: int, total_ranges: int):
    """Wrap `progress_callback` so several ranges report one continuous
    0-100 sweep instead of each range restarting its own 0-100 run.

    Mirrors the scaling `core/ocr_worker.py` used to apply itself when it
    looped over ranges and called `get_subtitles` once per range; now that
    the loop lives here (see `get_subtitles`), the scaling moved with it.
    """
    if progress_callback is None or total_ranges <= 1:
        return progress_callback

    base = int(range_index * 100 / total_ranges)
    span = int(100 / total_ranges)

    def scaled(phase_name, percent):
        progress_callback(phase_name, base + int(percent * span / 100))

    return scaled


def _get_subtitles_for_range(
        v: Video, video_path: str, time_start: str, time_end: str,
        lang: str, conf_threshold: int, sim_threshold: int, use_fullframe: bool,
        det_model_dir, rec_model_dir, use_gpu: bool,
        brightness_threshold, similar_image_threshold: float, similar_pixel_threshold: int, frames_to_skip: int,
        crop_x, crop_y, crop_width, crop_height,
        detect_labels: bool, only_labels: bool,
        label_min_duration: float, label_max_duration: float, label_conf_threshold: int, label_conf_threshold_min: int,
        label_mask_crops,
        progress_callback, subtitle_callback, cancel_event) -> str:
    """Run one time range's dialogue + label passes against an already-open
    `Video`, and return its ASS text.

    This is the body `get_subtitles` used to run once, directly against a
    freshly-constructed `Video`, before it could take multiple ranges. It is
    unchanged except for taking `v` as a parameter instead of constructing
    it -- which is what lets several ranges share one `Video` (and the
    container probe / engine registry lookups that come with it) instead of
    each range paying for its own. The label pass still runs once per call,
    i.e. once per range, exactly as before.
    """
    progress = ProgressTracker(
        include_dialogue=not only_labels,
        include_labels=detect_labels or only_labels,
        progress_callback=progress_callback,
    )
    progress.start()

    if not only_labels:
        ocr = v.run_ocr(use_gpu, lang, time_start, time_end, conf_threshold, use_fullframe, brightness_threshold, similar_image_threshold, similar_pixel_threshold, frames_to_skip, crop_x, crop_y, crop_width, crop_height, progress=progress, subtitle_callback=subtitle_callback, cancel_event=cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            return ""
        dialogue_ass = v.get_subtitles(sim_threshold)
    else:
        ocr = engine_registry.get_ocr_engine(lang, det_model_dir, rec_model_dir, use_gpu)
        dialogue_ass = None

    if detect_labels or only_labels:
        if cancel_event is not None and cancel_event.is_set():
            return dialogue_ass if dialogue_ass else ""

        from .label_scanner import LabelScanner
        scanner = LabelScanner(
            video_path=video_path, fps=v.fps, width=v.width, height=v.height,
            num_frames=v.num_frames,
            crop_x=crop_x, crop_y=crop_y, crop_width=crop_width, crop_height=crop_height,
            label_min_duration=label_min_duration, label_max_duration=label_max_duration,
            conf_threshold=label_conf_threshold, conf_threshold_min=label_conf_threshold_min, brightness_threshold=brightness_threshold,
            label_mask_crops=label_mask_crops,
        )
        det_engine = engine_registry.get_detection_engine(det_model_dir, use_gpu)

        # Get container-level start_time (set during run_ocr, or fetch independently).
        # This is the playback offset; for MKV it's 0, for MP4 it may be non-zero.
        stream_start_time = getattr(v, '_stream_start_time', None)
        if stream_start_time is None:
            from .pyav_adapter import Capture
            with Capture(video_path) as cap:
                stream_start_time = cap.get_stream_start_time() if hasattr(cap, 'get_stream_start_time') else 0.0

        labels = scanner.scan(det_engine, ocr, time_start, time_end, stream_start_time, progress=progress, cancel_event=cancel_event)

        if cancel_event is not None and cancel_event.is_set():
            return dialogue_ass if dialogue_ass else ""

        if only_labels:
            return utils.format_labels_only_ass(labels, v.width, v.height)
        elif labels:
            return utils.merge_ass_output(dialogue_ass, labels, v.width, v.height)

    return dialogue_ass if dialogue_ass else ""


def get_subtitles(
        video_path: str, lang='ch', time_start='0:00', time_end='',
        conf_threshold=75, sim_threshold=80, use_fullframe=False,
        det_model_dir=None, rec_model_dir=None, use_gpu=True,
        brightness_threshold=None, similar_image_threshold=1.0, similar_pixel_threshold=25, frames_to_skip=1,
        crop_x=None, crop_y=None, crop_width=None, crop_height=None,
        detect_labels=True, only_labels=False,
        label_min_duration=1.0, label_max_duration=5.0, label_conf_threshold=95, label_conf_threshold_min=75,
        label_mask_crops=None,
        time_ranges: list[tuple[str, str]] | None = None,
        progress_callback=None, subtitle_callback=None, cancel_event=None) -> str:

    # Every production OCR run enters here (core/ocr_worker.py calls this and
    # save_subtitles_to_file, which wraps it). Refuse before decoding a single
    # frame if the bit-exact reference backend is not the one that would run:
    # an `av` that fails to import silently demotes the whole run to the
    # PTS-estimating fallback, which is how a previous release shipped output
    # that could not be reproduced. OCR_ALLOW_FFMPEG_FALLBACK=1 opts in.
    assert_reference_backend()

    if time_ranges is not None:
        # Caller opted into the multi-range path explicitly, so an empty
        # list is a caller error, not "no ranges given" -- silently falling
        # back to time_start/time_end here would OCR the entire file instead
        # of failing on a clearly-wrong argument. Checked before opening
        # anything, so a bad call fails instantly rather than after paying
        # for a container probe first.
        if not time_ranges:
            raise ValueError("time_ranges must contain at least one (start, end) pair")
        ranges = time_ranges
    else:
        ranges = [(time_start, time_end)]

    # One `Video` -- and the container probe its constructor does -- shared
    # by every range below, instead of one per range. `run_ocr` still opens
    # its own decode session per range (each range decodes a different part
    # of the file, so that part is unavoidable); what this sharing removes is
    # the repeated `Video` construction/probe, since Task 1's engine registry
    # already made repeat engine construction free within a process.
    v = Video(video_path, det_model_dir, rec_model_dir)

    ass_parts = []
    for i, (t_start, t_end) in enumerate(ranges):
        if cancel_event is not None and cancel_event.is_set():
            break
        part = _get_subtitles_for_range(
            v, video_path, t_start, t_end,
            lang, conf_threshold, sim_threshold, use_fullframe,
            det_model_dir, rec_model_dir, use_gpu,
            brightness_threshold, similar_image_threshold, similar_pixel_threshold, frames_to_skip,
            crop_x, crop_y, crop_width, crop_height,
            detect_labels, only_labels,
            label_min_duration, label_max_duration, label_conf_threshold, label_conf_threshold_min,
            label_mask_crops,
            _scaled_progress_callback(progress_callback, i, len(ranges)),
            subtitle_callback, cancel_event,
        )
        if part:
            ass_parts.append(part)

    if not ass_parts:
        return ""
    if len(ass_parts) == 1:
        return ass_parts[0]

    # Multiple ranges: merge with the same ordering/dedup rules the old
    # per-range loop in core/ocr_worker.py used -- header from the first
    # part, every Dialogue line from every part, sorted by start timestamp.
    return utils.merge_ass_documents(ass_parts)


def save_subtitles_to_file(
        video_path: str, file_path='subtitle.ass', lang='ch',
        time_start='0:00', time_end='', conf_threshold=75, sim_threshold=80,
        use_fullframe=False, det_model_dir=None, rec_model_dir=None, use_gpu=True,
        brightness_threshold=None, similar_image_threshold=1.0, similar_pixel_threshold=25, frames_to_skip=1,
        crop_x=None, crop_y=None, crop_width=None, crop_height=None,
        detect_labels=True, only_labels=False,
        label_min_duration=1.0, label_max_duration=5.0, label_conf_threshold=95, label_conf_threshold_min=75,
        label_mask_crops=None,
        progress_callback=None, subtitle_callback=None, cancel_event=None) -> None:
    # Produce the text first, create the file second. Opening 'w+' up front
    # truncates before any work happens, so anything that refuses to run --
    # the backend guard above, a filter graph that cannot be built -- would
    # leave a zero-byte .ass next to the video, which reads as "OCR produced
    # nothing" rather than "OCR did not run".
    ass = get_subtitles(
        video_path, lang, time_start, time_end, conf_threshold,
        sim_threshold, use_fullframe, det_model_dir, rec_model_dir, use_gpu, brightness_threshold, similar_image_threshold, similar_pixel_threshold, frames_to_skip, crop_x, crop_y, crop_width, crop_height,
        detect_labels, only_labels, label_min_duration, label_max_duration, label_conf_threshold, label_conf_threshold_min,
        label_mask_crops,
        progress_callback=progress_callback, subtitle_callback=subtitle_callback, cancel_event=cancel_event)
    with open(file_path, 'w+', encoding='utf-8') as f:
        f.write(ass)
