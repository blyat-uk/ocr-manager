from . import utils
from .video import Video
from .progress import ProgressTracker
from .pyav_adapter import assert_reference_backend


def get_subtitles(
        video_path: str, lang='ch', time_start='0:00', time_end='',
        conf_threshold=75, sim_threshold=80, use_fullframe=False,
        det_model_dir=None, rec_model_dir=None, use_gpu=True,
        brightness_threshold=None, similar_image_threshold=1.0, similar_pixel_threshold=25, frames_to_skip=1,
        crop_x=None, crop_y=None, crop_width=None, crop_height=None,
        detect_labels=True, only_labels=False,
        label_min_duration=1.0, label_max_duration=5.0, label_conf_threshold=95, label_conf_threshold_min=75,
        label_mask_crops=None,
        progress_callback=None, subtitle_callback=None, cancel_event=None) -> str:

    # Every production OCR run enters here (core/ocr_worker.py calls this and
    # save_subtitles_to_file, which wraps it). Refuse before decoding a single
    # frame if the bit-exact reference backend is not the one that would run:
    # an `av` that fails to import silently demotes the whole run to the
    # PTS-estimating fallback, which is how a previous release shipped output
    # that could not be reproduced. OCR_ALLOW_FFMPEG_FALLBACK=1 opts in.
    assert_reference_backend()

    v = Video(video_path, det_model_dir, rec_model_dir)

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
        ocr = utils.create_ocr_engine(lang, det_model_dir, rec_model_dir, use_gpu)
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
        det_engine = utils.create_detection_engine(det_model_dir, use_gpu)

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
    with open(file_path, 'w+', encoding='utf-8') as f:
        f.write(get_subtitles(
            video_path, lang, time_start, time_end, conf_threshold,
            sim_threshold, use_fullframe, det_model_dir, rec_model_dir, use_gpu, brightness_threshold, similar_image_threshold, similar_pixel_threshold, frames_to_skip, crop_x, crop_y, crop_width, crop_height,
            detect_labels, only_labels, label_min_duration, label_max_duration, label_conf_threshold, label_conf_threshold_min,
            label_mask_crops,
            progress_callback=progress_callback, subtitle_callback=subtitle_callback, cancel_event=cancel_event))
