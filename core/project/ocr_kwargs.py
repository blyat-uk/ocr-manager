"""The exact OCR call for one file (fidelity bridge).

`ocr_call_for` must reproduce, for the same resolved settings, exactly the
`videocr.api.get_subtitles` / `save_subtitles_to_file` keyword arguments and
time ranges the old `core/ocr_worker.py`'s `OCRWorker._build_ocr_kwargs()` /
`_get_time_ranges()` derived from `core.config.Config`/`FileConfig` -- see
task-2-brief.md. Reduced OCR fidelity from a changed argument is the one
unforgivable failure this module exists to prevent, so every rule here
mirrors the old code's rule one-to-one rather than being "improved".

This module has no PyQt6 dependency, same as the rest of `core/project/`.
"""
import os
from dataclasses import dataclass

from core.project.model import FileEntry, FolderSettings

DEFAULT_BRIGHTNESS = 230   # the v1 app default the OCR worker fell back to


@dataclass(frozen=True)
class OcrCall:
    kwargs: dict              # keyword arguments for videocr.api.get_subtitles / save_subtitles_to_file, WITHOUT time fields
    time_ranges: list[tuple[str, str]]   # [] = whole file; (start or "0:00", end or "")


def ocr_call_for(entry: FileEntry, folder: FolderSettings, project_dir: str) -> OcrCall:
    """Resolve `entry`/`folder` into the exact OCR call for this file."""
    brightness = entry.brightness.value if entry.brightness is not None else DEFAULT_BRIGHTNESS

    kwargs = {
        "video_path": os.path.join(project_dir, entry.name),
        "lang": folder.ocr_lang,
        "conf_threshold": folder.conf_threshold,
        "sim_threshold": folder.sim_threshold,
        "brightness_threshold": brightness,
        "similar_image_threshold": folder.similar_image,
        "frames_to_skip": folder.frames_to_skip,
        "use_gpu": folder.use_gpu,
    }

    crop = entry.crop
    if crop is not None and crop.width > 0 and crop.height > 0:
        kwargs["crop_x"] = crop.x
        kwargs["crop_y"] = crop.y
        kwargs["crop_width"] = crop.width
        kwargs["crop_height"] = crop.height

    if not folder.labels_enabled:
        kwargs["detect_labels"] = False
    else:
        kwargs["detect_labels"] = True
        if folder.labels_only:
            kwargs["only_labels"] = True
        kwargs["label_min_duration"] = folder.label_min_duration
        kwargs["label_max_duration"] = folder.label_max_duration
        kwargs["label_conf_threshold"] = folder.label_conf_threshold
        kwargs["label_conf_threshold_min"] = folder.label_conf_threshold_min
        if folder.label_mask_crops:
            kwargs["label_mask_crops"] = [
                (mask[0], mask[1], mask[2], mask[3])
                for mask in folder.label_mask_crops
            ]

    if entry.time_ranges is not None:
        time_ranges = [(r.start or "0:00", r.end or "") for r in entry.time_ranges.ranges]
    else:
        time_ranges = []

    return OcrCall(kwargs=kwargs, time_ranges=time_ranges)
