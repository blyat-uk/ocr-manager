"""v1 -> v2 project migration.

The v1 `.ocr.json` schema (see docs/superpowers/stage3-research/
current-app-inventory.md SS2) stored `global`, `videocr`, `files`,
`labels`, `autodetect` and `automation` sections, almost everything as
strings. This module converts that into a Qt-free `core.project.model.
Project` with typed `FolderSettings` and per-file `FileEntry` values,
per the migration rules in task-1-brief.md and ruling A5/C1
(docs/superpowers/specs/2026-09-17-stage3-rulings.md).
"""
from core.project.model import (
    Brightness,
    Crop,
    FileEntry,
    FolderSettings,
    Media,
    Project,
    ReviewState,
    Source,
    TimeRange,
    TimeRanges,
)


def _to_int(value, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _to_float(value, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _map_crop_vertical_padding(raw, default: float) -> float:
    # Ruling A5: the old app's default was the literal string "0"; map
    # exactly that string to the new measured default, and carry any
    # other value over as a float (e.g. an explicit "0.0" is NOT the
    # same literal and is preserved, not remapped).
    if raw is None:
        return default
    if raw == "0":
        return 0.003
    return _to_float(raw, default)


def _map_bottom_half_cutoff(raw, default: float) -> float:
    # Same rule as above: old default literal "0.50" -> new default;
    # any other value (including "0.5") carries over as a float.
    if raw is None:
        return default
    if raw == "0.50":
        return 0.55
    return _to_float(raw, default)


def _migrate_folder(data: dict) -> FolderSettings:
    defaults = FolderSettings()

    global_ = data.get("global") or {}
    videocr = data.get("videocr") or {}
    labels = data.get("labels") or {}
    autodetect = data.get("autodetect") or {}
    automation = data.get("automation") or {}

    label_mask_crops = [tuple(c) for c in (labels.get("mask_crops") or [])]

    return FolderSettings(
        dialogue_enabled=_to_bool(global_.get("dialogue_enabled"), defaults.dialogue_enabled),
        labels_enabled=_to_bool(global_.get("labels_enabled"), defaults.labels_enabled),
        ocr_lang=videocr.get("ocr_lang", defaults.ocr_lang),
        conf_threshold=_to_int(videocr.get("conf_threshold"), defaults.conf_threshold),
        sim_threshold=_to_int(videocr.get("sim_threshold"), defaults.sim_threshold),
        similar_image=_to_float(videocr.get("similar_image"), defaults.similar_image),
        frames_to_skip=defaults.frames_to_skip,
        use_gpu=defaults.use_gpu,
        label_min_duration=_to_float(labels.get("label_min_duration"), defaults.label_min_duration),
        label_max_duration=_to_float(labels.get("label_max_duration"), defaults.label_max_duration),
        label_conf_threshold=_to_int(labels.get("label_conf_threshold"), defaults.label_conf_threshold),
        label_conf_threshold_min=_to_int(
            labels.get("label_conf_threshold_min"), defaults.label_conf_threshold_min
        ),
        label_mask_crops=label_mask_crops,
        ocr_parallel=_to_int(global_.get("ocr_parallel"), defaults.ocr_parallel),
        autopilot_enabled=defaults.autopilot_enabled,
        brightness_full_detect_files=defaults.brightness_full_detect_files,
        min_segment_length=_to_float(autodetect.get("min_segment_length"), defaults.min_segment_length),
        merge_repeating_silences=_to_bool(
            autodetect.get("merge_repeating_silences"), defaults.merge_repeating_silences
        ),
        crop_width_fraction=_to_float(automation.get("crop_width_fraction"), defaults.crop_width_fraction),
        crop_vertical_padding=_map_crop_vertical_padding(
            automation.get("crop_vertical_padding"), defaults.crop_vertical_padding
        ),
        crop_min_height_fraction=_to_float(
            automation.get("crop_min_height_fraction"), defaults.crop_min_height_fraction
        ),
        bottom_half_cutoff=_map_bottom_half_cutoff(
            automation.get("bottom_half_cutoff"), defaults.bottom_half_cutoff
        ),
        # detection_batch_size is intentionally dropped (ruling A5).
    )


def _parse_crop_dict(raw: dict) -> Crop:
    return Crop(
        x=_to_int(raw.get("x"), 0),
        y=_to_int(raw.get("y"), 0),
        width=_to_int(raw.get("width"), 0),
        height=_to_int(raw.get("height"), 0),
        source=Source.IMPORTED,
    )


def _migrate_crop(v1_entry: dict, global_crop: dict | None) -> Crop | None:
    own = v1_entry.get("crop")
    if own:
        return _parse_crop_dict(own)
    if global_crop:
        return _parse_crop_dict(global_crop)
    return None


def _migrate_brightness(v1_entry: dict, global_brightness) -> Brightness | None:
    own = v1_entry.get("brightness")
    if own is not None:
        return Brightness(value=_to_int(own, 0), source=Source.IMPORTED)
    if global_brightness is not None:
        return Brightness(value=_to_int(global_brightness, 0), source=Source.IMPORTED)
    return None


def _one_range(start, end) -> TimeRanges:
    return TimeRanges(ranges=[TimeRange(start=start or None, end=end or None)], source=Source.IMPORTED)


def _migrate_time_ranges(v1_entry: dict, global_time_range: dict | None) -> TimeRanges | None:
    own = v1_entry.get("time_ranges")
    if isinstance(own, list) and len(own) > 0:
        ranges = [TimeRange(start=r.get("start") or None, end=r.get("end") or None) for r in own]
        return TimeRanges(ranges=ranges, source=Source.IMPORTED)

    if "time_start" in v1_entry or "time_end" in v1_entry:
        return _one_range(v1_entry.get("time_start"), v1_entry.get("time_end"))

    if global_time_range:
        start = global_time_range.get("start")
        end = global_time_range.get("end")
        if start or end:
            return _one_range(start, end)

    return None


def _migrate_media(v1_entry: dict) -> Media:
    resolution = v1_entry.get("resolution") or {}
    return Media(
        width=_to_int(resolution.get("width"), 0),
        height=_to_int(resolution.get("height"), 0),
        duration=_to_float(v1_entry.get("duration"), 0.0),
        fps=0.0,
    )


def _migrate_sample_time(v1_entry: dict, duration: float) -> float | None:
    position = v1_entry.get("subtitle_position")
    if position is None or duration <= 0:
        return None
    return _to_float(position, 0.0) / 10000 * duration


def migrate_v1(data: dict, project_dir: str, video_names: list[str]) -> Project:
    folder = _migrate_folder(data)

    files_v1 = data.get("files") or {}
    global_ = data.get("global") or {}
    global_crop = global_.get("crop")
    global_brightness = global_.get("brightness")
    global_time_range = global_.get("time_range")

    files: dict[str, FileEntry] = {}
    for name in sorted(video_names):
        v1_entry = files_v1.get(name) or {}

        crop = _migrate_crop(v1_entry, global_crop)
        brightness = _migrate_brightness(v1_entry, global_brightness)
        time_ranges = _migrate_time_ranges(v1_entry, global_time_range)
        media = _migrate_media(v1_entry)
        sample_time = _migrate_sample_time(v1_entry, media.duration)

        review = ReviewState.PENDING
        if (crop is not None or folder.labels_only) and brightness is not None:
            review = ReviewState.REVIEWED

        files[name] = FileEntry(
            name=name,
            crop=crop,
            brightness=brightness,
            time_ranges=time_ranges,
            media=media,
            review=review,
            skipped=False,
            sample_time=sample_time,
            flags={},
            evidence={},
        )

    return Project(path=project_dir, folder=folder, files=files, migrated_from_v1=True)
