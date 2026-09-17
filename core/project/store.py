"""Project JSON store: v2 `.ocr.json` read/write, v1 migration dispatch,
and reconciliation against the video files actually on disk.
"""
import json
import logging
import shutil
import time
from pathlib import Path

from core.project.migrate import migrate_v1
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

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mkv", ".mp4")
CONFIG_FILENAME = ".ocr.json"


def list_video_files(project_dir: str) -> list[str]:
    """Sorted video file names in `project_dir` -- same rule as
    core.pipeline.get_video_files (glob on VIDEO_EXTENSIONS, case-sensitive).
    """
    directory = Path(project_dir)
    names = [
        f.name
        for ext in VIDEO_EXTENSIONS
        for f in directory.glob(f"*{ext}")
    ]
    return sorted(names)


def load_project(project_dir: str) -> Project:
    directory = Path(project_dir)
    config_path = directory / CONFIG_FILENAME
    video_names = list_video_files(project_dir)

    project: Project
    if config_path.exists():
        try:
            raw = config_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Corrupt %s in %s (%s) -- renaming and starting a fresh project",
                CONFIG_FILENAME, project_dir, exc,
            )
            corrupt_path = directory / f"{CONFIG_FILENAME}.corrupt-{int(time.time())}"
            try:
                config_path.rename(corrupt_path)
            except OSError:
                logger.warning("Could not rename corrupt config %s", config_path)
            project = Project(path=str(directory), folder=FolderSettings(), files={})
        else:
            version = data.get("version")
            if version == 2:
                project = from_json(data, str(directory))
            else:
                project = migrate_v1(data, str(directory), video_names)
    else:
        project = Project(path=str(directory), folder=FolderSettings(), files={})

    reconcile_files(project, video_names)
    return project


def save_project(project: Project) -> None:
    directory = Path(project.path)
    config_path = directory / CONFIG_FILENAME
    backup_path = directory / f"{CONFIG_FILENAME}.v1.bak"
    tmp_path = directory / f"{CONFIG_FILENAME}.tmp"

    if project.migrated_from_v1:
        if not backup_path.exists() and config_path.exists():
            shutil.copy2(config_path, backup_path)
        project.migrated_from_v1 = False

    tmp_path.write_text(
        json.dumps(to_json(project), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(config_path)


def reconcile_files(project: Project, video_names: list[str]) -> tuple[list[str], list[str]]:
    existing = set(project.files.keys())
    incoming = set(video_names)

    added = sorted(incoming - existing)
    removed = sorted(existing - incoming)

    for name in removed:
        del project.files[name]
    for name in added:
        project.files[name] = FileEntry(name=name)

    ordered = {name: project.files[name] for name in sorted(project.files.keys())}
    project.files.clear()
    project.files.update(ordered)

    return added, removed


def to_json(project: Project) -> dict:
    folder = project.folder
    folder_dict = {
        "dialogue_enabled": folder.dialogue_enabled,
        "labels_enabled": folder.labels_enabled,
        "ocr_lang": folder.ocr_lang,
        "conf_threshold": folder.conf_threshold,
        "sim_threshold": folder.sim_threshold,
        "similar_image": folder.similar_image,
        "frames_to_skip": folder.frames_to_skip,
        "use_gpu": folder.use_gpu,
        "label_min_duration": folder.label_min_duration,
        "label_max_duration": folder.label_max_duration,
        "label_conf_threshold": folder.label_conf_threshold,
        "label_conf_threshold_min": folder.label_conf_threshold_min,
        "label_mask_crops": [list(c) for c in folder.label_mask_crops],
        "ocr_parallel": folder.ocr_parallel,
        "autopilot_enabled": folder.autopilot_enabled,
        "brightness_full_detect_files": folder.brightness_full_detect_files,
        "min_segment_length": folder.min_segment_length,
        "merge_repeating_silences": folder.merge_repeating_silences,
        "crop_width_fraction": folder.crop_width_fraction,
        "crop_vertical_padding": folder.crop_vertical_padding,
        "crop_min_height_fraction": folder.crop_min_height_fraction,
        "bottom_half_cutoff": folder.bottom_half_cutoff,
    }

    files_dict = {}
    for name, entry in project.files.items():
        files_dict[name] = {
            "crop": None if entry.crop is None else {
                "x": entry.crop.x,
                "y": entry.crop.y,
                "width": entry.crop.width,
                "height": entry.crop.height,
                "source": entry.crop.source.value,
            },
            "brightness": None if entry.brightness is None else {
                "value": entry.brightness.value,
                "source": entry.brightness.source.value,
            },
            "time_ranges": None if entry.time_ranges is None else {
                "ranges": [{"start": r.start, "end": r.end} for r in entry.time_ranges.ranges],
                "source": entry.time_ranges.source.value,
            },
            "media": {
                "width": entry.media.width,
                "height": entry.media.height,
                "duration": entry.media.duration,
                "fps": entry.media.fps,
            },
            "review": entry.review.value,
            "skipped": entry.skipped,
            "sample_time": entry.sample_time,
            "flags": dict(entry.flags),
            "evidence": entry.evidence,
        }

    return {
        "version": 2,
        "folder": folder_dict,
        "files": files_dict,
    }


def from_json(data: dict, project_dir: str) -> Project:
    defaults = FolderSettings()
    folder_data = data.get("folder") or {}

    folder = FolderSettings(
        dialogue_enabled=folder_data.get("dialogue_enabled", defaults.dialogue_enabled),
        labels_enabled=folder_data.get("labels_enabled", defaults.labels_enabled),
        ocr_lang=folder_data.get("ocr_lang", defaults.ocr_lang),
        conf_threshold=folder_data.get("conf_threshold", defaults.conf_threshold),
        sim_threshold=folder_data.get("sim_threshold", defaults.sim_threshold),
        similar_image=folder_data.get("similar_image", defaults.similar_image),
        frames_to_skip=folder_data.get("frames_to_skip", defaults.frames_to_skip),
        use_gpu=folder_data.get("use_gpu", defaults.use_gpu),
        label_min_duration=folder_data.get("label_min_duration", defaults.label_min_duration),
        label_max_duration=folder_data.get("label_max_duration", defaults.label_max_duration),
        label_conf_threshold=folder_data.get("label_conf_threshold", defaults.label_conf_threshold),
        label_conf_threshold_min=folder_data.get(
            "label_conf_threshold_min", defaults.label_conf_threshold_min
        ),
        label_mask_crops=[tuple(c) for c in folder_data.get("label_mask_crops", [])],
        ocr_parallel=folder_data.get("ocr_parallel", defaults.ocr_parallel),
        autopilot_enabled=folder_data.get("autopilot_enabled", defaults.autopilot_enabled),
        brightness_full_detect_files=folder_data.get(
            "brightness_full_detect_files", defaults.brightness_full_detect_files
        ),
        min_segment_length=folder_data.get("min_segment_length", defaults.min_segment_length),
        merge_repeating_silences=folder_data.get(
            "merge_repeating_silences", defaults.merge_repeating_silences
        ),
        crop_width_fraction=folder_data.get("crop_width_fraction", defaults.crop_width_fraction),
        crop_vertical_padding=folder_data.get("crop_vertical_padding", defaults.crop_vertical_padding),
        crop_min_height_fraction=folder_data.get(
            "crop_min_height_fraction", defaults.crop_min_height_fraction
        ),
        bottom_half_cutoff=folder_data.get("bottom_half_cutoff", defaults.bottom_half_cutoff),
    )

    files: dict[str, FileEntry] = {}
    for name, fd in (data.get("files") or {}).items():
        crop_d = fd.get("crop")
        crop = None
        if crop_d:
            crop = Crop(
                x=crop_d["x"], y=crop_d["y"], width=crop_d["width"], height=crop_d["height"],
                source=Source(crop_d["source"]),
            )

        brightness_d = fd.get("brightness")
        brightness = None
        if brightness_d:
            brightness = Brightness(value=brightness_d["value"], source=Source(brightness_d["source"]))

        tr_d = fd.get("time_ranges")
        time_ranges = None
        if tr_d:
            ranges = [TimeRange(start=r.get("start"), end=r.get("end")) for r in tr_d.get("ranges", [])]
            time_ranges = TimeRanges(ranges=ranges, source=Source(tr_d["source"]))

        media_d = fd.get("media") or {}
        media = Media(
            width=media_d.get("width", 0),
            height=media_d.get("height", 0),
            duration=media_d.get("duration", 0.0),
            fps=media_d.get("fps", 0.0),
        )

        files[name] = FileEntry(
            name=name,
            crop=crop,
            brightness=brightness,
            time_ranges=time_ranges,
            media=media,
            review=ReviewState(fd.get("review", ReviewState.PENDING.value)),
            skipped=fd.get("skipped", False),
            sample_time=fd.get("sample_time"),
            flags=dict(fd.get("flags") or {}),
            evidence=dict(fd.get("evidence") or {}),
        )

    return Project(path=project_dir, folder=folder, files=files, migrated_from_v1=False)
