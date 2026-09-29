"""Qt-free per-project domain model, JSON store and v1 migration.

No PyQt6 import belongs anywhere under this package -- it is the shared
model used by the pipeline, the store, migration and the Stage 3 window.
Where a project's files live (a folder's, or one episode's under the cache
root) is core.project.layout's; the cache root's own bookkeeping is
core.project.episode_cache's.
"""
from core.project.layout import (
    ProjectLayout,
    cache_root,
    episode_layout,
    episode_target,
    folder_layout,
    layout_of,
    video_key,
)
from core.project.migrate import migrate_v1
from core.project.model import (
    MIN_CROP_SIDE,
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
    clamp_crop_box,
    frame_size_known,
)
from core.project.ocr_kwargs import DEFAULT_BRIGHTNESS, OcrCall, ocr_call_for
from core.project.store import (
    CONFIG_FILENAME,
    VIDEO_EXTENSIONS,
    UnsupportedProjectVersion,
    from_json,
    list_video_files,
    load_project,
    reconcile_files,
    save_project,
    to_json,
)

__all__ = [
    "MIN_CROP_SIDE",
    "Brightness",
    "Crop",
    "FileEntry",
    "FolderSettings",
    "Media",
    "Project",
    "ReviewState",
    "Source",
    "TimeRange",
    "TimeRanges",
    "clamp_crop_box",
    "frame_size_known",
    "CONFIG_FILENAME",
    "VIDEO_EXTENSIONS",
    "UnsupportedProjectVersion",
    "ProjectLayout",
    "cache_root",
    "episode_layout",
    "episode_target",
    "folder_layout",
    "layout_of",
    "video_key",
    "DEFAULT_BRIGHTNESS",
    "OcrCall",
    "ocr_call_for",
    "from_json",
    "list_video_files",
    "load_project",
    "migrate_v1",
    "reconcile_files",
    "save_project",
    "to_json",
]
