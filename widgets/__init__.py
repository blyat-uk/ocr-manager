"""GUI widgets for translator application."""

from widgets.folder_drop_zone import FolderDropZone
from widgets.phase_indicator import PhaseIndicator, PhaseState, PhaseBadge
from widgets.progress_table import ProgressTableWidget
from widgets.time_range_slider import TimeRangeSlider, RangeSlider
from widgets.videocr_settings_dialog import VideoCRSettingsDialog

__all__ = [
    'FolderDropZone',
    'PhaseIndicator',
    'PhaseState',
    'PhaseBadge',
    'ProgressTableWidget',
    'RangeSlider',
    'TimeRangeSlider',
    'VideoCRSettingsDialog',
]
