"""The episode view's screens (docs spec 2026-09-28, "Episode view"):
Preparing, Working and Done, around the workbench's own Review area.

Each is a QWidget over the controller, with `set_file(name)`; the window
owns the flow between them. Like every view, nothing here imports `core`.
"""
from app.views.episode.done import DoneView
from app.views.episode.prepare import PrepareView
from app.views.episode.script import ScriptPanel
from app.views.episode.slideshow import FramePreview, SlideShow
from app.views.episode.working import WorkingView

__all__ = ["DoneView", "FramePreview", "PrepareView", "ScriptPanel", "SlideShow", "WorkingView"]
