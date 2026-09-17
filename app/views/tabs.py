"""`evidence_tabs(controller)`: the stage's three real tabs (plan 3C).

The factory `MainWindow` is built with. Each tab lives in its own module and
implements `StageTab`; a tab plan 3C has not landed yet keeps the Task-1
`PlaceholderTab`, so the stage always shows all three titles in the same
order. Replacing one is a one-line change here.
"""
from __future__ import annotations

from app.views.crop_view import CropTab
from app.views.stage import StageTab, placeholder_tabs


def evidence_tabs(controller) -> list[StageTab]:
    placeholders = {tab.title: tab for tab in placeholder_tabs(controller)}
    return [
        CropTab(controller),
        placeholders["Brightness"],
        placeholders["Time ranges"],
    ]
