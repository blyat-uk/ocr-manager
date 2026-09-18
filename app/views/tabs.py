"""`evidence_tabs(controller)`: the stage's three real tabs (plan 3C).

The factory `MainWindow` is built with. Each tab lives in its own module and
implements `StageTab`, and the stage shows the three titles in this order.
Plan 3C Task 4 replaced the last `PlaceholderTab` with the real Time ranges
view, so nothing here stands in for anything any more.
"""
from __future__ import annotations

from app.views.brightness_view import BrightnessTab
from app.views.crop_view import CropTab
from app.views.ranges_view import RangesTab
from app.views.stage import StageTab


def evidence_tabs(controller) -> list[StageTab]:
    return [
        CropTab(controller),
        BrightnessTab(controller),
        RangesTab(controller),
    ]
