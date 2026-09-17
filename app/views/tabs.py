"""The stage's real tabs (plan 3C): `evidence_tabs(controller)` is the
factory `MainWindow` is given in place of `placeholder_tabs`.

The three titles and their order -- Crop, Brightness, Time ranges -- are the
stage's (ui-spec §3.3), and the inspector follows whichever is current
(ruling B4). Each task of plan 3C replaces one placeholder with its real
view, so a tab still being built keeps showing its values as key/value rows
rather than an empty page.
"""
from __future__ import annotations

from app.views.brightness_view import BrightnessTab
from app.views.stage import StageTab, placeholder_tabs


def evidence_tabs(controller) -> list[StageTab]:
    tabs = placeholder_tabs(controller)
    tabs[1] = BrightnessTab(controller)           # plan 3C Task 3
    return tabs
