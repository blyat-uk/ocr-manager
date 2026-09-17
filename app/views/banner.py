"""A non-modal message strip under the top bar: startup dependency
warnings, and folders that could not be opened or saved. No message boxes
(design §10)."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.widgets.base import Button, repolish


class Banner(QWidget):
    """`tone`: "warn" (amber) or "bad" (red). Hidden until `show_message`;
    "✕" hides it again."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Banner")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setProperty("tone", "warn")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 7, 14, 7)
        layout.setSpacing(10)
        texts = QVBoxLayout()
        texts.setSpacing(2)
        self._title = QLabel()
        self._title.setObjectName("BannerTitle")
        self._text = QLabel()
        self._text.setObjectName("BannerText")
        self._text.setWordWrap(True)
        self._text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        texts.addWidget(self._title)
        texts.addWidget(self._text)
        layout.addLayout(texts, 1)
        self.dismiss_button = Button("✕", "ghost", small=True)
        self.dismiss_button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.dismiss_button.clicked.connect(self.hide)
        layout.addWidget(self.dismiss_button, 0, Qt.AlignmentFlag.AlignTop)
        self.hide()

    def show_message(self, title: str, text: str, tone: str = "warn") -> None:
        self._title.setText(title)
        self._text.setText(text)
        self.setProperty("tone", tone)
        repolish(self)
        for label in (self._title, self._text):
            repolish(label)
        self.show()

    def title(self) -> str:
        return self._title.text()

    def text(self) -> str:
        return self._text.text()
