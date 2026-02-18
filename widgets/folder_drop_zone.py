"""Drag-and-drop folder selector widget."""

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QFrame
)
from PyQt6.QtGui import QDragEnterEvent, QDropEvent


class FolderDropZone(QFrame):
    """Drag-and-drop folder selector with visual feedback."""

    folder_selected = pyqtSignal(str)  # Emitted with folder path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._folder_path = None
        self._is_drag_over = False

        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        # Icon/hint row
        hint_layout = QHBoxLayout()
        hint_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._icon_label = QLabel("📁")

        hint_layout.addWidget(self._icon_label)

        self._hint_label = QLabel("Drop folder here or click to browse")

        hint_layout.addWidget(self._hint_label)

        layout.addLayout(hint_layout)

        # Path display row
        self._path_layout = QHBoxLayout()
        self._path_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._path_label = QLabel("")

        self._path_label.setWordWrap(True)
        self._path_layout.addWidget(self._path_label)

        # Open folder button (hidden initially)
        self._open_btn = QPushButton("Open")
        self._open_btn.setFixedWidth(60)
        self._open_btn.clicked.connect(self._open_folder_in_manager)
        self._open_btn.setVisible(False)
        self._path_layout.addWidget(self._open_btn)

        layout.addLayout(self._path_layout)

    def mousePressEvent(self, event):
        """Handle click to browse."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._browse_folder()

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and len(urls) == 1:
                path = urls[0].toLocalFile()
                if path and Path(path).is_dir():
                    event.acceptProposedAction()
                    self._set_drag_over(True)
                    return

        event.ignore()

    def dragLeaveEvent(self, event):
        """Handle drag leave."""
        self._set_drag_over(False)

    def dropEvent(self, event: QDropEvent):
        """Handle drop."""
        self._set_drag_over(False)

        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and len(urls) == 1:
                path = urls[0].toLocalFile()
                if path and Path(path).is_dir():
                    self.set_folder(path)
                    event.acceptProposedAction()
                    return

        event.ignore()

    def _set_drag_over(self, is_over: bool):
        """Update drag-over visual state."""
        self._is_drag_over = is_over

    def _browse_folder(self):
        """Open folder browser dialog."""
        start_dir = self._folder_path or "/mnt/FAST/work/"
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Project Directory",
            start_dir
        )
        if folder:
            self.set_folder(folder)

    def _open_folder_in_manager(self):
        """Open folder in system file manager."""
        if self._folder_path:
            import subprocess
            import platform

            system = platform.system()
            if system == "Linux":
                subprocess.Popen(["xdg-open", self._folder_path])
            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", self._folder_path])
            elif system == "Windows":
                subprocess.Popen(["explorer", self._folder_path])

    def set_folder(self, path: str):
        """Set the selected folder path."""
        self._folder_path = path
        folder_name = Path(path).name
        self._path_label.setText(f"<b>{folder_name}</b><br><small>{path}</small>")
        self._hint_label.setText("Project folder selected")
        self._open_btn.setVisible(True)
        self.folder_selected.emit(path)

    def get_folder(self) -> str | None:
        """Get the selected folder path."""
        return self._folder_path

    def clear(self):
        """Clear the selected folder."""
        self._folder_path = None
        self._path_label.setText("")
        self._hint_label.setText("Drop folder here or click to browse")
        self._open_btn.setVisible(False)
