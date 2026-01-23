"""Pipeline phase indicator widget."""

from enum import Enum

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame


class PhaseState(Enum):
    """State of a pipeline phase."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE = "complete"
    ERROR = "error"


class PhaseBadge(QFrame):
    """Single phase badge with state indicator."""

    clicked = pyqtSignal()

    def __init__(self, name: str, clickable: bool = False, parent=None):
        super().__init__(parent)
        self._name = name
        self._state = PhaseState.PENDING
        self._clickable = clickable

        self._setup_ui()
        self._update_style()

        if clickable:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        """Handle mouse press to emit clicked signal."""
        if self._clickable and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Circular indicator
        self._indicator = QFrame()
        self._indicator.setObjectName("phase-badge")
        self._indicator.setFixedSize(24, 24)
        layout.addWidget(self._indicator, 0, Qt.AlignmentFlag.AlignCenter)

        # Phase name label
        self._label = QLabel(self._name)
        self._label.setObjectName("phase-label")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)

    def set_state(self, state: PhaseState):
        """Set the phase state."""
        self._state = state
        self._update_style()

    def _update_style(self):
        """Update visual style based on state."""
        state_classes = {
            PhaseState.PENDING: "phase-badge-pending",
            PhaseState.ACTIVE: "phase-badge-active",
            PhaseState.COMPLETE: "phase-badge-complete",
            PhaseState.ERROR: "phase-badge-error",
        }

        self._indicator.setObjectName(state_classes.get(self._state, "phase-badge"))
        # Force style refresh
        self._indicator.style().unpolish(self._indicator)
        self._indicator.style().polish(self._indicator)

    def get_state(self) -> PhaseState:
        """Get current phase state."""
        return self._state


class PhaseConnector(QFrame):
    """Connecting line between phase badges."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("phase-connector")
        self.setFixedHeight(2)
        self.setMinimumWidth(30)


class PhaseIndicator(QWidget):
    """Pipeline progress visualization with phase badges."""

    badge_clicked = pyqtSignal(int)  # Emits badge index

    def __init__(self, phases: list[str], clickable_indices: list[int] = None, parent=None):
        super().__init__(parent)
        self._phases = phases
        self._clickable_indices = clickable_indices or []
        self._badges: list[PhaseBadge] = []
        self._connectors: list[PhaseConnector] = []

        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        for i, phase_name in enumerate(self._phases):
            # Add badge
            clickable = i in self._clickable_indices
            badge = PhaseBadge(phase_name, clickable=clickable)
            if clickable:
                badge.clicked.connect(lambda idx=i: self.badge_clicked.emit(idx))
            self._badges.append(badge)
            layout.addWidget(badge)

            # Add connector between badges (not after last one)
            if i < len(self._phases) - 1:
                connector = PhaseConnector()
                self._connectors.append(connector)
                layout.addWidget(connector)

    def set_phase_state(self, index: int, state: PhaseState):
        """Set state for a specific phase."""
        if 0 <= index < len(self._badges):
            self._badges[index].set_state(state)

    def set_active_phase(self, index: int):
        """Set a phase as active and mark previous phases as complete."""
        for i, badge in enumerate(self._badges):
            if i < index:
                badge.set_state(PhaseState.COMPLETE)
            elif i == index:
                badge.set_state(PhaseState.ACTIVE)
            else:
                badge.set_state(PhaseState.PENDING)

    def mark_complete(self):
        """Mark all phases as complete."""
        for badge in self._badges:
            badge.set_state(PhaseState.COMPLETE)

    def mark_error(self, phase_index: int):
        """Mark a specific phase as error."""
        if 0 <= phase_index < len(self._badges):
            self._badges[phase_index].set_state(PhaseState.ERROR)

    def reset(self):
        """Reset all phases to pending state."""
        for badge in self._badges:
            badge.set_state(PhaseState.PENDING)

    def get_phase_count(self) -> int:
        """Get number of phases."""
        return len(self._badges)

    def get_phase_state(self, index: int) -> PhaseState | None:
        """Get state of a specific phase."""
        if 0 <= index < len(self._badges):
            return self._badges[index].get_state()
        return None
