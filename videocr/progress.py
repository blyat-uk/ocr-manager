"""Unified progress tracking across the video processing pipeline.

Dispatches weighted progress to an external callback for dialogue
extraction and label scanning phases.
"""

from __future__ import annotations


class ProgressTracker:
    """Progress tracking with callback dispatch for dialogue and labels.

    Manages two independent progress ranges (0-100%) for:
    - "Extracting dialogue" during dialogue extraction
    - "Extracting labels" during all label scanning phases

    Each range independently tracks 0-100% progress for its task.
    """

    def __init__(self, include_dialogue: bool = True, include_labels: bool = True,
                 progress_callback: callable = None):
        """Initialize the progress tracker.

        Args:
            include_dialogue: Whether dialogue extraction will run.
            include_labels: Whether label scanning will run.
            progress_callback: Optional callback(task_name: str, percent: int) for external progress reporting.
        """
        self.include_dialogue = include_dialogue
        self.include_labels = include_labels
        self.progress_callback = progress_callback
        self._current_task = None  # 'dialogue' or 'labels'
        self._current_phase = None
        self._phase_progress = 0.0
        self._phase_total = 0
        self._completed_weight = 0.0  # Accumulated weight from completed phases within current task
        self._configure_label_weights()

    def _configure_label_weights(self):
        """Set up internal weights for label phases.

        Label phases are weighted internally within the labels progress range:
        - Phase 1: 20% (detection scan - sparse sampling at 720p)
        - Phase 2: 10% (position grouping - pure spatial, fast)
        - Phase 3: 40% (crop, clean, recognize - dual OCR at intervals)
        - Phase 4: 30% (timing refinement - detection scan for start/end)
        """
        self._label_weights = {
            'label_p1': 20,
            'label_p2': 10,
            'label_p3': 40,
            'label_p4': 30,
        }

    def start(self, desc: str = None):
        """Mark the tracker as ready to display progress.

        Progress reporting begins when the first phase starts via set_phase().
        """
        pass

    def set_phase(self, phase: str, total_units: int, desc: str = None):
        """Begin a new phase with known work units.

        Args:
            phase: Phase identifier ('dialogue', 'label_p1', 'label_p2', 'label_p3', 'label_p4').
            total_units: Total work units in this phase.
            desc: Optional description (ignored, uses task-based descriptions).
        """
        is_label_phase = phase.startswith('label_')

        # Detect task switch (dialogue → labels)
        if is_label_phase and self._current_task == 'dialogue':
            self._current_task = 'labels'
            self._completed_weight = 0.0
            self._current_phase = None
            if self.progress_callback:
                self.progress_callback('Extracting labels', 0)
        elif is_label_phase and self._current_task is None:
            # Starting directly with labels (only_labels mode)
            self._current_task = 'labels'
            self._completed_weight = 0.0
            if self.progress_callback:
                self.progress_callback('Extracting labels', 0)
        elif not is_label_phase and phase == 'dialogue':
            # Starting dialogue phase
            if self._current_task is None:
                self._current_task = 'dialogue'
                if self.progress_callback:
                    self.progress_callback('Extracting dialogue', 0)

        # Complete previous phase within the same task (for label phases)
        if is_label_phase and self._current_phase is not None and self._current_phase in self._label_weights:
            self._completed_weight += self._label_weights[self._current_phase]

        self._current_phase = phase
        self._phase_progress = 0.0
        self._phase_total = max(1, total_units)  # Avoid division by zero

    def update(self, n: int = 1):
        """Increment progress within current phase.

        Args:
            n: Number of units completed.
        """
        if self._current_phase is None:
            return

        self._phase_progress += n

        # Calculate progress based on current task
        if self._current_task == 'dialogue':
            # Dialogue: simple 0-100% based on frames
            phase_fraction = min(1.0, self._phase_progress / self._phase_total)
            total_progress = phase_fraction * 100
        else:
            # Labels: weighted across phases
            phase_weight = self._label_weights.get(self._current_phase, 0)
            phase_fraction = min(1.0, self._phase_progress / self._phase_total)
            current_contribution = phase_weight * phase_fraction
            total_progress = self._completed_weight + current_contribution

        if self.progress_callback:
            task_name = 'Extracting dialogue' if self._current_task == 'dialogue' else 'Extracting labels'
            self.progress_callback(task_name, int(total_progress))
