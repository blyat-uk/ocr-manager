"""Pipeline execution and workflow management."""

import shutil
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal, QProcess

from core.config import Config, GlobalConfig, save_header_to_translate


class Pipeline(QObject):
    """Manages workflow execution."""

    # Signals
    command_started = pyqtSignal(str)
    output_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    phase_completed = pyqtSignal(str)
    pipeline_finished = pyqtSignal(bool)

    PHASES = ["Preparation", "OCR Extraction", "Quality Assurance", "Translation", "Cleanup", "Styling", "Embedding"]

    def __init__(self, config: Config, global_config: GlobalConfig, stop_at_phase: int = None):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.process = None
        self.current_phase = 0
        self.subphase = 0
        self.stop_at_phase = stop_at_phase  # None = run all, 0-6 = stop after that phase

    def detect_resume_phase(self) -> int:
        """Detect which phase to resume from based on file structure.

        Checkpoints (ass-* commands are idempotent):
        - eng-ass/*.eng.ass exists → resume from Phase 7 (Embedding)
        - translate/*.eng.ass exists → resume from Phase 5 (Cleanup)
        - chi-ass/*.ass exists → resume from Phase 3 (QA)
        """
        project_path = Path(self.config.project_path)

        eng_ass = project_path / "eng-ass"
        translate = project_path / "translate"
        chi_ass = project_path / "chi-ass"

        # Check 1: Styling complete? (files moved to eng-ass)
        if eng_ass.exists() and list(eng_ass.glob("*.eng.ass")):
            return 6  # Resume from Phase 7 (Embedding)

        # Check 2: Translation complete? (files still in translate)
        if translate.exists() and list(translate.glob("*.eng.ass")):
            return 4  # Resume from Phase 5 (Cleanup)

        # Check 3: OCR complete? (files in chi-ass)
        if chi_ass.exists() and list(chi_ass.glob("*.ass")):
            return 2  # Resume from Phase 3 (QA) - idempotent

        # Fresh start
        return 0

    def start(self):
        """Start pipeline execution, resuming from detected phase."""
        # Always ensure directories exist (even when resuming)
        self.ensure_directories()

        detected_phase = self.detect_resume_phase()

        self.current_phase = detected_phase
        self.subphase = 0

        if detected_phase > 0:
            phase_name = self.PHASES[detected_phase]
            self.output_received.emit(f"Resuming from phase {detected_phase + 1}: {phase_name}...\n")

        # Check if resume phase already exceeds stop_at_phase
        if self.stop_at_phase is not None and detected_phase > self.stop_at_phase:
            self.output_received.emit(f"Resume phase ({detected_phase + 1}) is beyond stop-at phase ({self.stop_at_phase + 1}). Already complete.\n")
            self.pipeline_finished.emit(True)
            return

        self.run_next_phase()

    def ensure_directories(self):
        """Ensure all required directories exist."""
        project_path = Path(self.config.project_path)
        dirs = ["chi-ass", "translate", "eng-ass", ".translation-project"]
        for d in dirs:
            (project_path / d).mkdir(exist_ok=True)

    def run_next_phase(self):
        """Execute next phase in sequence."""
        if self.current_phase >= len(self.PHASES):
            self.pipeline_finished.emit(True)
            return

        # Check if we should stop after completing target phase
        if self.stop_at_phase is not None and self.current_phase > self.stop_at_phase:
            self.output_received.emit(f"\n✓ Stopped after phase {self.stop_at_phase + 1}: {self.PHASES[self.stop_at_phase]}\n")
            self.pipeline_finished.emit(True)
            return

        phase_method = getattr(self, f"phase_{self.current_phase + 1}")
        phase_method()

    def phase_1(self):
        """Phase 1: Preparation - Create directory structure and copy fonts."""
        project_path = Path(self.config.project_path)

        # Create directories (already called in start(), but safe to call again)
        self.ensure_directories()

        # Save header template to translate/header.txt for persistence
        if self.config.header_template:
            save_header_to_translate(self.config.project_path, self.config.header_template)
            self.output_received.emit("Saved header template to translate/header.txt\n")

        # Copy fonts
        if self.global_config.fonts_directory:
            fonts_src = Path(self.global_config.fonts_directory)
            fonts_dst = project_path / "Fonts"
            if fonts_src.exists() and not fonts_dst.exists():
                shutil.copytree(fonts_src, fonts_dst)

        self.phase_completed.emit("Preparation")
        self.current_phase += 1
        self.run_next_phase()

    def phase_2(self):
        """Phase 2: OCR Extraction - ocrp outputs .ass files directly."""
        project_path = Path(self.config.project_path)
        chi_ass_dir = project_path / "chi-ass"

        if self.subphase == 0:
            # Pre-process: Move already OCR'd files to 'ocr-ready' (created on demand)
            ocr_ready_dir = project_path / "ocr-ready"

            # Check each .mkv file for existing .ass
            for mkv_file in project_path.glob("*.mkv"):
                ass_name = mkv_file.stem + ".ass"

                # Check if .ass exists in root or chi-ass folder
                ass_in_root = project_path / ass_name
                ass_in_chi = chi_ass_dir / ass_name

                ass_file = None
                if ass_in_root.exists() and ass_in_root.stat().st_size > 0:
                    ass_file = ass_in_root
                elif ass_in_chi.exists() and ass_in_chi.stat().st_size > 0:
                    ass_file = ass_in_chi

                if ass_file:
                    # Create ocr-ready directory only when needed
                    ocr_ready_dir.mkdir(exist_ok=True)
                    # Move .mkv to ocr-ready
                    shutil.move(str(mkv_file), str(ocr_ready_dir / mkv_file.name))
                    # Move .ass to chi-ass (if not already there)
                    if ass_file != ass_in_chi:
                        shutil.move(str(ass_file), str(chi_ass_dir / ass_name))

                    self.output_received.emit(f"Skipping OCR for {mkv_file.name} (already has ASS)\n")

            # Continue to next subphase
            self.subphase += 1
            self.run_next_phase()

        elif self.subphase == 1:
            # Run OCR on remaining files
            cmd = ["ocrp"]

            # Full frame and/or crop region (can be combined)
            if self.config.fullframe:
                cmd.append("--fullframe")
            if self.config.crop_width > 0 and self.config.crop_height > 0:
                crops = f"{self.config.crop_x},{self.config.crop_y},{self.config.crop_width},{self.config.crop_height}"
                cmd.extend(["--crop", crops])

            # Width parameter for downscaling
            # if self.config.ocr_width > 0:
            #     cmd.extend(["--width", str(self.config.ocr_width)])

            cmd.extend(["-b", str(self.config.brightness)])
            cmd.extend(["--max", str(self.config.ocr_parallel)])

            if self.config.time_start:
                cmd.extend(["-ts", self.config.time_start])

            if self.config.time_end:
                cmd.extend(["-te", self.config.time_end])

            self.run_command(cmd, cwd=self.config.project_path)

        elif self.subphase == 2:
            # Post-process: Move ASS files from root to chi-ass
            for ass_file in project_path.glob("*.ass"):
                shutil.move(str(ass_file), str(chi_ass_dir / ass_file.name))
                self.output_received.emit(f"Moved {ass_file.name} to chi-ass/\n")

            # Move MKV files back from 'ocr-ready' to root
            ocr_ready_dir = project_path / "ocr-ready"
            if ocr_ready_dir.exists():
                for mkv_file in ocr_ready_dir.glob("*.mkv"):
                    shutil.move(str(mkv_file), str(project_path / mkv_file.name))
                    self.output_received.emit(f"Restored {mkv_file.name} to project root\n")

                # Delete ocr-ready folder only if empty
                if not any(ocr_ready_dir.iterdir()):
                    ocr_ready_dir.rmdir()

            self.phase_completed.emit("OCR Extraction")
            self.current_phase += 1
            self.subphase = 0
            self.run_next_phase()

    def phase_3(self):
        """Phase 3: Quality Assurance - Run ass-qafix (twice) and ass-credits."""
        chi_ass = Path(self.config.project_path) / "chi-ass"

        if self.subphase == 0:
            # Run ass-qafix first pass
            cmd = ["ass-qafix", "--inplace"]
            self.run_command(cmd, cwd=str(chi_ass))
        elif self.subphase == 1:
            # Run ass-qafix second pass (sometimes first pass doesn't fix all issues)
            cmd = ["ass-qafix", "--inplace"]
            self.run_command(cmd, cwd=str(chi_ass))
        elif self.subphase == 2:
            # Run ass-credits if remove_credits is enabled
            if self.config.remove_credits:
                cmd = ["ass-credits", "--yes"]
                self.run_command(cmd, cwd=str(chi_ass))
            else:
                # Skip ass-credits, proceed to copy files
                self.output_received.emit("Skipping ass-credits (disabled)\n")
                self.subphase += 1
                self.run_next_phase()
        elif self.subphase == 3:
            # Move to translate directory
            translate_dir = Path(self.config.project_path) / "translate"
            for ass_file in chi_ass.glob("*.ass"):
                shutil.copy(str(ass_file), str(translate_dir / ass_file.name))

            self.phase_completed.emit("Quality Assurance")
            self.current_phase += 1
            self.subphase = 0
            self.run_next_phase()

    def phase_4(self):
        """Phase 4: Translation."""
        translate_dir = Path(self.config.project_path) / "translate"

        if self.subphase == 0:
            cmd = ["subs-translator"]

            # Check if glossary.json exists and is not empty
            glossary_file = translate_dir / "glossary.json"
            if glossary_file.exists() and glossary_file.stat().st_size > 0:
                cmd.append("--only-translate")
                self.output_received.emit("Found existing glossary.json, using --only-translate mode\n")

            self.run_command(cmd, cwd=str(translate_dir))
        elif self.subphase == 1:
            self.phase_completed.emit("Translation")
            self.current_phase += 1
            self.subphase = 0
            self.run_next_phase()

    def phase_5(self):
        """Phase 5: Cleanup - Remove Chinese .ass files."""
        translate_dir = Path(self.config.project_path) / "translate"

        # Remove *.ass files (not *.eng.ass)
        for ass_file in translate_dir.glob("*.ass"):
            if not ass_file.name.endswith(".eng.ass"):
                ass_file.unlink()

        self.phase_completed.emit("Cleanup")
        self.current_phase += 1
        self.run_next_phase()

    def phase_6(self):
        """Phase 6: Styling - Write header, run ass-header, compare, and QA fix (twice)."""
        translate_dir = Path(self.config.project_path) / "translate"
        eng_ass_dir = Path(self.config.project_path) / "eng-ass"

        if self.subphase == 0:
            # Ensure eng-ass directory exists (for projects started before this dir was added)
            eng_ass_dir.mkdir(exist_ok=True)

            # Write header file to translate directory
            header_file = translate_dir / "header.txt"
            with open(header_file, "w", encoding="utf-8") as f:
                f.write(self.config.header_template)

            # Move *.eng.ass files from translate to eng-ass
            for eng_file in translate_dir.glob("*.eng.ass"):
                shutil.move(str(eng_file), str(eng_ass_dir / eng_file.name))
                self.output_received.emit(f"Moved {eng_file.name} to eng-ass/\n")

            # Run ass-header in eng-ass, referencing header in translate
            cmd = ["ass-header", "../translate/header.txt"]
            self.run_command(cmd, cwd=str(eng_ass_dir))
        elif self.subphase == 1:
            # Run ass-compare QA check
            cmd = ["ass-compare", "--chi", "chi-ass", "--eng", "eng-ass", "--no-color"]
            self.run_command(cmd, cwd=self.config.project_path)
        elif self.subphase == 2:
            # Run ass-qafix on English files - first pass
            cmd = ["ass-qafix", "--inplace"]
            self.run_command(cmd, cwd=str(eng_ass_dir))
        elif self.subphase == 3:
            # Run ass-qafix on English files - second pass (sometimes first pass doesn't fix all issues)
            cmd = ["ass-qafix", "--inplace"]
            self.run_command(cmd, cwd=str(eng_ass_dir))
        elif self.subphase == 4:
            self.phase_completed.emit("Styling")
            self.current_phase += 1
            self.subphase = 0
            self.run_next_phase()

    def phase_7(self):
        """Phase 7: Embedding - Copy files and run submerge."""
        eng_ass_dir = Path(self.config.project_path) / "eng-ass"
        project_root = Path(self.config.project_path)

        if self.subphase == 0:
            # Copy *.eng.ass files from eng-ass to root
            for eng_file in eng_ass_dir.glob("*.eng.ass"):
                shutil.copy(str(eng_file), str(project_root / eng_file.name))
                self.output_received.emit(f"Copied {eng_file.name} to project root\n")

            # Run submerge
            cmd = ["submerge", "-p", str(self.global_config.parallel_workers_muxing)]
            self.run_command(cmd, cwd=str(project_root))
        elif self.subphase == 1:
            self.phase_completed.emit("Embedding")
            self.current_phase += 1
            self.subphase = 0
            self.run_next_phase()

    def run_command(self, cmd: list, cwd: str = None):
        """Execute command and emit output signals."""
        self.command_started.emit(" ".join(cmd))

        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.on_stdout)
        self.process.readyReadStandardError.connect(self.on_stderr)
        self.process.finished.connect(self.on_process_finished)

        if cwd:
            self.process.setWorkingDirectory(cwd)

        self.process.start(cmd[0], cmd[1:])

    def on_stdout(self):
        """Handle stdout output."""
        data = self.process.readAllStandardOutput().data().decode("utf-8")
        self.output_received.emit(data)

    def on_stderr(self):
        """Handle stderr output."""
        data = self.process.readAllStandardError().data().decode("utf-8")
        self.output_received.emit(data)

    def on_process_finished(self, exit_code, exit_status):
        """Handle process completion."""
        if exit_code != 0:
            self.error_occurred.emit(f"Command failed with exit code {exit_code}")
            self.pipeline_finished.emit(False)
            return

        # Advance subphase or phase
        self.subphase += 1

        # Check if we need to continue with subphases
        phase_method = getattr(self, f"phase_{self.current_phase + 1}")
        phase_method()

    def stop(self):
        """Stop pipeline execution."""
        if self.process and self.process.state() == QProcess.ProcessState.Running:
            self.process.terminate()
            self.process.waitForFinished(3000)
            if self.process.state() == QProcess.ProcessState.Running:
                self.process.kill()

    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self.process is not None and self.process.state() == QProcess.ProcessState.Running
