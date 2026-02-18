# OCR Manager

A GPU-accelerated video OCR tool for batch-extracting hardcoded subtitles and on-screen labels from video files. Built with PyQt6 and PaddleOCR.

OCR Manager processes folders of video files in parallel, producing ASS subtitle files with accurate timing. It handles both dialogue subtitles (bottom of frame) and positioned text overlays like nameplates, episode titles, and location labels.

## Features

### OCR Engine

- **PaddleOCR-based extraction** — Uses PaddleOCR for text detection and recognition with GPU acceleration via PaddlePaddle
- **Dialogue subtitles** — Producer-consumer pipeline that decodes video frames, applies crop/brightness preprocessing, and runs OCR with configurable confidence thresholds
- **Label/nameplate detection** — 4-phase pipeline that detects positioned text overlays anywhere in the frame:
  1. Detection scan at 720p to find frames with text
  2. Spatial clustering of detection boxes across frames
  3. Dual OCR (brightness-filtered + raw) at regular intervals with content change detection
  4. Timing refinement via binary search for precise start/end times
- **ASS output** — Generates Advanced SubStation Alpha files with proper timestamp formatting and positioned label events

### Parallel Processing

- **Queue-based concurrency** — Processes multiple video files simultaneously with configurable worker count (1-8)
- **Per-file progress tracking** — Real-time progress display for each file with ETA calculation
- **Cooperative cancellation** — Clean shutdown that terminates workers and removes partial output

### GUI

- **Interactive crop selection** — Visual frame-based crop region selector with frame scrubbing across episodes and a timeline slider
- **Brightness tester** — Preview brightness threshold effects on cropped frames with carousel navigation and zoom/pan
- **Per-file configuration** — Override crop region, brightness, and time range per file with copy/paste support between files
- **Label mask regions** — Define mask areas to exclude from label detection (drawn visually in the crop selector)
- **Time range slider** — Restrict OCR to a portion of the video, auto-scaled to the longest file's duration
- **Pipeline status indicator** — Phase badges showing Create Directory, OCR Extraction, and Quality Assurance progress
- **File table** — Sortable table with status indicators (queued/processing/completed/failed/done), per-file progress bars, resolution labels, and config override markers
- **Real-time subtitle preview** — Live subtitle feed dialog showing detected lines as OCR runs
- **Logs viewer** — Per-file log output and QA phase logs accessible during and after processing
- **Project configuration persistence** — Settings saved to `.ocr.json` per project with async writes
- **Desktop notifications** — Sends completion/failure notifications via `notify-send`
- **Crash recovery** — Automatically detects completed files and resumes from where it left off
- **Dark theme** — Fusion-based dark theme applied by default

### Post-Processing

- **Quality assurance** — Runs `ass-qafix` (if available) twice on output files to clean OCR artifacts
- **Output organization** — Creates `chi/`, `eng/`, and `translate/` directories in the project folder

## Requirements

- **Python** 3.11 or 3.12
- **NVIDIA GPU** with CUDA 12.9 support (for PaddlePaddle GPU acceleration)
- **ffmpeg** in PATH (used for frame extraction in crop selector and brightness tester)
- **ass-qafix** in PATH (optional, for post-OCR quality assurance)

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd ocr-manager
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Install PaddlePaddle GPU

PaddlePaddle must be installed from Paddle's own package index before the other dependencies:

```bash
.venv/bin/pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

### 4. Install remaining dependencies

```bash
.venv/bin/pip install -r requirements.txt
```

### 5. Install PyQt6

```bash
.venv/bin/pip install PyQt6
```

### 6. Verify

```bash
.venv/bin/python -c "from videocr.api import get_subtitles; print('OK')"
```

## Usage

### Launch the GUI

```bash
.venv/bin/python main.py
```

### Desktop entry (Linux)

To launch from your application menu, create a `.desktop` file:

```ini
[Desktop Entry]
Version=1.5
Type=Application
Name=OCR Manager
Exec=/path/to/ocr-manager/.venv/bin/python /path/to/ocr-manager/main.py
Path=/path/to/ocr-manager
Icon=subtitles
Terminal=false
Categories=AudioVideo;Video;Utility;
```

### Typical workflow

1. Click **Select Folder** and choose a directory containing `.mkv` or `.mp4` files
2. Use **Select** to draw a crop region around the subtitle area
3. Use **Test** to verify the brightness threshold produces clean white-on-black text
4. Optionally enable **label detection** and configure its settings
5. Adjust the **time range** slider if you only need a portion of each video
6. Set the **parallel worker** count based on your GPU memory
7. Click **Start Processing** — the pipeline creates directories, runs OCR in parallel, and applies QA fixes
8. Output `.ass` files appear in the `chi/` subdirectory

### Per-file overrides

Select one or more files in the table, then adjust crop, brightness, or time range. The settings apply only to the selected files. A dot indicator appears next to files with custom configuration. Right-click for copy/paste operations.

## Project Structure

```
ocr-manager/
├── main.py                  # Application entry point and main window
├── theme.py                 # Dark theme stylesheet
├── core/
│   ├── config.py            # Config dataclasses, per-file config store, project persistence
│   ├── config_saver.py      # Async config writer (background thread)
│   ├── image_processing.py  # Brightness filtering for preview
│   ├── log_store.py         # Per-file log accumulator
│   ├── ocr_manager.py       # Parallel worker orchestration with timing
│   ├── ocr_worker.py        # Single-file OCR via QThread + videocr API
│   ├── pipeline.py          # 3-phase state machine with crash recovery
│   └── video_utils.py       # Frame extraction and metadata scanning
├── videocr/                 # Embedded OCR engine (forked from videocr-PaddleOCR)
│   ├── api.py               # Public API: get_subtitles, save_subtitles_to_file
│   ├── video.py             # Frame decoding, producer-consumer OCR pipeline
│   ├── label_scanner.py     # 4-phase label/nameplate detection
│   ├── models.py            # PredictedFrame, PredictedSubtitle dataclasses
│   ├── progress.py          # Callback-based progress dispatcher
│   ├── pyav_adapter.py      # PyAV video capture wrapper
│   └── utils.py             # ASS formatting, OCR engine creation, label merging
├── widgets/
│   ├── brightness_tester.py # Brightness preview dialog with zoom/pan
│   ├── crop_selector.py     # Visual crop region + label mask selector
│   ├── file_table.py        # Sortable file list with status/progress
│   ├── label_settings_dialog.py
│   ├── logs_dialog.py       # Per-file log viewer
│   ├── phase_indicator.py   # Pipeline phase badges
│   ├── subtitle_preview_dialog.py  # Real-time subtitle feed
│   ├── time_range_slider.py # Dual-handle time range control
│   └── videocr_settings_dialog.py
└── resources/               # Icons
```

## Output Directory Layout

After processing, the project directory contains:

```
project/
├── *.mkv                    # Original video files
├── chi/                     # Chinese ASS subtitle files (OCR output, cleaned)
│   └── *.ass
├── eng/                     # English ASS files (manual or translated)
├── translate/               # Translation working files
└── .ocr.json                # Saved project configuration
```

## Configuration Reference

### Global settings (UI)

| Setting | Default | Description |
|---------|---------|-------------|
| Crop Region | — | `x, y, width, height` of the subtitle area |
| Brightness | 230 | Threshold for brightness filtering (0-255) |
| Time Range | Full | Start/end timestamps to limit OCR scope |
| Parallel | 4 | Number of concurrent OCR workers |
| Labels | Enabled | Detect positioned text overlays |

### VideoCR settings (via phase badge click)

| Setting | Default | Description |
|---------|---------|-------------|
| OCR Language | `ch` | PaddleOCR language model |
| Confidence | 95 | Minimum OCR confidence threshold |
| Similarity | 82 | Threshold for merging similar adjacent subtitles |
| Similar Image | 0.3 | Image similarity threshold for frame deduplication |

### Label settings (via gear icon)

| Setting | Default | Description |
|---------|---------|-------------|
| Labels Only | Off | Skip dialogue extraction, detect only labels |
| Min Duration | 0.5s | Minimum label display duration |
| Max Duration | 5.0s | Maximum label display duration |
| Confidence | 95 | Label OCR confidence threshold |
| Min Confidence | 80 | Lower bound for fuzzy matching |

## License

See [LICENSE](LICENSE) for details.
