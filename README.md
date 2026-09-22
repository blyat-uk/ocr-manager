# OCR Manager

A GPU-accelerated video OCR tool for batch-extracting hardcoded subtitles and on-screen labels from video files. Built with PyQt6 and PaddleOCR.

OCR Manager takes a folder of video episodes, works out the settings each file needs, and produces ASS subtitle files with accurate timing. It handles both dialogue subtitles (bottom of frame) and positioned text overlays like nameplates, episode titles and location labels.

## Features

### OCR engine

- **PaddleOCR-based extraction** — Uses PaddleOCR for text detection and recognition with GPU acceleration via PaddlePaddle
- **Dialogue subtitles** — Producer-consumer pipeline that decodes video frames, applies crop/brightness preprocessing, and runs OCR with configurable confidence thresholds
- **Label/nameplate detection** — 4-phase pipeline that detects positioned text overlays anywhere in the frame:
  1. Detection scan at 720p to find frames with text
  2. Spatial clustering of detection boxes across frames
  3. Dual OCR (brightness-filtered + raw) at regular intervals with content change detection
  4. Timing refinement via binary search for precise start/end times
- **ASS output** — Generates Advanced SubStation Alpha files with proper timestamp formatting and positioned label events
- **Bit-exact decoding** — PyAV (pinned `av==18.1.0`) is the reference decoder; HDR (PQ and HLG) sources are tone-mapped in the decode graph, and the detectors sample through the same path the OCR pass reads

### Detection, so you do not have to configure each file

An **auto-pilot** runs as soon as a folder is opened and fills in what is missing, file by file:

- **Crop box** — speech-guided probing finds the subtitle band, and files in a series validate each other through a running consensus
- **Brightness threshold** — measured on the frames OCR actually sees, not on a preview approximation
- **Time ranges** — audio fingerprinting finds the repeating intro/outro across the folder, so OCR skips them
- **Metadata and thumbnails** for the queue

Every result carries flags. A result that is not confidently applicable is **never applied silently**: the file is marked for review and the inspector shows what was detected and why it is doubtful. A value you set by hand is never overwritten by detection.

### Running a batch

- **Parallel processing** — several files at once, with a configurable limit you can raise mid-run
- **Live progress** — per-file phase and percentage, ETA, a live feed of subtitles as they are recognised, and GPU utilisation
- **Cooperative cancellation** — pause and stop are cooperative; nothing is ever killed mid-frame
- **Safe output** — each file writes `chi/<name>.ass.partial`, is QA-fixed, then atomically replaces its final file. **An existing subtitle file is never deleted before its replacement is ready**, so stopping a run leaves your previous output intact
- **Crash recovery / resume** — a file with a non-empty `chi/<name>.ass` counts as done and is left out of the next run unless you confirm replacing it
- **Folder watcher** — episodes added to or removed from the folder join or leave the queue while the app is open
- **Desktop notification** on completion, via `notify-send`
- **Logs** — a pipeline log and a per-file log, live during the run

### The window

Dark-only, single window: a top bar with counts and Start, a review queue of episodes down the left, the evidence for the selected file in the middle, and an inspector showing that file's resolved settings on the right. "Test OCR" runs real OCR on a 30-second window so you can check a setting before committing to a whole run. Folder-wide settings live in one sheet. Window geometry and the last folder opened are remembered.

## Download

Ready-to-run builds are on the [releases page](https://github.com/blyat-uk/ocr-manager/releases):

| OS | File |
|---|---|
| Windows 10/11 (x64) | `ocr-manager-vX.Y.Z-win.exe` — installer, per user, no administrator rights; or `-win.zip`, portable |
| macOS 14.5+ (Apple Silicon) | `ocr-manager-vX.Y.Z-mac.dmg` |
| Linux x86_64 (glibc 2.34+: Ubuntu 22.04, Debian 12, Fedora 36 or newer) | `ocr-manager-vX.Y.Z-linux.AppImage`, or `-linux.tar.gz`, portable |

Each carries its own Python, every library and ffmpeg/ffprobe; nothing else needs installing.

**The OCR engine is installed on first run.** PaddlePaddle is too large to ship in every download, so the first start opens a setup window: on an NVIDIA GPU with a recent driver it recommends the matching CUDA build (CUDA 12.9 for RTX 20xx–50xx on driver 575+, older builds for older drivers and cards; 2–5.5 GB to download), otherwise the CPU build (about 0.2 GB). A GPU install is tested before it is used, and falls back to the CPU build by itself if the test fails. The OCR models (about 0.2 GB) download on the first run too. macOS always uses the CPU build.

The engine and logs live in your user data folder, never inside the app:

| OS | Folder |
|---|---|
| Windows | `%LOCALAPPDATA%\ocr-manager` |
| macOS | `~/Library/Application Support/ocr-manager` |
| Linux | `~/.local/share/ocr-manager` (or `$XDG_DATA_HOME/ocr-manager`) |

To switch between the GPU and CPU builds later, or reinstall the engine, start the app with `--setup-engine` (Windows: the Start menu's "OCR Manager (OCR engine setup)"; Linux: the launcher's "OCR engine setup" action; macOS: `open -a "OCR Manager" --args --setup-engine`). The models are cached in `~/.paddlex`.

Notes:

- **Windows:** the installer is not code-signed; SmartScreen may ask you to confirm (More info → Run anyway). `ocr-manager-cli.cmd` in the install folder runs the app with a console, for `--version`, `--self-test` and troubleshooting.
- **macOS:** the app is not notarized. Open it once with right-click → Open (or allow it under System Settings → Privacy & Security), or run `xattr -dr com.apple.quarantine "/Applications/OCR Manager.app"`.
- **Linux:** on X11, Qt needs `libxcb-cursor0` (Debian/Ubuntu), `xcb-util-cursor` (Fedora, Arch).

## Running from source

### Requirements

- **Python** 3.11 or 3.12
- **NVIDIA GPU** with CUDA 12.9 support (for PaddlePaddle GPU acceleration)
- **ffmpeg** / **ffprobe** in PATH (audio extraction and probing for the detectors)

Optional, each degrading gracefully when absent:

- **kdialog** — used as the folder picker when present; Qt's dialog otherwise
- **notify-send** — the completion notification
- **nvidia-smi** — the GPU readout in the run view

Quality assurance of the OCR output is built in (`core/ass_qafix.py`); no external tool is needed for it.

### Installation

#### 1. Clone the repository

```bash
git clone <repo-url>
cd ocr-manager
```

#### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

#### 3. Install PaddlePaddle GPU

PaddlePaddle must be installed from Paddle's own package index before the other dependencies:

```bash
.venv/bin/pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

#### 4. Install remaining dependencies

```bash
.venv/bin/pip install -r requirements.txt
```

#### 5. Install PyQt6

```bash
.venv/bin/pip install PyQt6
```

#### 6. Verify

```bash
.venv/bin/python -c "from videocr.api import get_subtitles; print('OK')"
```

## Usage

### Launch

```bash
.venv/bin/python main.py [folder]     # thin launcher
.venv/bin/python -m app [folder]      # the same window, directly
```

The folder argument is optional; without it the window opens its "Open a folder of episodes" state. You can also drop a folder anywhere on the window.

### Desktop entry (Linux)

To launch from your application menu, create a `.desktop` file:

```ini
[Desktop Entry]
Version=2.0
Type=Application
Name=OCR Manager
Exec=/path/to/ocr-manager/.venv/bin/python /path/to/ocr-manager/main.py
Path=/path/to/ocr-manager
Icon=subtitles
Terminal=false
Categories=AudioVideo;Video;Utility;
```

### Typical workflow

1. **Open a folder** of `.mkv` / `.mp4` episodes (Ctrl+O, the button, or drag and drop). The last folder you opened is where the picker starts.
2. **Wait for the queue to settle.** Detection runs on its own: each row shows what it is still waiting for, then `ready` once its crop, brightness and time ranges are known.
3. **Look at the rows that ask for you.** A `check …` badge means a detector produced something it will not apply unreviewed. Select the file: the inspector shows the detected values and the flag, and "Test OCR (T)" runs real OCR on 30 seconds of that file so you can see what a run would produce.
4. **Mark it reviewed** (Space) once you are happy, or **skip** the file to leave it out entirely.
5. **Adjust folder-wide settings** if needed — "⚙ Folder settings" holds what to extract (dialogue, labels, or labels only), the OCR engine settings, label settings, how many files to run at once, and the auto-pilot's own thresholds.
6. **Press Start.** Files that already have output in `chi/` are asked about once — answer No and they stay out of the run; nothing is deleted either way.
7. **Watch the run.** The run view shows each file's phase and progress, a live feed of recognised lines, and an idle-worker hint offering to raise the parallel count if files are waiting. Pause, resume and stop are all cooperative.
8. Output `.ass` files appear in the `chi/` subdirectory.

### Per-file settings

Everything in the inspector is per-file: crop, brightness threshold and OCR time ranges. Right-click a row in the queue to copy one file's settings and paste them onto another, to re-detect it, to test OCR on it, to open its log, or to skip it. When you correct a value by hand, the app offers to re-detect the other files using your correction as a hint.

## Project structure

```
ocr-manager/
├── main.py                  # Thin launcher: freeze_support(), then app.__main__.main()
├── app/                     # The ONLY Qt layer
│   ├── __main__.py          #   python -m app: QApplication, theme, window, crash guard
│   ├── controller.py        #   ProjectController: owns the project, the job runner and the
│   │                        #   auto-pilot; the only code that mutates the model
│   ├── main_window.py       #   The workbench window
│   ├── views/               #   queue, stage, inspector, folder settings, run view, logs, top bar
│   ├── theme/               #   Dark design tokens and the generated stylesheet
│   └── widgets/             #   Shared UI primitives
├── core/                    # Qt-free: no PyQt6 import belongs anywhere below here
│   ├── project/             #   The model, `.ocr.json` v2 store, v1 migration, and
│   │                        #   ocr_kwargs.py — the exact OCR call for one file
│   ├── jobs/                #   runner.py (GPU/CPU/RUN lanes), detect_jobs.py, apply.py
│   │                        #   (result → model rules), autopilot.py, run.py (the OCR run)
│   ├── detect/              #   crop, brightness, ocr_view, vad, ranges/ (audio fingerprints)
│   └── ass_qafix.py         #   Post-OCR cleanup, applied to every file a run writes
├── videocr/                 # Embedded OCR engine (forked from videocr-PaddleOCR)
│   ├── api.py               #   Public API: get_subtitles, save_subtitles_to_file
│   ├── video.py             #   Frame decoding, producer-consumer OCR pipeline
│   ├── label_scanner.py     #   4-phase label/nameplate detection
│   ├── engine_registry.py   #   Exclusive-lease pool: engines are never shared between threads
│   ├── pyav_adapter.py      #   PyAV capture, tone mapping, the reference decoder
│   └── utils.py             #   ASS formatting, engine creation, label merging
├── tests/                   # Fast unit/behaviour tests, plus tests/ui/ for the window
└── tools/                   # fidelity_check.py (the golden gate) and bench.py
```

Two rules hold the architecture together, and tests enforce both: **`app/` is the only Qt layer**, and **views talk only to `ProjectController`**, never to `core/` directly.

## Output directory layout

After processing, the project directory contains:

```
project/
├── *.mkv                    # Original video files
├── .ocr.json                # Folder settings and per-file values (version 2)
├── .ocr-cache/              # Detector caches — safe to delete, they are rebuilt
│   ├── evidence/            #   What each detector measured, per file
│   └── ...                  #   Audio fingerprints
├── chi/                     # Chinese ASS subtitle files (OCR output, QA-fixed)
│   └── *.ass
├── eng/                     # English ASS files (manual or translated)
└── translate/               # Translation working files
```

`.ocr.json` from an older version is migrated the first time the folder is opened, and the original is kept as `.ocr.json.v1.bak`.

## Configuration reference

### Per file (the inspector)

| Setting | Default | Description |
|---------|---------|-------------|
| Crop region | detected | `x, y, width, height` of the subtitle area |
| Brightness | detected (230 if nothing could be measured) | Threshold for brightness filtering (0-255) |
| Time ranges | detected | Parts of the video to OCR; empty means the whole file |

### Folder settings — what to extract

| Setting | Default | Description |
|---------|---------|-------------|
| Dialogue subtitles | On | Extract bottom-of-frame dialogue |
| Positioned labels | On | Detect nameplates and other overlays. Dialogue off + labels on is labels-only mode, which needs no crop |
| Mask regions | none | Areas to exclude from label detection |

### Folder settings — OCR engine

| Setting | Default | Description |
|---------|---------|-------------|
| Language | `ch` | PaddleOCR language model |
| Confidence threshold | 95 | Minimum OCR confidence |
| Merge similar lines above | 82 | Threshold for merging similar adjacent subtitles |
| Similar-frame threshold | 0.3 | Image similarity threshold for frame deduplication |

### Folder settings — labels

| Setting | Default | Description |
|---------|---------|-------------|
| Minimum duration | 0.5 s | Shortest label display time |
| Maximum duration | 5.0 s | Longest label display time |
| Minimum confidence | 80 | Lower bound for fuzzy matching |

### Folder settings — performance and auto-pilot

| Setting | Default | Description |
|---------|---------|-------------|
| Parallel files | 4 | How many files OCR at once (1-8); can be raised during a run |
| Run detections when a folder opens | On | The auto-pilot |
| Full brightness detection on the first | 3 | Files measured thoroughly before the rest use the folder's plateau |
| Minimum repeating segment | 30 s | Shortest intro/outro the range detector will report |
| Merge repeating silences | Off | Treat a silence between two repeating segments as part of them |
| Crop geometry | — | Width fraction, vertical padding, minimum height, subtitle band start |

## Development

### Tests

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ -m "not slow"   # fast suite
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/                  # full suite
```

`QT_QPA_PLATFORM=offscreen` is required: `tests/ui/` builds the real window headlessly. Markers (`pytest.ini`): `slow`, and `needs_media` for the tests that read the reference video projects listed in `tests/fixtures/media_manifest.json`.

`tests/ui/test_parity.py` is the feature-parity checklist — one row per behaviour that may not be lost, each naming the tests that keep it.

### Fidelity

OCR output is the product, so it is pinned byte for byte:

```bash
.venv/bin/python tools/fidelity_check.py     # must print four OKs
```

It compares against `tests/goldens/*.ass`. **Never run it with `--capture`** — that re-records the goldens — without reviewing and approving the diff first. `tests/test_concurrent_fidelity.py` and `tests/test_run_job.py` pin the same digests from the test suite.

Changes to `videocr/` or `core/ass_qafix.py` move those bytes. Treat them as the fidelity surface.

### Benchmarks

```bash
.venv/bin/python tools/bench.py --suite all --label NAME --out FILE
```

### Building a release

`packaging/build.py` builds the bundle and the artifacts for the OS it runs on (stdlib only, any Python 3.11+): a python-build-standalone interpreter with every dependency except Paddle, the sources, static ffmpeg/ffprobe and the platform launcher, then the `.tar.gz` + `.AppImage`, the Inno Setup installer + `.zip`, or the signed `.app` in a `.dmg`. Every download is pinned by SHA-256. `python packaging/build.py run -- --self-test` runs the built bundle the way its launcher does.

Pushing a tag `vX.Y.Z` that matches `app/version.py` runs `.github/workflows/release.yml`: it builds on Linux, Windows and macOS, smoke-tests each build (self-test, the CPU engine install, OCR of a test line, the window, the launchers) and publishes a GitHub release with the artifacts and `SHA256SUMS.txt`.

## License

See [LICENSE](LICENSE) for details.
