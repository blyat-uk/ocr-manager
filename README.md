# Donghua Translation Automation Tool

A PyQt6-based GUI application for automating the Chinese donghua to English subtitle translation workflow.

## Features

- **Visual Crop Selection**: Interactive video player to select OCR crop regions
- **Brightness Testing**: Visual brightness level testing with preview gallery
- **Automated Pipeline**: One-click execution of the complete 8-phase workflow
- **Terminal Output**: Real-time command output display
- **Configuration Persistence**: Save and load project-specific settings

## System Requirements

### Python Environment
- Python 3.13
- python-pyqt6 6.10.0 (system package)
- python-opencv-cuda 4.12.0 (system package)

### Operating System
- Linux (tested on Arch Linux)

### CLI Tools (must be in PATH)
- `ocrp` - OCR extraction
- `ass-credits`, `ass-header`, `ass-qafix` - ASS subtitle processing
- `sub-visualize` - Brightness testing
- `subs-translator` - Translation
- `submerge` - Subtitle embedding
- `srt-to-ass` - Format conversion

### Multimedia Tools
- ffmpeg 8.0.1
- ffprobe 8.0.1

## Installation

### Arch Linux

```bash
# Install system dependencies
sudo pacman -S python python-pyqt6 python-opencv-cuda ffmpeg

# Ensure CLI tools are installed and in PATH
# (Installation method depends on your setup)

# Clone repository
git clone <repository-url>
cd translator

# Run application (no virtual environment needed)
python main.py
```

## Usage

1. **Select Project Directory**: Click "Select Directory" and choose a folder containing your MKV files

2. **Configure OCR Parameters**:
   - Click "Select" next to Crop Region to visually select the subtitle area (or enter coordinates manually)
   - Click "Test" next to Brightness to find optimal brightness level
   - Set Time Start (optional, when opening credits end, e.g., "01:45")
   - Set Time End (optional, to process only part of the video)
   - Set Parallel (number of files to OCR simultaneously, default: 4)

3. **Configure Cleanup**:
   - Set Credits Start minute (when ending credits begin, e.g., 18)

4. **Configure Header Template**:
   - Enter your ASS header styling template

5. **Start Processing**: Click "Start Processing" to run the complete workflow
   - All configuration changes are automatically saved

The pipeline will execute these phases automatically:
1. Preparation (create directories, copy fonts)
2. OCR Extraction
3. Format Conversion (SRT to ASS)
4. Quality Assurance (credits removal, fixes)
5. Translation
6. Cleanup
7. Styling (apply header template)
8. Embedding (mux subtitles into MKV)

## Configuration Files

- **Project config**: `.translation-project/config.json` (in project directory, auto-saved on any change)
- **Global config**: `~/.config/translator-gui/config.json`

## Keyboard Shortcuts

- `Ctrl+Q` - Quit application
- `Ctrl+C` - Stop running pipeline (when focused on main window)

## Troubleshooting

### Missing Dependencies Error
If you see "Required tools not found in PATH", ensure all CLI tools are installed and accessible:
```bash
which ocrp ass-credits ass-header ass-qafix sub-visualize subs-translator submerge ffmpeg
```

### Video Player Not Working
Ensure PyQt6 multimedia backend is properly installed:
```bash
sudo pacman -S python-pyqt6 gstreamer
```

### Frame Extraction Fails
Verify ffmpeg is working:
```bash
ffmpeg -version
```

## Development

### Project Structure
```
translator/
├── main.py                 # Entry point + main window
├── widgets/
│   ├── __init__.py
│   ├── crop_selector.py    # Crop region selection dialog
│   ├── brightness_tester.py # Brightness testing dialog
│   └── terminal_output.py  # Terminal output widget
├── core/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── pipeline.py         # Workflow execution
│   └── video_utils.py      # Frame extraction utilities
├── requirements.txt        # System dependencies reference
├── README.md
└── .gitignore
```

## License

[Your License Here]

## Credits

Built for automating the Chinese donghua translation workflow.
