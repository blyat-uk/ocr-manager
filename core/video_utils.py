"""Utility functions for video frame extraction."""
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

# Cache for HDR metadata to avoid repeated FFprobe calls
_hdr_cache: dict[str, "HDRMetadata"] = {}


@dataclass
class HDRMetadata:
    """HDR video metadata."""
    is_hdr: bool
    transfer: Optional[str] = None  # 'smpte2084' (PQ) or 'arib-std-b67' (HLG)
    color_primaries: Optional[str] = None
    color_space: Optional[str] = None


def detect_hdr(video_path: str) -> HDRMetadata:
    """Detect HDR metadata from video file.

    Args:
        video_path: Path to video file

    Returns:
        HDRMetadata with detection results
    """
    # Check cache first
    if video_path in _hdr_cache:
        return _hdr_cache[video_path]

    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=color_transfer,color_primaries,color_space',
            '-of', 'json',
            video_path
        ], capture_output=True, text=True, check=True)

        data = json.loads(result.stdout)
        streams = data.get('streams', [])

        if not streams:
            metadata = HDRMetadata(is_hdr=False)
            _hdr_cache[video_path] = metadata
            return metadata

        stream = streams[0]
        transfer = stream.get('color_transfer')
        primaries = stream.get('color_primaries')
        color_space = stream.get('color_space')

        # HDR transfer functions: PQ (HDR10/Dolby Vision) or HLG
        hdr_transfers = {'smpte2084', 'arib-std-b67'}
        is_hdr = transfer in hdr_transfers

        metadata = HDRMetadata(
            is_hdr=is_hdr,
            transfer=transfer,
            color_primaries=primaries,
            color_space=color_space
        )
        _hdr_cache[video_path] = metadata
        return metadata

    except (subprocess.CalledProcessError, json.JSONDecodeError):
        metadata = HDRMetadata(is_hdr=False)
        _hdr_cache[video_path] = metadata
        return metadata


def extract_frame(mkv_path: str, timestamp: str, output_path: str) -> bool:
    """Extract single frame from video at timestamp.

    Automatically detects HDR content and applies tone mapping for correct
    SDR display.

    Args:
        mkv_path: Path to MKV file
        timestamp: Timestamp string (HH:MM:SS or HH:MM:SS.cs)
        output_path: Where to save extracted frame (PNG)

    Returns:
        True if successful, False otherwise
    """
    try:
        hdr = detect_hdr(mkv_path)

        if hdr.is_hdr:
            # Apply tone mapping for HDR content
            # zscale converts HDR to linear, tonemap applies Hable curve,
            # then convert to BT.709 SDR
            if hdr.transfer == 'smpte2084':
                # HDR10/PQ tone mapping
                vf = 'zscale=t=linear:npl=100,format=gbrpf32le,tonemap=hable,zscale=t=bt709,format=yuv420p'
            else:
                # HLG tone mapping (arib-std-b67)
                vf = 'zscale=t=linear,format=gbrpf32le,tonemap=hable,zscale=t=bt709,format=yuv420p'

            subprocess.run([
                'ffmpeg', '-ss', timestamp, '-i', mkv_path,
                '-vf', vf,
                '-vframes', '1', '-y', output_path
            ], check=True, capture_output=True)
        else:
            # SDR video - no filter needed
            subprocess.run([
                'ffmpeg', '-ss', timestamp, '-i', mkv_path,
                '-vframes', '1', '-y', output_path
            ], check=True, capture_output=True)

        return True
    except subprocess.CalledProcessError:
        return False


def get_video_duration(mkv_path: str) -> int:
    """Get video duration in seconds using ffprobe.

    Args:
        mkv_path: Path to MKV file

    Returns:
        Duration in seconds
    """
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        mkv_path
    ], capture_output=True, text=True)

    return int(float(result.stdout.strip()))


def get_video_resolution(video_path: str) -> tuple[int, int]:
    """Get video resolution (width, height) using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (width, height), or (0, 0) if detection fails
    """
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            video_path
        ], capture_output=True, text=True, check=True)

        data = json.loads(result.stdout)
        streams = data.get('streams', [])

        if streams:
            stream = streams[0]
            width = stream.get('width', 0)
            height = stream.get('height', 0)
            return (width, height)
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        pass

    return (0, 0)


def timestamp_to_seconds(timestamp: str) -> int:
    """Convert MM:SS or HH:MM:SS to seconds."""
    parts = timestamp.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def seconds_to_timestamp(seconds: int) -> str:
    """Convert seconds to HH:MM:SS."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class VideoMetadataScanner(QThread):
    """Background thread for scanning video file metadata."""

    # Emitted for each file: (filename, width, height, duration_seconds)
    file_scanned = pyqtSignal(str, int, int, int)
    # Emitted when all files are scanned: (longest_filename, longest_duration)
    scan_complete = pyqtSignal(str, int)

    def __init__(self, video_files: list[Path], parent=None):
        super().__init__(parent)
        self._video_files = video_files
        self._stop_requested = False

    def run(self):
        """Scan all video files for metadata."""
        longest_file = ""
        longest_duration = 0

        for video in self._video_files:
            if self._stop_requested:
                break

            try:
                width, height = get_video_resolution(str(video))
                duration = get_video_duration(str(video))

                self.file_scanned.emit(video.name, width, height, duration)

                if duration > longest_duration:
                    longest_duration = duration
                    longest_file = video.name
            except Exception:
                # Emit with zeros on failure
                self.file_scanned.emit(video.name, 0, 0, 0)

        if not self._stop_requested:
            self.scan_complete.emit(longest_file, longest_duration)

    def stop(self):
        """Request the scanner to stop."""
        self._stop_requested = True
