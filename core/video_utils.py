"""Utility functions for video frame extraction."""
import subprocess


def extract_frame(mkv_path: str, timestamp: str, output_path: str) -> bool:
    """Extract single frame from video at timestamp.

    Args:
        mkv_path: Path to MKV file
        timestamp: Timestamp in HH:MM:SS format
        output_path: Where to save extracted frame (PNG)

    Returns:
        True if successful, False otherwise
    """
    try:
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
