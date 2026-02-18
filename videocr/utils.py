import contextlib
import datetime
import io
import json
import logging
import os
import subprocess
import sys
import warnings
import yaml
from importlib.metadata import version
from packaging import version as version_parser

PADDLEOCR_FORMAT_CHANGE_VERSION = "3.0.3"


@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout, stderr, warnings, and logging during model loading."""
    # Redirect at OS file descriptor level (captures subprocess/C extension output)
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)

    # Also capture Python-level stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    # Suppress paddle/paddleocr loggers
    loggers_to_suppress = [
        "ppocr",
        "paddle",
        "paddleocr",
        "paddlex",
    ]
    old_levels = {}
    for logger_name in loggers_to_suppress:
        logger = logging.getLogger(logger_name)
        old_levels[logger_name] = logger.level
        logger.setLevel(logging.CRITICAL + 1)

    # Also suppress root logger temporarily
    root_logger = logging.getLogger()
    old_root_level = root_logger.level
    root_logger.setLevel(logging.CRITICAL + 1)

    # Suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            yield
        finally:
            # Restore OS-level file descriptors
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
            # Restore Python-level streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            # Restore logger levels
            for logger_name, level in old_levels.items():
                logging.getLogger(logger_name).setLevel(level)
            root_logger.setLevel(old_root_level)


def get_video_start_time(video_path: str) -> float:
    """Get the video stream start_time using ffprobe.

    Some HEVC streams have a non-zero start_time (e.g., 0.042s = 1 frame at 25fps)
    which causes subtitle timestamps to be offset. OpenCV doesn't expose this
    correctly, so we use ffprobe to get the actual value.

    Returns:
        Start time in seconds, or 0.0 if not available.
    """
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
            '-show_entries', 'stream=start_time',
            '-of', 'json', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get('streams', [])
            if streams:
                start_time_str = streams[0].get('start_time', '0')
                return float(start_time_str)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, FileNotFoundError):
        pass
    return 0.0

# convert time string to frame index
def get_frame_index(time_str: str, fps: float):
    t = time_str.split(':')
    t = list(map(float, t))
    if len(t) == 3:
        td = datetime.timedelta(hours=t[0], minutes=t[1], seconds=t[2])
    elif len(t) == 2:
        td = datetime.timedelta(minutes=t[0], seconds=t[1])
    else:
        raise ValueError(
            'Time data "{}" does not match format "%H:%M:%S"'.format(time_str))
    index = int(td.total_seconds() * fps)
    return index


# convert frame index into ASS timestamp (H:MM:SS.cc format with centiseconds)
def get_ass_timestamp(frame_index: int, fps: float) -> str:
    td = datetime.timedelta(seconds=frame_index / fps)
    cs = td.microseconds // 10000  # centiseconds (1/100th second)
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    return '{:d}:{:02d}:{:02d}.{:02d}'.format(h, m, s, cs)


def get_ass_timestamp_from_seconds(pts_seconds: float) -> str:
    """Convert PTS (seconds) directly to ASS timestamp format.

    This is the canonical method for converting timing - uses raw PTS values
    instead of going through frame index calculations which can drift.

    Args:
        pts_seconds: Presentation timestamp in seconds

    Returns:
        ASS timestamp string in H:MM:SS.cc format
    """
    if pts_seconds is None or pts_seconds < 0:
        pts_seconds = 0.0

    td = datetime.timedelta(seconds=pts_seconds)
    cs = td.microseconds // 10000  # centiseconds (1/100th second)
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    return '{:d}:{:02d}:{:02d}.{:02d}'.format(h, m, s, cs)


def format_ass_dialogue(start_time: str, end_time: str, text: str) -> str:
    """Format a single ASS Dialogue line."""
    # Replace newlines with ASS line break syntax \N
    text_escaped = text.replace('\n', '\\N')
    return f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text_escaped}\n"


# check if format conversion is required
def needs_conversion():
    current_version = version("paddleocr")
    return version_parser.parse(current_version) >= version_parser.parse(PADDLEOCR_FORMAT_CHANGE_VERSION)


# convert returned format from paddleocr to old format
def convert_pred_data_to_old_format(new_pred_data):
    old_format = []

    for item in new_pred_data:
        result = []

        rec_texts = item.get('rec_texts', [])
        rec_scores = item.get('rec_scores', [])
        rec_polys = item.get('rec_polys', [])

        for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
            box = [[float(x), float(y)] for x, y in poly.tolist()]
            result.append([box, (text, score)])

        old_format.append(result)

    return old_format


# reads the model name from inference.yml inside the given model directory.
def get_model_name_from_dir(model_dir):
    if model_dir == None:
        return None

    yml_path = os.path.join(model_dir, "inference.yml")
    if not os.path.exists(yml_path):
        return None

    try:
        with open(yml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config.get("Global", {}).get("model_name")
    except Exception:
        return None


def create_ocr_engine(lang, det_model_dir, rec_model_dir, use_gpu):
    """Create a full PaddleOCR engine (detection + recognition)."""
    from paddleocr import PaddleOCR
    with suppress_output():
        return PaddleOCR(
            lang=lang,
            text_recognition_model_dir=rec_model_dir,
            text_detection_model_dir=det_model_dir,
            text_detection_model_name=get_model_name_from_dir(det_model_dir),
            text_recognition_model_name=get_model_name_from_dir(rec_model_dir),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="gpu" if use_gpu else "cpu",
        )


def create_detection_engine(det_model_dir, use_gpu):
    """Create a detection-only engine for label scanning (PaddleOCR 3.x TextDetection)."""
    from paddleocr import TextDetection
    model_name = get_model_name_from_dir(det_model_dir) or "PP-OCRv5_server_det"
    with suppress_output():
        return TextDetection(
            model_name=model_name,
            model_dir=det_model_dir,
            device="gpu" if use_gpu else "cpu",
        )


def format_ass_header(play_res_x: int, play_res_y: int, include_label_style: bool = False) -> str:
    """Generate ASS file header with Script Info and Styles sections."""
    label_style = ""
    if include_label_style:
        label_style = "\nStyle: Label,Arial,36,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1"

    return f"""[Script Info]
Title: Video Subtitles
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,30,1{label_style}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def format_ass_label_dialogue(start_time: str, end_time: str, text: str, pos_x: int, pos_y: int) -> str:
    """Format a single ASS Dialogue line for a positioned label."""
    text_escaped = text.replace('\n', '\\N')
    return f"Dialogue: 0,{start_time},{end_time},Label,,0,0,0,,{{\\pos({pos_x},{pos_y})}}{text_escaped}\n"


def parse_ass_timestamp(timestamp: str) -> float:
    """Parse ASS timestamp (H:MM:SS.cc) to seconds."""
    # Format: H:MM:SS.cc (e.g., "0:03:36.35")
    parts = timestamp.split(':')
    h = int(parts[0])
    m = int(parts[1])
    s_cs = parts[2].split('.')
    s = int(s_cs[0])
    cs = int(s_cs[1])
    return h * 3600 + m * 60 + s + cs / 100.0


def parse_ass_dialogue_line(line: str) -> dict | None:
    """Parse an ASS Dialogue line into components.

    Returns dict with: start, end, style, text, full_line
    Or None if not a valid Dialogue line.
    """
    if not line.startswith("Dialogue:"):
        return None

    # Format: Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
    # Split on comma, but only first 9 commas (text may contain commas)
    content = line[len("Dialogue:"):].strip()
    parts = content.split(',', 9)
    if len(parts) < 10:
        return None

    return {
        'layer': parts[0].strip(),
        'start': parts[1].strip(),
        'end': parts[2].strip(),
        'style': parts[3].strip(),
        'name': parts[4].strip(),
        'margin_l': parts[5].strip(),
        'margin_r': parts[6].strip(),
        'margin_v': parts[7].strip(),
        'effect': parts[8].strip(),
        'text': parts[9].rstrip('\n'),
        'start_seconds': parse_ass_timestamp(parts[1].strip()),
        'end_seconds': parse_ass_timestamp(parts[2].strip()),
    }


def sanitize_ass_dialogues(lines: list[str]) -> list[str]:
    """Sanitize ASS dialogue lines: sort by time and merge consecutive identical lines.

    Args:
        lines: List of ASS Dialogue lines (strings)

    Returns:
        Sanitized list of Dialogue lines
    """
    # Parse all lines
    parsed = []
    for line in lines:
        p = parse_ass_dialogue_line(line)
        if p:
            parsed.append(p)

    if not parsed:
        return lines

    # Sort by start time
    parsed.sort(key=lambda x: x['start_seconds'])

    # Merge consecutive lines with same text (and style/position)
    merged = []
    for p in parsed:
        if merged:
            prev = merged[-1]
            # Check if this line can be merged with previous:
            # - Same text (including position tags)
            # - Same style
            # - Previous end == current start (adjacent)
            if (prev['text'] == p['text'] and
                prev['style'] == p['style'] and
                abs(prev['end_seconds'] - p['start_seconds']) < 0.05):  # 50ms tolerance
                # Extend previous line's end time
                prev['end'] = p['end']
                prev['end_seconds'] = p['end_seconds']
                continue
        merged.append(p)

    # Reconstruct dialogue lines
    result = []
    for p in merged:
        line = f"Dialogue: {p['layer']},{p['start']},{p['end']},{p['style']},{p['name']},"
        line += f"{p['margin_l']},{p['margin_r']},{p['margin_v']},{p['effect']},{p['text']}\n"
        result.append(line)

    return result


def merge_ass_output(dialogue_ass: str, labels, play_res_x: int, play_res_y: int) -> str:
    """Merge dialogue ASS output with label events into a single ASS file.

    Re-generates the header with the Label style included, preserves existing
    dialogue lines, and appends label lines. All lines are sorted by start time
    and consecutive identical lines are merged.
    """
    # Extract dialogue lines from existing ASS (everything after the Events format line)
    dialogue_lines = []
    in_events = False
    for line in dialogue_ass.splitlines(keepends=True):
        if line.startswith("Format: Layer,"):
            in_events = True
            continue
        if in_events and line.startswith("Dialogue:"):
            dialogue_lines.append(line if line.endswith('\n') else line + '\n')

    # Generate label lines
    label_lines = []
    for label in labels:
        start = get_ass_timestamp_from_seconds(label.start_pts)
        end = get_ass_timestamp_from_seconds(label.end_pts)
        label_lines.append(format_ass_label_dialogue(start, end, label.text, label.pos_x, label.pos_y))

    # Combine and sanitize all lines (sort by time, merge consecutive identical)
    all_lines = dialogue_lines + label_lines
    sanitized_lines = sanitize_ass_dialogues(all_lines)

    # Re-generate header with Label style
    header = format_ass_header(play_res_x, play_res_y, include_label_style=True)

    return header + "".join(sanitized_lines)


def format_labels_only_ass(labels, play_res_x: int, play_res_y: int) -> str:
    """Generate a complete ASS file with only label events (no dialogue).

    Lines are sorted by start time and consecutive identical lines are merged.
    """
    header = format_ass_header(play_res_x, play_res_y, include_label_style=True)

    label_lines = []
    for label in labels:
        start = get_ass_timestamp_from_seconds(label.start_pts)
        end = get_ass_timestamp_from_seconds(label.end_pts)
        label_lines.append(format_ass_label_dialogue(start, end, label.text, label.pos_x, label.pos_y))

    # Sanitize (sort and merge consecutive identical lines)
    sanitized_lines = sanitize_ass_dialogues(label_lines)

    return header + "".join(sanitized_lines)
