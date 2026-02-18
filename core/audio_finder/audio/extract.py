from __future__ import annotations

import hashlib
import json
import logging
import subprocess

import numpy as np

logger = logging.getLogger(__name__)

_LOSSLESS_CODECS = frozenset({
    "flac", "alac",
    "pcm_s16le", "pcm_s16be", "pcm_s24le", "pcm_s24be",
    "pcm_s32le", "pcm_s32be", "pcm_f32le", "pcm_f32be",
    "wavpack", "truehd", "mlp", "dts",
})


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _score_audio_stream(stream: dict) -> float:
    score = 0.0
    codec = stream.get("codec_name", "").lower()
    bitrate = _safe_int(stream.get("bit_rate"))
    sample_rate = _safe_int(stream.get("sample_rate"))
    channels = _safe_int(stream.get("channels"))

    if codec in _LOSSLESS_CODECS:
        score += 50
    elif codec in ("aac", "opus"):
        score += 30
    elif codec in ("mp3", "ac3", "eac3", "vorbis"):
        score += 20
    else:
        score += 10

    if bitrate is not None and bitrate > 0:
        score += min(bitrate / 320_000 * 30, 30)
    elif codec in _LOSSLESS_CODECS:
        score += 30

    if sample_rate is not None and sample_rate > 0:
        score += min(sample_rate / 48_000 * 10, 10)

    if channels is not None and channels > 0:
        score += min(channels / 6 * 10, 10)

    return score


def _select_best_audio_stream(path: str) -> int | None:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index,codec_name,sample_rate,channels,bit_rate",
        "-of", "json",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, check=True, text=True)
    info = json.loads(result.stdout)
    streams = info.get("streams", [])

    if len(streams) == 0:
        raise RuntimeError(f"No audio streams found in {path}")
    if len(streams) == 1:
        return None

    scored = [(i, _score_audio_stream(s), s) for i, s in enumerate(streams)]
    best_idx, best_score, best_stream = max(scored, key=lambda t: t[1])

    codec = best_stream.get("codec_name", "unknown")
    logger.debug(
        "%d audio streams detected; selected stream a:%d (codec=%s, score=%.1f)",
        len(streams), best_idx, codec, best_score,
    )

    return best_idx


def extract_audio(path: str, sample_rate: int = 11025) -> np.ndarray:
    audio_idx = _select_best_audio_stream(path)
    cmd = ["ffmpeg", "-i", path]
    if audio_idx is not None:
        cmd += ["-map", f"0:a:{audio_idx}"]
    cmd += [
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "s16le",
        "-af", "highpass=f=80,lowpass=f=5000",
        "-loglevel", "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    samples = np.frombuffer(result.stdout, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def get_duration(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, check=True, text=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()
