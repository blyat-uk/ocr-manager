from __future__ import annotations

from core.audio_finder.config import HashConfig
from core.audio_finder.util.hashing import pack_hash


def generate_fingerprints(
    peaks: list[tuple[int, int]],
    cfg: HashConfig,
) -> list[tuple[int, int]]:
    result = []
    n = len(peaks)

    for i in range(n):
        f1, t1 = peaks[i]
        paired = 0

        for j in range(i + 1, n):
            if paired >= cfg.fanout:
                break

            f2, t2 = peaks[j]
            dt = t2 - t1

            if dt < cfg.dt_min:
                continue
            if dt > cfg.dt_max:
                break

            f1_c = min(f1, (1 << cfg.f1_bits) - 1)
            f2_c = min(f2, (1 << cfg.f2_bits) - 1)
            dt_c = min(dt, (1 << cfg.dt_bits) - 1)

            h = pack_hash(f1_c, f2_c, dt_c)
            result.append((h, t1))
            paired += 1

    return result
