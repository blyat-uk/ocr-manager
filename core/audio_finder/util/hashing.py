from __future__ import annotations


def pack_hash(f1: int, f2: int, dt: int) -> int:
    return (f1 << 20) | (f2 << 10) | dt


def unpack_hash(h: int) -> tuple[int, int, int]:
    dt = h & 0x3FF
    f2 = (h >> 10) & 0x3FF
    f1 = (h >> 20) & 0x3FF
    return f1, f2, dt
