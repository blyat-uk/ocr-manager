"""The golden cases must come out byte-identical when they run at the same
time, in threads of one process -- which is how OCRManager runs its workers
(QThreads, default 4) in the shipped app.

tools/fidelity_check.py runs one case at a time, so on its own it cannot see
anything that only goes wrong when workers overlap, such as two threads
running inference on one shared engine (see videocr/engine_registry.py).
"""
from __future__ import annotations

import threading

import pytest

from tools.fidelity_check import GOLDEN_DIR, digest, load_cases, media_root, run_case

# Pinned here as well as in tests/goldens/, so a golden rewritten by
# `fidelity_check.py --capture` cannot quietly move what this test accepts.
EXPECTED_DIGEST_PREFIXES = {
    "slay_1080p_dialogue": "c76ee2af4ecd",
    "slay_1080p_multirange": "273ba6ac6c03",
    "slay_1080p_labels": "af324c0c56bd",
    "slay_1080p_multirange_labels": "3ff9b1b329b3",
}


@pytest.mark.needs_media
@pytest.mark.slow
def test_all_fidelity_cases_run_concurrently_match_their_goldens(reference_media):
    cases = load_cases()
    assert {c.name for c in cases} == set(EXPECTED_DIGEST_PREFIXES)
    root = media_root()
    for case in cases:
        if not (root / case.project / case.file).exists():
            pytest.skip(f"missing input for {case.name}")

    produced: dict[str, str] = {}
    errors: dict[str, str] = {}
    start = threading.Barrier(len(cases))

    def run(case):
        try:
            start.wait(timeout=60)
            produced[case.name] = run_case(case, root)
        except Exception as exc:
            errors[case.name] = repr(exc)

    threads = [threading.Thread(target=run, args=(case,), name=case.name) for case in cases]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    mismatches = {}
    for case in cases:
        golden = digest((GOLDEN_DIR / f"{case.name}.ass").read_text(encoding="utf-8"))
        assert golden.startswith(EXPECTED_DIGEST_PREFIXES[case.name]), case.name
        got = digest(produced[case.name])
        if got != golden:
            mismatches[case.name] = f"golden {golden[:12]} != produced {got[:12]}"
    assert not mismatches, mismatches
