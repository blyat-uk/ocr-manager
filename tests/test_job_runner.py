"""Behaviour of core.jobs.runner: lanes, ordering, cancellation, pause, shutdown.

Every wait is bounded. WAIT bounds things that should happen promptly, so a
regression fails instead of hanging. QUIET is the window in which something
must NOT happen (a held or excluded job starting). A fake job that waits for
cancellation gives up after JOB_PATIENCE and reports so, which keeps the
cancel tests fast even when cancellation is broken.
"""
import logging
import threading
import time

import pytest

from core.jobs import JobContext, JobEvent, JobRunner, Lane

pytestmark = pytest.mark.timeout(60)

WAIT = 5.0
QUIET = 0.25
JOB_PATIENCE = 3.0
TERMINAL = ("finished", "failed", "cancelled")


class Recorder:
    """on_event listener: keeps every event with the thread it arrived on."""

    def __init__(self):
        self._cond = threading.Condition()
        self._events: list[tuple[JobEvent, int]] = []

    def __call__(self, event: JobEvent) -> None:
        with self._cond:
            self._events.append((event, threading.get_ident()))
            self._cond.notify_all()

    def with_threads(self, key: str) -> list[tuple[JobEvent, int]]:
        with self._cond:
            return [(e, t) for e, t in self._events if e.key == key]

    def of(self, key: str) -> list[JobEvent]:
        return [e for e, _ in self.with_threads(key)]

    def types(self, key: str) -> list[str]:
        return [e.type for e in self.of(key)]

    def wait_for(self, predicate, timeout: float = WAIT) -> bool:
        with self._cond:
            return self._cond.wait_for(predicate, timeout)

    def wait_terminal(self, key: str, count: int = 1) -> JobEvent | None:
        """The count-th terminal event for key, or None if it never came."""
        def terminals():
            return [e for e in self.of(key) if e.type in TERMINAL]
        if not self.wait_for(lambda: len(terminals()) >= count):
            return None
        return terminals()[count - 1]


class FakeJob:
    def __init__(self, key, lane=Lane.GPU, *, priority=0, kind="test",
                 file=None, body=None):
        self.key = key
        self.kind = kind
        self.lane = lane
        self.file = file
        self.priority = priority
        self.body = body
        self.ran = threading.Event()
        self.ctx: JobContext | None = None

    def run(self, ctx: JobContext):
        self.ctx = ctx
        self.ran.set()
        return self.body(ctx) if self.body is not None else None

    def __repr__(self):
        return f"FakeJob({self.key!r})"


class Harness:
    def __init__(self, cpu_workers: int = 2):
        self.rec = Recorder()
        self.runner = JobRunner(self.rec, cpu_workers=cpu_workers)
        self._gates: list[threading.Event] = []

    def gate(self) -> threading.Event:
        gate = threading.Event()
        self._gates.append(gate)
        return gate

    def blocker(self, key: str, lane: Lane = Lane.GPU) -> FakeJob:
        """Submit a job that holds its lane until the returned job's gate opens."""
        gate = self.gate()
        job = FakeJob(key, lane, body=lambda ctx: gate.wait(WAIT))
        job.gate = gate
        self.runner.submit(job)
        assert job.ran.wait(WAIT), f"{key} never started"
        return job

    def all_finished(self, keys) -> bool:
        return self.rec.wait_for(
            lambda: all("finished" in self.rec.types(k) for k in keys))

    def close(self):
        for gate in self._gates:
            gate.set()
        self.runner.shutdown(timeout=2.0)


@pytest.fixture
def make_harness():
    made = []

    def make(cpu_workers: int = 2) -> Harness:
        harness = Harness(cpu_workers)
        made.append(harness)
        return harness

    yield make
    for harness in made:
        harness.close()


@pytest.fixture
def h(make_harness) -> Harness:
    return make_harness()


# --- lanes -----------------------------------------------------------------

@pytest.mark.parametrize("lane", [Lane.GPU, Lane.RUN])
def test_single_worker_lane_never_starts_a_second_job_while_one_runs(h, lane):
    first = h.blocker("first", lane)
    second = FakeJob("second", lane)
    h.runner.submit(second)

    assert not second.ran.wait(QUIET)
    assert h.runner.running() == [first]
    assert h.runner.queued() == [second]

    first.gate.set()
    assert h.all_finished(["first", "second"])


def test_gpu_lane_max_concurrency_is_one_over_20_jobs(h):
    lock = threading.Lock()
    active = 0
    peak = 0

    def body(ctx):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.005)
        with lock:
            active -= 1

    keys = [f"gpu-{i}" for i in range(20)]
    for key in keys:
        h.runner.submit(FakeJob(key, Lane.GPU, body=body))

    assert h.all_finished(keys)
    assert peak == 1


def test_cpu_lane_runs_up_to_cpu_workers_in_parallel_and_no_more(make_harness):
    h = make_harness(cpu_workers=3)
    gate = h.gate()
    jobs = [FakeJob(f"cpu-{i}", Lane.CPU, body=lambda ctx: gate.wait(WAIT))
            for i in range(4)]
    for job in jobs:
        h.runner.submit(job)

    for job in jobs[:3]:
        assert job.ran.wait(WAIT), f"{job.key} did not start alongside the others"
    assert not jobs[3].ran.wait(QUIET)
    assert len(h.runner.running()) == 3

    # A saturated CPU lane does not hold up the GPU lane.
    gpu = FakeJob("gpu", Lane.GPU, body=lambda ctx: "gpu ran")
    h.runner.submit(gpu)
    assert h.rec.wait_terminal("gpu").result == "gpu ran"

    gate.set()
    assert h.all_finished([job.key for job in jobs])


# --- ordering and replacement -----------------------------------------------

def test_higher_priority_runs_first_and_fifo_within_a_priority(h):
    blocker = h.blocker("blocker")
    order = []
    specs = [("a", 0), ("b", 5), ("c", 0), ("d", 5), ("e", 1), ("f", 0)]
    for key, priority in specs:
        h.runner.submit(FakeJob(key, Lane.GPU, priority=priority,
                                body=lambda ctx, key=key: order.append(key)))

    expected = ["b", "d", "e", "a", "c", "f"]
    assert [job.key for job in h.runner.queued()] == expected

    blocker.gate.set()
    assert h.all_finished(expected)
    assert order == expected


def test_submitting_a_queued_key_replaces_it_in_place(h):
    blocker = h.blocker("blocker")
    order = []
    old_x = FakeJob("x", body=lambda ctx: order.append("old x"))
    y = FakeJob("y", body=lambda ctx: order.append("y"))
    z = FakeJob("z", body=lambda ctx: order.append("z"))
    for job in (old_x, y, z):
        h.runner.submit(job)

    new_x = FakeJob("x", body=lambda ctx: order.append("new x"))
    h.runner.submit(new_x)

    # Old one cancelled before the new one is queued, both before submit returns.
    assert h.rec.types("x") == ["queued", "cancelled", "queued"]
    assert h.runner.queued() == [new_x, y, z]

    blocker.gate.set()
    assert h.all_finished(["x", "y", "z"])
    assert order == ["new x", "y", "z"]
    assert not old_x.ran.is_set()


def test_submitting_a_running_key_queues_a_second_job_without_cancelling(h):
    gate = h.gate()
    first = FakeJob("k", body=lambda ctx: (gate.wait(WAIT), "first")[1])
    h.runner.submit(first)
    assert first.ran.wait(WAIT)

    second = FakeJob("k", body=lambda ctx: "second")
    h.runner.submit(second)

    assert h.runner.running() == [first]
    assert h.runner.queued() == [second]
    assert h.rec.types("k") == ["queued", "started", "queued"]

    gate.set()
    assert h.rec.wait_terminal("k", count=2) is not None
    assert h.rec.types("k") == ["queued", "started", "queued",
                                "finished", "started", "finished"]
    assert [e.result for e in h.rec.of("k") if e.type == "finished"] == ["first", "second"]


def test_job_ids_tell_instances_with_the_same_key_apart(h):
    blocker = h.blocker("blocker")
    h.runner.submit(FakeJob("k"))
    h.runner.submit(FakeJob("k", body=lambda ctx: ctx.emit("run_file_started", file="a.mp4")))
    blocker.gate.set()
    assert h.rec.wait_terminal("k", count=2) is not None

    events = h.rec.of("k")
    assert [e.type for e in events] == ["queued", "cancelled", "queued", "started",
                                        "run_file_started", "finished"]
    replaced, replacement = events[0].job_id, events[2].job_id
    assert [e.job_id for e in events] == [replaced] * 2 + [replacement] * 4
    assert 0 < h.rec.of("blocker")[0].job_id < replaced < replacement


@pytest.mark.parametrize("old_lane", [Lane.CPU, Lane.GPU], ids=["same-lane", "other-lane"])
def test_a_job_waits_until_the_running_job_with_its_key_has_ended(old_lane):
    """Same-key jobs never overlap, even with a free worker; other keys run past."""
    rec = Recorder()
    slow_terminal_for = {}

    def listener(event):
        # Deliver the old job's terminal event slowly: the new job must not
        # start until that delivery is over.
        if event.type in TERMINAL and event.job_id == slow_terminal_for.get("id"):
            time.sleep(QUIET)
        rec(event)

    runner = JobRunner(listener, cpu_workers=2)
    gate = threading.Event()
    try:
        old = FakeJob("metadata:a.mp4", old_lane, body=lambda ctx: (gate.wait(WAIT), "old")[1])
        runner.submit(old)
        assert old.ran.wait(WAIT)
        slow_terminal_for["id"] = rec.of("metadata:a.mp4")[0].job_id

        new = FakeJob("metadata:a.mp4", Lane.CPU, body=lambda ctx: "new")
        other = FakeJob("metadata:b.mp4", Lane.CPU, body=lambda ctx: "other")
        runner.submit(new)
        runner.submit(other)

        assert rec.wait_terminal("metadata:b.mp4").result == "other"  # not blocked behind it
        assert not new.ran.wait(QUIET)
        assert runner.queued() == [new]

        gate.set()
        assert rec.wait_terminal("metadata:a.mp4", count=2) is not None
    finally:
        gate.set()
        runner.shutdown(timeout=2.0)

    events = rec.of("metadata:a.mp4")
    assert [(e.type, e.result) for e in events] == [
        ("queued", None), ("started", None), ("queued", None),
        ("finished", "old"), ("started", None), ("finished", "new"),
    ]
    old_id, new_id = events[0].job_id, events[2].job_id
    assert old_id != new_id
    assert [e.job_id for e in events] == [old_id, old_id, new_id, old_id, new_id, new_id]


def test_a_listener_resubmitting_a_key_during_its_replacement_keeps_every_job_consistent():
    rec = Recorder()
    runner = None

    def listener(event):
        rec(event)
        if event.type == "cancelled" and event.file == "old":
            runner.submit(FakeJob("k", file="from-listener"))

    runner = JobRunner(listener)
    try:
        runner.pause(Lane.GPU)
        runner.submit(FakeJob("k", file="old"))
        runner.submit(FakeJob("k", file="new"))
        queued = [job.file for job in runner.queued()]
    finally:
        runner.shutdown(timeout=2.0)

    def types(file):
        return [e.type for e in rec.of("k") if e.file == file]

    assert types("old") == ["queued", "cancelled"]
    assert types("from-listener") == ["queued", "cancelled"]
    assert types("new") == ["queued", "cancelled"]  # its own "queued"; cancelled by shutdown
    assert queued == ["new"]


def test_lifecycle_events_and_cancel_use_the_identity_captured_at_submit(h):
    renamed = threading.Event()

    def body(ctx):
        job.key, job.kind, job.file = "renamed", "other", "other.mp4"
        renamed.set()
        return _wait_cancel_event(ctx)

    job = FakeJob("crop:a.mp4", kind="crop", file="a.mp4", body=body)
    h.runner.submit(job)
    assert renamed.wait(WAIT)

    h.runner.cancel("crop:a.mp4")

    terminal = h.rec.wait_terminal("crop:a.mp4")
    assert terminal is not None
    assert (terminal.type, terminal.result) == ("cancelled", "stopped")
    assert {(e.key, e.kind, e.file) for e in h.rec.of("crop:a.mp4")} == {("crop:a.mp4", "crop", "a.mp4")}
    assert h.rec.of("renamed") == []


# --- cancellation ------------------------------------------------------------

def test_cancelling_a_queued_job_removes_it_without_running(h):
    blocker = h.blocker("blocker")
    victim = FakeJob("victim")
    after = FakeJob("after")
    h.runner.submit(victim)
    h.runner.submit(after)

    h.runner.cancel("victim")

    assert h.rec.types("victim") == ["queued", "cancelled"]
    assert h.runner.queued() == [after]
    blocker.gate.set()
    assert h.all_finished(["after"])
    assert not victim.ran.is_set()
    assert h.rec.types("victim") == ["queued", "cancelled"]


def _poll_cancelled(ctx):
    deadline = time.monotonic() + JOB_PATIENCE
    while not ctx.cancelled():
        if time.monotonic() > deadline:
            return "gave up"
        time.sleep(0.005)
    return "stopped"


def _poll_cancel_check(ctx):
    check = ctx.cancel_check()
    deadline = time.monotonic() + JOB_PATIENCE
    while not check():
        if time.monotonic() > deadline:
            return "gave up"
        time.sleep(0.005)
    return "stopped"


def _wait_cancel_event(ctx):
    return "stopped" if ctx.cancel_event.wait(JOB_PATIENCE) else "gave up"


@pytest.mark.parametrize("watch", [_poll_cancelled, _poll_cancel_check, _wait_cancel_event],
                         ids=["cancelled", "cancel_check", "cancel_event"])
def test_cancelling_a_running_job_sets_its_event_and_reports_cancelled(h, watch):
    job = FakeJob("busy", body=watch)
    h.runner.submit(job)
    assert job.ran.wait(WAIT)

    h.runner.cancel("busy")

    terminal = h.rec.wait_terminal("busy")
    assert terminal is not None
    assert terminal.type == "cancelled"
    assert terminal.result == "stopped"  # what run() returned after stopping
    assert job.ctx.cancel_event.is_set()
    assert h.rec.types("busy") == ["queued", "started", "cancelled"]


def test_a_job_that_ignores_cancel_and_completes_reports_finished(h):
    gate = h.gate()
    job = FakeJob("stubborn", body=lambda ctx: (gate.wait(WAIT), 42)[1])
    h.runner.submit(job)
    assert job.ran.wait(WAIT)

    h.runner.cancel("stubborn")
    gate.set()

    terminal = h.rec.wait_terminal("stubborn")
    assert terminal is not None
    assert (terminal.type, terminal.result) == ("finished", 42)
    assert job.ctx.cancel_event.is_set()  # the request did reach the job


def test_cancel_reaches_both_the_running_and_the_queued_job_with_that_key(h):
    running = FakeJob("k", body=_wait_cancel_event)
    h.runner.submit(running)
    assert running.ran.wait(WAIT)
    queued = FakeJob("k")
    h.runner.submit(queued)

    h.runner.cancel("k")

    assert h.rec.wait_terminal("k", count=2) is not None
    assert [t for t in h.rec.types("k") if t in TERMINAL] == ["cancelled", "cancelled"]
    assert not queued.ran.is_set()
    assert h.runner.queued() == []


def test_cancel_where_cancels_matching_queued_and_running_jobs_only(h):
    gate = h.gate()
    running_crop = FakeJob("crop:a", Lane.GPU, kind="crop", body=_wait_cancel_event)
    running_meta = FakeJob("metadata:a", Lane.CPU, kind="metadata",
                           body=lambda ctx: (gate.wait(WAIT), ctx.cancelled())[1])
    for job in (running_crop, running_meta):
        h.runner.submit(job)
        assert job.ran.wait(WAIT)
    queued_crop = FakeJob("crop:b", Lane.GPU, kind="crop")
    queued_bright = FakeJob("brightness:b", Lane.GPU, kind="brightness")
    h.runner.submit(queued_crop)
    h.runner.submit(queued_bright)

    h.runner.cancel_where(lambda job: job.kind == "crop")

    assert h.rec.types("crop:b") == ["queued", "cancelled"]
    assert h.rec.wait_terminal("crop:a").type == "cancelled"
    assert h.rec.wait_terminal("brightness:b").type == "finished"
    gate.set()
    meta = h.rec.wait_terminal("metadata:a")
    assert (meta.type, meta.result) == ("finished", False)
    assert not queued_crop.ran.is_set()


# --- failures and listeners --------------------------------------------------

def test_exception_reports_failed_with_traceback_and_the_lane_keeps_working(h):
    def explode(ctx):
        raise ValueError("bad crop box\nsecond line")

    h.runner.submit(FakeJob("bad", body=explode))
    failed = h.rec.wait_terminal("bad")
    assert failed is not None
    assert failed.type == "failed"
    assert failed.message == "bad crop box"
    assert "Traceback" in failed.error
    assert "ValueError: bad crop box" in failed.error
    assert "explode" in failed.error

    # The GPU lane has one worker thread: this only runs if it survived.
    h.runner.submit(FakeJob("good", body=lambda ctx: "ok"))
    good = h.rec.wait_terminal("good")
    assert good is not None
    assert (good.type, good.result) == ("finished", "ok")


class _Unprintable(Exception):
    def __str__(self):
        raise RuntimeError("str() is broken too")


def test_an_exception_that_cannot_be_printed_still_reports_failed(h):
    def explode(ctx):
        raise _Unprintable()

    h.runner.submit(FakeJob("unprintable", body=explode))
    failed = h.rec.wait_terminal("unprintable")
    assert failed is not None
    assert failed.type == "failed"
    assert failed.message == "_Unprintable"
    assert "_Unprintable" in failed.error
    assert h.runner.running() == []

    h.runner.submit(FakeJob("after", body=lambda ctx: "ok"))
    after = h.rec.wait_terminal("after")
    assert after is not None and after.result == "ok"


class _ListenerAbort(BaseException):
    pass


def test_a_listener_raising_baseexception_on_started_still_ends_the_job_failed():
    rec = Recorder()

    def listener(event):
        rec(event)
        if event.type == "started" and event.key == "fragile":
            raise _ListenerAbort("adapter blew up")

    runner = JobRunner(listener, cpu_workers=1)
    fragile = FakeJob("fragile", body=lambda ctx: "never")
    try:
        runner.submit(fragile)
        failed = rec.wait_terminal("fragile")
        runner.submit(FakeJob("next", body=lambda ctx: "ok"))
        after = rec.wait_terminal("next")
    finally:
        runner.shutdown(timeout=2.0)

    assert rec.types("fragile") == ["queued", "started", "failed"]
    assert failed.message == "adapter blew up"
    assert "_ListenerAbort" in failed.error
    assert not fragile.ran.is_set()
    assert runner.running() == []
    assert (after.type, after.result) == ("finished", "ok")


def test_listener_exceptions_are_logged_and_never_stop_a_lane(caplog):
    rec = Recorder()

    def listener(event):
        rec(event)
        raise RuntimeError("listener bug")

    runner = JobRunner(listener, cpu_workers=1)
    try:
        with caplog.at_level(logging.ERROR, logger="core.jobs.runner"):
            runner.submit(FakeJob("first", body=lambda ctx: ctx.progress(0.5)))
            assert rec.wait_terminal("first") is not None
            runner.submit(FakeJob("second", body=lambda ctx: "still alive"))
            second = rec.wait_terminal("second")
    finally:
        runner.shutdown(timeout=2.0)

    assert rec.types("first") == ["queued", "started", "progress", "finished"]
    assert (second.type, second.result) == ("finished", "still alive")
    assert any(r.exc_info and "listener bug" in str(r.exc_info[1]) for r in caplog.records)


# --- pause -------------------------------------------------------------------

def test_a_paused_lane_starts_nothing_until_resumed(h):
    running = h.blocker("running")
    h.runner.pause(Lane.GPU)
    assert h.runner.is_paused(Lane.GPU)
    assert not h.runner.is_paused(Lane.CPU)

    held = FakeJob("held")
    h.runner.submit(held)
    running.gate.set()
    assert h.rec.wait_terminal("running").type == "finished"  # in-flight job unaffected
    assert not held.ran.wait(QUIET)
    assert h.runner.queued() == [held]

    h.runner.submit(FakeJob("cpu", Lane.CPU))  # other lanes keep going
    assert h.rec.wait_terminal("cpu").type == "finished"

    h.runner.resume(Lane.GPU)
    assert not h.runner.is_paused(Lane.GPU)
    assert h.rec.wait_terminal("held").type == "finished"


def test_pause_only_holds_matching_jobs_and_lets_the_rest_run(h):
    h.runner.pause(Lane.GPU, only=lambda job: job.kind == "crop")
    assert h.runner.is_paused(Lane.GPU)

    crop = FakeJob("crop:a.mp4", kind="crop", priority=5)
    proof = FakeJob("proof:a.mp4", kind="proof", priority=0)
    h.runner.submit(crop)
    h.runner.submit(proof)

    assert h.rec.wait_terminal("proof:a.mp4").type == "finished"
    assert not crop.ran.wait(QUIET)
    assert h.runner.queued() == [crop]

    h.runner.resume(Lane.GPU)
    assert not h.runner.is_paused(Lane.GPU)
    assert h.rec.wait_terminal("crop:a.mp4").type == "finished"


def test_a_full_pause_holds_everything_and_resume_clears_both_holds(h):
    h.runner.pause(Lane.GPU, only=lambda job: job.kind == "crop")
    h.runner.pause(Lane.GPU)
    proof = FakeJob("proof", kind="proof")
    crop = FakeJob("crop", kind="crop")
    h.runner.submit(proof)
    h.runner.submit(crop)

    assert not proof.ran.wait(QUIET)

    h.runner.resume(Lane.GPU)
    assert not h.runner.is_paused(Lane.GPU)
    assert h.all_finished(["proof", "crop"])


# --- job-side events ---------------------------------------------------------

def test_progress_log_and_custom_events_arrive_in_order(h):
    caller = threading.get_ident()

    def body(ctx):
        ctx.progress(0.25, "decoding")
        ctx.log("engine leased")
        ctx.emit("run_file_started", file="ep01.mp4")
        ctx.progress(1.7)
        ctx.progress(-0.3, "clamped")
        ctx.emit("run_subtitle", file="ep01.mp4", result=("0:01", "0:02", "你好"))
        return "done"

    h.runner.submit(FakeJob("run", Lane.RUN, kind="run", body=body))

    first_event, first_thread = h.rec.with_threads("run")[0]
    assert (first_event.type, first_thread) == ("queued", caller)

    assert h.rec.wait_terminal("run") is not None
    seen = [(e.type, e.progress, e.message, e.file, e.result) for e in h.rec.of("run")]
    assert seen == [
        ("queued", None, "", None, None),
        ("started", None, "", None, None),
        ("progress", 0.25, "decoding", None, None),
        ("log", None, "engine leased", None, None),
        ("run_file_started", None, "", "ep01.mp4", None),
        ("progress", 1.0, "", None, None),
        ("progress", 0.0, "clamped", None, None),
        ("run_subtitle", None, "", "ep01.mp4", ("0:01", "0:02", "你好")),
        ("finished", None, "", None, "done"),
    ]
    assert all((e.key, e.kind) == ("run", "run") for e in h.rec.of("run"))
    assert len({e.job_id for e in h.rec.of("run")}) == 1  # custom events included
    assert h.rec.of("run")[0].job_id > 0
    assert all(thread != caller for _, thread in h.rec.with_threads("run")[1:])


def test_events_from_a_jobs_helper_threads_stay_between_started_and_terminal(h):
    def body(ctx):
        def helper(n):
            for i in range(50):
                ctx.emit("tick", message=f"{n}:{i}")
        helpers = [threading.Thread(target=helper, args=(n,)) for n in range(4)]
        for thread in helpers:
            thread.start()
        for thread in helpers:
            thread.join(WAIT)
        return "joined"

    job = FakeJob("fan-out", Lane.RUN, body=body)
    h.runner.submit(job)
    assert h.rec.wait_terminal("fan-out") is not None

    events = h.rec.of("fan-out")
    types = [e.type for e in events]
    assert types[:2] == ["queued", "started"]
    assert types[-1] == "finished"
    ticks = [e.message for e in events[2:-1]]
    assert len(ticks) == 200 and all(t == "tick" for t in types[2:-1])
    for n in range(4):
        assert [int(m.split(":")[1]) for m in ticks if m.startswith(f"{n}:")] == list(range(50))

    job.ctx.log("late line")  # a straggler after the terminal event is dropped
    job.ctx.progress(0.5)
    assert len(h.rec.of("fan-out")) == len(events)


def test_emit_rejects_lifecycle_types_and_identity_overrides(h):
    errors = {}

    def body(ctx):
        attempts = {
            "lifecycle": lambda: ctx.emit("finished"),
            "key": lambda: ctx.emit("run_file_started", key="other"),
            "unknown": lambda: ctx.emit("run_file_started", colour="red"),
        }
        for name, attempt in attempts.items():
            try:
                attempt()
            except Exception as exc:  # noqa: BLE001 - recording the type
                errors[name] = type(exc)

    h.runner.submit(FakeJob("strict", body=body))
    assert h.rec.wait_terminal("strict") is not None
    assert errors == {"lifecycle": ValueError, "key": TypeError, "unknown": TypeError}
    assert h.rec.types("strict") == ["queued", "started", "finished"]


# --- shutdown ----------------------------------------------------------------

def test_shutdown_cancels_cooperatively_clears_queues_and_joins():
    rec = Recorder()
    runner = JobRunner(rec, cpu_workers=2)
    busy_gpu = FakeJob("busy-gpu", Lane.GPU, body=_wait_cancel_event)
    busy_cpu = FakeJob("busy-cpu", Lane.CPU, body=_wait_cancel_event)
    try:
        for job in (busy_gpu, busy_cpu):
            runner.submit(job)
            assert job.ran.wait(WAIT)
        waiting = [FakeJob("wait-1"), FakeJob("wait-2")]
        for job in waiting:
            runner.submit(job)

        started = time.monotonic()
        assert runner.shutdown(timeout=WAIT) is True
        elapsed = time.monotonic() - started
    finally:
        runner.shutdown(timeout=0.1)

    assert elapsed < JOB_PATIENCE / 2  # joined because the jobs saw the cancel
    for job in waiting:
        assert rec.types(job.key) == ["queued", "cancelled"]
        assert not job.ran.is_set()
    assert rec.types("busy-gpu") == ["queued", "started", "cancelled"]
    assert rec.types("busy-cpu") == ["queued", "started", "cancelled"]
    assert runner.queued() == [] and runner.running() == []
    with pytest.raises(RuntimeError):
        runner.submit(FakeJob("too-late"))


def test_shutdown_times_out_on_an_uncooperative_job_without_killing_it():
    rec = Recorder()
    runner = JobRunner(rec)
    gate = threading.Event()
    completed = threading.Event()

    def stubborn(ctx):
        gate.wait(WAIT)
        completed.set()
        return "done anyway"

    job = FakeJob("stubborn", body=stubborn)
    runner.submit(job)
    assert job.ran.wait(WAIT)
    try:
        started = time.monotonic()
        assert runner.shutdown(timeout=0.3) is False
        elapsed = time.monotonic() - started
        assert 0.25 <= elapsed < 2.0
        assert not completed.is_set()
        assert runner.running() == [job]
    finally:
        gate.set()

    assert completed.wait(WAIT)
    terminal = rec.wait_terminal("stubborn")
    assert terminal is not None
    assert (terminal.type, terminal.result) == ("finished", "done anyway")
    assert runner.shutdown(timeout=WAIT) is True  # its worker survived and exited


def test_shutdown_called_from_inside_a_job_leaves_its_own_thread_out():
    rec = Recorder()
    runner = JobRunner(rec)
    try:
        runner.submit(FakeJob("self-stop", body=lambda ctx: runner.shutdown(timeout=WAIT)))
        terminal = rec.wait_terminal("self-stop")
    finally:
        runner.shutdown(timeout=2.0)

    assert terminal is not None
    assert (terminal.type, terminal.result) == ("finished", True)


# --- identity ------------------------------------------------------------------

def test_a_jobs_identity_cannot_be_rewritten_through_its_context(h):
    gate = h.gate()
    refused = set()

    def body(ctx):
        for name, value in (("key", "mutated"), ("kind", "other"),
                            ("file", "other.mp4"), ("job_id", 999)):
            try:
                setattr(ctx, name, value)
            except AttributeError:
                refused.add(name)
        gate.wait(WAIT)
        return "first"

    first = FakeJob("k", Lane.CPU, file="a.mp4", body=body)
    h.runner.submit(first)
    assert first.ran.wait(WAIT)
    follow_up = FakeJob("k", Lane.CPU, body=lambda ctx: "second")
    unrelated = FakeJob("mutated", Lane.CPU, body=lambda ctx: "unrelated")
    h.runner.submit(follow_up)
    h.runner.submit(unrelated)

    assert h.rec.wait_terminal("mutated").result == "unrelated"  # not gated by "k"
    assert not follow_up.ran.wait(QUIET)                            # gated by "k"
    gate.set()
    second = h.rec.wait_terminal("k", count=2)                     # "k" was released
    assert second is not None and second.result == "second"

    assert refused == {"key", "kind", "file", "job_id"}
    first_id = h.rec.of("k")[0].job_id
    first_events = [e for e in h.rec.of("k") if e.job_id == first_id]
    assert [e.type for e in first_events] == ["queued", "started", "finished"]
    assert {(e.kind, e.file) for e in first_events} == {("test", "a.mp4")}


def test_event_fields_after_file_are_keyword_only():
    with pytest.raises(TypeError):
        JobEvent("progress", "k", "test", None, 0.5)
    event = JobEvent("progress", "k", "test", None, progress=0.5)
    assert (event.job_id, event.progress) == (0, 0.5)
