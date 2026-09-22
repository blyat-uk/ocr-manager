"""Video capture adapters with optional GPU acceleration."""
import cv2
import itertools
import logging
import math
import numpy as np
import subprocess
import json
import shutil

from core.proc import TEXT_ENCODING, hidden_child

logger = logging.getLogger(__name__)

# Try PyAV first (FFmpeg bindings with hardware acceleration support)
try:
    import av
    PYAV_AVAILABLE = True
    PYAV_IMPORT_ERROR = None
except Exception as exc:  # ImportError, or a linker error from an ABI mismatch
    PYAV_AVAILABLE = False
    PYAV_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# Check if FFmpeg is available
FFMPEG_AVAILABLE = shutil.which('ffmpeg') is not None and shutil.which('ffprobe') is not None

# HDR transfer characteristic constants (from ITU-T H.273)
_TRC_SMPTE2084 = 16   # PQ (HDR10)
_TRC_ARIB_STD_B67 = 18  # HLG

_ZSCALE_AVAILABLE = None  # Lazy-cached (system ffmpeg CLI)
_PYAV_ZSCALE_AVAILABLE = None  # Lazy-cached (PyAV's own filter registry)

# Downscale 4K+ to 1080p at decode level for performance.
# Frames arrive pre-scaled so the OCR pipeline processes the same pixel count
# regardless of source resolution.  get(CAP_PROP_FRAME_*) still returns native
# dimensions; only read() produces scaled frames.
DECODE_TARGET_HEIGHT = 1080

# Largest output-space alignment step we'll accept when planning an
# in-graph crop. If neither axis's exact native/output ratio yields a
# qualifying alignment within this cap, the crop stage is refused rather
# than risk a phase mismatch between crop and scale (see _crop_axis_plan).
_CROP_ALIGN_CAP = 64

# seek_to_display_time(t) counts a frame as shown at `t` when its PTS is at
# most `t` plus this, so a time that is a frame's PTS give or take float
# rounding (a PTS plus a sum of 0.5 s or 0.2 s steps) names that frame.
DISPLAY_TIME_TOLERANCE = 1e-6


def _crop_axis_plan(out_dim: int, native_dim: int):
    """Plan one axis of an in-graph crop so native-space coordinates are
    both exact integers and even (required for 4:2:0 chroma safety).

    `out_dim` and `native_dim` are generally related by a rounded scale
    factor (e.g. `_output_width` is `int(width * sf)` rounded up to even),
    not a clean ratio, so we derive the *exact* rational mapping from their
    gcd instead of trusting a single float scale factor for both axes.

    Reducing native_dim/out_dim by g = gcd(out_dim, native_dim) gives
    out_dim = g * out_step and native_dim = g * native_step with
    out_step and native_step coprime. An output-space coordinate maps to
    an exact integer native coordinate iff it is a multiple of out_step
    (since gcd(out_step, native_step) == 1). If native_step is odd, the
    resulting native coordinate needs a further factor of 2 to guarantee
    it lands on an even (chroma-safe) native pixel; if native_step is
    already even any multiple of out_step is automatically even.

    Returns (align, native_step, out_step), where any output coordinate
    that is a multiple of `align` maps via
    `native = (coord // out_step) * native_step`
    to an exact, even native coordinate -- or None if no such alignment
    exists within `_CROP_ALIGN_CAP`.
    """
    g = math.gcd(out_dim, native_dim)
    out_step = out_dim // g
    native_step = native_dim // g

    # Empirical finding (not just theory): libswscale's scale filter is
    # only reliably bit-identical across two differently-sized invocations
    # of the *same* ratio when the ratio's reduced denominator (out_step)
    # is a power of two. Ratios such as 4:3 or 10:9 (out_step 3 or 9) were
    # measured to differ from the full-frame reference by up to 1 LSB on a
    # scattering of pixels, and raising the padding from 8 to 64 changed
    # nothing -- so this is not an edge/tap-support shortfall, it is
    # accumulated floating-point drift in the filter's phase computation
    # that depends on the *absolute* native offset a crop starts at, which
    # a locally-restarted sub-buffer scale cannot reproduce unless 1/out_step
    # is exactly representable in binary floating point (true only for
    # powers of two). Refuse the crop stage rather than risk it.
    if out_step & (out_step - 1) != 0:
        return None

    align = out_step if native_step % 2 == 0 else 2 * out_step
    if align > _CROP_ALIGN_CAP:
        return None
    return align, native_step, out_step


def _has_zscale() -> bool:
    """Check if the *system* FFmpeg CLI has the zscale filter (requires zimg).

    Only meaningful for FFmpegNVDECCapture, which shells out to that binary.
    PyAV graphs run against the FFmpeg bundled in the `av` wheel, whose
    filter inventory is unrelated -- use `_pyav_has_zscale()` for those.
    """
    global _ZSCALE_AVAILABLE
    if _ZSCALE_AVAILABLE is None:
        try:
            result = subprocess.run(
                ['ffmpeg', '-filters'],
                capture_output=True, text=True, timeout=5, **TEXT_ENCODING,
                **hidden_child()
            )
            _ZSCALE_AVAILABLE = 'zscale' in result.stdout
        except Exception:
            _ZSCALE_AVAILABLE = False
    return _ZSCALE_AVAILABLE


def _pyav_has_zscale() -> bool:
    """Check if PyAV's own bundled FFmpeg exposes the zscale filter.

    The binary wheels for av>=18 are not built against zimg, so `zscale` is
    absent from PyAV's registry even on machines whose system ffmpeg has it.
    Probing the CLI instead (see `_has_zscale`) makes every HDR source build
    a graph that cannot configure.
    """
    global _PYAV_ZSCALE_AVAILABLE
    if _PYAV_ZSCALE_AVAILABLE is None:
        if not PYAV_AVAILABLE:
            _PYAV_ZSCALE_AVAILABLE = False
        else:
            try:
                import av.filter
                _PYAV_ZSCALE_AVAILABLE = 'zscale' in av.filter.filters_available
            except Exception:
                _PYAV_ZSCALE_AVAILABLE = False
    return _PYAV_ZSCALE_AVAILABLE


def _build_ffmpeg_tonemap_vf(transfer: str) -> str:
    """Build FFmpeg -vf filter chain for HDR→SDR tone mapping.

    Args:
        transfer: color_transfer string from ffprobe (e.g. 'smpte2084', 'arib-std-b67')

    Returns:
        Filter chain string for -vf argument.
    """
    if _has_zscale():
        # PQ needs npl=100 (nominal peak luminance); HLG does not
        zscale_linear = 'zscale=t=linear:npl=100' if transfer == 'smpte2084' else 'zscale=t=linear'
        return (
            f'{zscale_linear},format=gbrpf32le,'
            'tonemap=hable,zscale=t=bt709,format=bgr24'
        )
    return 'format=gbrpf32le,tonemap=hable,format=bgr24'


class FFmpegNVDECCapture:
    """Video capture using FFmpeg subprocess with NVDEC hardware acceleration.

    Uses NVIDIA's dedicated video decoder hardware for fast decoding.
    Falls back to CPU decoding if NVDEC is not available.
    """

    def __init__(self, video_path, use_gpu=True, decode_target_height=None, crop_rect=None):
        self.path = video_path
        self.use_gpu = use_gpu
        self._decode_target_height = decode_target_height
        self._crop_request = crop_rect  # Accepted for interface parity; ignored.
        self._crop_slice = None  # Never set: callers keep slicing in Python.
        self.proc = None
        self._pos = 0
        self._frame_count = None
        self._fps = None
        self._width = None
        self._height = None
        self._output_width = None
        self._output_height = None
        self._scale_factor = 1.0
        self._frame_size = None
        self._seek_pos = 0
        self._last_pts = None  # Estimated PTS in seconds
        self._hdr_transfer = None  # Color transfer from ffprobe
        self._container_start_time = 0.0
        # See PyAVCapture.seek_past_end.
        self.seek_past_end = False

    def _probe_video(self):
        """Get video metadata using ffprobe."""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            self.path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, **TEXT_ENCODING, **hidden_child())
        if result.returncode != 0:
            raise IOError(f'Cannot probe video {self.path}')

        data = json.loads(result.stdout)
        video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
        if not video_stream:
            raise IOError(f'No video stream found in {self.path}')

        self._width = int(video_stream['width'])
        self._height = int(video_stream['height'])
        self._frame_size = self._width * self._height * 3  # BGR24

        # Get FPS
        if 'avg_frame_rate' in video_stream:
            num, den = map(int, video_stream['avg_frame_rate'].split('/'))
            self._fps = num / den if den else 25.0
        else:
            self._fps = 25.0

        # Detect HDR transfer characteristics
        self._hdr_transfer = video_stream.get('color_transfer')

        # Get frame count
        if 'nb_frames' in video_stream:
            self._frame_count = int(video_stream['nb_frames'])
        elif 'duration' in data['format']:
            duration = float(data['format']['duration'])
            self._frame_count = int(duration * self._fps)
        else:
            self._frame_count = 100000  # Fallback

        fmt = data.get("format", {})
        try:
            self._container_start_time = max(0.0, float(fmt.get("start_time", 0.0)))
        except (TypeError, ValueError):
            self._container_start_time = 0.0

    def _start_ffmpeg(self, seek_time=None):
        """Start FFmpeg subprocess with optional seek."""
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error']

        # Add hardware acceleration
        if self.use_gpu:
            cmd.extend(['-hwaccel', 'cuda'])

        # Add seek before input (fast seek)
        if seek_time and seek_time > 0:
            cmd.extend(['-ss', str(seek_time)])

        cmd.extend(['-i', self.path])

        # Build combined -vf chain (tone mapping + scaling)
        vf_parts = []
        if self._hdr_transfer in ('smpte2084', 'arib-std-b67'):
            vf_parts.append(_build_ffmpeg_tonemap_vf(self._hdr_transfer))
        if self._scale_factor < 1.0:
            vf_parts.append(f'scale={self._output_width}:{self._output_height}')
        if vf_parts:
            cmd.extend(['-vf', ','.join(vf_parts)])

        cmd.extend([
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-'
        ])

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self._frame_size * 10,  # Buffer 10 frames
            **hidden_child()
        )

    def __enter__(self):
        if not FFMPEG_AVAILABLE:
            # Fall back to OpenCV
            self.cap = cv2.VideoCapture(self.path)
            if not self.cap.isOpened():
                raise IOError(f'Cannot open video {self.path}')
            return self.cap

        self._probe_video()

        # Compute decode-level scaling for high-res videos
        if (self._decode_target_height is not None
                and self._height > self._decode_target_height):
            self._scale_factor = self._decode_target_height / self._height
            self._output_width = int(self._width * self._scale_factor)
            self._output_width += self._output_width % 2  # Ensure even
            self._output_height = self._decode_target_height
            self._frame_size = self._output_width * self._output_height * 3
        else:
            self._output_width = self._width
            self._output_height = self._height

        self._start_ffmpeg()
        self._pos = 0
        self._seek_pos = 0
        return self

    def configure_crop(self, crop_rect) -> None:
        """Interface parity with PyAVCapture; this backend never crops in
        its decode graph (crop_rect was already accepted-but-ignored by
        the constructor for the same reason - see _crop_request above).
        Callers can call this unconditionally regardless of backend.
        """
        self._crop_request = crop_rect

    def __exit__(self, exc_type, exc_value, traceback):
        if not FFMPEG_AVAILABLE:
            self.cap.release()
            return
        if self.proc:
            self.proc.stdout.close()
            self.proc.terminate()
            self.proc.wait()
            self.proc = None

    def get(self, prop):
        """Get video property (compatible with cv2.VideoCapture.get)."""
        if not FFMPEG_AVAILABLE:
            return self.cap.get(prop)

        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self._frame_count
        elif prop == cv2.CAP_PROP_FPS:
            return self._fps
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._height
        elif prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self._width
        elif prop == cv2.CAP_PROP_POS_FRAMES:
            return self._seek_pos + self._pos
        return 0

    def set(self, prop, value):
        """Set video property (compatible with cv2.VideoCapture.set)."""
        if not FFMPEG_AVAILABLE:
            return self.cap.set(prop, value)

        if prop == cv2.CAP_PROP_POS_FRAMES:
            target_pos = int(value)
            # Positions <= 0 leave a running pipe where it is, as they always
            # have; but a pipe a past-the-end seek stopped restarts for any
            # position, including the one it was at.
            if target_pos > 0 or self.proc is None:
                self._reposition(max(0, target_pos))
            return True
        return False

    def seek_to_pts(self, pts):
        """Position the pipe so the next read()/grab() returns the first
        frame whose PTS -- as get_last_pts() reports it -- is at or after
        `pts` (interface parity with PyAVCapture.seek_to_pts).

        This backend reports PTS as ordinal / fps + container start time,
        where the ordinal counts frames from the start of the file (ffmpeg's
        -ss is relative to the start time), so this inverts that. Unlike
        set(), it also repositions to ordinal 0.
        """
        self.seek_past_end = False
        # The 1e-6 absorbs float noise when `pts` is exactly a frame's PTS.
        self._reposition(max(0, math.ceil((pts - self._container_start_time) * self._fps - 1e-6)))
        return True

    def seek_to_display_time(self, t):
        """Position the pipe so the next read()/grab() returns the frame on
        screen at time `t` (interface parity with
        PyAVCapture.seek_to_display_time): the last frame whose PTS, as
        get_last_pts() reports it, is at most `t` + DISPLAY_TIME_TOLERANCE,
        or the first frame when `t` precedes it.

        This backend reports PTS as ordinal / fps + container start time, so
        this finds the last such ordinal with that same expression.

        Returns:
            bool: False when that ordinal is at or past the frame count
                  (ffprobe's nb_frames, or failing that the duration
                  estimate the pipeline bounds its scans with); read()/grab()
                  then fail until the next seek, and seek_past_end is True.
                  ffmpeg itself would still deliver a frame for a -ss past
                  the end.
        """
        limit = t + DISPLAY_TIME_TOLERANCE
        start = self._container_start_time
        ordinal = max(0, math.floor((limit - start) * self._fps))
        # Settle float rounding against read()'s own PTS expression.
        while ordinal > 0 and ordinal / self._fps + start > limit:
            ordinal -= 1
        while (ordinal + 1) / self._fps + start <= limit:
            ordinal += 1
        self.seek_past_end = ordinal >= self._frame_count
        if self.seek_past_end:
            self._stop_ffmpeg()
            return False
        self._reposition(ordinal)
        return True

    def _reposition(self, ordinal):
        """Make `ordinal` the next frame the pipe delivers, restarting ffmpeg
        unless it is already next."""
        if self.proc is not None and ordinal == self._seek_pos + self._pos:
            return
        self._stop_ffmpeg()
        self._start_ffmpeg(ordinal / self._fps if ordinal > 0 else None)
        self._seek_pos = ordinal
        self._pos = 0

    def _stop_ffmpeg(self):
        """End the pipe; read()/grab() fail until the next seek starts one."""
        if self.proc:
            self.proc.stdout.close()
            self.proc.terminate()
            self.proc.wait()
            self.proc = None

    def read(self):
        """Read next frame from FFmpeg pipe.

        Returns:
            tuple: (success, frame) where frame is BGR numpy array
        """
        if not FFMPEG_AVAILABLE:
            ret, frame = self.cap.read()
            if ret:
                self._last_pts = self._pos / self._fps if self._fps else None
            return ret, frame

        if self.proc is None:
            return False, None  # a seek past the last frame stopped the pipe
        try:
            raw = self.proc.stdout.read(self._frame_size)
            if len(raw) != self._frame_size:
                return False, None
            # A writable copy: callers modify frames in place (label masks),
            # and a view of the immutable pipe bytes would refuse that.
            frame = np.frombuffer(bytearray(raw), dtype=np.uint8).reshape(self._output_height, self._output_width, 3)
            # PTS of THIS frame is derived from its index, which is the position
            # before the increment. Add the seek origin and the container start.
            self._last_pts = (
                (self._seek_pos + self._pos) / self._fps + self._container_start_time
                if self._fps else None
            )
            self._pos += 1
            return True, frame
        except Exception:
            return False, None

    def grab(self):
        """Advance past the next frame (compatible with cv2.VideoCapture.grab).

        Interface parity with PyAVCapture.grab. The pipe delivers every
        frame already converted, so there is nothing to skip here: this is
        read() with the frame discarded, keeping position and PTS exactly
        as read() leaves them.
        """
        ok, _ = self.read()
        return ok

    def get_last_pts(self) -> float:
        """Get estimated PTS of the last read frame in seconds."""
        return self._last_pts

    def get_stream_start_time(self) -> float:
        """The container's start_time in seconds: time 0 of the output, as
        PyAVCapture.get_stream_start_time() explains.

        read() counts PTS from the same place: ffmpeg's pipe begins at the
        container start, padding with copies of the first frame when the
        video starts later, so the real frames keep their true PTS.
        """
        return getattr(self, "_container_start_time", 0.0)


class PyAVCapture:
    """Video capture using PyAV with optional CUDA hardware acceleration.

    Uses FFmpeg's NVDEC for hardware-accelerated video decoding on NVIDIA GPUs.
    """

    _CROP_PAD = 8  # margin decoded around the crop, sliced off after conversion

    def __init__(self, video_path, use_gpu=True, decode_target_height=None, crop_rect=None):
        self.path = video_path
        self.use_gpu = use_gpu
        self._decode_target_height = decode_target_height
        self.container = None
        self.stream = None
        self._pos = 0
        # True once read() or set() has been called; configure_crop()
        # refuses to (re)build the filter graph after that point.
        self._read_started = False
        self._frame_count = None
        self._fps = None
        self._width = None
        self._height = None
        self._scale_factor = 1.0
        self._output_width = None
        self._output_height = None
        # PTS tracking - canonical timestamp source
        self._last_pts = None  # Last frame's PTS in seconds
        # Why the last seek_to_pts()/seek_to_display_time() found no frame:
        # True when its time is past the last frame (or there is nothing to
        # decode), so every later time is too. False after a seek that found
        # a frame, and after one whose retries all landed late -- a later
        # time may still have a frame.
        self.seek_past_end = False
        # Combined filter graph for tone mapping and/or scaling (None = fast path)
        self._filter_graph = None
        # Crop-in-filter-graph state
        self._crop_request = crop_rect   # (x, y, w, h) in decode-output coords
        self._crop_slice = None          # (y0, y1, x0, x1) applied after conversion
        self._crop_native = None         # (x, y, w, h) in native decode coords
        self._crop_scaled_size = None    # (w, h) the scale stage should target
        self._crop_graph_active = False  # True once the graph actually cropped
        # Filled in by _detect_tonemap() during __enter__, before _plan_crop()
        # needs to know whether a filter graph will exist at all.
        self._needs_tonemap = False
        self._tonemap_trc = None

    def __enter__(self):
        if not PYAV_AVAILABLE:
            # Fall back to OpenCV
            self.cap = cv2.VideoCapture(self.path)
            if not self.cap.isOpened():
                raise IOError(f'Cannot open video {self.path}')
            return self.cap

        # Open container
        self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]

        # Enable threading for faster decoding
        self.stream.thread_type = 'AUTO'

        # Cache video properties
        self._fps = float(self.stream.average_rate) if self.stream.average_rate else 25.0
        if self.stream.frames:
            self._frame_count = self.stream.frames
        elif self.stream.duration and self.stream.time_base:
            duration_sec = float(self.stream.duration * self.stream.time_base)
            self._frame_count = int(duration_sec * self._fps)
        elif self.container.duration:
            # Use container duration (in microseconds)
            duration_sec = self.container.duration / 1_000_000
            self._frame_count = int(duration_sec * self._fps)
        else:
            # Estimate based on file size (rough fallback, avoids counting)
            self._frame_count = 100000  # Large number, will stop at actual end
        self._width = self.stream.width
        self._height = self.stream.height

        # Compute decode-level scaling for high-res videos
        if (self._decode_target_height is not None
                and self._height > self._decode_target_height):
            self._scale_factor = self._decode_target_height / self._height
            self._output_width = int(self._width * self._scale_factor)
            self._output_width += self._output_width % 2  # Ensure even
            self._output_height = self._decode_target_height
        else:
            self._scale_factor = 1.0
            self._output_width = self._width
            self._output_height = self._height

        self._pos = 0
        self._read_started = False
        self._frame_generator = self.container.decode(video=0)

        # Whether the stream needs HDR->SDR tone mapping. Resolved before the
        # crop is planned because _plan_crop's admissibility rule depends on
        # whether a filter graph is going to exist for some other reason.
        self._needs_tonemap, self._tonemap_trc = self._detect_tonemap()

        # Plan-crop + build-graph is a separate, public step (configure_crop,
        # below) so a caller that does not know its crop until after opening
        # -- it needs this capture's own self.get(CAP_PROP_FRAME_HEIGHT) etc.
        # first, e.g. videocr/video.py's run_ocr() -- can still get a crop
        # baked into the filter graph, by calling configure_crop() itself
        # once it knows one, before the first read()/set(). Everything
        # __enter__ already knows by this point (_width/_height/
        # _output_width/_output_height/_scale_factor/_needs_tonemap/
        # self.stream) is exactly what that step needs, so nothing here
        # depends on the crop being known yet. Run it now unconditionally
        # (with whatever crop_rect the constructor was given, None by
        # default) so every *existing* caller -- everything that already
        # passes crop_rect= to the constructor, or none at all -- keeps
        # getting it fully configured by the time __enter__ returns, same
        # as before this split.
        self.configure_crop(self._crop_request)

        return self

    def configure_crop(self, crop_rect) -> None:
        """(Re)plan the crop and (re)build the filter graph (tone-map
        and/or downscale and/or crop) for this already-open capture.

        Must be called before the first read()/set() -- the filter graph,
        once built, is what every read() pushes decoded frames through,
        so rebuilding it after frames have already been pulled would
        silently start converting later frames differently mid-stream.
        __enter__() already calls this once, with whatever crop_rect the
        constructor was given (None if none), so most callers never need
        to call it again. A caller that only learns its crop after
        opening -- because planning it needs native height, which this
        capture is the one place that has it before decoding starts --
        calls it again itself, with the crop now known, before touching
        read()/set(). That rebuilds the graph a second time; harmless,
        since nothing has been pushed through the first one yet.

        Re-configuring (to a different crop, to no crop at all, or to one
        _plan_crop ends up refusing) never leaves a previous call's crop
        state behind: everything crop-shaped is reset before (re)planning,
        below, since _plan_crop only ever *sets* these fields, never
        clears a stale value left over from an earlier, different request.
        """
        if self._read_started:
            # Rejected -- leave _crop_request exactly as an earlier,
            # successful call last left it, not as whatever was passed to
            # this failed one.
            raise RuntimeError(
                "configure_crop() called after decoding already started; "
                "it must run before the first read()/set()."
            )
        self._crop_request = crop_rect
        if not PYAV_AVAILABLE:
            return  # cv2.VideoCapture fallback: no filter-graph crop support
        self._crop_native = None
        self._crop_scaled_size = None
        self._crop_slice = None
        self._crop_graph_active = False
        try:
            # Translate the requested output-space crop into a native-space,
            # aligned, padded decode box (no-op if no crop was requested).
            self._plan_crop()

            # Set up combined filter graph (tone mapping + scaling)
            self._setup_filter_graph()
        except Exception:
            # __exit__ does not run when __enter__ raises, so if this is
            # __enter__'s own call (see above), release the container here
            # rather than leaking an open demuxer. A later, explicit call
            # (this capture already returned from __enter__) is inside the
            # caller's `with` block, so __exit__ still runs normally on the
            # way out -- closing here first is harmless, __exit__ just finds
            # self.container already None and does nothing.
            if self.container is not None:
                self.container.close()
                self.container = None
            raise

    def _detect_tonemap(self):
        """(needs_tonemap, transfer_characteristic) for this stream."""
        try:
            trc = self.stream.codec_context.color_trc
        except Exception:
            return False, None
        return trc in (_TRC_SMPTE2084, _TRC_ARIB_STD_B67), trc

    def _plan_crop(self):
        """Translate the requested output-space crop into a native-space,
        aligned, padded decode box plus the exact slice to take afterwards.

        The invariant this must satisfy: EITHER the crop stage is planned
        such that the graph's crop+scale is `np.array_equal` to slicing the
        full-frame reference, OR the crop stage is refused entirely (this
        method returns without setting `self._crop_slice`), leaving
        `video.py` to slice the full frame in Python. "Planned but only
        approximately correct" must never happen.

        The two axes are planned independently via `_crop_axis_plan`
        because `_output_height` is exactly the decode target but
        `_output_width` is `int(width * sf)` rounded up to even, so the
        two axes' native/output ratios are not generally identical. A
        padding margin absorbs swscale's filter taps at the crop edges;
        it is sliced off after conversion via `self._crop_slice`.
        """
        if self._crop_request is None:
            return

        # The crop stage is only admissible when a filter graph is going to
        # exist anyway (tone map and/or downscale). Without one, read()'s
        # reference path converts with `frame.to_ndarray(format='bgr24')`,
        # whereas any graph converts through a `format=bgr24` filter *node*.
        # Those are not the same function: swscale's filter node dithers,
        # the reformatter behind to_ndarray does not, so for any source
        # deeper than 8 bits the two disagree. Measured on 320x240
        # testsrc2, frame 0, to_ndarray vs format-filter:
        #
        #   yuv420p    (h264/h265)  max abs diff 0
        #   yuv420p10le h264         max abs diff 148
        #   yuv420p10le h265         max abs diff 148
        #
        # So merely *asking* for a crop would silently change every pixel of
        # a 10-bit source. Refuse instead, leaving `_crop_slice` None so
        # video.py slices the full frame in Python off the identical
        # reference path. Where a graph exists for another reason the
        # conversion path is the same with and without the crop node, and
        # the crop stays exact (verified by the crop-invariant matrix,
        # which covers 10-bit downscale geometries).
        if not self._needs_tonemap and self._scale_factor >= 1.0:
            return

        x, y, w, h = (int(v) for v in self._crop_request)
        out_w, out_h = self._output_width, self._output_height
        x = max(0, min(x, out_w - 1)); y = max(0, min(y, out_h - 1))
        w = max(1, min(w, out_w - x)); h = max(1, min(h, out_h - y))

        plan_x = _crop_axis_plan(out_w, self._width)
        plan_y = _crop_axis_plan(out_h, self._height)
        if plan_x is None or plan_y is None:
            # Some axis either has no alignment within the cap, or its
            # ratio's reduced denominator isn't a power of two (see
            # _crop_axis_plan) -- refuse the crop stage rather than risk a
            # phase/precision mismatch against the reference.
            # `self._crop_slice` stays None.
            return
        align_x, native_step_x, out_step_x = plan_x
        align_y, native_step_y, out_step_y = plan_y

        def to_native_x(coord):
            return (coord // out_step_x) * native_step_x

        def to_native_y(coord):
            return (coord // out_step_y) * native_step_y

        # Padded box in OUTPUT space, aligned down/up per-axis.
        px0 = max(0, ((x - self._CROP_PAD) // align_x) * align_x)
        py0 = max(0, ((y - self._CROP_PAD) // align_y) * align_y)
        px1 = min(out_w, -(-(x + w + self._CROP_PAD) // align_x) * align_x)
        py1 = min(out_h, -(-(y + h + self._CROP_PAD) // align_y) * align_y)

        # Same box in NATIVE space, where the crop filter runs. Clamp
        # against the actual decoded frame: `_output_width`/`_output_height`
        # can round up past the exact native/scale ratio (e.g. to force an
        # even output width), so defensively cap the native box too.
        n_x0 = to_native_x(px0)
        n_y0 = to_native_y(py0)
        n_x1 = min(to_native_x(px1), self._width)
        n_y1 = min(to_native_y(py1), self._height)
        n_w = n_x1 - n_x0
        n_h = n_y1 - n_y0
        if n_w <= 0 or n_h <= 0:
            return

        self._crop_native = (n_x0, n_y0, n_w, n_h)
        self._crop_scaled_size = (px1 - px0, py1 - py0)
        self._crop_slice = (y - py0, y - py0 + h, x - px0, x - px0 + w)

    @staticmethod
    def _add_tonemap_chain(graph, last, trc):
        """Append the best HDR->SDR tone-map chain PyAV can actually build.

        Returns the new tail filter. The zscale variant is preferred because
        it linearises and returns to BT.709 properly, but PyAV's bundled
        FFmpeg usually lacks zimg; the zscale-free variant is then the same
        chain `_build_ffmpeg_tonemap_vf` hands the subprocess backend in the
        equivalent situation. Building the chain against PyAV's own registry
        rather than the system ffmpeg CLI is the whole point: probing the CLI
        made every PQ/HLG source raise inside graph.configure().
        """
        if _pyav_has_zscale():
            zscale_linear_args = 't=linear:npl=100' if trc == _TRC_SMPTE2084 else 't=linear'
            linearize = graph.add('zscale', zscale_linear_args)
            fmt_in = graph.add('format', 'gbrpf32le')
            tonemap = graph.add('tonemap', 'hable')
            bt709 = graph.add('zscale', 't=bt709')
            last.link_to(linearize)
            linearize.link_to(fmt_in)
            fmt_in.link_to(tonemap)
            tonemap.link_to(bt709)
            return bt709

        fmt_in = graph.add('format', 'gbrpf32le')
        tonemap = graph.add('tonemap', 'hable')
        last.link_to(fmt_in)
        fmt_in.link_to(tonemap)
        return tonemap

    def _setup_filter_graph(self):
        """Build a unified PyAV filter graph for HDR tone mapping and/or downscaling."""
        needs_tonemap, trc = self._needs_tonemap, self._tonemap_trc
        needs_scale = self._scale_factor < 1.0
        # _plan_crop only plans a crop when one of the two above is true, so
        # the crop node never brings a graph (and with it a different bgr24
        # conversion path) into existence on its own.
        needs_crop = self._crop_slice is not None

        if not needs_tonemap and not needs_scale and not needs_crop:
            self._filter_graph = None
            return

        try:
            graph = av.filter.Graph()
            buf = graph.add_buffer(template=self.stream)
            last = buf

            if needs_tonemap:
                last = self._add_tonemap_chain(graph, last, trc)

            if needs_crop:
                n_x0, n_y0, n_w, n_h = self._crop_native
                crop = graph.add('crop', f'{n_w}:{n_h}:{n_x0}:{n_y0}')
                last.link_to(crop)
                last = crop

            # Scale after tone mapping (operates on uint8 bgr24 for efficiency)
            fmt_out = graph.add('format', 'bgr24')
            last.link_to(fmt_out)
            last = fmt_out

            if needs_scale:
                if needs_crop:
                    sw, sh = self._crop_scaled_size
                else:
                    sw, sh = self._output_width, self._output_height
                scale = graph.add('scale', f'{sw}:{sh}')
                last.link_to(scale)
                last = scale

            sink = graph.add('buffersink')
            last.link_to(sink)

            graph.configure()
            self._filter_graph = graph
            if needs_crop:
                self._crop_graph_active = True
        except Exception as exc:
            # Fatal, deliberately. The graph is the only thing that honours
            # `decode_target_height` and the only thing that tone maps, while
            # callers (videocr/video.py) have already rescaled their crop
            # coordinates into output space. Continuing without it would hand
            # them native-resolution, still-HDR frames that they would then
            # slice with stale output-space coordinates -- the wrong region of
            # the wrong picture, silently. Reset the crop state so a caller
            # that catches this cannot find a half-built plan.
            self._filter_graph = None
            self._crop_graph_active = False
            self._crop_slice = None
            raise RuntimeError(
                f"Could not build the PyAV filter graph for {self.path} "
                f"(tone map={needs_tonemap}, downscale={needs_scale}, "
                f"crop={needs_crop}). Decoding without it would silently "
                f"return {self._width}x{self._height} frames where "
                f"{self._output_width}x{self._output_height} was requested. "
                f"Refusing rather than degrading. Cause: {type(exc).__name__}: {exc}"
            ) from exc

    def __exit__(self, exc_type, exc_value, traceback):
        if not PYAV_AVAILABLE:
            self.cap.release()
            return
        if self.container:
            self.container.close()
            self.container = None

    def get(self, prop):
        """Get video property (compatible with cv2.VideoCapture.get)."""
        if not PYAV_AVAILABLE:
            return self.cap.get(prop)

        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self._frame_count
        elif prop == cv2.CAP_PROP_FPS:
            return self._fps
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._height
        elif prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self._width
        elif prop == cv2.CAP_PROP_POS_FRAMES:
            return self._pos
        return 0

    def set(self, prop, value):
        """Set video property (compatible with cv2.VideoCapture.set)."""
        self._read_started = True
        if not PYAV_AVAILABLE:
            return self.cap.set(prop, value)

        if prop == cv2.CAP_PROP_POS_FRAMES:
            target_pos = int(value)
            if target_pos > 0:
                # Seek to timestamp - PyAV seeks to nearest keyframe
                target_pts = int(target_pos / self._fps / self.stream.time_base)
                self.container.seek(target_pts, stream=self.stream)
                self._frame_generator = self.container.decode(video=0)

                # Skip frames until we reach the target position
                # PyAV seek lands on keyframe, which may be before target
                for frame in self._frame_generator:
                    frame_pos = int(round(frame.pts * float(self.stream.time_base) * self._fps))
                    if frame_pos >= target_pos:
                        # Found target frame - store it for next read()
                        self._pos = frame_pos
                        self._pending_frame = frame
                        return True
                # Reached end without finding target
                self._pos = target_pos
            return True
        return False

    # How far before its target a seek (_seek_with_retries) retries when it
    # lands after it, doubling per retry.
    _SEEK_BACKOFF_SECONDS = 1.0
    _SEEK_MAX_RETRIES = 8

    def seek_to_pts(self, pts):
        """Position the capture so the next read()/grab() returns the first
        frame whose PTS -- as get_last_pts() reports it -- is at or after
        `pts`. Passing a PTS that get_last_pts() returned earlier lands on
        exactly that frame.

        set(CAP_PROP_POS_FRAMES, n) cannot promise that. It targets
        round(PTS * fps), so on a file whose first frame is not at PTS 0 (an
        MP4 edit list, a container start time) an index counted from the
        first frame read names a different frame, and two frames whose
        PTS * fps straddle a .5 map to the same position. This compares
        float(frame.pts * time_base) -- the value get_last_pts() returns --
        directly, and skips the frames before the target without
        converting them, exactly as set() does.

        If a seek lands *after* the target (a demuxer indexing keyframes by
        DTS can) or yields no frame at all, it retries from further back
        (_SEEK_BACKOFF_SECONDS, then twice that, and so on), until the first
        decoded frame is at or before the target or the seek reaches the
        start of the stream. What a seek to the start of the stream decodes
        first is the first frame, so that seek is never retried.

        Returns:
            bool: True if a frame at or after `pts` exists. False at end of
                  stream, and also when all _SEEK_MAX_RETRIES retries still
                  land after `pts`: frames between `pts` and where the seek
                  landed may exist, so the landed frame is not handed out as
                  if it were the first one (a warning is logged). After
                  False, read()/grab() fail until the next seek.
        """
        def choose(frames, from_stream_start):
            decoded_any = False
            for frame, frame_pts in frames:
                if not decoded_any and frame_pts > pts and not from_stream_start:
                    return self._LANDED_LATE
                decoded_any = True
                if frame_pts >= pts:
                    self._park(frame)
                    return True
            if decoded_any or from_stream_start:
                return False  # end of stream before reaching the target
            return self._LANDED_LATE  # the seek decoded nothing

        return self._seek_with_retries(int(round(pts / self.stream.time_base)), choose,
                                       f"seek_to_pts({pts!r})")

    def seek_to_display_time(self, t):
        """Position the capture so the next read()/grab() returns the frame
        on screen at time `t`: the last frame whose PTS -- as get_last_pts()
        reports it -- is at most `t` + DISPLAY_TIME_TOLERANCE, or the first
        frame when `t` precedes it. Reading on continues with the frames
        after it.

        This is what a scan that records a decision under a time must read.
        seek_to_pts() instead returns the first frame at or after a time,
        which for a time between two frames is the one not yet shown. And
        set(CAP_PROP_POS_FRAMES, int(t * fps)) matches neither: it truncates
        (int(150.2 * 25) == 3754), maps positions back through
        round(PTS * fps) (one frame early throughout a file whose first frame
        is 0.525 frame in), never retries a seek that lands late, and does
        not move at all for position 0. This always seeks.

        It decodes up to the first frame past `t`, and keeps that one for the
        read after next. If the first frame decoded is already past `t` or
        the seek yields no frame, it retries from further back exactly as
        seek_to_pts() does.

        Returns:
            bool: True if a frame is on screen at `t`. False when `t` is
                  past the last frame (its PTS plus one frame at the nominal
                  rate), and when all _SEEK_MAX_RETRIES retries still land
                  after `t` (a warning is logged). After False, read()/grab()
                  fail until the next seek.
        """
        limit = t + DISPLAY_TIME_TOLERANCE

        def choose(frames, from_stream_start):
            on_screen = on_screen_pts = after = None
            for frame, frame_pts in frames:
                if frame_pts > limit:
                    after = frame
                    break
                on_screen, on_screen_pts = frame, frame_pts
            if on_screen is None:
                if after is not None and from_stream_start:
                    self._park(after)  # `t` precedes the first frame
                    return True
                # Landed after `t`, or decoded nothing (which, from the start
                # of the stream, means there is nothing to decode).
                return False if from_stream_start else self._LANDED_LATE
            if after is None and limit >= on_screen_pts + 1.0 / self._fps:
                return False  # past the last frame
            self._park(on_screen, after)
            return True

        return self._seek_with_retries(math.floor(limit / self.stream.time_base), choose,
                                       f"seek_to_display_time({t!r})")

    # What a seek's frame chooser returns when the seek landed after its
    # target (or decoded nothing), so _seek_with_retries seeks from further
    # back.
    _LANDED_LATE = object()

    def _seek_with_retries(self, target_ts, choose, description):
        """The seek loop seek_to_pts() and seek_to_display_time() share.

        Seeks to `target_ts` (in the stream's time base) and hands
        `choose(frames, from_stream_start)` the decoded frames, as
        (frame, float(frame.pts * time_base)) pairs, skipping frames without a
        PTS. `choose` parks the frame it wants and returns True, returns False
        when there is none, or returns _LANDED_LATE. Then this seeks again from
        _SEEK_BACKOFF_SECONDS further back, then twice that, and so on. A seek
        that reached the start of the stream is never retried (`choose` is
        told, since what it decodes first is the first frame). When all
        _SEEK_MAX_RETRIES retries still land late, it logs a warning naming
        `description` and reports no frame. After False, read()/grab() fail
        until the next seek, and seek_past_end tells the two apart.
        """
        self._read_started = True
        self.seek_past_end = False
        time_base = self.stream.time_base
        stream_start = self.stream.start_time if self.stream.start_time is not None else 0
        backoff = max(1, int(round(self._SEEK_BACKOFF_SECONDS / time_base)))
        seek_ts = max(target_ts, stream_start)
        for attempt in range(self._SEEK_MAX_RETRIES + 1):
            self.container.seek(seek_ts, stream=self.stream)
            self._frame_generator = self.container.decode(video=0)
            self._pending_frame = None
            from_stream_start = seek_ts <= stream_start
            frames = ((frame, float(frame.pts * time_base))
                      for frame in self._frame_generator if frame.pts is not None)
            outcome = choose(frames, from_stream_start)
            if outcome is True:
                return True
            if outcome is False:
                return self._no_frame()
            if attempt == self._SEEK_MAX_RETRIES:
                return self._no_frame(description)
            seek_ts = max(target_ts - backoff, stream_start)
            backoff *= 2

    def _park(self, frame, following=None):
        """Make a frame a seek decoded the next one read()/grab() returns,
        followed by `following` (also already decoded) if given, then the
        rest of the stream."""
        self._pending_frame = frame
        if following is not None:
            self._frame_generator = itertools.chain((following,), self._frame_generator)
        self._pos = int(round(float(frame.pts * self.stream.time_base) * self._fps))

    def _no_frame(self, retries_exhausted_by=None):
        """A seek's False: leave nothing for read()/grab() to return until
        the next seek, rather than whatever frame the seek stopped at."""
        self.seek_past_end = retries_exhausted_by is None
        if retries_exhausted_by is not None:
            logger.warning(
                "%s on %s: every seek, including %d retries from further back, "
                "landed after the target; reporting no frame rather than a later one",
                retries_exhausted_by, self.path, self._SEEK_MAX_RETRIES)
        self._pending_frame = None
        self._frame_generator = iter(())
        return False

    def read(self):
        """Read next frame (compatible with cv2.VideoCapture.read).

        Returns:
            tuple: (success, frame) where frame is BGR numpy array
                   (downscaled if decode_target_height was set)
        """
        self._read_started = True
        if not PYAV_AVAILABLE:
            ret, frame = self.cap.read()
            if ret:
                # OpenCV fallback: estimate PTS from frame position
                self._last_pts = self._pos / self._fps if self._fps else None
            return ret, frame

        try:
            frame = self._next_decoded()

            # Apply filter graph (tone mapping + scaling) if set up
            if self._filter_graph is not None:
                self._filter_graph.vpush(frame)
                frame = self._filter_graph.vpull()
                img = frame.to_ndarray()
            else:
                img = frame.to_ndarray(format='bgr24')

            # `_crop_slice` is only ever set once the graph has actually
            # cropped: _plan_crop refuses unless a graph is needed anyway,
            # and a graph that fails to build aborts __enter__ outright
            # rather than leaving a plan behind. So there is no "planned but
            # inactive" case to special-case here.
            if self._crop_slice is not None:
                y0, y1, x0, x1 = self._crop_slice
                img = img[y0:y1, x0:x1]
                img = np.ascontiguousarray(img)

            return True, img
        except StopIteration:
            return False, None
        except Exception:
            return False, None

    def grab(self):
        """Advance past the next frame without converting it
        (compatible with cv2.VideoCapture.grab).

        The frame is still decoded -- later frames may reference it -- and
        position/PTS advance exactly as in read(), but the frame is neither
        pushed through the filter graph nor converted to BGR. For a 10-bit
        4K source that conversion is most of what read() costs the calling
        thread, so a sequential scan that only looks at every Nth frame
        should grab() the others. The frames it does read() are
        byte-identical to reading every frame: neither the BGR conversion
        nor any graph this class builds (tone map, downscale, crop) carries
        state from one frame to the next, which tests/test_label_sampling.py
        checks for each graph configuration.

        Returns:
            bool: True if a frame was consumed, False at end of stream.
        """
        self._read_started = True
        if not PYAV_AVAILABLE:
            return self.cap.grab()

        try:
            self._next_decoded()
            return True
        except StopIteration:
            return False
        except Exception:
            return False

    def _next_decoded(self):
        """Consume the next decoded frame -- the one a seek parked, if any --
        and advance position/PTS to it. Shared by read() and grab() so the
        two can never disagree about which frame comes next.

        Raises StopIteration at end of stream.
        """
        # Check if we have a pending frame from seek
        if getattr(self, '_pending_frame', None) is not None:
            frame = self._pending_frame
            self._pending_frame = None
        else:
            frame = next(self._frame_generator)

        # Store canonical PTS timestamp in seconds (ground truth for timing)
        self._last_pts = float(frame.pts * self.stream.time_base)

        # Calculate frame position from PTS (for compatibility)
        self._pos = int(round(self._last_pts * self._fps))
        return frame

    def get_last_pts(self) -> float:
        """Get the PTS (presentation timestamp) of the last read frame in seconds.

        This is the canonical timing source for subtitle synchronization.
        Returns None if no frame has been read yet.
        """
        return self._last_pts

    def get_stream_start_time(self) -> float:
        """The container's start_time in seconds: time 0 of the output.

        Subtitle times are frame PTS minus this, so every line lands on the
        frame it was read from *as the player paints it*. A player's zero is
        the container start -- libavformat's format start_time, the earliest
        of all the streams -- which is what mpv rebases to under its default
        --rebase-start-time=yes. The video stream's start_time is NOT that
        zero: it merely says when the first frame arrives, and subtracting it
        shifts every line earlier by (video start - container start).

        In "Tales of Demon and Gods - 173 [4K].mkv" the audio and the
        container start at 0.015999 s while the first frame starts at
        0.100994 s, and counting from the frame put every line 85 ms -- two
        frames -- early: rendered through mpv, the line was on screen two
        frames before the hardsub it was read from. Counting from the
        container puts them on the same frame. For MKV this is typically 0;
        for MP4 it can be non-zero due to edit lists.
        """
        if not PYAV_AVAILABLE or self.container is None:
            return 0.0
        if self.container.start_time is not None:
            return max(0.0, self.container.start_time / 1_000_000)  # microseconds -> seconds
        return 0.0


# Use PyAV as default (fastest in practice), fall back to OpenCV
# Note: FFmpegNVDECCapture has fast hardware decode but pipe overhead makes it slower
if PYAV_AVAILABLE:
    Capture = PyAVCapture
elif FFMPEG_AVAILABLE:
    Capture = FFmpegNVDECCapture
else:
    # Fallback to OpenCV wrapper
    class OpenCVCapture:
        def __init__(self, video_path, use_gpu=True, decode_target_height=None, crop_rect=None):
            self.path = video_path
            self._decode_target_height = decode_target_height
            self._crop_request = crop_rect  # Accepted for interface parity; ignored.
            self._crop_slice = None  # Never set: callers keep slicing in Python.
            self._last_pts = None
            self._scale_factor = 1.0
            self._output_width = None
            self._output_height = None
            self.seek_past_end = False  # See PyAVCapture.seek_past_end.
        def __enter__(self):
            self.cap = cv2.VideoCapture(self.path)
            if not self.cap.isOpened():
                raise IOError(f'Cannot open video {self.path}')
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            if (self._decode_target_height is not None
                    and h > self._decode_target_height):
                self._scale_factor = self._decode_target_height / h
                self._output_width = int(w * self._scale_factor)
                self._output_width += self._output_width % 2
                self._output_height = self._decode_target_height
            else:
                self._output_width = w
                self._output_height = h
            return self
        def configure_crop(self, crop_rect) -> None:
            """Interface parity with PyAVCapture; this backend never crops
            in its decode graph. Callers can call this unconditionally
            regardless of backend."""
            self._crop_request = crop_rect
        def __exit__(self, exc_type, exc_value, traceback):
            self.cap.release()
        def get(self, prop):
            return self.cap.get(prop)
        def set(self, prop, value):
            return self.cap.set(prop, value)
        def read(self):
            ret, frame = self.cap.read()
            if ret:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self._last_pts = pos / fps if fps else None
                if self._scale_factor < 1.0:
                    frame = cv2.resize(frame,
                                       (self._output_width, self._output_height),
                                       interpolation=cv2.INTER_AREA)
            return ret, frame
        def grab(self):
            """Interface parity with PyAVCapture.grab: advance one frame
            without retrieving or resizing it. read() derives PTS from the
            stream position, so the next read() is unaffected."""
            return self.cap.grab()
        def seek_to_pts(self, pts):
            """Interface parity with PyAVCapture.seek_to_pts. read() reports
            PTS as the post-read position / fps, i.e. (ordinal + 1) / fps,
            so the frame reported as `pts` is ordinal ceil(pts * fps) - 1."""
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            ordinal = max(0, math.ceil(pts * fps - 1e-6) - 1) if fps else 0
            self.seek_past_end = False
            return self.cap.set(cv2.CAP_PROP_POS_FRAMES, ordinal)
        def seek_to_display_time(self, t):
            """Interface parity with PyAVCapture.seek_to_display_time. read()
            reports PTS as (ordinal + 1) / fps, so the frame on screen at `t`
            is the last ordinal whose reported PTS is at most
            `t` + DISPLAY_TIME_TOLERANCE, or ordinal 0 when `t` precedes it.
            Returns False when that ordinal is at or past the frame count
            OpenCV reports (seek_past_end is then True), leaving the capture
            at the end so read() fails."""
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            ordinal = 0
            if fps:
                limit = t + DISPLAY_TIME_TOLERANCE
                ordinal = max(0, math.floor(limit * fps) - 1)
                # Settle float rounding against read()'s own PTS expression.
                while ordinal > 0 and (ordinal + 1) / fps > limit:
                    ordinal -= 1
                while (ordinal + 2) / fps <= limit:
                    ordinal += 1
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.seek_past_end = frame_count > 0 and ordinal >= frame_count
            if self.seek_past_end:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                return False
            return self.cap.set(cv2.CAP_PROP_POS_FRAMES, ordinal)
        def get_last_pts(self) -> float:
            return self._last_pts
        def get_stream_start_time(self) -> float:
            return 0.0  # OpenCV doesn't expose start_time
    Capture = OpenCVCapture


ALLOW_FALLBACK_ENV = "OCR_ALLOW_FFMPEG_FALLBACK"


def capture_backend_name() -> str:
    """Name of the capture backend actually in use."""
    if Capture is PyAVCapture:
        return "pyav"
    if FFMPEG_AVAILABLE and Capture is FFmpegNVDECCapture:
        return "ffmpeg"
    return "opencv"


def assert_reference_backend() -> None:
    """Raise unless the bit-exact reference backend (PyAV) is in use.

    The ffmpeg-subprocess fallback is not bit-exact and estimates timestamps,
    so it must never be used for OCR without an explicit opt-in.
    """
    import os
    if PYAV_AVAILABLE:
        return
    if os.environ.get(ALLOW_FALLBACK_ENV) == "1":
        return
    raise RuntimeError(
        "PyAV is unavailable, so video would be decoded by the non-bit-exact "
        f"fallback backend. Fix with: .venv/bin/pip install -U --only-binary=:all: av\n"
        f"Import error was: {PYAV_IMPORT_ERROR}\n"
        f"To proceed anyway (output will differ), set {ALLOW_FALLBACK_ENV}=1."
    )
