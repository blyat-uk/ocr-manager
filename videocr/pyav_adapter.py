"""Video capture adapters with optional GPU acceleration."""
import cv2
import numpy as np
import subprocess
import json
import shutil

# Try PyAV first (FFmpeg bindings with hardware acceleration support)
try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False

# Check if FFmpeg is available
FFMPEG_AVAILABLE = shutil.which('ffmpeg') is not None and shutil.which('ffprobe') is not None

# HDR transfer characteristic constants (from ITU-T H.273)
_TRC_SMPTE2084 = 16   # PQ (HDR10)
_TRC_ARIB_STD_B67 = 18  # HLG

_ZSCALE_AVAILABLE = None  # Lazy-cached

# Downscale 4K+ to 1080p at decode level for performance.
# Frames arrive pre-scaled so the OCR pipeline processes the same pixel count
# regardless of source resolution.  get(CAP_PROP_FRAME_*) still returns native
# dimensions; only read() produces scaled frames.
DECODE_TARGET_HEIGHT = 1080


def _has_zscale() -> bool:
    """Check if system FFmpeg has zscale filter (requires zimg)."""
    global _ZSCALE_AVAILABLE
    if _ZSCALE_AVAILABLE is None:
        try:
            result = subprocess.run(
                ['ffmpeg', '-filters'],
                capture_output=True, text=True, timeout=5
            )
            _ZSCALE_AVAILABLE = 'zscale' in result.stdout
        except Exception:
            _ZSCALE_AVAILABLE = False
    return _ZSCALE_AVAILABLE


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

    def __init__(self, video_path, use_gpu=True, decode_target_height=None):
        self.path = video_path
        self.use_gpu = use_gpu
        self._decode_target_height = decode_target_height
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

    def _probe_video(self):
        """Get video metadata using ffprobe."""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            self.path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
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
            bufsize=self._frame_size * 10  # Buffer 10 frames
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
        return self

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
            return self._pos
        return 0

    def set(self, prop, value):
        """Set video property (compatible with cv2.VideoCapture.set)."""
        if not FFMPEG_AVAILABLE:
            return self.cap.set(prop, value)

        if prop == cv2.CAP_PROP_POS_FRAMES:
            target_pos = int(value)
            if target_pos != self._pos and target_pos > 0:
                # Restart FFmpeg with seek
                if self.proc:
                    self.proc.stdout.close()
                    self.proc.terminate()
                    self.proc.wait()
                seek_time = target_pos / self._fps
                self._start_ffmpeg(seek_time)
                self._pos = target_pos
            return True
        return False

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

        try:
            raw = self.proc.stdout.read(self._frame_size)
            if len(raw) != self._frame_size:
                return False, None
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(self._output_height, self._output_width, 3)
            self._pos += 1
            # Estimate PTS from frame position (FFmpeg subprocess doesn't expose true PTS)
            self._last_pts = self._pos / self._fps if self._fps else None
            return True, frame
        except Exception:
            return False, None

    def get_last_pts(self) -> float:
        """Get estimated PTS of the last read frame in seconds."""
        return self._last_pts

    def get_scale_factor(self):
        """Get the downscaling factor (1.0 = no scaling, <1.0 = downscaled)."""
        return self._scale_factor

    def get_stream_start_time(self) -> float:
        """Get the stream's start_time (estimated as 0 for FFmpeg pipe)."""
        # FFmpeg subprocess doesn't expose start_time directly
        # Return 0 as we can't reliably get this info through the pipe
        return 0.0


class PyAVCapture:
    """Video capture using PyAV with optional CUDA hardware acceleration.

    Uses FFmpeg's NVDEC for hardware-accelerated video decoding on NVIDIA GPUs.
    """

    def __init__(self, video_path, use_gpu=True, decode_target_height=None):
        self.path = video_path
        self.use_gpu = use_gpu
        self._decode_target_height = decode_target_height
        self.container = None
        self.stream = None
        self._pos = 0
        self._frame_count = None
        self._fps = None
        self._width = None
        self._height = None
        self._scale_factor = 1.0
        self._output_width = None
        self._output_height = None
        # PTS tracking - canonical timestamp source
        self._last_pts = None  # Last frame's PTS in seconds
        # Combined filter graph for tone mapping and/or scaling (None = fast path)
        self._filter_graph = None

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
        self._frame_generator = self.container.decode(video=0)

        # Set up combined filter graph (tone mapping + scaling)
        self._setup_filter_graph()

        return self

    def _setup_filter_graph(self):
        """Build a unified PyAV filter graph for HDR tone mapping and/or downscaling."""
        needs_tonemap = False
        trc = None
        try:
            trc = self.stream.codec_context.color_trc
            needs_tonemap = trc in (_TRC_SMPTE2084, _TRC_ARIB_STD_B67)
        except Exception:
            pass

        needs_scale = self._scale_factor < 1.0

        if not needs_tonemap and not needs_scale:
            self._filter_graph = None
            return

        try:
            graph = av.filter.Graph()
            buf = graph.add_buffer(template=self.stream)
            last = buf

            if needs_tonemap:
                zscale_linear_args = 't=linear:npl=100' if trc == _TRC_SMPTE2084 else 't=linear'
                linearize = graph.add('zscale', zscale_linear_args)
                fmt_in = graph.add('format', 'gbrpf32le')
                tonemap = graph.add('tonemap', 'hable')
                bt709 = graph.add('zscale', 't=bt709')
                last.link_to(linearize)
                linearize.link_to(fmt_in)
                fmt_in.link_to(tonemap)
                tonemap.link_to(bt709)
                last = bt709

            # Scale after tone mapping (operates on uint8 bgr24 for efficiency)
            fmt_out = graph.add('format', 'bgr24')
            last.link_to(fmt_out)
            last = fmt_out

            if needs_scale:
                scale = graph.add('scale', f'{self._output_width}:{self._output_height}')
                last.link_to(scale)
                last = scale

            sink = graph.add('buffersink')
            last.link_to(sink)

            graph.configure()
            self._filter_graph = graph
        except Exception:
            # Graceful fallback: proceed without filtering
            self._filter_graph = None
            self._scale_factor = 1.0

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

    def get_scale_factor(self):
        """Get the downscaling factor (1.0 = no scaling, <1.0 = downscaled)."""
        if not PYAV_AVAILABLE:
            return 1.0
        return self._scale_factor

    def set(self, prop, value):
        """Set video property (compatible with cv2.VideoCapture.set)."""
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

    def read(self):
        """Read next frame (compatible with cv2.VideoCapture.read).

        Returns:
            tuple: (success, frame) where frame is BGR numpy array
                   (downscaled if decode_target_height was set)
        """
        if not PYAV_AVAILABLE:
            ret, frame = self.cap.read()
            if ret:
                # OpenCV fallback: estimate PTS from frame position
                self._last_pts = self._pos / self._fps if self._fps else None
            return ret, frame

        try:
            # Check if we have a pending frame from seek
            if hasattr(self, '_pending_frame') and self._pending_frame is not None:
                frame = self._pending_frame
                self._pending_frame = None
            else:
                frame = next(self._frame_generator)

            # Store canonical PTS timestamp in seconds (ground truth for timing)
            self._last_pts = float(frame.pts * self.stream.time_base)

            # Calculate frame position from PTS (for compatibility)
            self._pos = int(round(self._last_pts * self._fps))

            # Apply filter graph (tone mapping + scaling) if set up
            if self._filter_graph is not None:
                self._filter_graph.vpush(frame)
                frame = self._filter_graph.vpull()
                img = frame.to_ndarray()
            else:
                img = frame.to_ndarray(format='bgr24')

            return True, img
        except StopIteration:
            return False, None
        except Exception:
            return False, None

    def get_last_pts(self) -> float:
        """Get the PTS (presentation timestamp) of the last read frame in seconds.

        This is the canonical timing source for subtitle synchronization.
        Returns None if no frame has been read yet.
        """
        return self._last_pts

    def get_stream_start_time(self) -> float:
        """Get the container-level start_time in seconds.

        Players use PTS values directly for playback, offset only by the
        container start_time (not the stream start_time). For MKV this is
        typically 0; for MP4 it can be non-zero due to edit lists.
        Stream start_time merely indicates when the first sample appears
        and is NOT a playback offset to subtract.
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
        def __init__(self, video_path, use_gpu=True, decode_target_height=None):
            self.path = video_path
            self._decode_target_height = decode_target_height
            self._last_pts = None
            self._scale_factor = 1.0
            self._output_width = None
            self._output_height = None
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
        def get_last_pts(self) -> float:
            return self._last_pts
        def get_scale_factor(self):
            return self._scale_factor
        def get_stream_start_time(self) -> float:
            return 0.0  # OpenCV doesn't expose start_time
    Capture = OpenCVCapture
