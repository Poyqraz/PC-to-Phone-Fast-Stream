"""Adaptive streaming control algorithm for MJPEG screen sharing.

Provides intelligent frame-rate, quality, and resolution management
based on real-time motion detection, bandwidth estimation, and
configurable parameter bounds.
"""

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class StreamConfig:
    """User-configurable bounds and defaults for the control algorithm."""

    min_fps: float = 1.0
    max_fps: float = 30.0
    default_fps: float = 11.0

    min_quality: int = 10
    max_quality: int = 95
    default_quality: int = 70

    min_width: int = 320
    max_width: int = 3840
    default_width: int = 1280

    min_height: int = 240
    max_height: int = 2160
    default_height: int = 920

    motion_threshold: float = 0.02
    motion_high_threshold: float = 0.10

    bandwidth_window: int = 30
    quality_step: int = 5
    fps_step: float = 1.0

    target_bandwidth_kbps: float = 5000.0

    def validate(self) -> list[str]:
        """Return a list of validation error messages (empty if valid)."""
        errors = []
        if self.min_fps <= 0 or self.min_fps > self.max_fps:
            errors.append(f"Invalid FPS range: [{self.min_fps}, {self.max_fps}]")
        if not (1 <= self.min_quality <= self.max_quality <= 100):
            errors.append(f"Invalid quality range: [{self.min_quality}, {self.max_quality}]")
        if self.min_width <= 0 or self.min_width > self.max_width:
            errors.append(f"Invalid width range: [{self.min_width}, {self.max_width}]")
        if self.min_height <= 0 or self.min_height > self.max_height:
            errors.append(f"Invalid height range: [{self.min_height}, {self.max_height}]")
        if self.default_fps < self.min_fps or self.default_fps > self.max_fps:
            errors.append(f"Default FPS {self.default_fps} outside range")
        if self.default_quality < self.min_quality or self.default_quality > self.max_quality:
            errors.append(f"Default quality {self.default_quality} outside range")
        return errors


@dataclass
class StreamMetrics:
    """Real-time telemetry from the streaming pipeline."""

    current_fps: float = 0.0
    target_fps: float = 11.0
    current_quality: int = 70
    current_width: int = 1280
    current_height: int = 920
    motion_score: float = 0.0
    bandwidth_kbps: float = 0.0
    frame_count: int = 0
    avg_frame_size_bytes: float = 0.0
    avg_encode_ms: float = 0.0
    avg_capture_ms: float = 0.0
    dropped_frames: int = 0
    mode: str = "adaptive"

    def to_dict(self) -> dict:
        return {
            "current_fps": round(self.current_fps, 2),
            "target_fps": round(self.target_fps, 2),
            "current_quality": self.current_quality,
            "resolution": f"{self.current_width}x{self.current_height}",
            "motion_score": round(self.motion_score, 4),
            "bandwidth_kbps": round(self.bandwidth_kbps, 2),
            "frame_count": self.frame_count,
            "avg_frame_size_bytes": round(self.avg_frame_size_bytes, 1),
            "avg_encode_ms": round(self.avg_encode_ms, 2),
            "avg_capture_ms": round(self.avg_capture_ms, 2),
            "dropped_frames": self.dropped_frames,
            "mode": self.mode,
        }


@dataclass
class _BandwidthSample:
    timestamp: float
    size_bytes: int


@dataclass
class _TimingSample:
    encode_ms: float
    capture_ms: float


class StreamController:
    """Adaptive control algorithm for MJPEG streaming.

    Operating modes:
      - "adaptive": algorithm adjusts FPS, quality, and resolution
        automatically based on motion detection and bandwidth.
      - "manual": all parameters are locked to user-set values.

    The control loop runs each frame via `process_frame()` which:
      1. Detects motion via structural similarity (absolute diff + threshold).
      2. Estimates bandwidth from recent frame sizes.
      3. Adjusts quality and FPS to stay within bandwidth target.
      4. Returns the encoded JPEG bytes and the computed frame delay.
    """

    def __init__(self, config: StreamConfig | None = None):
        self._config = config or StreamConfig()
        errors = self._config.validate()
        if errors:
            raise ValueError(f"Invalid StreamConfig: {'; '.join(errors)}")

        self._metrics = StreamMetrics(
            target_fps=self._config.default_fps,
            current_quality=self._config.default_quality,
            current_width=self._config.default_width,
            current_height=self._config.default_height,
            mode="adaptive",
        )

        self._lock = threading.Lock()
        self._prev_gray: np.ndarray | None = None
        self._bandwidth_samples: list[_BandwidthSample] = []
        self._timing_samples: list[_TimingSample] = []
        self._frame_sizes: list[int] = []
        self._last_frame_time: float = 0.0
        self._fps_measurement_times: list[float] = []

    @property
    def config(self) -> StreamConfig:
        return self._config

    def get_metrics(self) -> StreamMetrics:
        with self._lock:
            return StreamMetrics(**self._metrics.__dict__)

    def set_mode(self, mode: str) -> None:
        if mode not in ("adaptive", "manual"):
            raise ValueError(f"Unknown mode: {mode}")
        with self._lock:
            self._metrics.mode = mode

    def set_manual_params(
        self,
        fps: float | None = None,
        quality: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> dict:
        """Set manual override parameters. Returns the applied values."""
        cfg = self._config
        applied = {}
        with self._lock:
            if fps is not None:
                self._metrics.target_fps = max(cfg.min_fps, min(cfg.max_fps, fps))
                applied["fps"] = self._metrics.target_fps
            if quality is not None:
                self._metrics.current_quality = max(cfg.min_quality, min(cfg.max_quality, quality))
                applied["quality"] = self._metrics.current_quality
            if width is not None:
                self._metrics.current_width = max(cfg.min_width, min(cfg.max_width, width))
                applied["width"] = self._metrics.current_width
            if height is not None:
                self._metrics.current_height = max(cfg.min_height, min(cfg.max_height, height))
                applied["height"] = self._metrics.current_height
        return applied

    def compute_motion_score(self, current_gray: np.ndarray) -> float:
        """Compute motion score between current and previous grayscale frames.

        Returns a float in [0, 1] representing the fraction of pixels
        that changed beyond a fixed intensity threshold.
        """
        if self._prev_gray is None or self._prev_gray.shape != current_gray.shape:
            return 0.0

        diff = cv2.absdiff(self._prev_gray, current_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        changed_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        return changed_pixels / total_pixels if total_pixels > 0 else 0.0

    def _estimate_bandwidth(self) -> float:
        """Estimate current bandwidth in kbps from recent frame samples."""
        now = time.time()
        window = self._config.bandwidth_window
        self._bandwidth_samples = [
            s for s in self._bandwidth_samples if now - s.timestamp < window
        ]
        if len(self._bandwidth_samples) < 2:
            return 0.0

        total_bytes = sum(s.size_bytes for s in self._bandwidth_samples)
        duration = self._bandwidth_samples[-1].timestamp - self._bandwidth_samples[0].timestamp
        if duration <= 0:
            return 0.0
        return (total_bytes * 8) / (duration * 1000)

    def _adapt_quality(self, motion_score: float, bandwidth_kbps: float) -> None:
        """Adjust JPEG quality based on motion and bandwidth."""
        cfg = self._config
        q = self._metrics.current_quality
        target_bw = cfg.target_bandwidth_kbps

        if bandwidth_kbps > target_bw * 1.2:
            q -= cfg.quality_step
        elif bandwidth_kbps < target_bw * 0.5:
            q += cfg.quality_step
        elif motion_score > cfg.motion_high_threshold:
            q -= cfg.quality_step
        elif motion_score < cfg.motion_threshold:
            q += cfg.quality_step // 2

        self._metrics.current_quality = max(cfg.min_quality, min(cfg.max_quality, q))

    def _adapt_fps(self, motion_score: float, bandwidth_kbps: float) -> None:
        """Adjust target FPS based on motion and bandwidth."""
        cfg = self._config
        fps = self._metrics.target_fps
        target_bw = cfg.target_bandwidth_kbps

        if motion_score > cfg.motion_high_threshold:
            fps += cfg.fps_step
        elif motion_score < cfg.motion_threshold:
            fps -= cfg.fps_step * 0.5

        if bandwidth_kbps > target_bw * 1.3:
            fps -= cfg.fps_step

        self._metrics.target_fps = max(cfg.min_fps, min(cfg.max_fps, fps))

    def _update_fps_measurement(self, now: float) -> None:
        """Track actual delivered FPS over a sliding 2-second window."""
        self._fps_measurement_times.append(now)
        cutoff = now - 2.0
        self._fps_measurement_times = [
            t for t in self._fps_measurement_times if t > cutoff
        ]
        if len(self._fps_measurement_times) >= 2:
            span = self._fps_measurement_times[-1] - self._fps_measurement_times[0]
            if span > 0:
                self._metrics.current_fps = (len(self._fps_measurement_times) - 1) / span

    def process_frame(self, raw_bgra: np.ndarray) -> tuple[bytes, float]:
        """Run the full control pipeline on one captured frame.

        Args:
            raw_bgra: Raw BGRA frame from mss.grab().

        Returns:
            (jpeg_bytes, frame_delay_seconds)
        """
        now = time.time()
        capture_start = now

        with self._lock:
            w = self._metrics.current_width
            h = self._metrics.current_height
            quality = self._metrics.current_quality
            mode = self._metrics.mode

        bgr = cv2.cvtColor(raw_bgra, cv2.COLOR_BGRA2BGR)
        resized = cv2.resize(bgr, (w, h))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        capture_ms = (time.time() - capture_start) * 1000

        motion_score = self.compute_motion_score(gray)
        self._prev_gray = gray.copy()

        encode_start = time.time()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode(".jpg", resized, encode_param)
        jpeg_bytes = buffer.tobytes()
        encode_ms = (time.time() - encode_start) * 1000

        frame_size = len(jpeg_bytes)

        with self._lock:
            self._bandwidth_samples.append(_BandwidthSample(now, frame_size))
            bandwidth_kbps = self._estimate_bandwidth()

            if mode == "adaptive":
                self._adapt_quality(motion_score, bandwidth_kbps)
                self._adapt_fps(motion_score, bandwidth_kbps)

            self._timing_samples.append(_TimingSample(encode_ms, capture_ms))
            if len(self._timing_samples) > 60:
                self._timing_samples = self._timing_samples[-60:]

            self._frame_sizes.append(frame_size)
            if len(self._frame_sizes) > 60:
                self._frame_sizes = self._frame_sizes[-60:]

            self._metrics.motion_score = motion_score
            self._metrics.bandwidth_kbps = bandwidth_kbps
            self._metrics.frame_count += 1
            self._metrics.avg_frame_size_bytes = (
                sum(self._frame_sizes) / len(self._frame_sizes)
            )
            self._metrics.avg_encode_ms = (
                sum(s.encode_ms for s in self._timing_samples) / len(self._timing_samples)
            )
            self._metrics.avg_capture_ms = (
                sum(s.capture_ms for s in self._timing_samples) / len(self._timing_samples)
            )

            self._update_fps_measurement(now)
            frame_delay = 1.0 / self._metrics.target_fps

        return jpeg_bytes, frame_delay

    def reset(self) -> None:
        """Reset all internal state and metrics to defaults."""
        cfg = self._config
        with self._lock:
            self._prev_gray = None
            self._bandwidth_samples.clear()
            self._timing_samples.clear()
            self._frame_sizes.clear()
            self._fps_measurement_times.clear()
            self._last_frame_time = 0.0
            self._metrics = StreamMetrics(
                target_fps=cfg.default_fps,
                current_quality=cfg.default_quality,
                current_width=cfg.default_width,
                current_height=cfg.default_height,
                mode=self._metrics.mode,
            )
