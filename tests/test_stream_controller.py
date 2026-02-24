"""Unit tests for the StreamController adaptive control algorithm."""

import cv2
import numpy as np
import pytest

from stream_controller import StreamConfig, StreamController, StreamMetrics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return StreamConfig()


@pytest.fixture
def controller(config):
    return StreamController(config)


def _make_frame(width=1920, height=1080, color=(0, 0, 0, 255)):
    """Create a synthetic BGRA frame (like mss.grab output)."""
    frame = np.zeros((height, width, 4), dtype=np.uint8)
    frame[:, :] = color
    return frame


def _make_noisy_frame(width=1920, height=1080, seed=42):
    """Create a random BGRA frame to simulate high-motion content."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (height, width, 4), dtype=np.uint8)


# ---------------------------------------------------------------------------
# StreamConfig tests
# ---------------------------------------------------------------------------

class TestStreamConfig:
    def test_default_config_is_valid(self):
        cfg = StreamConfig()
        assert cfg.validate() == []

    def test_invalid_fps_range(self):
        cfg = StreamConfig(min_fps=30, max_fps=10)
        errors = cfg.validate()
        assert any("FPS" in e for e in errors)

    def test_invalid_quality_range(self):
        cfg = StreamConfig(min_quality=100, max_quality=10)
        errors = cfg.validate()
        assert any("quality" in e for e in errors)

    def test_invalid_width_range(self):
        cfg = StreamConfig(min_width=3840, max_width=320)
        errors = cfg.validate()
        assert any("width" in e for e in errors)

    def test_invalid_height_range(self):
        cfg = StreamConfig(min_height=2160, max_height=240)
        errors = cfg.validate()
        assert any("height" in e for e in errors)

    def test_default_fps_out_of_range(self):
        cfg = StreamConfig(min_fps=5, max_fps=10, default_fps=15)
        errors = cfg.validate()
        assert any("Default FPS" in e for e in errors)

    def test_default_quality_out_of_range(self):
        cfg = StreamConfig(min_quality=30, max_quality=80, default_quality=95)
        errors = cfg.validate()
        assert any("Default quality" in e for e in errors)

    def test_custom_valid_config(self):
        cfg = StreamConfig(min_fps=5, max_fps=25, default_fps=15, default_quality=60)
        assert cfg.validate() == []


# ---------------------------------------------------------------------------
# StreamController initialization tests
# ---------------------------------------------------------------------------

class TestStreamControllerInit:
    def test_default_initialization(self, controller):
        metrics = controller.get_metrics()
        assert metrics.target_fps == 11.0
        assert metrics.current_quality == 70
        assert metrics.current_width == 1280
        assert metrics.current_height == 920
        assert metrics.mode == "adaptive"
        assert metrics.frame_count == 0

    def test_custom_config_initialization(self):
        cfg = StreamConfig(default_fps=20, default_quality=85, default_width=640, default_height=480)
        ctrl = StreamController(cfg)
        metrics = ctrl.get_metrics()
        assert metrics.target_fps == 20.0
        assert metrics.current_quality == 85
        assert metrics.current_width == 640
        assert metrics.current_height == 480

    def test_invalid_config_raises(self):
        cfg = StreamConfig(min_fps=30, max_fps=10)
        with pytest.raises(ValueError, match="Invalid StreamConfig"):
            StreamController(cfg)


# ---------------------------------------------------------------------------
# StreamMetrics tests
# ---------------------------------------------------------------------------

class TestStreamMetrics:
    def test_to_dict_keys(self):
        m = StreamMetrics()
        d = m.to_dict()
        expected_keys = {
            "current_fps", "target_fps", "current_quality", "resolution",
            "motion_score", "bandwidth_kbps", "frame_count",
            "avg_frame_size_bytes", "avg_encode_ms", "avg_capture_ms",
            "dropped_frames", "mode",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_resolution_format(self):
        m = StreamMetrics(current_width=1920, current_height=1080)
        assert m.to_dict()["resolution"] == "1920x1080"

    def test_to_dict_rounding(self):
        m = StreamMetrics(current_fps=10.12345, motion_score=0.00123456)
        d = m.to_dict()
        assert d["current_fps"] == 10.12
        assert d["motion_score"] == 0.0012


# ---------------------------------------------------------------------------
# Mode switching tests
# ---------------------------------------------------------------------------

class TestModeSwitch:
    def test_set_manual_mode(self, controller):
        controller.set_mode("manual")
        assert controller.get_metrics().mode == "manual"

    def test_set_adaptive_mode(self, controller):
        controller.set_mode("manual")
        controller.set_mode("adaptive")
        assert controller.get_metrics().mode == "adaptive"

    def test_invalid_mode_raises(self, controller):
        with pytest.raises(ValueError, match="Unknown mode"):
            controller.set_mode("turbo")


# ---------------------------------------------------------------------------
# Manual parameter override tests
# ---------------------------------------------------------------------------

class TestManualParams:
    def test_set_fps(self, controller):
        applied = controller.set_manual_params(fps=20)
        assert applied["fps"] == 20
        assert controller.get_metrics().target_fps == 20

    def test_fps_clamped_to_max(self, controller):
        applied = controller.set_manual_params(fps=999)
        assert applied["fps"] == controller.config.max_fps

    def test_fps_clamped_to_min(self, controller):
        applied = controller.set_manual_params(fps=0.01)
        assert applied["fps"] == controller.config.min_fps

    def test_set_quality(self, controller):
        applied = controller.set_manual_params(quality=50)
        assert applied["quality"] == 50

    def test_quality_clamped(self, controller):
        applied = controller.set_manual_params(quality=200)
        assert applied["quality"] == controller.config.max_quality

    def test_set_resolution(self, controller):
        applied = controller.set_manual_params(width=800, height=600)
        assert applied["width"] == 800
        assert applied["height"] == 600

    def test_multiple_params(self, controller):
        applied = controller.set_manual_params(fps=15, quality=80, width=640, height=480)
        assert len(applied) == 4

    def test_no_params_noop(self, controller):
        applied = controller.set_manual_params()
        assert applied == {}


# ---------------------------------------------------------------------------
# Motion detection tests
# ---------------------------------------------------------------------------

class TestMotionDetection:
    def test_no_previous_frame(self, controller):
        gray = np.zeros((480, 640), dtype=np.uint8)
        score = controller.compute_motion_score(gray)
        assert score == 0.0

    def test_identical_frames_zero_motion(self, controller):
        gray = np.zeros((480, 640), dtype=np.uint8)
        controller._prev_gray = gray.copy()
        score = controller.compute_motion_score(gray)
        assert score == 0.0

    def test_different_frames_high_motion(self, controller):
        black = np.zeros((480, 640), dtype=np.uint8)
        white = np.full((480, 640), 255, dtype=np.uint8)
        controller._prev_gray = black
        score = controller.compute_motion_score(white)
        assert score > 0.9

    def test_partial_motion(self, controller):
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = frame1.copy()
        frame2[0:50, :] = 255  # top half changed
        controller._prev_gray = frame1
        score = controller.compute_motion_score(frame2)
        assert 0.3 < score < 0.7

    def test_shape_mismatch_returns_zero(self, controller):
        controller._prev_gray = np.zeros((480, 640), dtype=np.uint8)
        different_size = np.zeros((240, 320), dtype=np.uint8)
        score = controller.compute_motion_score(different_size)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Process frame integration tests
# ---------------------------------------------------------------------------

class TestProcessFrame:
    def test_returns_jpeg_bytes(self, controller):
        frame = _make_frame()
        jpeg, _delay = controller.process_frame(frame)
        assert isinstance(jpeg, bytes)
        assert len(jpeg) > 0
        assert jpeg[:2] == b"\xff\xd8"  # JPEG SOI marker

    def test_returns_positive_delay(self, controller):
        frame = _make_frame()
        _, delay = controller.process_frame(frame)
        assert delay > 0

    def test_increments_frame_count(self, controller):
        frame = _make_frame()
        controller.process_frame(frame)
        controller.process_frame(frame)
        assert controller.get_metrics().frame_count == 2

    def test_updates_avg_frame_size(self, controller):
        frame = _make_frame()
        controller.process_frame(frame)
        m = controller.get_metrics()
        assert m.avg_frame_size_bytes > 0

    def test_updates_encode_timing(self, controller):
        frame = _make_frame()
        controller.process_frame(frame)
        m = controller.get_metrics()
        assert m.avg_encode_ms >= 0

    def test_motion_detected_on_change(self, controller):
        black = _make_frame(color=(0, 0, 0, 255))
        white = _make_frame(color=(255, 255, 255, 255))
        controller.process_frame(black)
        controller.process_frame(white)
        m = controller.get_metrics()
        assert m.motion_score > 0.5

    def test_no_motion_on_static(self, controller):
        frame = _make_frame(color=(128, 128, 128, 255))
        controller.process_frame(frame)
        controller.process_frame(frame)
        m = controller.get_metrics()
        assert m.motion_score == 0.0

    def test_jpeg_decodable(self, controller):
        frame = _make_noisy_frame()
        jpeg, _ = controller.process_frame(frame)
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape[1] == controller.get_metrics().current_width
        assert decoded.shape[0] == controller.get_metrics().current_height


# ---------------------------------------------------------------------------
# Adaptive algorithm behavior tests
# ---------------------------------------------------------------------------

class TestAdaptiveAlgorithm:
    def test_quality_drops_on_high_motion(self):
        cfg = StreamConfig(default_quality=70)
        ctrl = StreamController(cfg)
        black = _make_frame(color=(0, 0, 0, 255))
        white = _make_frame(color=(255, 255, 255, 255))

        ctrl.process_frame(black)
        initial_q = ctrl.get_metrics().current_quality

        for i in range(5):
            f = white if i % 2 == 0 else black
            ctrl.process_frame(f)

        final_q = ctrl.get_metrics().current_quality
        assert final_q <= initial_q

    def test_fps_increases_on_high_motion(self):
        cfg = StreamConfig(default_fps=10)
        ctrl = StreamController(cfg)
        black = _make_frame(color=(0, 0, 0, 255))
        white = _make_frame(color=(255, 255, 255, 255))

        ctrl.process_frame(black)
        initial_fps = ctrl.get_metrics().target_fps

        for i in range(5):
            f = white if i % 2 == 0 else black
            ctrl.process_frame(f)

        final_fps = ctrl.get_metrics().target_fps
        assert final_fps >= initial_fps

    def test_manual_mode_skips_adaptation(self):
        ctrl = StreamController()
        ctrl.set_mode("manual")
        ctrl.set_manual_params(fps=15, quality=50)

        black = _make_frame(color=(0, 0, 0, 255))
        white = _make_frame(color=(255, 255, 255, 255))

        for i in range(10):
            f = white if i % 2 == 0 else black
            ctrl.process_frame(f)

        m = ctrl.get_metrics()
        assert m.target_fps == 15
        assert m.current_quality == 50


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_frame_count(self, controller):
        frame = _make_frame()
        controller.process_frame(frame)
        controller.process_frame(frame)
        controller.reset()
        assert controller.get_metrics().frame_count == 0

    def test_reset_restores_defaults(self, controller):
        controller.set_manual_params(fps=25, quality=90, width=640, height=480)
        controller.reset()
        m = controller.get_metrics()
        assert m.target_fps == controller.config.default_fps
        assert m.current_quality == controller.config.default_quality
        assert m.current_width == controller.config.default_width
        assert m.current_height == controller.config.default_height

    def test_reset_preserves_mode(self, controller):
        controller.set_mode("manual")
        controller.reset()
        assert controller.get_metrics().mode == "manual"

    def test_reset_clears_motion(self, controller):
        black = _make_frame(color=(0, 0, 0, 255))
        white = _make_frame(color=(255, 255, 255, 255))
        controller.process_frame(black)
        controller.process_frame(white)
        controller.reset()
        assert controller.get_metrics().motion_score == 0.0
