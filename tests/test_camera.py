"""Tests for camera availability and frame capture.

These tests require a physical webcam and macOS camera permission granted
to the terminal app (System Settings → Privacy & Security → Camera).
Tests are automatically skipped if no camera is available.
"""

import numpy as np
import pytest


def _open_camera():
    """Try to open the camera, skip test if unavailable."""
    from pronun.workflow.camera import Camera
    cam = Camera()
    try:
        cam.open()
    except RuntimeError as e:
        pytest.skip(f"Camera not available: {e}")
    return cam


def test_camera_opens():
    """Camera can be opened and closed without error."""
    cam = _open_camera()
    assert cam.is_opened()
    cam.close()
    assert not cam.is_opened()


def test_camera_returns_frame():
    """Camera returns a non-None frame on read."""
    cam = _open_camera()
    try:
        frame = cam.read_frame()
        assert frame is not None, (
            "Camera opened but read_frame() returned None — "
            "check macOS camera permission for Terminal in "
            "System Settings → Privacy & Security → Camera"
        )
    finally:
        cam.close()


def test_camera_frame_is_rgb():
    """Frame is a uint8 RGB numpy array with 3 channels."""
    cam = _open_camera()
    try:
        frame = cam.read_frame()
        if frame is None:
            pytest.skip("Camera returned no frame")
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3, f"Expected (H, W, 3), got shape {frame.shape}"
        assert frame.shape[2] == 3, f"Expected 3 channels (RGB), got {frame.shape[2]}"
        assert frame.dtype == np.uint8, f"Expected uint8, got {frame.dtype}"
    finally:
        cam.close()


def test_camera_frame_dimensions():
    """Frame has reasonable resolution (at least 320×240)."""
    cam = _open_camera()
    try:
        frame = cam.read_frame()
        if frame is None:
            pytest.skip("Camera returned no frame")
        h, w, _ = frame.shape
        assert w >= 320, f"Frame width too small: {w}px"
        assert h >= 240, f"Frame height too small: {h}px"
    finally:
        cam.close()


def test_camera_context_manager():
    """Camera works correctly as a context manager."""
    from pronun.workflow.camera import Camera
    try:
        with Camera() as cam:
            assert cam.is_opened()
            frame = cam.read_frame()
            assert frame is not None
    except RuntimeError:
        pytest.skip("Camera not available")
