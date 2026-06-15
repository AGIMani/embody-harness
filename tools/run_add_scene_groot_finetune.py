#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from functools import partial
import json
import math
import multiprocessing
import os
from pathlib import Path
import queue
import shutil
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
ISAAC_GROOT_ROOT = Path(os.environ.get("ISAAC_GROOT_ROOT", ROOT_DIR.parent / "Isaac-GR00T"))
DEFAULT_POLICY_CHECKPOINT = ROOT_DIR / "checkpoints" / "finetune" / "checkpoint-59000"
DEFAULT_COSMOS_MODEL = ROOT_DIR / "checkpoints" / "nvidia" / "Cosmos-Reason2-2B"
DEFAULT_TASK = "pick up the bottle with green cap and place it in the white rectangle area"
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_EGO_ROI_ZOOM = 2.0
DEFAULT_EGO_ROI_CENTER_X = 0.50
DEFAULT_EGO_ROI_CENTER_Y = 0.65
DEFAULT_BOTTLE_POS = (-0.016, 0.32889, 0.82667)
DEFAULT_BOTTLE_EULER = (0.0, 0.0, 0.0)
DEFAULT_SCENE_SUPPORT_COLLIDER_POS = (-0.541071, -0.112500, 0.678571)
DEFAULT_SCENE_SUPPORT_COLLIDER_SIZE = (0.700000, 0.700000, 0.040000)
DEFAULT_INITIAL_RIGHT_ARM_Q = (
    0.2724284429,
    1.6012174157,
    1.4535451076,
    1.2643514167,
    0.2993937799,
    -0.0534419817,
    0.1828232391,
)
DEFAULT_INITIAL_LEFT_ARM_Q = (
    -0.1575159650,
    1.6297011890,
    -1.4525677233,
    1.2456240339,
    0.0059690260,
    -0.0214850031,
    0.1212131166,
)
DEFAULT_INITIAL_HAND_Q = (
    0.072044,
    0.422158,
    0.136070,
    0.136070,
    0.136070,
    0.204105,
    0.034896,
    0.026172,
    0.062802,
    0.113390,
)
DEFAULT_INITIAL_REFERENCE_EEF_9D = (
    -0.382481151,
    0.157008573,
    0.614672903,
    -0.117517303,
    -0.277327377,
    -0.948701119,
    0.055898530,
    -0.955272723,
    0.272063020,
)
DEFAULT_INITIAL_REFERENCE_HAND_Q = (
    0.072044,
    0.422158,
    0.136070,
    0.136070,
    0.136070,
    0.204105,
    0.034896,
    0.026172,
    0.062802,
    0.113390,
)
POLICY_HAND_JOINT_NAMES = (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
    "index_mcp_roll",
    "ring_mcp_roll",
    "pinky_mcp_roll",
    "thumb_cmc_roll",
)
POLICY_HAND_CLIP_LOWER = -0.6
POLICY_HAND_CLIP_UPPER = 1.6
DEFAULT_MAX_HAND_JOINT_DELTA = 0.0
L10_HAND_ACTIVE_LOCAL_INDICES = (0, 2, 3, 4, 5, 9)
L10_HAND_MASKED_LOCAL_INDICES = (1, 6, 7, 8)
DEFAULT_IK_J4_LIMIT_RAD = (0.0, 2.14)
DEFAULT_POLICY_DEBUG_JSONL = ROOT_DIR / "logs" / "groot_finetune_policy_debug.jsonl"
DEFAULT_POLICY_TELEOP_DEBUG_JSONL = ROOT_DIR / "logs" / "groot_finetune_teleop_bridge.jsonl"
DEFAULT_POLICY_TRANSLATION_SCALE = (1.0, 1.0, 1.0)
DEFAULT_POLICY_ORIENTATION_MAX_SPEED_RAD_S = 8.0

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import add_scene_glb as harness  # noqa: E402
from teleop_stack.teleop.openxr_genesis_adapter import (  # noqa: E402
    adapter_debug_payload as canonical_adapter_debug_payload,
)
from teleop_stack.teleop.openxr_genesis_adapter import (  # noqa: E402
    map_openxr_quaternion_to_genesis_parent,
    map_openxr_vector_to_genesis,
)
from teleop_stack.teleop.orientation_tracker import OrientationTargetTracker, OrientationTrackerConfig  # noqa: E402
from teleop_stack.teleop.spatial_frames import matrix_to_quat_xyzw, quat_xyzw_to_matrix  # noqa: E402


def _vec3(text: str) -> tuple[float, float, float]:
    parts = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected numeric x,y,z") from exc


def _vec2(text: str) -> tuple[float, float]:
    parts = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected a,b")
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected numeric a,b") from exc


def _vec7(text: str) -> tuple[float, ...]:
    parts = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if len(parts) != 7:
        raise argparse.ArgumentTypeError("expected 7 comma-separated numbers")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected numeric comma-separated values") from exc


def _vec9(text: str) -> tuple[float, ...]:
    parts = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if len(parts) != 9:
        raise argparse.ArgumentTypeError("expected 9 comma-separated numbers")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected numeric comma-separated values") from exc


def _vec10(text: str) -> tuple[float, ...]:
    parts = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if len(parts) != 10:
        raise argparse.ArgumentTypeError("expected 10 comma-separated numbers")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected numeric comma-separated values") from exc


def _axis_map(text: str) -> tuple[str, str, str]:
    parts = tuple(part.strip().lower() for part in str(text).split(",") if part.strip())
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected 3 comma-separated axis tokens, e.g. x,y,z")
    for part in parts:
        axis = part[1:] if part.startswith(("-", "+")) else part
        if axis not in {"x", "y", "z"}:
            raise argparse.ArgumentTypeError(f"unsupported axis token: {part!r}")
    return parts  # type: ignore[return-value]


def _image_size(text: str) -> tuple[int, int]:
    parts = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected height,width, e.g. 224,224")
    try:
        height, width = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected integer height,width") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("height and width must be positive")
    return (height, width)


def _tensor_to_np(value: object) -> np.ndarray:
    return harness._tensor_to_np(value)  # noqa: SLF001


def _rotmat_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    # GR00T's Nero/L10 dataset stores rot6d as the first two rotation columns.
    # Keep this convention aligned with remote Teleop's groot_policy.py.
    return rotation[:, :2].reshape(6, order="F").astype(np.float32)


def _rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(6)
    col0 = rot6d[:3]
    col1 = rot6d[3:6]
    col0 = col0 / max(float(np.linalg.norm(col0)), 1e-12)
    col1 = col1 - float(np.dot(col0, col1)) * col0
    col1 = col1 / max(float(np.linalg.norm(col1)), 1e-12)
    col2 = np.cross(col0, col1)
    return np.column_stack([col0, col1, col2])


def _rotation_to_quat_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    quat_wxyz = harness._quat_wxyz_from_rotation(rotation)  # noqa: SLF001
    return (float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]), float(quat_wxyz[0]))


def _quat_xyzw_to_rotation(quat_xyzw: tuple[float, float, float, float]) -> np.ndarray:
    matrix = quat_xyzw_to_matrix(tuple(float(v) for v in quat_xyzw))
    return np.asarray(matrix, dtype=np.float64)


def _quat_wxyz_to_rotation(quat_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return _quat_xyzw_to_rotation((x, y, z, w))


def _rotation_to_quat_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    quat_xyzw = _rotation_to_quat_xyzw(rotation)
    return (float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    arr = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    return tuple(tuple(float(value) for value in row) for row in arr)


def _yaw_rotation_matrix(rad: float) -> np.ndarray:
    cos_v = math.cos(float(rad))
    sin_v = math.sin(float(rad))
    return np.asarray(
        (
            (cos_v, -sin_v, 0.0),
            (sin_v, cos_v, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _apply_yaw_to_quat_xyzw(
    quaternion_xyzw: tuple[float, float, float, float],
    yaw_rad: float | None,
) -> tuple[float, float, float, float]:
    if yaw_rad is None:
        return tuple(float(v) for v in quaternion_xyzw)  # type: ignore[return-value]
    rotation = _yaw_rotation_matrix(float(yaw_rad)) @ _quat_xyzw_to_rotation(quaternion_xyzw)
    return matrix_to_quat_xyzw(_matrix_tuple(rotation))  # type: ignore[return-value]


def _resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[-3:-1] == (height, width):
        return image.astype(np.uint8, copy=False)
    original_shape = image.shape
    image = image.reshape(-1, *original_shape[-3:])
    resized = [_resize_one_with_pad(Image.fromarray(frame), height, width) for frame in image]
    return np.stack(resized).reshape(*original_shape[:-3], height, width, original_shape[-1])


def _resize_one_with_pad(image: Image.Image, height: int, width: int) -> np.ndarray:
    cur_width, cur_height = image.size
    ratio = max(cur_width / width, cur_height / height)
    resized_width = max(1, int(cur_width / ratio))
    resized_height = max(1, int(cur_height / ratio))
    resized_image = image.resize((resized_width, resized_height), resample=Image.BILINEAR)
    output = Image.new(resized_image.mode, (width, height), 0)
    output.paste(resized_image, ((width - resized_width) // 2, (height - resized_height) // 2))
    return np.asarray(output)


def _roi_crop_zoom_hwc(image: np.ndarray, *, zoom: float, center_x: float, center_y: float) -> np.ndarray:
    zoom = float(zoom)
    if zoom <= 1.0:
        return image
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image
    center_x = min(max(float(center_x), 0.0), 1.0)
    center_y = min(max(float(center_y), 0.0), 1.0)
    crop_width = max(1, min(width, int(round(width / zoom))))
    crop_height = max(1, min(height, int(round(height / zoom))))
    crop_x = int(round(center_x * width - crop_width / 2.0))
    crop_y = int(round(center_y * height - crop_height / 2.0))
    crop_x = min(max(0, crop_x), max(0, width - crop_width))
    crop_y = min(max(0, crop_y), max(0, height - crop_height))
    return image[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]


def _as_hwc_uint8(
    value: Any,
    *,
    image_size: tuple[int, int],
    roi_zoom: float = 1.0,
    roi_center_x: float = 0.5,
    roi_center_y: float = 0.5,
) -> np.ndarray:
    if value is None:
        return np.zeros((*image_size, 3), dtype=np.uint8)
    image = _tensor_to_np(value)
    if isinstance(image, np.ndarray) and image.dtype.fields is not None:
        image = image.view(np.uint8).reshape(image.shape + (-1,))
    elif isinstance(image, np.ndarray) and image.dtype == np.uint32:
        image = image.view(np.uint8).reshape(image.shape + (4,))
    while image.ndim > 3:
        image = image[0]
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3:
        return np.zeros((*image_size, 3), dtype=np.uint8)
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        return np.zeros((*image_size, 3), dtype=np.uint8)
    if np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(image)) if image.size else 0.0
        if max_value <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255)
    image = image.astype(np.uint8, copy=False)
    image = _roi_crop_zoom_hwc(image, zoom=roi_zoom, center_x=roi_center_x, center_y=roi_center_y)
    return _resize_with_pad(image, image_size[0], image_size[1])


def _render_camera_rgb(
    camera: object | None,
    *,
    image_size: tuple[int, int],
    roi_zoom: float = 1.0,
    roi_center_x: float = 0.5,
    roi_center_y: float = 0.5,
) -> np.ndarray:
    if camera is None:
        return np.zeros((*image_size, 3), dtype=np.uint8)
    try:
        rendered = camera.render(rgb=True, depth=False, segmentation=False, normal=False, force_render=True)
    except TypeError:
        rendered = camera.render()
    except Exception as exc:
        print(f"[camera] render failed: {exc}", flush=True)
        return np.zeros((*image_size, 3), dtype=np.uint8)
    if isinstance(rendered, dict):
        rendered = rendered.get("rgb", rendered.get("color", rendered.get("image")))
    elif isinstance(rendered, (tuple, list)):
        rendered = rendered[0] if rendered else None
    return _as_hwc_uint8(
        rendered,
        image_size=image_size,
        roi_zoom=roi_zoom,
        roi_center_x=roi_center_x,
        roi_center_y=roi_center_y,
    )


def _bottle_panel_main(initial_values, values, policy_running, sim_running, reset_counter, stop_flag) -> None:
    import tkinter as tk
    from tkinter import ttk

    specs = (
        ("x", -0.60, 0.60, "m", 0.001),
        ("y", -0.20, 0.80, "m", 0.001),
        ("z", 0.20, 1.20, "m", 0.001),
        ("roll", -180.0, 180.0, "deg", 0.1),
        ("pitch", -180.0, 180.0, "deg", 0.1),
        ("yaw", -180.0, 180.0, "deg", 0.1),
    )
    sliders = []
    labels = []

    def set_value(idx: int, value: float | str, *, update_slider: bool = False) -> None:
        _, lower, upper, unit, _ = specs[idx]
        current = max(float(lower), min(float(upper), float(value)))
        values[idx] = current
        precision = 5 if unit == "m" else 3
        labels[idx].config(text=f"{current: .{precision}f}")
        if update_slider:
            sliders[idx].set(current)

    def nudge(idx: int, delta: float) -> None:
        set_value(idx, float(values[idx]) + delta, update_slider=True)

    def set_policy_running(enabled: bool) -> None:
        policy_running.value = bool(enabled)
        policy_button.config(text="Pause Policy" if policy_running.value else "Start Policy")
        status_label.config(text="Policy inference running" if policy_running.value else "Policy inference paused")

    def set_sim_running(enabled: bool) -> None:
        sim_running.value = bool(enabled)
        sim_button.config(text="Pause Physics Step" if sim_running.value else "Start Physics Step")

    def reset() -> None:
        set_policy_running(False)
        set_sim_running(False)
        for idx, value in enumerate(initial_values):
            set_value(idx, value, update_slider=True)
        reset_counter.value += 1

    def print_pose() -> None:
        current = [float(values[idx]) for idx in range(6)]
        print(
            "[bottle-policy-debug] relative_to_scene "
            f"pos=({current[0]:.5f}, {current[1]:.5f}, {current[2]:.5f}) "
            f"euler_deg=({current[3]:.3f}, {current[4]:.3f}, {current[5]:.3f})",
            flush=True,
        )

    def close() -> None:
        stop_flag.value = True
        root.destroy()
        root.quit()

    def refresh_buttons() -> None:
        policy_button.config(text="Pause Policy" if policy_running.value else "Start Policy")
        sim_button.config(text="Pause Physics Step" if sim_running.value else "Start Physics Step")
        status_label.config(text="Policy inference running" if policy_running.value else "Policy inference paused")
        root.after(200, refresh_buttons)

    root = tk.Tk()
    root.title("Bottle Pose Control")
    root.geometry("760x430")
    root.minsize(660, 380)

    ttk.Label(root, text="Bottle pose relative to scene", font=("Arial", 12, "bold")).pack(
        fill=tk.X, padx=12, pady=(12, 4)
    )
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

    for idx, (label, lower, upper, unit, step) in enumerate(specs):
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text=f"{label} ({unit})", width=12).pack(side=tk.LEFT)
        ttk.Button(row, text="-", width=3, command=lambda i=idx, s=step: nudge(i, -s)).pack(side=tk.LEFT)
        slider = ttk.Scale(
            row,
            from_=lower,
            to=upper,
            orient=tk.HORIZONTAL,
            command=lambda value, i=idx: set_value(i, value),
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(row, text="+", width=3, command=lambda i=idx, s=step: nudge(i, s)).pack(side=tk.LEFT)
        value_label = ttk.Label(row, text="", width=12)
        value_label.pack(side=tk.RIGHT)
        sliders.append(slider)
        labels.append(value_label)
        set_value(idx, float(initial_values[idx]), update_slider=True)

    buttons = ttk.Frame(root)
    buttons.pack(fill=tk.X, padx=12, pady=(4, 12))
    policy_button = ttk.Button(buttons, text="Start Policy", command=lambda: set_policy_running(not bool(policy_running.value)))
    policy_button.pack(side=tk.LEFT)
    sim_button = ttk.Button(buttons, text="Start Physics Step", command=lambda: set_sim_running(not bool(sim_running.value)))
    sim_button.pack(side=tk.LEFT, padx=8)
    ttk.Button(buttons, text="Reset Bottle", command=reset).pack(side=tk.LEFT, padx=8)
    ttk.Button(buttons, text="Print Pose", command=print_pose).pack(side=tk.LEFT)
    ttk.Button(buttons, text="Close", command=close).pack(side=tk.RIGHT)
    status_label = ttk.Label(buttons, text="Policy inference paused")
    status_label.pack(side=tk.RIGHT, padx=12)

    set_policy_running(bool(policy_running.value))
    set_sim_running(bool(sim_running.value))
    refresh_buttons()

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()


def _create_bottle_panel(
    initial_pos: tuple[float, float, float],
    initial_euler: tuple[float, float, float],
    policy_running,
) -> dict[str, object]:
    initial_values = tuple(float(v) for v in (*initial_pos, *initial_euler))
    values = multiprocessing.RawArray("d", initial_values)
    sim_running = multiprocessing.RawValue("b", False)
    reset_counter = multiprocessing.RawValue("i", 0)
    stop_flag = multiprocessing.RawValue("b", False)
    process = multiprocessing.Process(
        target=_bottle_panel_main,
        args=(initial_values, values, policy_running, sim_running, reset_counter, stop_flag),
        daemon=True,
    )
    process.start()
    return {
        "values": values,
        "policy_running": policy_running,
        "sim_running": sim_running,
        "reset_counter": reset_counter,
        "stop_flag": stop_flag,
        "process": process,
    }


def _read_bottle_panel(
    panel: dict[str, object],
) -> tuple[tuple[float, float, float], tuple[float, float, float], bool, bool, int, bool]:
    values = panel["values"]
    current = tuple(float(values[idx]) for idx in range(6))
    return (
        current[:3],
        current[3:],
        bool(panel["policy_running"].value),
        bool(panel["sim_running"].value),
        int(panel["reset_counter"].value),
        bool(panel["stop_flag"].value),
    )


def _shutdown_panel(panel: dict[str, object] | None) -> None:
    if not panel:
        return
    panel["stop_flag"].value = True
    process = panel["process"]
    if process.is_alive():
        process.join(timeout=1.0)


class ConsoleController:
    def __init__(self, policy_running) -> None:
        self._policy_running = policy_running
        self.quit_requested = False
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)

    @property
    def policy_running(self) -> bool:
        return bool(self._policy_running.value)

    @policy_running.setter
    def policy_running(self, value: bool) -> None:
        self._policy_running.value = bool(value)

    def start(self) -> None:
        self._thread.start()
        print(
            "[console] commands: start | stop | toggle | reset | status | quit",
            flush=True,
        )

    def _read_loop(self) -> None:
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:
                time.sleep(0.1)
                continue
            self._queue.put(line.strip().lower())

    def update(self, executor: "RightArmPolicyExecutor | None" = None) -> None:
        while True:
            try:
                command = self._queue.get_nowait()
            except queue.Empty:
                return
            if command in {"start", "s", "开始"}:
                self.policy_running = True
                print("[console] policy inference started", flush=True)
            elif command in {"stop", "pause", "p", "停止", "暂停"}:
                self.policy_running = False
                print("[console] policy inference stopped", flush=True)
            elif command in {"toggle", "t"}:
                self.policy_running = not self.policy_running
                print(f"[console] policy_running={self.policy_running}", flush=True)
            elif command in {"reset", "r", "重置"}:
                self.policy_running = False
                if executor is not None:
                    executor.reset()
                print("[console] policy stopped and robot reset", flush=True)
            elif command in {"status", "st"}:
                print(f"[console] policy_running={self.policy_running}", flush=True)
            elif command in {"quit", "q", "exit"}:
                self.quit_requested = True
                print("[console] quit requested", flush=True)
            elif command:
                print(f"[console] unknown command: {command}", flush=True)


class CameraPreview:
    def __init__(self, *, enabled: bool, scale: int = 2) -> None:
        self.enabled = bool(enabled)
        self.scale = max(1, int(scale))
        self._cv2 = None
        self._available = False
        self._warned = False
        self._windows = ("GR00T ego_view model input", "GR00T wrist_view model input")

    def _ensure(self) -> bool:
        if not self.enabled:
            return False
        if self._available:
            return True
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            if not self._warned:
                print("[camera-preview] disabled: DISPLAY/WAYLAND_DISPLAY is not set", flush=True)
                self._warned = True
            self.enabled = False
            return False
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception as exc:
            if not self._warned:
                print(f"[camera-preview] disabled: failed to import cv2: {exc}", flush=True)
                self._warned = True
            self.enabled = False
            return False
        self._cv2 = cv2
        try:
            for name in self._windows:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        except Exception as exc:
            if not self._warned:
                print(f"[camera-preview] disabled: failed to create OpenCV windows: {exc}", flush=True)
                self._warned = True
            self.enabled = False
            return False
        self._available = True
        print("[camera-preview] showing ego_view and wrist_view model inputs; press q in a preview window to close previews", flush=True)
        return True

    def show(self, *, ego: np.ndarray, wrist: np.ndarray) -> None:
        if not self._ensure():
            return
        assert self._cv2 is not None
        cv2 = self._cv2
        for name, image in ((self._windows[0], ego), (self._windows[1], wrist)):
            frame = np.asarray(image)
            if frame.ndim != 3 or frame.shape[-1] != 3:
                continue
            if self.scale > 1:
                height, width = frame.shape[:2]
                frame = cv2.resize(frame, (width * self.scale, height * self.scale), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self.close()
            self.enabled = False
            print("[camera-preview] closed", flush=True)

    def close(self) -> None:
        if not self._available or self._cv2 is None:
            return
        for name in self._windows:
            try:
                self._cv2.destroyWindow(name)
            except Exception:
                pass
        self._available = False


class EefChunkTrajectoryOverlay:
    def __init__(
        self,
        scene: object,
        *,
        enabled: bool,
        max_points: int,
        line_radius: float,
        point_radius: float,
        show_orientation: bool,
        orientation_max_frames: int,
        orientation_axis_length: float,
        orientation_axis_radius: float,
    ) -> None:
        self.scene = scene
        self.enabled = bool(enabled)
        self.max_points = max(2, int(max_points))
        self.line_radius = max(float(line_radius), 1.0e-5)
        self.point_radius = max(float(point_radius), 1.0e-5)
        self.show_orientation = bool(show_orientation)
        self.orientation_max_frames = max(1, int(orientation_max_frames))
        self.orientation_axis_length = max(float(orientation_axis_length), 1.0e-4)
        self.orientation_axis_radius = max(float(orientation_axis_radius), 1.0e-5)
        self._objects: list[object] = []
        self._active_obj: object | None = None
        self._last_points: np.ndarray | None = None
        self._warned = False

    @staticmethod
    def _chunk_eef(action: dict[str, np.ndarray] | None, *, max_points: int) -> np.ndarray | None:
        if not isinstance(action, dict) or "eef_9d" not in action:
            return None
        arr = np.asarray(action["eef_9d"], dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[-1] < 3 or arr.shape[0] <= 0:
            return None
        eef = arr[:, : max(3, min(arr.shape[-1], 9))]
        points = eef[:, :3]
        finite = np.all(np.isfinite(points), axis=1)
        eef = eef[finite]
        if eef.shape[0] <= 0:
            return None
        if eef.shape[0] > max_points:
            indices = np.linspace(0, eef.shape[0] - 1, max_points).round().astype(int)
            eef = eef[indices]
        return eef.astype(np.float32, copy=False)

    @staticmethod
    def _orientation_frames(eef: np.ndarray, *, max_frames: int) -> np.ndarray | None:
        if eef.ndim != 2 or eef.shape[-1] < 9:
            return None
        frames = eef
        if frames.shape[0] > max_frames:
            indices = np.linspace(0, frames.shape[0] - 1, max_frames).round().astype(int)
            frames = frames[indices]
        transforms = []
        for row in frames:
            transform = np.eye(4, dtype=np.float32)
            transform[:3, 3] = np.asarray(row[:3], dtype=np.float32)
            transform[:3, :3] = _rot6d_to_rotmat(np.asarray(row[3:9], dtype=np.float32)).astype(np.float32)
            transforms.append(transform)
        return np.stack(transforms, axis=0) if transforms else None

    @staticmethod
    def _pose_at(pos: np.ndarray) -> np.ndarray:
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = np.asarray(pos, dtype=np.float32).reshape(3)
        return pose

    def clear(self) -> None:
        if not self._objects:
            self._active_obj = None
            self._last_points = None
            return
        for obj in tuple(self._objects):
            try:
                self.scene.clear_debug_object(obj)
            except Exception:
                try:
                    self.scene.clear_debug_objects()
                except Exception:
                    pass
                break
        self._objects.clear()
        self._active_obj = None
        self._last_points = None

    def _disable_after_error(self, exc: Exception) -> None:
        if not self._warned:
            print(f"[eef-trajectory] disabled: Genesis debug draw failed: {exc}", flush=True)
            self._warned = True
        self.enabled = False

    def draw_chunk(self, action: dict[str, np.ndarray] | None) -> None:
        if not self.enabled:
            return
        eef = self._chunk_eef(action, max_points=self.max_points)
        if eef is None:
            self.clear()
            return
        points = eef[:, :3]
        try:
            self.clear()
            if points.shape[0] >= 2:
                self._objects.append(
                    self.scene.draw_debug_trajectory(
                        points,
                        radius=self.line_radius,
                        color=(1.0, 0.55, 0.02, 0.92),
                    )
                )
            self._objects.append(
                self.scene.draw_debug_spheres(
                    points,
                    radius=self.point_radius,
                    color=(0.0, 0.85, 1.0, 0.75),
                )
            )
            self._active_obj = self.scene.draw_debug_sphere(
                points[0],
                radius=self.point_radius * 1.6,
                color=(0.1, 1.0, 0.15, 0.95),
            )
            self._objects.append(self._active_obj)
            orientation_frames = (
                self._orientation_frames(eef, max_frames=self.orientation_max_frames) if self.show_orientation else None
            )
            if orientation_frames is not None:
                self._objects.append(
                    self.scene.draw_debug_frames(
                        orientation_frames,
                        axis_length=self.orientation_axis_length,
                        origin_size=self.point_radius * 0.55,
                        axis_radius=self.orientation_axis_radius,
                    )
                )
            self._last_points = points
            print(
                "[eef-trajectory] chunk "
                f"points={points.shape[0]} first={tuple(round(float(v), 4) for v in points[0])} "
                f"last={tuple(round(float(v), 4) for v in points[-1])} "
                f"orientation_frames={0 if orientation_frames is None else orientation_frames.shape[0]}",
                flush=True,
            )
        except Exception as exc:
            self.clear()
            self._disable_after_error(exc)

    def update_active(self, action_index: int) -> None:
        if not self.enabled or self._active_obj is None or self._last_points is None:
            return
        idx = min(max(int(action_index), 0), self._last_points.shape[0] - 1)
        try:
            self.scene.update_debug_objects((self._active_obj,), (self._pose_at(self._last_points[idx]),))
        except Exception as exc:
            self._disable_after_error(exc)


class Gr00tObservationBuilder:
    def __init__(
        self,
        *,
        modality_config: dict[str, Any],
        instruction: str,
        image_size: tuple[int, int],
    ) -> None:
        self.modality_config = modality_config
        self.instruction = str(instruction)
        self.image_size = tuple(int(v) for v in image_size)
        video_delta = list(self.modality_config["video"].delta_indices)
        self.video_history_len = max([abs(int(i)) for i in video_delta] + [0]) + 1
        self.frame_buffer: deque[dict[str, np.ndarray]] = deque(maxlen=self.video_history_len)

    def append_frame(self, *, ego: np.ndarray, wrist: np.ndarray) -> None:
        self.frame_buffer.append({"ego": ego, "wrist": wrist})

    def _select_history_frames(self) -> list[dict[str, np.ndarray]]:
        if not self.frame_buffer:
            blank = np.zeros((*self.image_size, 3), dtype=np.uint8)
            self.append_frame(ego=blank, wrist=blank)
        delta_indices = list(self.modality_config["video"].delta_indices)
        buffer = list(self.frame_buffer)
        selected = []
        for delta in delta_indices:
            delta = int(delta)
            selected.append(buffer[-1] if delta == 0 else buffer[max(delta, -len(buffer))])
        return selected

    def build(
        self,
        *,
        arm_q: np.ndarray,
        eef_pose: np.ndarray,
        policy_eef_9d: np.ndarray | None = None,
        hand_q: np.ndarray,
        reference_action: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        frames = self._select_history_frames()
        video: dict[str, np.ndarray] = {}
        for key in self.modality_config["video"].modality_keys:
            lowered = str(key).lower()
            frame_key = "wrist" if "wrist" in lowered or "hand" in lowered else "ego"
            video[key] = np.stack([frame[frame_key] for frame in frames], axis=0)[None, ...].astype(np.uint8)

        arm_q = np.asarray(arm_q, dtype=np.float32).reshape(-1)
        arm_7 = np.zeros(7, dtype=np.float32)
        arm_7[: min(7, arm_q.size)] = arm_q[: min(7, arm_q.size)]
        hand_10 = np.zeros(10, dtype=np.float32)
        hand_10[: min(10, hand_q.size)] = np.asarray(hand_q, dtype=np.float32).reshape(-1)[: min(10, hand_q.size)]
        if policy_eef_9d is None:
            eef_xyz = np.asarray(eef_pose[:3, 3], dtype=np.float32).reshape(3)
            eef_rot6d = _rotmat_to_rot6d(eef_pose[:3, :3])
            eef_9d = np.concatenate([eef_xyz, eef_rot6d]).astype(np.float32)
        else:
            eef_9d = np.asarray(policy_eef_9d, dtype=np.float32).reshape(-1)[:9]
            if eef_9d.size != 9:
                raise ValueError(f"policy_eef_9d must contain 9 values, got {eef_9d.shape}")
            eef_xyz = eef_9d[:3].astype(np.float32, copy=False)
            eef_rot6d = eef_9d[3:9].astype(np.float32, copy=False)
        source = {
            "eef_9d": eef_9d,
            "arm_eef_pos": eef_xyz,
            "arm_eef_rot6d": eef_rot6d,
            "hand_joint_pos": hand_10,
            "arm_joint_pos": arm_7,
            "hand_joint_target": reference_action["hand_joint_target"].reshape(-1).astype(np.float32),
            "arm_joint_target": reference_action["arm_joint_target"].reshape(-1).astype(np.float32),
        }
        state = {}
        for key in self.modality_config["state"].modality_keys:
            if key not in source:
                raise KeyError(f"Cannot build GR00T state key {key!r}; available={sorted(source)}")
            state[key] = source[key][None, None, ...].astype(np.float32, copy=False)

        language = {key: [[self.instruction]] for key in self.modality_config["language"].modality_keys}
        return {"video": video, "state": state, "language": language}


class PolicyTeleopBridge:
    """Interpret policy eef_9d as OpenXR-like source wrist motion."""

    def __init__(
        self,
        *,
        coordinate_adapter: str,
        input_axis_map: tuple[str, str, str],
        translation_scale_xyz: tuple[float, float, float],
        yaw_recenter: bool,
        orientation_reference_mode: str,
        orientation_axis_map: tuple[str, str, str],
        orientation_max_speed_rad_s: float,
        workspace_min: tuple[float, float, float],
        workspace_max: tuple[float, float, float],
    ) -> None:
        self.coordinate_adapter = str(coordinate_adapter)
        if self.coordinate_adapter not in {"openxr_genesis", "none"}:
            raise ValueError(f"unsupported policy coordinate adapter: {self.coordinate_adapter!r}")
        self.input_axis_map = tuple(str(v) for v in input_axis_map)
        self.translation_scale_xyz = np.asarray(translation_scale_xyz, dtype=np.float64).reshape(3)
        self.yaw_recenter_enabled = bool(yaw_recenter)
        self.workspace_min = np.asarray(workspace_min, dtype=np.float64).reshape(3)
        self.workspace_max = np.asarray(workspace_max, dtype=np.float64).reshape(3)
        self.orientation_tracker = OrientationTargetTracker(
            OrientationTrackerConfig(
                axis_map=tuple(str(v) for v in orientation_axis_map),  # type: ignore[arg-type]
                max_speed_rad_s=float(orientation_max_speed_rad_s),
                reference_mode=str(orientation_reference_mode),  # type: ignore[arg-type]
            )
        )
        self.source_anchor_xyz = np.zeros(3, dtype=np.float64)
        self.source_anchor_rotation = np.eye(3, dtype=np.float64)
        self.source_command_xyz = np.zeros(3, dtype=np.float64)
        self.source_command_rotation = np.eye(3, dtype=np.float64)
        self.target_anchor_xyz = np.zeros(3, dtype=np.float64)
        self.target_anchor_quat_wxyz = (1.0, 0.0, 0.0, 0.0)
        self.yaw_correction_rad: float | None = None
        self.last_debug: dict[str, Any] = {}
        self.reset_generation = 0

    def reset_from_eef(self, eef_pose: np.ndarray) -> None:
        pose = np.asarray(eef_pose, dtype=np.float64).reshape(4, 4)
        self.source_anchor_xyz = np.zeros(3, dtype=np.float64)
        self.source_anchor_rotation = np.eye(3, dtype=np.float64)
        self.source_command_xyz = self.source_anchor_xyz.copy()
        self.source_command_rotation = self.source_anchor_rotation.copy()
        self.target_anchor_xyz = pose[:3, 3].astype(np.float64, copy=True)
        self.target_anchor_quat_wxyz = _rotation_to_quat_wxyz(pose[:3, :3])
        source_quat = self._source_rotation_to_genesis_quat(self.source_anchor_rotation)
        self._reset_yaw_correction(source_quat)
        source_quat = _apply_yaw_to_quat_xyzw(source_quat, self.yaw_correction_rad)
        self.orientation_tracker.reset_anchor(source_quat, self.target_anchor_quat_wxyz)
        self.reset_generation += 1
        self.last_debug = {
            "event": "reset",
            "coordinate_adapter": self.coordinate_adapter,
            "adapter_payload": self._adapter_debug_payload(),
            "source_anchor_xyz": self.source_anchor_xyz.tolist(),
            "target_anchor_xyz": self.target_anchor_xyz.tolist(),
            "target_anchor_quat_wxyz": list(self.target_anchor_quat_wxyz),
            "yaw_correction_rad": self.yaw_correction_rad,
            "yaw_correction_deg": None if self.yaw_correction_rad is None else math.degrees(self.yaw_correction_rad),
        }
        print(
            "[policy-teleop] reset "
            f"adapter={self.coordinate_adapter} "
            f"target_anchor={tuple(round(float(v), 4) for v in self.target_anchor_xyz)} "
            f"yaw_deg={None if self.yaw_correction_rad is None else round(math.degrees(self.yaw_correction_rad), 3)}",
            flush=True,
        )

    def current_reference_eef_9d(self) -> np.ndarray:
        return np.concatenate(
            [
                self.source_command_xyz.astype(np.float32),
                _rotmat_to_rot6d(self.source_command_rotation),
            ]
        ).astype(np.float32)

    def step_source_eef(self, eef_9d: np.ndarray, *, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        source_xyz, source_rotation = self._decode_source_eef(eef_9d)
        target_xyz, mapped_delta, source_delta = self._target_position_from_source(source_xyz)
        source_quat = self._source_rotation_to_genesis_quat(source_rotation)
        source_quat = _apply_yaw_to_quat_xyzw(source_quat, self.yaw_correction_rad)
        orientation_result = self.orientation_tracker.update(source_quat, dt_s=max(float(dt_s), 1.0e-6))
        target_rotation = _quat_wxyz_to_rotation(orientation_result.cmd_target_quat_wxyz)
        self.source_command_xyz = source_xyz
        self.source_command_rotation = source_rotation
        self.last_debug = {
            "event": "step",
            "coordinate_adapter": self.coordinate_adapter,
            "source_xyz": source_xyz.tolist(),
            "source_delta_openxr": source_delta.tolist(),
            "mapped_delta_genesis": mapped_delta.tolist(),
            "target_xyz": target_xyz.tolist(),
            "source_quat_xyzw_genesis": list(source_quat),
            "orientation": orientation_result.as_dict(),
        }
        return target_xyz.astype(np.float64), target_rotation.astype(np.float64)

    def preview_action_chunk(self, action: dict[str, np.ndarray] | None, *, max_dt_s: float) -> dict[str, np.ndarray] | None:
        if not isinstance(action, dict) or "eef_9d" not in action:
            return action
        preview = {key: np.asarray(value, dtype=np.float32, copy=True) for key, value in action.items()}
        arr = np.asarray(preview["eef_9d"], dtype=np.float32)
        batched = arr.ndim == 3
        chunk = arr[0] if batched else arr
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        if chunk.ndim != 2 or chunk.shape[-1] < 9:
            return preview
        mapped_rows = []
        for row in chunk:
            source_xyz, source_rotation = self._decode_source_eef(row)
            target_xyz, _, _ = self._target_position_from_source(source_xyz)
            source_quat = self._source_rotation_to_genesis_quat(source_rotation)
            source_quat = _apply_yaw_to_quat_xyzw(source_quat, self.yaw_correction_rad)
            target_rotation = _quat_xyzw_to_rotation(source_quat)
            mapped_rows.append(np.concatenate([target_xyz.astype(np.float32), _rotmat_to_rot6d(target_rotation)]))
        mapped = np.stack(mapped_rows, axis=0).astype(np.float32)
        if batched:
            preview["eef_9d"] = mapped[None, ...]
        else:
            preview["eef_9d"] = mapped
        return preview

    def _decode_source_eef(self, eef_9d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eef = np.asarray(eef_9d, dtype=np.float64).reshape(-1)
        source_xyz = np.zeros(3, dtype=np.float64)
        source_xyz[: min(3, eef.size)] = eef[: min(3, eef.size)]
        if eef.size >= 9:
            source_rotation = _rot6d_to_rotmat(eef[3:9])
        else:
            source_rotation = self.source_command_rotation.copy()
        return source_xyz, source_rotation

    def _target_position_from_source(self, source_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source_delta = np.asarray(source_xyz, dtype=np.float64).reshape(3) - self.source_anchor_xyz
        if self.coordinate_adapter == "openxr_genesis":
            mapped = np.asarray(map_openxr_vector_to_genesis(tuple(float(v) for v in source_delta)), dtype=np.float64)
        else:
            mapped = source_delta.copy()
        mapped = np.asarray(harness._map_vec3_axes(tuple(float(v) for v in mapped), self.input_axis_map), dtype=np.float64)  # noqa: SLF001
        if self.yaw_correction_rad is not None:
            mapped = _yaw_rotation_matrix(self.yaw_correction_rad) @ mapped
        mapped = mapped * self.translation_scale_xyz
        target = np.clip(self.target_anchor_xyz + mapped, self.workspace_min, self.workspace_max)
        return target, mapped, source_delta

    def _source_rotation_to_genesis_quat(self, source_rotation: np.ndarray) -> tuple[float, float, float, float]:
        source_quat = _rotation_to_quat_xyzw(np.asarray(source_rotation, dtype=np.float64).reshape(3, 3))
        if self.coordinate_adapter == "openxr_genesis":
            return map_openxr_quaternion_to_genesis_parent(source_quat)
        return source_quat

    def _reset_yaw_correction(self, source_quat_xyzw: tuple[float, float, float, float]) -> None:
        if not self.yaw_recenter_enabled or self.coordinate_adapter == "none":
            self.yaw_correction_rad = None
            return
        rotation = _quat_xyzw_to_rotation(source_quat_xyzw)
        source_forward = -rotation[:, 2]
        horizontal = np.asarray((source_forward[0], source_forward[1]), dtype=np.float64)
        norm = float(np.linalg.norm(horizontal))
        if norm <= 1.0e-9:
            self.yaw_correction_rad = None
            return
        horizontal /= norm
        target = np.asarray((-1.0, 0.0), dtype=np.float64)
        cross_z = float(horizontal[0] * target[1] - horizontal[1] * target[0])
        dot = float(np.dot(horizontal, target))
        self.yaw_correction_rad = float(math.atan2(cross_z, dot))

    def _adapter_debug_payload(self) -> dict[str, object]:
        if self.coordinate_adapter == "openxr_genesis":
            payload = canonical_adapter_debug_payload()
            payload["adapter"] = "openxr_genesis"
            return payload
        return {"adapter": "none"}


class RightArmPolicyExecutor:
    def __init__(
        self,
        scene: object,
        *,
        max_joint_step: float,
        ik_solver_max_joint_step: float,
        min_joint_step: float,
        pos_tol: float,
        ik_j4_limit: bool,
        ik_j4_limit_rad: tuple[float, float],
        workspace_min: tuple[float, float, float],
        workspace_max: tuple[float, float, float],
        print_hand_every: int,
        max_hand_joint_delta: float,
        initial_hand_q: tuple[float, ...],
        initial_reference_eef_9d: tuple[float, ...],
        initial_reference_hand_q: tuple[float, ...],
        initial_right_arm_q: tuple[float, ...],
        initial_left_arm_q: tuple[float, ...],
        policy_execution_mode: str,
        policy_openxr_coordinate_adapter: str,
        policy_input_axis_map: tuple[str, str, str],
        policy_translation_scale: tuple[float, float, float],
        policy_yaw_recenter: bool,
        policy_orientation_reference_mode: str,
        policy_orientation_axis_map: tuple[str, str, str],
        policy_orientation_max_speed_rad_s: float,
        action_dt_s: float,
    ) -> None:
        self.scene = scene
        assembly = getattr(scene, "nero_assembly_info", None)
        if not isinstance(assembly, dict):
            raise RuntimeError("Scene does not contain Nero assembly")
        self.assembly = assembly
        self.arm = assembly["right"]
        self.left_arm = assembly.get("left")
        self.eef_link = self.arm.get_link(str(assembly.get("eef_link", harness.DEFAULT_EEF_LINK)))
        self.arm_dofs = harness._arm_dofs(self.arm)  # noqa: SLF001
        self.max_joint_step = float(max_joint_step)
        self.ik_solver_max_joint_step = float(ik_solver_max_joint_step)
        self.min_joint_step = max(float(min_joint_step), 0.0)
        self.pos_tol = float(pos_tol)
        self.ik_j4_limit = bool(ik_j4_limit)
        self.ik_j4_limit_rad = tuple(float(v) for v in ik_j4_limit_rad)
        self.workspace_min = np.asarray(workspace_min, dtype=np.float64)
        self.workspace_max = np.asarray(workspace_max, dtype=np.float64)
        self.max_hand_joint_delta = max(0.0, float(max_hand_joint_delta))
        self.policy_execution_mode = str(policy_execution_mode)
        if self.policy_execution_mode not in {"teleop_source", "robot_target"}:
            raise ValueError(f"unsupported policy execution mode: {self.policy_execution_mode!r}")
        self.action_dt_s = float(action_dt_s)
        self.initial_right_arm_q = np.asarray(initial_right_arm_q, dtype=np.float32).reshape(7)
        self.initial_left_arm_q = np.asarray(initial_left_arm_q, dtype=np.float32).reshape(7)
        self._ik_joint_limit_hit_count = 0
        self.reset_generation = 0
        self.initial_hand_q = self._clip_policy_hand_q(np.asarray(initial_hand_q, dtype=np.float32))
        self.initial_reference_eef_9d = np.asarray(initial_reference_eef_9d, dtype=np.float32).reshape(9)
        self.initial_reference_hand_q = self._clip_policy_hand_q(
            np.asarray(initial_reference_hand_q, dtype=np.float32)
        )
        self.reference_eef_9d = self.initial_reference_eef_9d.copy()
        self.policy_hand_q_raw = self.initial_reference_hand_q.copy()
        self.reference_hand_q = self.initial_reference_hand_q.copy()
        self.sim_hand_q = self._project_hand_q_for_sim(self.initial_hand_q)
        self.hand_q = self.sim_hand_q.copy()
        self._set_initial_arm_poses()
        self.q_cmd = _tensor_to_np(self.arm.get_qpos()).reshape(-1)[self.arm_dofs].astype(np.float32)
        self.target_rotation = self.current_eef_pose()[:3, :3].copy()
        self.target_xyz = self.current_eef_pose()[:3, 3].copy()
        self.reference_eef_9d = self.target_policy_eef_9d()
        self.teleop_bridge = (
            PolicyTeleopBridge(
                coordinate_adapter=policy_openxr_coordinate_adapter,
                input_axis_map=policy_input_axis_map,
                translation_scale_xyz=policy_translation_scale,
                yaw_recenter=policy_yaw_recenter,
                orientation_reference_mode=policy_orientation_reference_mode,
                orientation_axis_map=policy_orientation_axis_map,
                orientation_max_speed_rad_s=policy_orientation_max_speed_rad_s,
                workspace_min=workspace_min,
                workspace_max=workspace_max,
            )
            if self.policy_execution_mode == "teleop_source"
            else None
        )
        if self.teleop_bridge is not None:
            self.teleop_bridge.reset_from_eef(self.current_eef_pose())
        self._hand_target_print_count = 0
        self.print_hand_every = int(print_hand_every)
        print(
            "[policy-ik] "
            f"eef_link={getattr(self.eef_link, 'name', 'revo2_flange')} "
            f"max_joint_step={self.max_joint_step:.4f} "
            f"ik_solver_max_joint_step={self.ik_solver_max_joint_step:.4f} "
            f"min_joint_step={self.min_joint_step:.4f} "
            f"pos_tol={self.pos_tol:.1e} "
            f"ik_j4_limit={self.ik_j4_limit} range={self.ik_j4_limit_rad} "
            f"policy_execution_mode={self.policy_execution_mode}",
            flush=True,
        )
        if self.teleop_bridge is not None:
            print(
                "[policy-teleop] "
                f"adapter={self.teleop_bridge.coordinate_adapter} "
                f"axis_map={self.teleop_bridge.input_axis_map} "
                f"translation_scale={tuple(float(v) for v in self.teleop_bridge.translation_scale_xyz)} "
                f"yaw_recenter={self.teleop_bridge.yaw_recenter_enabled}",
                flush=True,
            )
        print(
            "[policy-ik] initial_right_arm_q="
            + str(tuple(round(float(v), 6) for v in self.initial_right_arm_q)),
            flush=True,
        )
        print(f"[policy-hand] canonical_order={POLICY_HAND_JOINT_NAMES}", flush=True)
        print(
            "[policy-hand] gr00t_l10_groups "
            f"active={tuple(POLICY_HAND_JOINT_NAMES[idx] for idx in L10_HAND_ACTIVE_LOCAL_INDICES)} "
            f"masked={tuple(POLICY_HAND_JOINT_NAMES[idx] for idx in L10_HAND_MASKED_LOCAL_INDICES)} "
            f"policy_clip=({POLICY_HAND_CLIP_LOWER}, {POLICY_HAND_CLIP_UPPER}) "
            f"max_hand_joint_delta={self.max_hand_joint_delta:.3f}",
            flush=True,
        )
        print(
            "[policy-hand] initial_reference="
            + str(
                {
                    name: round(float(value), 4)
                    for name, value in zip(POLICY_HAND_JOINT_NAMES, self.reference_hand_q, strict=False)
                }
            ),
            flush=True,
        )
        print(
            "[policy-eef] initial_reference_eef_9d="
            + str(tuple(round(float(v), 6) for v in self.reference_eef_9d)),
            flush=True,
        )
        print(
            "[policy-hand] initial_sim_pose="
            + str(
                {
                    name: round(float(value), 4)
                    for name, value in zip(POLICY_HAND_JOINT_NAMES, self.sim_hand_q, strict=False)
                }
            ),
            flush=True,
        )
        self._apply_linker_hand_target()
        harness._step_scene_with_attached_parts(self.scene)  # noqa: SLF001
        self._freeze_target_at_current_eef()
        self.reference_eef_9d = self.target_policy_eef_9d()
        if self.teleop_bridge is not None:
            self.teleop_bridge.reset_from_eef(self.current_eef_pose())
        self._warmup_ik()
        self._hand_target_print_count = 0

    def reset(self) -> None:
        self._set_initial_arm_poses()
        self.q_cmd = _tensor_to_np(self.arm.get_qpos()).reshape(-1)[self.arm_dofs].astype(np.float32)
        self.reference_eef_9d = self.initial_reference_eef_9d.copy()
        self.policy_hand_q_raw = self.initial_reference_hand_q.copy()
        self.reference_hand_q = self.initial_reference_hand_q.copy()
        self.sim_hand_q = self._project_hand_q_for_sim(self.initial_hand_q)
        self.hand_q = self.sim_hand_q.copy()
        self._apply_linker_hand_target()
        harness._step_scene_with_attached_parts(self.scene)  # noqa: SLF001
        self._freeze_target_at_current_eef()
        self.reference_eef_9d = self.target_policy_eef_9d()
        if self.teleop_bridge is not None:
            self.teleop_bridge.reset_from_eef(self.current_eef_pose())
        self._warmup_ik()
        self.reset_generation += 1

    def current_arm_q(self) -> np.ndarray:
        return self.current_actual_arm_q()

    def current_actual_arm_q(self) -> np.ndarray:
        qpos = _tensor_to_np(self.arm.get_qpos()).reshape(-1).astype(np.float32)
        out = np.zeros(len(self.arm_dofs), dtype=np.float32)
        for idx, dof in enumerate(self.arm_dofs):
            if int(dof) < qpos.size:
                out[idx] = float(qpos[int(dof)])
        return out

    def current_eef_pose(self) -> np.ndarray:
        pos = _tensor_to_np(self.eef_link.get_pos()).reshape(3).astype(np.float64)
        quat = _tensor_to_np(self.eef_link.get_quat()).reshape(4).astype(np.float64)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = harness._rotation_from_quat_wxyz(quat)  # noqa: SLF001
        pose[:3, 3] = pos
        return pose

    def _arm_root_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = _tensor_to_np(self.arm.get_pos()).reshape(3).astype(np.float64)
        quat = _tensor_to_np(self.arm.get_quat()).reshape(4).astype(np.float64)
        rotation = harness._rotation_from_quat_wxyz(quat)  # noqa: SLF001
        return pos, np.asarray(rotation, dtype=np.float64).reshape(3, 3)

    def world_pose_to_policy_pose(self, pose: np.ndarray) -> np.ndarray:
        """Convert Genesis world EEF pose to the remote Teleop robot-base policy frame."""
        pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        arm_pos, arm_rotation = self._arm_root_pose()
        out = np.eye(4, dtype=np.float64)
        out[:3, 3] = arm_rotation.T @ (pose[:3, 3] - arm_pos)
        out[:3, :3] = arm_rotation.T @ pose[:3, :3]
        return out

    def policy_pose_to_world_pose(self, pose: np.ndarray) -> np.ndarray:
        """Convert remote Teleop robot-base policy frame pose to Genesis world."""
        pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        arm_pos, arm_rotation = self._arm_root_pose()
        out = np.eye(4, dtype=np.float64)
        out[:3, 3] = arm_pos + arm_rotation @ pose[:3, 3]
        out[:3, :3] = arm_rotation @ pose[:3, :3]
        return out

    def current_policy_eef_pose(self) -> np.ndarray:
        return self.world_pose_to_policy_pose(self.current_eef_pose())

    def current_policy_eef_9d(self) -> np.ndarray:
        pose = self.current_policy_eef_pose()
        return np.concatenate([pose[:3, 3].astype(np.float32), _rotmat_to_rot6d(pose[:3, :3])]).astype(np.float32)

    def target_policy_eef_9d(self) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = self.target_xyz
        pose[:3, :3] = self.target_rotation
        policy_pose = self.world_pose_to_policy_pose(pose)
        return np.concatenate(
            [policy_pose[:3, 3].astype(np.float32), _rotmat_to_rot6d(policy_pose[:3, :3])]
        ).astype(np.float32)

    def policy_eef_9d_to_world_pose(self, eef_9d: np.ndarray) -> np.ndarray:
        eef = np.asarray(eef_9d, dtype=np.float64).reshape(-1)
        policy_pose = np.eye(4, dtype=np.float64)
        policy_pose[:3, 3] = 0.0
        policy_pose[:3, 3][: min(3, eef.size)] = eef[: min(3, eef.size)]
        if eef.size >= 9:
            policy_pose[:3, :3] = _rot6d_to_rotmat(eef[3:9])
        else:
            policy_pose[:3, :3] = self.world_pose_to_policy_pose(
                np.block(
                    [
                        [self.target_rotation, self.target_xyz.reshape(3, 1)],
                        [np.zeros((1, 3), dtype=np.float64), np.ones((1, 1), dtype=np.float64)],
                    ]
                )
            )[:3, :3]
        return self.policy_pose_to_world_pose(policy_pose)

    def _command_eef_9d_from_action_step(self, eef: np.ndarray) -> np.ndarray:
        values = np.asarray(eef, dtype=np.float32).reshape(-1)
        out = self.reference_eef_9d.astype(np.float32, copy=True)
        out[: min(9, values.size)] = values[: min(9, values.size)]
        out = self._clamp_policy_eef_command(out)
        return out

    def _clamp_policy_eef_command(self, eef_9d: np.ndarray) -> np.ndarray:
        out = np.asarray(eef_9d, dtype=np.float32).reshape(-1).copy()
        if out.size >= 3:
            out[:3] = np.clip(out[:3], self.workspace_min.astype(np.float32), self.workspace_max.astype(np.float32))
        return out

    def current_reference_action(self) -> dict[str, np.ndarray]:
        if self.teleop_bridge is not None:
            eef_9d = self.teleop_bridge.current_reference_eef_9d()
        else:
            eef_9d = self.reference_eef_9d.astype(np.float32, copy=True)
        arm_joint_target = np.zeros(7, dtype=np.float32)
        actual_arm_q = self.current_actual_arm_q()
        arm_joint_target[: min(7, actual_arm_q.size)] = actual_arm_q[: min(7, actual_arm_q.size)]
        return {
            "eef_9d": eef_9d[None, :],
            "hand_joint_target": self.reference_hand_q[:10][None, :].astype(np.float32),
            "arm_joint_target": arm_joint_target[None, :],
        }

    def rtc_seed_action_from_command_chunk(
        self,
        action: dict[str, np.ndarray],
        *,
        observation_arm_q: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Mirror remote _commands_to_groot_action_seed for arm_joint_target.

        EEF and hand commands are action targets. The remote command->seed adapter
        fills arm_joint_target from the reported arm_joint_pos because policy
        commands do not carry an arm-joint target when EEF control is active.
        """
        seed = {key: np.asarray(value, dtype=np.float32, copy=True) for key, value in action.items()}
        if "arm_joint_target" not in seed:
            return seed
        arm_q = self.current_actual_arm_q() if observation_arm_q is None else np.asarray(observation_arm_q, dtype=np.float32)
        arm_7 = np.zeros(7, dtype=np.float32)
        arm_7[: min(7, arm_q.size)] = arm_q[: min(7, arm_q.size)]
        arr = np.asarray(seed["arm_joint_target"], dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 2:
            seed["arm_joint_target"] = np.repeat(arm_7[None, :], arr.shape[0], axis=0).astype(np.float32)
        elif arr.ndim == 1:
            seed["arm_joint_target"] = arm_7.astype(np.float32)
        return seed

    def current_observation_hand_q(self) -> np.ndarray:
        return self._current_linker_hand_q(fallback=self.sim_hand_q).astype(np.float32, copy=True)

    def step_action(self, action: dict[str, np.ndarray], action_index: int) -> None:
        applied_hand = False
        if "eef_9d" in action:
            eef = self._action_step(action["eef_9d"], action_index)
            if eef.size >= 3:
                if self.teleop_bridge is not None:
                    self.target_xyz, self.target_rotation = self.teleop_bridge.step_source_eef(
                        eef,
                        dt_s=self.action_dt_s,
                    )
                else:
                    self.reference_eef_9d = self._command_eef_9d_from_action_step(eef)
                    world_pose = self.policy_eef_9d_to_world_pose(self.reference_eef_9d)
                    self.target_xyz = world_pose[:3, 3]
                    self.target_rotation = world_pose[:3, :3]
            if eef.size >= 9:
                if self.teleop_bridge is None:
                    self.reference_eef_9d = self._command_eef_9d_from_action_step(eef)
                    self.target_rotation = self.policy_eef_9d_to_world_pose(self.reference_eef_9d)[:3, :3]
            self._solve_and_apply_eef_target()
        elif "arm_joint_target" in action:
            self._apply_joint_target(self._action_step(action["arm_joint_target"], action_index))

        if "hand_joint_target" in action:
            self.policy_hand_q_raw = np.asarray(
                self._action_step(action["hand_joint_target"], action_index)[:10],
                dtype=np.float32,
            )
            self.reference_hand_q = self._guard_hand_q_for_command(
                self.policy_hand_q_raw,
                previous_hand_q=self.reference_hand_q,
            )
            self.sim_hand_q = self._project_hand_q_for_sim(self.reference_hand_q)
            self.hand_q = self.sim_hand_q.copy()
            self._apply_linker_hand_target()
            applied_hand = True
        harness._step_scene_with_attached_parts(self.scene)  # noqa: SLF001
        if applied_hand and self._should_print_hand():
            print(f"[policy-hand] actual_after_step={self._current_linker_hand_positions()}", flush=True)

    def preview_action_chunk_for_overlay(self, action: dict[str, np.ndarray] | None) -> dict[str, np.ndarray] | None:
        if self.teleop_bridge is None:
            return self._preview_robot_target_action_chunk_in_world(action)
        return self.teleop_bridge.preview_action_chunk(action, max_dt_s=self.action_dt_s)

    def _preview_robot_target_action_chunk_in_world(
        self,
        action: dict[str, np.ndarray] | None,
    ) -> dict[str, np.ndarray] | None:
        if not isinstance(action, dict) or "eef_9d" not in action:
            return action
        preview = {key: np.asarray(value, dtype=np.float32, copy=True) for key, value in action.items()}
        arr = np.asarray(preview["eef_9d"], dtype=np.float32)
        batched = arr.ndim == 3
        chunk = arr[0] if batched else arr
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        if chunk.ndim != 2 or chunk.shape[-1] < 3:
            return preview
        mapped_rows = []
        for row in chunk:
            world_pose = self.policy_eef_9d_to_world_pose(row)
            mapped_rows.append(np.concatenate([world_pose[:3, 3].astype(np.float32), _rotmat_to_rot6d(world_pose[:3, :3])]))
        mapped = np.stack(mapped_rows, axis=0).astype(np.float32)
        if batched:
            preview["eef_9d"] = mapped[None, ...]
        else:
            preview["eef_9d"] = mapped
        return preview

    def teleop_debug_snapshot(self) -> dict[str, Any] | None:
        if self.teleop_bridge is None:
            return None
        return dict(self.teleop_bridge.last_debug)

    def eef_frame_debug_snapshot(self) -> dict[str, Any]:
        current_world = self.current_eef_pose()
        current_policy = self.world_pose_to_policy_pose(current_world)
        target_world = np.eye(4, dtype=np.float64)
        target_world[:3, 3] = self.target_xyz
        target_world[:3, :3] = self.target_rotation
        target_policy = self.world_pose_to_policy_pose(target_world)
        arm_pos, arm_rotation = self._arm_root_pose()
        return {
            "frame": "right_arm_entity_local_rokae_base",
            "arm_root_world_pos": arm_pos.tolist(),
            "arm_root_world_rot6d": _rotmat_to_rot6d(arm_rotation).tolist(),
            "current_world_xyz": current_world[:3, 3].tolist(),
            "current_policy_xyz": current_policy[:3, 3].tolist(),
            "target_world_xyz": target_world[:3, 3].tolist(),
            "target_policy_xyz": target_policy[:3, 3].tolist(),
            "reference_action_policy_xyz": self.reference_eef_9d[:3].tolist(),
            "reference_action_policy_rot6d": self.reference_eef_9d[3:9].tolist(),
        }

    def hand_policy_clip_delta(self, action: dict[str, np.ndarray]) -> float:
        max_delta = 0.0
        if "hand_joint_target" in action:
            hand = np.asarray(action["hand_joint_target"], dtype=np.float32)
            clipped = hand.copy()
            flat = clipped.reshape(-1, clipped.shape[-1])
            for row_idx in range(flat.shape[0]):
                flat[row_idx, :10] = self._clip_policy_hand_q(flat[row_idx, :10])
            max_delta = max(max_delta, float(np.max(np.abs(clipped - hand))))
        return max_delta

    def probe_clip_action_chunk(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        clipped = {key: np.asarray(value, dtype=np.float32, copy=True) for key, value in action.items()}
        if "hand_joint_target" not in clipped:
            return clipped
        hand = clipped["hand_joint_target"]
        if hand.ndim < 2 or hand.shape[-1] < 10:
            return clipped
        flat = hand.reshape(-1, hand.shape[-1])
        for row_idx in range(flat.shape[0]):
            flat[row_idx, :10] = self._clip_policy_hand_q(flat[row_idx, :10])
        return clipped

    def sim_project_action_chunk(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        projected = {key: np.asarray(value, dtype=np.float32, copy=True) for key, value in action.items()}
        if "hand_joint_target" not in projected:
            return projected
        hand = projected["hand_joint_target"]
        if hand.ndim < 2 or hand.shape[-1] < 10:
            return projected
        flat = hand.reshape(-1, hand.shape[-1])
        for row_idx in range(flat.shape[0]):
            flat[row_idx, :10] = self._project_hand_q_for_sim(flat[row_idx, :10])
        return projected

    def guard_action_chunk_for_execution(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        guarded = {key: np.asarray(value, dtype=np.float32, copy=True) for key, value in action.items()}
        if "eef_9d" in guarded:
            eef = guarded["eef_9d"]
            if eef.ndim >= 2 and eef.shape[-1] >= 3:
                flat_eef = eef.reshape(-1, eef.shape[-1])
                for row_idx in range(flat_eef.shape[0]):
                    flat_eef[row_idx, : min(9, flat_eef.shape[1])] = self._clamp_policy_eef_command(
                        flat_eef[row_idx, : min(9, flat_eef.shape[1])]
                    )
        return guarded

    def sim_project_action_chunk_for_execution(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Project command/reference actions to Genesis physical hand limits only for execution."""
        projected = self.guard_action_chunk_for_execution(action)
        if "hand_joint_target" not in projected:
            return projected
        hand = projected["hand_joint_target"]
        if hand.ndim < 2 or hand.shape[-1] < 10:
            return projected
        flat = hand.reshape(-1, hand.shape[-1])
        for row_idx in range(flat.shape[0]):
            flat[row_idx, :10] = self._project_hand_q_for_sim(flat[row_idx, :10])
        return projected

    def limit_command_hand_chunk(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Optional command-space rate limit without applying Genesis/URDF physical projection."""
        limited = {key: np.asarray(value, dtype=np.float32, copy=True) for key, value in action.items()}
        if "hand_joint_target" not in limited or self.max_hand_joint_delta <= 0.0:
            return limited
        hand = limited["hand_joint_target"]
        if hand.ndim < 2 or hand.shape[-1] < 10:
            return limited
        flat = hand.reshape(-1, hand.shape[-1])
        previous = self.reference_hand_q.astype(np.float32, copy=True)
        for row_idx in range(flat.shape[0]):
            values = self._clip_policy_hand_q(flat[row_idx, :10])
            lower = previous - self.max_hand_joint_delta
            upper = previous + self.max_hand_joint_delta
            values = self._clip_policy_hand_q(np.clip(values, lower, upper))
            flat[row_idx, :10] = values
            previous = values
        return limited

    def execution_hand_action_from_command(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self.sim_project_action_chunk_for_execution(self.limit_command_hand_chunk(action))

    def _guard_hand_q_for_command(self, hand_q: np.ndarray, *, previous_hand_q: np.ndarray) -> np.ndarray:
        """Keep GR00T command/reference in policy space; Genesis projection happens separately."""
        values = self._clip_policy_hand_q(hand_q)
        if self.max_hand_joint_delta > 0.0:
            previous = self._clip_policy_hand_q(previous_hand_q)
            lower = previous - self.max_hand_joint_delta
            upper = previous + self.max_hand_joint_delta
            values = self._clip_policy_hand_q(np.clip(values, lower, upper))
        return values

    def hand_debug_snapshot(
        self,
        *,
        policy_action_chunk: dict[str, np.ndarray],
        clipped_action_chunk: dict[str, np.ndarray],
        guarded_action_chunk: dict[str, np.ndarray],
        reference_action: dict[str, np.ndarray],
        rtc_seed_action: dict[str, np.ndarray] | None,
        reference_action_source: str,
        observation_hand_q: np.ndarray,
    ) -> dict[str, Any]:
        raw_hand = _first_batch_chunk(policy_action_chunk, "hand_joint_target") if "hand_joint_target" in policy_action_chunk else None
        clipped_hand = (
            _first_batch_chunk(clipped_action_chunk, "hand_joint_target") if "hand_joint_target" in clipped_action_chunk else None
        )
        guarded_hand = (
            _first_batch_chunk(guarded_action_chunk, "hand_joint_target")
            if "hand_joint_target" in guarded_action_chunk
            else None
        )
        sim_hand = (
            _first_batch_chunk(self.sim_project_action_chunk(guarded_action_chunk), "hand_joint_target")
            if "hand_joint_target" in guarded_action_chunk
            else None
        )
        reference_hand = np.asarray(reference_action.get("hand_joint_target", self.reference_hand_q[None, :]), dtype=np.float32)
        rtc_seed_hand = None
        if rtc_seed_action is not None and "hand_joint_target" in rtc_seed_action:
            rtc_seed_hand = _first_batch_chunk(rtc_seed_action, "hand_joint_target")[:1]
        reference_hand_flat = reference_hand.reshape(-1)[:10].astype(np.float32, copy=True)
        return {
            "reference_hand_source": str(reference_action_source),
            "policy_raw_unclipped": _hand_chunk_debug(raw_hand),
            "policy_raw_delta_from_reference": _hand_chunk_delta_debug(raw_hand, reference_hand_flat),
            "policy_clipped_probe_range": _hand_chunk_debug(clipped_hand),
            "guarded_command_range": _hand_chunk_debug(guarded_hand),
            "guarded_delta_from_reference": _hand_chunk_delta_debug(guarded_hand, reference_hand_flat),
            "sim_hand_projected_range": _hand_chunk_debug(sim_hand),
            "reference_hand_first": _named_hand_values(reference_hand_flat),
            "rtc_seed_hand_first": None if rtc_seed_hand is None else _named_hand_values(rtc_seed_hand.reshape(-1)[:10]),
            "observation_hand_state": _named_hand_values(observation_hand_q),
            "executor_reference_hand": _named_hand_values(self.reference_hand_q),
            "executor_sim_hand": _named_hand_values(self.sim_hand_q),
            "negative_floor_warning": _negative_floor_warning(clipped_hand),
        }

    def _clip_policy_hand_q(self, hand_q: np.ndarray) -> np.ndarray:
        values = np.asarray(hand_q, dtype=np.float32).reshape(-1)[:10].copy()
        if values.size < 10:
            padded = np.zeros(10, dtype=np.float32)
            padded[: values.size] = values
            values = padded
        values = np.clip(values, POLICY_HAND_CLIP_LOWER, POLICY_HAND_CLIP_UPPER)
        return values.astype(np.float32, copy=False)

    def _project_hand_q_for_sim(self, hand_q: np.ndarray) -> np.ndarray:
        values = self._clip_policy_hand_q(hand_q)
        limits_by_name = self.assembly.get("linker_hand_joint_limits_by_name", {})
        if not isinstance(limits_by_name, dict):
            limits_by_name = {}
        projected = values.copy()
        for idx, name in enumerate(POLICY_HAND_JOINT_NAMES):
            limit = limits_by_name.get(name)
            if isinstance(limit, (tuple, list)) and len(limit) == 2:
                lower, upper = float(limit[0]), float(limit[1])
            else:
                lower, upper = 0.0, POLICY_HAND_CLIP_UPPER
            projected[idx] = float(np.clip(projected[idx], lower, upper))
        return projected.astype(np.float32, copy=False)

    @staticmethod
    def _action_step(value: np.ndarray, index: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 2:
            arr = arr[min(index, arr.shape[0] - 1)]
        return arr.reshape(-1)

    def _solve_and_apply_eef_target(self) -> None:
        target_quat = np.asarray(_rotation_to_quat_xyzw(self.target_rotation), dtype=np.float32)
        target_quat_wxyz = np.asarray(
            (target_quat[3], target_quat[0], target_quat[1], target_quat[2]),
            dtype=np.float32,
        )
        qpos_init = _tensor_to_np(self.arm.get_qpos()).reshape(-1).astype(np.float32)
        qpos_init[self.arm_dofs] = self.q_cmd
        qpos_init = self._apply_ik_joint4_search_limit(qpos_init)
        try:
            qpos, error = self.arm.inverse_kinematics(
                link=self.eef_link,
                pos=self.target_xyz.astype(np.float32),
                quat=target_quat_wxyz,
                init_qpos=qpos_init,
                dofs_idx_local=self.arm_dofs,
                max_samples=1,
                max_solver_iters=32,
                damping=0.02,
                pos_tol=self.pos_tol,
                max_step_size=self.ik_solver_max_joint_step,
                return_error=True,
            )
            limited_qpos = self._apply_ik_joint4_search_limit(_tensor_to_np(qpos).reshape(-1).astype(np.float32))
            solved = limited_qpos[self.arm_dofs].astype(np.float32)
            dq = np.clip(solved - self.q_cmd, -self.max_joint_step, self.max_joint_step)
            if self.min_joint_step > 0.0:
                dq = np.where(np.abs(dq) < self.min_joint_step, 0.0, dq)
            self.q_cmd = self.q_cmd + dq
            self.arm.set_dofs_position(self.q_cmd, self.arm_dofs, zero_velocity=True)
            self.arm.control_dofs_position(self.q_cmd, self.arm_dofs)
            err = tuple(round(float(v), 5) for v in _tensor_to_np(error).reshape(-1))
            print(
                f"[policy-step] eef_target={tuple(round(float(v), 4) for v in self.target_xyz)} "
                f"ik_error={err} ik_seed_source=command_q_state",
                flush=True,
            )
        except Exception as exc:
            print(f"[policy-step] IK failed: {exc}", flush=True)

    def _apply_joint_target(self, joint_target: np.ndarray) -> None:
        if joint_target.size < 7:
            return
        target_q = np.asarray(joint_target[:7], dtype=np.float32)
        dq = np.clip(target_q - self.q_cmd, -self.max_joint_step, self.max_joint_step)
        if self.min_joint_step > 0.0:
            dq = np.where(np.abs(dq) < self.min_joint_step, 0.0, dq)
        self.q_cmd = self.q_cmd + dq
        self.arm.set_dofs_position(self.q_cmd, self.arm_dofs, zero_velocity=True)
        self.arm.control_dofs_position(self.q_cmd, self.arm_dofs)
        print(f"[policy-step] arm_joint_target={tuple(round(float(v), 4) for v in self.q_cmd)}", flush=True)

    def _apply_linker_hand_target(self) -> None:
        joint_names = POLICY_HAND_JOINT_NAMES
        if not joint_names:
            return

        class _HandTarget:
            def __init__(self, names: tuple[str, ...], values: np.ndarray) -> None:
                self.joint_names = names
                self.joint_positions = tuple(float(v) for v in values[: len(names)])

        self._hand_target_print_count += 1
        if self._should_print_hand():
            print(
                f"[policy-hand] target={_named_hand_values(self.reference_hand_q)} "
                f"sim={_named_hand_values(self.sim_hand_q)}",
                flush=True,
            )
        harness._set_linker_hand_target(self.assembly, "right", _HandTarget(joint_names, self.sim_hand_q))  # noqa: SLF001
        self._force_linker_hand_q(self.sim_hand_q)

    def _force_linker_hand_q(self, hand_q: np.ndarray) -> bool:
        """Synchronize Genesis' actual L10 qpos with the current command.

        Remote Nero runtime applies LinkerHand targets by setting and controlling
        the DOFs in the same update. Keep this policy rollout on that path so
        reported hand state, command target, and the next RTC reference agree.
        """
        linker_hand = self.assembly.get("linker_hand")
        if linker_hand is None:
            return False
        assembly_names = list(self.assembly.get("linker_hand_joint_names", ()))
        assembly_dofs = list(self.assembly.get("linker_hand_dofs", ()))
        if not assembly_names or not assembly_dofs:
            return False

        command = self._project_hand_q_for_sim(hand_q)
        values_by_name = {
            name: float(value)
            for name, value in zip(POLICY_HAND_JOINT_NAMES, command, strict=False)
        }
        mimic_by_name = self.assembly.get("linker_hand_mimic_by_name", {})
        if isinstance(mimic_by_name, dict):
            for mimic_name, spec in mimic_by_name.items():
                try:
                    source_name, multiplier, offset = spec
                except (TypeError, ValueError):
                    continue
                if str(mimic_name) not in values_by_name and str(source_name) in values_by_name:
                    values_by_name[str(mimic_name)] = float(multiplier) * float(values_by_name[str(source_name)]) + float(offset)

        limits_by_name = self.assembly.get("linker_hand_joint_limits_by_name", {})
        default_limits = (-np.inf, np.inf)
        values = np.asarray(
            [
                float(
                    np.clip(
                        values_by_name.get(str(name), 0.0),
                        *(limits_by_name.get(str(name), default_limits) if isinstance(limits_by_name, dict) else default_limits),
                    )
                )
                for name in assembly_names
            ],
            dtype=np.float32,
        )
        try:
            linker_hand.set_dofs_position(values, assembly_dofs, zero_velocity=True)
            linker_hand.control_dofs_position(values, assembly_dofs)
        except Exception as exc:
            print(f"[policy-hand] force_qpos skipped: {exc}", flush=True)
            return False
        return True

    def _set_initial_arm_poses(self) -> None:
        harness._set_arm_initial_pose(self.arm, self.initial_right_arm_q)  # noqa: SLF001
        if self.left_arm is not None:
            try:
                harness._set_arm_initial_pose(self.left_arm, self.initial_left_arm_q)  # noqa: SLF001
            except Exception as exc:
                print(f"[policy-ik] left initial pose skipped: {exc}", flush=True)

    def _freeze_target_at_current_eef(self) -> None:
        pose = self.current_eef_pose()
        self.target_rotation = pose[:3, :3].copy()
        self.target_xyz = pose[:3, 3].copy()

    def _warmup_ik(self) -> None:
        qpos_init = _tensor_to_np(self.arm.get_qpos()).reshape(-1).astype(np.float32)
        qpos_init[self.arm_dofs] = self.q_cmd
        qpos_init = self._apply_ik_joint4_search_limit(qpos_init)
        target_quat = np.asarray(_rotation_to_quat_xyzw(self.target_rotation), dtype=np.float32)
        target_quat_wxyz = np.asarray((target_quat[3], target_quat[0], target_quat[1], target_quat[2]), dtype=np.float32)
        started = time.perf_counter()
        try:
            self.arm.inverse_kinematics(
                link=self.eef_link,
                pos=self.target_xyz.astype(np.float32),
                quat=target_quat_wxyz,
                init_qpos=qpos_init,
                dofs_idx_local=self.arm_dofs,
                max_samples=1,
                max_solver_iters=32,
                damping=0.02,
                pos_tol=self.pos_tol,
                max_step_size=self.ik_solver_max_joint_step,
                return_error=True,
            )
            print(f"[policy-ik] warmup_s={time.perf_counter() - started:.4f}", flush=True)
        except Exception as exc:
            print(f"[policy-ik] warmup failed: {exc}", flush=True)

    def _apply_ik_joint4_search_limit(self, qpos: np.ndarray) -> np.ndarray:
        limited = np.asarray(qpos, dtype=np.float32).copy()
        if not self.ik_j4_limit or len(self.arm_dofs) < 4:
            return limited
        dof_index = int(self.arm_dofs[3])
        lower, upper = (float(v) for v in self.ik_j4_limit_rad)
        if lower > upper:
            lower, upper = upper, lower
        before = float(limited[dof_index])
        after = float(np.clip(before, lower, upper))
        if before != after:
            limited[dof_index] = after
            self._ik_joint_limit_hit_count += 1
            if self._ik_joint_limit_hit_count == 1 or self._ik_joint_limit_hit_count % 60 == 0:
                print(
                    f"[policy-ik] joint4_limit before={before:+.3f} after={after:+.3f} "
                    f"range=({lower:+.3f},{upper:+.3f})",
                    flush=True,
                )
        return limited

    def _should_print_hand(self) -> bool:
        every = int(self.print_hand_every)
        return every > 0 and (self._hand_target_print_count == 1 or self._hand_target_print_count % every == 0)

    def _current_linker_hand_positions(self) -> dict[str, float]:
        qpos = self._current_linker_hand_q(fallback=None)
        if qpos is None:
            return {}
        return {
            name: round(float(value), 4)
            for name, value in zip(POLICY_HAND_JOINT_NAMES, qpos, strict=False)
        }

    def _current_linker_hand_q(self, *, fallback: np.ndarray | None) -> np.ndarray | None:
        linker_hand = self.assembly.get("linker_hand")
        if linker_hand is None:
            return None if fallback is None else np.asarray(fallback, dtype=np.float32).reshape(-1)[:10].copy()
        assembly_names = list(self.assembly.get("linker_hand_joint_names", ()))
        assembly_dofs = list(self.assembly.get("linker_hand_dofs", ()))
        if not assembly_names or not assembly_dofs:
            return None if fallback is None else np.asarray(fallback, dtype=np.float32).reshape(-1)[:10].copy()
        try:
            qpos = _tensor_to_np(linker_hand.get_qpos()).reshape(-1)
        except Exception:
            return None if fallback is None else np.asarray(fallback, dtype=np.float32).reshape(-1)[:10].copy()
        by_name = {
            name: float(qpos[int(dof)])
            for name, dof in zip(assembly_names, assembly_dofs, strict=False)
            if int(dof) < qpos.size
        }
        values = np.asarray(fallback if fallback is not None else np.zeros(10, dtype=np.float32), dtype=np.float32).reshape(-1)
        out = np.zeros(10, dtype=np.float32)
        out[: min(10, values.size)] = values[: min(10, values.size)]
        for idx, name in enumerate(POLICY_HAND_JOINT_NAMES):
            if name in by_name:
                out[idx] = float(by_name[name])
        return out


def _named_hand_values(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        name: round(float(value), 4)
        for name, value in zip(POLICY_HAND_JOINT_NAMES, arr[: len(POLICY_HAND_JOINT_NAMES)], strict=False)
    }


def _hand_chunk_debug(values: np.ndarray | None) -> dict[str, object] | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[-1] < len(POLICY_HAND_JOINT_NAMES):
        return {"shape": list(arr.shape)}
    return {
        "shape": list(arr.shape),
        "first": _named_hand_values(arr[0, :10]),
        "range": {
            name: [
                round(float(np.min(arr[:, idx])), 4),
                round(float(np.max(arr[:, idx])), 4),
            ]
            for idx, name in enumerate(POLICY_HAND_JOINT_NAMES)
        },
    }


def _hand_chunk_delta_debug(values: np.ndarray | None, reference: np.ndarray) -> dict[str, object] | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[-1] < len(POLICY_HAND_JOINT_NAMES):
        return {"shape": list(arr.shape)}
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    if ref.size < len(POLICY_HAND_JOINT_NAMES):
        padded = np.zeros(len(POLICY_HAND_JOINT_NAMES), dtype=np.float64)
        padded[: ref.size] = ref
        ref = padded
    delta = arr[:, : len(POLICY_HAND_JOINT_NAMES)] - ref[: len(POLICY_HAND_JOINT_NAMES)][None, :]
    return {
        "shape": list(delta.shape),
        "first": _named_hand_values(delta[0]),
        "range": {
            name: [
                round(float(np.min(delta[:, idx])), 4),
                round(float(np.max(delta[:, idx])), 4),
            ]
            for idx, name in enumerate(POLICY_HAND_JOINT_NAMES)
        },
    }


def _negative_floor_warning(values: np.ndarray | None) -> dict[str, object]:
    if values is None:
        return {"active": False}
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[None, :]
    watched = ("middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch")
    hits = {}
    for name in watched:
        idx = POLICY_HAND_JOINT_NAMES.index(name)
        if arr.ndim == 2 and arr.shape[-1] > idx:
            hits[name] = bool(np.any(arr[:, idx] <= POLICY_HAND_CLIP_LOWER + 1.0e-5))
    return {"active": any(hits.values()), "hits": hits}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _jsonable_action(action: dict[str, np.ndarray] | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return {
        str(key): _jsonable(np.asarray(value, dtype=np.float32))
        for key, value in action.items()
    }


def _append_jsonl(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


class DryRunPolicy:
    def __init__(self, modality_config: dict[str, Any]) -> None:
        self.modality_config = modality_config
        self.count = 0

    def get_modality_config(self) -> dict[str, Any]:
        return self.modality_config

    def reset(self) -> dict[str, Any]:
        self.count = 0
        return {}

    def get_action(self, observation: dict[str, Any], options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.count += 1
        reference = (options or {}).get("reference_action") or {}
        out: dict[str, np.ndarray] = {}
        horizon = len(self.modality_config["action"].delta_indices)
        if "eef_9d" in self.modality_config["action"].modality_keys:
            eef = np.asarray(reference.get("eef_9d", np.zeros((1, 9), dtype=np.float32)), dtype=np.float32)
            chunk = np.repeat(eef[:, None, :], horizon, axis=1)
            chunk[:, :, 0] += 0.015 * math.sin(self.count * 0.15)
            out["eef_9d"] = chunk.astype(np.float32)
        if "hand_joint_target" in self.modality_config["action"].modality_keys:
            hand = np.asarray(reference.get("hand_joint_target", np.zeros((1, 10), dtype=np.float32)), dtype=np.float32)
            out["hand_joint_target"] = np.repeat(hand[:, None, :], horizon, axis=1).astype(np.float32)
        if "arm_joint_target" in self.modality_config["action"].modality_keys:
            arm = np.asarray(reference.get("arm_joint_target", np.zeros((1, 7), dtype=np.float32)), dtype=np.float32)
            out["arm_joint_target"] = np.repeat(arm[:, None, :], horizon, axis=1).astype(np.float32)
        return out, {"dry_run": True}


def _prepare_local_checkpoint(checkpoint: Path, cosmos_model: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    checkpoint = checkpoint.expanduser().resolve()
    cosmos_model = cosmos_model.expanduser().resolve()
    config_path = checkpoint / "config.json"
    processor_path = checkpoint / "processor_config.json"
    if not config_path.exists() or not processor_path.exists():
        raise FileNotFoundError(f"Not a GR00T checkpoint: {checkpoint}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    processor_cfg = json.loads(processor_path.read_text(encoding="utf-8"))
    changed = False

    def model_path_is_accessible(path_text: str) -> bool:
        try:
            return Path(path_text).expanduser().exists()
        except OSError:
            return False

    for data in (cfg, processor_cfg.get("processor_kwargs", {})):
        old_model_name = str(data.get("model_name", ""))
        if old_model_name and not model_path_is_accessible(old_model_name):
            data["model_name"] = str(cosmos_model)
            changed = True
    if not changed:
        return checkpoint, None

    tmp = tempfile.TemporaryDirectory(prefix="harness_groot_checkpoint_")
    tmp_path = Path(tmp.name)
    for child in checkpoint.iterdir():
        dst = tmp_path / child.name
        if child.is_dir():
            if child.name == "experiment_cfg":
                shutil.copytree(child, dst)
            else:
                os.symlink(child, dst, target_is_directory=True)
        elif child.name == "config.json":
            dst.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        elif child.name == "processor_config.json":
            dst.write_text(json.dumps(processor_cfg, indent=2), encoding="utf-8")
        else:
            os.symlink(child, dst)
    print(f"[policy] patched temporary checkpoint model_name -> {cosmos_model}", flush=True)
    return tmp_path, tmp


def _load_groot_policy(args: argparse.Namespace):
    if str(ISAAC_GROOT_ROOT) not in sys.path:
        sys.path.insert(0, str(ISAAC_GROOT_ROOT))
    if not ISAAC_GROOT_ROOT.exists():
        raise FileNotFoundError(f"Isaac-GR00T source not found: {ISAAC_GROOT_ROOT}")

    import gr00t  # noqa: F401
    import gr00t.model  # noqa: F401
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy import Gr00tPolicy

    checkpoint, tmp = _prepare_local_checkpoint(Path(args.policy_checkpoint), Path(args.cosmos_model))
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=str(checkpoint),
        device=str(args.policy_device),
        strict=not bool(args.no_policy_strict),
    )
    _patch_pytorch_action_head_rtc(getattr(policy, "model", None))
    policy._harness_tempdir = tmp  # keep symlink tree alive
    return policy


def _load_modality_config_for_dry_run() -> dict[str, Any]:
    if str(ISAAC_GROOT_ROOT) not in sys.path:
        sys.path.insert(0, str(ISAAC_GROOT_ROOT))
    import gr00t.model  # noqa: F401
    from gr00t.data.embodiment_tags import EmbodimentTag

    config_path = ISAAC_GROOT_ROOT / "examples" / "IsaacLab" / "nero_right_l10_multiview_modality_config.py"
    sys.path.insert(0, str(config_path.parent))
    __import__(config_path.stem)
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    return MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]


def _make_policy(args: argparse.Namespace):
    if args.dry_run_policy:
        return DryRunPolicy(_load_modality_config_for_dry_run())
    return _load_groot_policy(args)


def _create_policy_cameras(scene: object, *, image_size: tuple[int, int]) -> tuple[object | None, object | None]:
    d455 = getattr(scene, "d455_rgb_camera", None)
    d405 = getattr(scene, "right_d405_camera", None)
    print(
        "[camera] model_inputs "
        f"ego_view={'d455_rgb_camera' if d455 is not None else 'missing'} "
        f"wrist_view={'right_d405_camera' if d405 is not None else 'missing'} "
        f"target_size={image_size}",
        flush=True,
    )
    if d455 is None or d405 is None:
        raise RuntimeError("GR00T finetune policy requires both D455 ego_view and D405 wrist_view cameras.")
    return d455, d405


def _validate_policy_video_schema(modality_config: dict[str, Any]) -> None:
    video_keys = tuple(str(key) for key in modality_config["video"].modality_keys)
    expected = {"ego_view", "wrist_view"}
    missing = expected.difference(video_keys)
    if missing:
        raise RuntimeError(
            "The finetune policy must receive two video streams named "
            f"ego_view and wrist_view; missing={sorted(missing)} actual={list(video_keys)}"
        )
    print(f"[policy] video schema ok: {list(video_keys)}", flush=True)


def _print_first_observation_video_shapes(observation: dict[str, Any]) -> None:
    video = observation.get("video", {})
    if not isinstance(video, dict):
        return
    parts = []
    for key, value in video.items():
        arr = np.asarray(value)
        parts.append(f"{key}:shape={arr.shape},dtype={arr.dtype}")
    print("[policy] observation video " + " ".join(parts), flush=True)


def _summarize_action_chunk(action: dict[str, np.ndarray]) -> str:
    parts = []
    for key in sorted(action):
        arr = np.asarray(action[key])
        if arr.size:
            finite = arr[np.isfinite(arr)]
            if finite.size:
                min_value = float(np.min(finite))
                max_value = float(np.max(finite))
                mean_value = float(np.mean(finite))
                summary = f"{key}:shape={arr.shape},min={min_value:.4f},max={max_value:.4f},mean={mean_value:.4f}"
                if key == "hand_joint_target":
                    first = np.asarray(arr, dtype=np.float64)
                    while first.ndim > 1:
                        first = first[0]
                    named = {
                        name: round(float(value), 4)
                        for name, value in zip(POLICY_HAND_JOINT_NAMES, first[: len(POLICY_HAND_JOINT_NAMES)], strict=False)
                    }
                    summary += f",first={named}"
                    by_step = np.asarray(arr, dtype=np.float64)
                    if by_step.ndim == 3:
                        by_step = by_step[0]
                    if by_step.ndim == 2 and by_step.shape[-1] >= len(POLICY_HAND_JOINT_NAMES):
                        hand_ranges = {
                            name: (
                                round(float(np.min(by_step[:, idx])), 4),
                                round(float(np.max(by_step[:, idx])), 4),
                            )
                            for idx, name in enumerate(POLICY_HAND_JOINT_NAMES)
                        }
                        summary += f",range={hand_ranges}"
                parts.append(summary)
            else:
                parts.append(f"{key}:shape={arr.shape},nonfinite")
        else:
            parts.append(f"{key}:shape={arr.shape},empty")
    return " ".join(parts)


def _unbatch_action_dict(action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    previous_action: dict[str, np.ndarray] = {}
    for key, value in action.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        previous_action[key] = arr.astype(np.float32, copy=True)
    return previous_action


def _first_batch_chunk(action: dict[str, np.ndarray], key: str) -> np.ndarray:
    value = np.asarray(action[key], dtype=np.float32)
    if value.ndim == 3:
        value = value[0]
    return value.astype(np.float32, copy=False)


def _first_step_reference(action: dict[str, np.ndarray] | None) -> dict[str, np.ndarray] | None:
    if action is None:
        return None
    reference: dict[str, np.ndarray] = {}
    for key, value in action.items():
        chunk = np.asarray(value, dtype=np.float32)
        if chunk.ndim == 3:
            chunk = chunk[0]
        if chunk.ndim != 2 or chunk.shape[0] == 0:
            continue
        reference[key] = chunk[:1].astype(np.float32, copy=True)
    return reference or None


def _stored_rtc_action_chunk(
    *,
    policy_action: dict[str, np.ndarray],
    rtc_seed_action: dict[str, np.ndarray] | None,
    action_keys: list[str],
    frozen_steps: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    stored = {
        key: _first_batch_chunk(policy_action, key).astype(np.float32, copy=True)
        for key in action_keys
        if key in policy_action
    }
    metadata: dict[str, Any] = {
        "raw_action_shape": "T,D",
        "policy_raw_action_field": "policy_raw_action",
        "frozen_seed_steps_requested": int(max(0, frozen_steps)),
        "frozen_seed_steps_applied": 0,
    }
    if rtc_seed_action is None or frozen_steps <= 0:
        return stored, metadata
    applied = 0
    for key in action_keys:
        if key not in stored or key not in rtc_seed_action:
            continue
        seed = _first_batch_chunk(rtc_seed_action, key)
        count = min(int(frozen_steps), stored[key].shape[0], seed.shape[0])
        if count <= 0:
            continue
        stored[key][:count] = seed[:count]
        applied = max(applied, count)
    metadata["frozen_seed_steps_applied"] = int(applied)
    return stored, metadata


def _elapsed_action_steps(
    *,
    previous_start_monotonic_s: float | None,
    current_observation_monotonic_s: float,
    action_dt_s: float,
    fallback_steps: int,
    max_steps: int,
) -> int:
    if previous_start_monotonic_s is None or action_dt_s <= 0.0:
        return max(0, min(int(fallback_steps), int(max_steps)))
    elapsed_s = max(0.0, float(current_observation_monotonic_s) - float(previous_start_monotonic_s))
    elapsed_steps = int(math.floor(elapsed_s / float(action_dt_s) + 0.5))
    return max(0, min(elapsed_steps, int(max_steps)))


class TeleopRtcSeedManager:
    """Minimal probe-style action trajectory seed manager."""

    def __init__(
        self,
        *,
        action_keys: list[str],
        action_dt_s: float,
        horizon: int,
        max_chunks: int = 4,
    ) -> None:
        self.action_keys = list(action_keys)
        self.action_dt_s = float(action_dt_s)
        self.horizon = int(horizon)
        self.max_chunks = max(1, int(max_chunks))
        self._epoch_s: float | None = None
        self._chunks: list[dict[str, Any]] = []

    def clear(self) -> None:
        self._epoch_s = None
        self._chunks.clear()

    def time_to_step(self, timestamp_s: float) -> int:
        if self._epoch_s is None:
            self._epoch_s = float(timestamp_s)
        return int(round((float(timestamp_s) - float(self._epoch_s)) / self.action_dt_s))

    def push(self, action: dict[str, np.ndarray], *, start_monotonic_s: float, frame_id: int) -> None:
        chunk = {
            "start_monotonic_s": float(start_monotonic_s),
            "start_step": self.time_to_step(float(start_monotonic_s)),
            "frame_id": int(frame_id),
            "action": {
                key: _first_batch_chunk(action, key).astype(np.float32, copy=True)
                for key in self.action_keys
                if key in action
            },
        }
        self._chunks.append(chunk)
        if len(self._chunks) > self.max_chunks:
            del self._chunks[: len(self._chunks) - self.max_chunks]

    def seed_window(
        self,
        *,
        anchor_start_monotonic_s: float,
        anchor_start_frame_id: int,
        horizon: int,
    ) -> tuple[dict[str, np.ndarray] | None, float | None, dict[str, Any]]:
        if not self._chunks:
            return None, None, {"reason": "empty"}
        start_step = self.time_to_step(float(anchor_start_monotonic_s))
        indexed = self._latest_action_rows_by_step()
        rows = []
        for action_step in range(start_step, start_step + int(horizon)):
            row = indexed.get(action_step)
            if row is None:
                break
            rows.append(row)
        if not rows:
            return None, None, {"reason": "no_seed_window", "start_action_step": int(start_step)}
        valid_steps = len(rows)
        if valid_steps < int(horizon):
            last = rows[-1]
            for _ in range(valid_steps, int(horizon)):
                rows.append(last)
        seed_action = {
            key: np.stack([row[key] for row in rows], axis=0).astype(np.float32)
            for key in self.action_keys
            if key in rows[0]
        }
        metadata = {
            "reason": "ok",
            "source": "teleop_trajectory_manager",
            "seed_steps": int(len(rows)),
            "seed_valid_steps": int(valid_steps),
            "seed_padded_steps": int(len(rows) - valid_steps),
            "seed_pad_mode": "repeat_last" if len(rows) > valid_steps else "none",
            "start_action_step": int(start_step),
            "anchor_start_monotonic_s": float(anchor_start_monotonic_s),
            "anchor_start_frame_id": int(anchor_start_frame_id),
            "action_keys": sorted(seed_action.keys()),
        }
        return seed_action, float(anchor_start_monotonic_s), metadata

    def _latest_action_rows_by_step(self) -> dict[int, dict[str, np.ndarray]]:
        indexed: dict[int, dict[str, np.ndarray]] = {}
        for chunk in self._chunks:
            start_step = int(chunk["start_step"])
            action = chunk["action"]
            if not action:
                continue
            chunk_horizon = min(value.shape[0] for value in action.values())
            for offset in range(chunk_horizon):
                indexed[start_step + offset] = {
                    key: value[offset].astype(np.float32, copy=True)
                    for key, value in action.items()
                }
        return indexed


def _teleop_rtc_options(
    *,
    enabled: bool,
    rtc_mode: str,
    previous_action: dict[str, np.ndarray] | None,
    previous_action_start_monotonic_s: float | None,
    current_observation_monotonic_s: float,
    action_dt_s: float,
    fallback_replan_horizon: int,
    max_overlap_steps: int | None,
    frozen_steps: int,
    ramp_rate: float,
    guidance_beta: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "enabled": bool(enabled),
        "mode": str(rtc_mode),
        "action_dt_s": float(action_dt_s),
        "fallback_replan_horizon": int(fallback_replan_horizon),
        "max_overlap_steps": max_overlap_steps,
        "guidance_beta": float(guidance_beta),
        "previous_action_source": "teleop_trajectory_manager" if previous_action is not None else "none",
    }
    if not enabled or str(rtc_mode) == "off" or previous_action is None:
        metadata["reason"] = "disabled_or_no_previous_action"
        return None, metadata
    previous_horizon = min(int(value.shape[0]) for value in previous_action.values())
    elapsed_steps = _elapsed_action_steps(
        previous_start_monotonic_s=previous_action_start_monotonic_s,
        current_observation_monotonic_s=float(current_observation_monotonic_s),
        action_dt_s=float(action_dt_s),
        fallback_steps=int(fallback_replan_horizon),
        max_steps=previous_horizon,
    )
    raw_overlap_steps = max(0, min(previous_horizon - elapsed_steps, previous_horizon))
    overlap_steps = (
        raw_overlap_steps
        if max_overlap_steps is None
        else min(raw_overlap_steps, max(0, int(max_overlap_steps)))
    )
    previous_start_step = max(0, min(int(elapsed_steps), int(previous_horizon)))
    metadata.update(
        {
            "previous_horizon": int(previous_horizon),
            "elapsed_steps": int(elapsed_steps),
            "raw_overlap_steps": int(raw_overlap_steps),
            "overlap_steps": int(overlap_steps),
            "previous_start_step": int(previous_start_step),
            "previous_action_start_monotonic_s": previous_action_start_monotonic_s,
            "current_observation_monotonic_s": float(current_observation_monotonic_s),
        }
    )
    if overlap_steps <= 0:
        metadata["reason"] = "no_overlap"
        return None, metadata
    options = {
        "action_horizon": int(previous_horizon),
        "rtc_mode": str(rtc_mode),
        "rtc_overlap_steps": int(overlap_steps),
        "rtc_frozen_steps": max(0, min(int(frozen_steps), int(overlap_steps))),
        "rtc_ramp_rate": float(ramp_rate),
        "rtc_guidance_beta": float(guidance_beta),
        "rtc_previous_start_step": int(previous_start_step),
    }
    metadata["options"] = dict(options)
    metadata["reason"] = "ok"
    return options, metadata


def _action_input_previous_action(action_input: Any):
    try:
        if "action" in action_input:
            return action_input["action"]
    except TypeError:
        pass
    return getattr(action_input, "action", None)


def _initial_actions_with_teleop_rtc(
    action_head: Any,
    action_input: Any,
    options: dict[str, Any] | None,
    *,
    batch_size: int,
    dtype: Any,
    device: Any,
):
    import torch

    if hasattr(action_head, "init_actions"):
        actions = (
            action_head.init_actions.expand((batch_size, -1, -1))
            .to(dtype=dtype, device=device)
            .clone()
        )
    else:
        actions = torch.randn(
            size=(batch_size, action_head.config.action_horizon, action_head.action_dim),
            dtype=dtype,
            device=device,
        )
    vel_strength = torch.ones_like(actions)
    previous_action = _action_input_previous_action(action_input)
    if previous_action is None:
        return actions, vel_strength
    if options is None:
        raise AssertionError("options is not None")
    for key in ("action_horizon", "rtc_overlap_steps", "rtc_frozen_steps", "rtc_ramp_rate"):
        if key not in options:
            raise AssertionError(f"{key} is not in options")

    previous_action = previous_action.to(dtype=dtype, device=device)
    action_slice_end = max(0, min(int(options["action_horizon"]), previous_action.shape[1]))
    overlap_steps = max(0, min(int(options["rtc_overlap_steps"]), action_slice_end, actions.shape[1]))
    frozen_steps = max(0, min(int(options["rtc_frozen_steps"]), overlap_steps))
    if overlap_steps <= 0:
        return actions, vel_strength
    if "rtc_previous_start_step" in options:
        start = max(0, min(int(options["rtc_previous_start_step"]), action_slice_end))
        end = min(action_slice_end, start + overlap_steps)
        overlap_steps = max(0, min(overlap_steps, end - start))
        frozen_steps = max(0, min(frozen_steps, overlap_steps))
        if overlap_steps <= 0:
            return actions, vel_strength
    else:
        end = action_slice_end
        start = end - overlap_steps

    actions[:, :overlap_steps, :] = previous_action[:, start:end, :]
    vel_strength[:, :frozen_steps, :] = 0.0
    intermediate_steps = overlap_steps - frozen_steps
    if intermediate_steps > 0:
        ramp = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
        ramp = 1 - torch.exp(-float(options["rtc_ramp_rate"]) * ramp)
        ramp = ramp / ramp[-1].clamp_min(1.0e-8)
        vel_strength[:, frozen_steps:overlap_steps, :] = ramp[1:-1][None, :, None].to(
            dtype=dtype,
            device=device,
        )
    return actions, vel_strength


def _action_head_get_action_with_features_with_teleop_rtc(
    self: Any,
    backbone_features: Any,
    state_features: Any,
    embodiment_id: Any,
    backbone_output: Any,
    action_input: Any,
    options: dict[str, Any] | None = None,
) -> Any:
    import torch
    from transformers.feature_extraction_utils import BatchFeature

    with torch.no_grad():
        vl_embeds = backbone_features
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions, vel_strength = _initial_actions_with_teleop_rtc(
            self,
            action_input,
            options,
            batch_size=batch_size,
            dtype=vl_embeds.dtype,
            device=device,
        )
        dt = 1.0 / self.num_inference_timesteps
        for timestep_index in range(self.num_inference_timesteps):
            t_cont = timestep_index / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,),
                fill_value=t_discretized,
                device=device,
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            sa_embs = torch.cat((state_features, action_features), dim=1)
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]
            actions = actions + dt * pred_velocity * vel_strength

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )


def _patch_pytorch_action_head_rtc(model: Any) -> None:
    action_head = getattr(model, "action_head", None)
    if action_head is None or not hasattr(action_head, "get_action_with_features"):
        return
    if getattr(action_head, "_harness_probe_rtc_patch_status", None) == "enabled":
        return
    action_head._harness_probe_original_get_action_with_features = action_head.get_action_with_features
    action_head.get_action_with_features = partial(
        _action_head_get_action_with_features_with_teleop_rtc,
        action_head,
    )
    action_head._harness_probe_rtc_patch_status = "enabled"
    print("[policy] patched PyTorch action head with probe Teleop RTC seed-window behavior", flush=True)


def _legacy_rtc_options(
    *,
    enabled: bool,
    action_horizon: int,
    replan_horizon: int,
    previous_action: dict[str, np.ndarray] | None,
    overlap_steps: int | None,
    frozen_steps: int,
    ramp_rate: float,
) -> dict[str, Any] | None:
    if not enabled or previous_action is None:
        return None
    previous_horizon = min(int(np.asarray(value).shape[0]) for value in previous_action.values())
    overlap = int(overlap_steps) if overlap_steps is not None else int(action_horizon) - int(replan_horizon)
    overlap = max(0, min(overlap, previous_horizon, int(action_horizon)))
    if overlap <= 0:
        print(
            "[policy] rtc disabled for this replan because overlap is 0; "
            f"action_horizon={action_horizon} replan_horizon={replan_horizon}",
            flush=True,
        )
        return None
    frozen = max(0, min(int(frozen_steps), overlap))
    return {
        "action_horizon": int(previous_horizon),
        "rtc_overlap_steps": int(overlap),
        "rtc_frozen_steps": int(frozen),
        "rtc_ramp_rate": float(ramp_rate),
    }


def _rec_to_dtype(value: Any, *, dtype: Any) -> Any:
    try:
        import torch
    except Exception:
        return value
    if isinstance(value, torch.Tensor):
        if torch.is_floating_point(value):
            return value.to(dtype=dtype)
        return value
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _rec_to_dtype(item, dtype=dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_rec_to_dtype(item, dtype=dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_rec_to_dtype(item, dtype=dtype) for item in value)
    return value


def _policy_get_action_cpu_processor(
    policy: object,
    observation: dict[str, Any],
    *,
    reference_action: dict[str, np.ndarray],
    previous_action: dict[str, np.ndarray] | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run GR00T inference with CPU preprocessing and optional previous action chunk.

    Genesis GPU mode can leave torch's default device as CUDA. GR00T's processor
    mixes torch.from_numpy(...) CPU tensors with torch.zeros(...) default-device
    tensors, so force the processor/collator phase back to CPU for consistency. This
    mirrors the Isaac-GR00T evaluation path: previous absolute action chunks are
    placed in VLAStepData.actions for RTC/action-context, while reference_action is
    passed in metadata/options for action-relative decoding.
    """
    try:
        import torch
    except Exception:
        merged_options = dict(options or {})
        merged_options["reference_action"] = reference_action
        return policy.get_action(observation, options=merged_options)

    if not hasattr(torch, "set_default_device") or not hasattr(torch, "get_default_device"):
        merged_options = dict(options or {})
        merged_options["reference_action"] = reference_action
        return policy.get_action(observation, options=merged_options)

    previous_device = torch.get_default_device()
    try:
        torch.set_default_device("cpu")
        if not all(
            hasattr(policy, name)
            for name in ("_unbatch_observation", "processor", "collate_fn", "model", "modality_configs")
        ):
            merged_options = dict(options or {})
            merged_options["reference_action"] = reference_action
            return policy.get_action(observation, options=merged_options)

        if getattr(policy, "strict", False):
            policy.check_observation(observation)

        from gr00t.data.types import MessageType, VLAStepData

        unbatched_observations = policy._unbatch_observation(observation)  # noqa: SLF001
        if previous_action is not None and len(unbatched_observations) != 1:
            raise ValueError("previous_action currently supports batch size 1.")

        processed_inputs = []
        states = []
        for obs in unbatched_observations:
            states.append(obs["state"])
            vla_step_data = VLAStepData(
                images=obs["video"],
                states=obs["state"],
                actions={} if previous_action is None else previous_action,
                text=obs["language"][policy.language_key][0],
                embodiment=policy.embodiment_tag,
                metadata={"reference_action": reference_action},
            )
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(policy.processor(messages))
        collated_inputs = policy.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        merged_options = dict(options or {})
        merged_options["reference_action"] = reference_action
        with torch.inference_mode():
            try:
                model_pred = policy.model.get_action(**collated_inputs, options=merged_options)
            except TypeError:
                model_pred = policy.model.get_action(**collated_inputs)
        normalized_action = model_pred["action_pred"].float()

        batched_states = {
            key: np.stack([state[key] for state in states], axis=0)
            for key in policy.modality_configs["state"].modality_keys
        }
        unnormalized_action = policy.processor.decode_action(
            normalized_action.cpu().numpy(),
            policy.embodiment_tag,
            batched_states,
            reference_action=reference_action,
        )
        casted_action = {key: value.astype(np.float32) for key, value in unnormalized_action.items()}
        if getattr(policy, "strict", False):
            policy.check_action(casted_action)
        return casted_action, {}
    finally:
        torch.set_default_device(previous_device)


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug bottle placement and run GR00T finetune policy on add_scene_glb scene.")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--policy-checkpoint", type=Path, default=DEFAULT_POLICY_CHECKPOINT)
    parser.add_argument("--cosmos-model", type=Path, default=DEFAULT_COSMOS_MODEL)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--dry-run-policy", action="store_true", help="Use a tiny deterministic fake policy for scene/control debugging.")
    parser.add_argument("--no-policy-strict", action="store_true")
    parser.add_argument("--instruction", default=DEFAULT_TASK)
    parser.add_argument("--image-size", type=_image_size, default=DEFAULT_IMAGE_SIZE, help="Model image height,width.")
    parser.add_argument(
        "--ego-roi-zoom",
        type=float,
        default=DEFAULT_EGO_ROI_ZOOM,
        help="D455 ego_view center-crop digital zoom. Use 1.0 to disable; default matches remote RealSense collection.",
    )
    parser.add_argument(
        "--ego-roi-center-x",
        type=float,
        default=DEFAULT_EGO_ROI_CENTER_X,
        help="D455 ego_view ROI center X in normalized image coordinates.",
    )
    parser.add_argument(
        "--ego-roi-center-y",
        type=float,
        default=DEFAULT_EGO_ROI_CENTER_Y,
        help="D455 ego_view ROI center Y in normalized image coordinates.",
    )
    parser.add_argument("--policy-hz", type=float, default=2.0)
    parser.add_argument(
        "--wall-clock-replan",
        action="store_true",
        help="Also replan by wall-clock policy Hz. Disabled by default to match probe chunk execution.",
    )
    parser.add_argument("--replan-horizon", type=int, default=8)
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-fps", type=float, default=10.0, help="Policy action timeline FPS used for probe-style RTC seed windows.")
    parser.add_argument("--rtc-mode", choices=("compat", "seed_window", "off"), default="compat")
    parser.add_argument("--rtc-overlap-steps", type=int, default=None)
    parser.add_argument("--rtc-max-overlap-steps", type=int, default=5)
    parser.add_argument("--rtc-frozen-steps", type=int, default=2)
    parser.add_argument("--rtc-ramp-rate", type=float, default=3.0)
    parser.add_argument("--rtc-guidance-beta", type=float, default=0.5)
    parser.add_argument("--max-joint-step", type=float, default=0.045)
    parser.add_argument("--ik-solver-max-joint-step", type=float, default=0.045)
    parser.add_argument("--min-joint-step", type=float, default=0.001)
    parser.add_argument("--ik-pos-tol", type=float, default=1e-3)
    parser.add_argument("--ik-j4-limit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ik-j4-limit-rad", type=_vec2, default=DEFAULT_IK_J4_LIMIT_RAD)
    parser.add_argument(
        "--print-hand-every",
        type=int,
        default=1,
        help="Print L10 hand target/actual every N executed policy steps; 0 disables hand step logs.",
    )
    parser.add_argument(
        "--max-hand-joint-delta",
        type=float,
        default=DEFAULT_MAX_HAND_JOINT_DELTA,
        help="Remote Teleop-style per-step L10 command guard in radians; 0 disables step limiting.",
    )
    parser.add_argument(
        "--workspace-min",
        type=_vec3,
        default=(-0.85, -0.60, 0.50),
        help="Policy robot-base frame EEF command minimum xyz, matching remote RealShadow defaults.",
    )
    parser.add_argument(
        "--workspace-max",
        type=_vec3,
        default=(-0.20, 0.60, 0.70),
        help="Policy robot-base frame EEF command maximum xyz, matching remote RealShadow defaults.",
    )
    parser.add_argument("--bottle-pos", type=_vec3, default=DEFAULT_BOTTLE_POS)
    parser.add_argument("--bottle-euler", type=_vec3, default=DEFAULT_BOTTLE_EULER)
    parser.add_argument(
        "--initial-hand-q",
        type=_vec10,
        default=DEFAULT_INITIAL_HAND_Q,
        help="Initial 10D L10 hand target in GR00T canonical order; defaults to the remote Teleop open pose.",
    )
    parser.add_argument(
        "--initial-reference-eef-9d",
        type=_vec9,
        default=DEFAULT_INITIAL_REFERENCE_EEF_9D,
        help="Initial action-relative eef_9d command reference in the policy/robot-base frame.",
    )
    parser.add_argument(
        "--initial-reference-hand-q",
        type=_vec10,
        default=DEFAULT_INITIAL_REFERENCE_HAND_Q,
        help="Initial action-relative 10D hand command reference in GR00T canonical order.",
    )
    parser.add_argument("--initial-right-arm-q", type=_vec7, default=DEFAULT_INITIAL_RIGHT_ARM_Q)
    parser.add_argument("--initial-left-arm-q", type=_vec7, default=DEFAULT_INITIAL_LEFT_ARM_Q)
    parser.add_argument("--policy-debug-jsonl", type=Path, default=DEFAULT_POLICY_DEBUG_JSONL)
    parser.add_argument("--policy-teleop-debug-jsonl", type=Path, default=DEFAULT_POLICY_TELEOP_DEBUG_JSONL)
    parser.add_argument(
        "--policy-execution-mode",
        choices=("teleop_source", "robot_target"),
        default="robot_target",
        help=(
            "robot_target matches remote GR00T/Teleop: decoded eef_9d is the robot-frame EEF command target. "
            "teleop_source is only for explicit OpenXR-source debug."
        ),
    )
    parser.add_argument(
        "--policy-openxr-coordinate-adapter",
        choices=("openxr_genesis", "none"),
        default="openxr_genesis",
        help=(
            "Coordinate adapter for policy teleop_source mode. "
            "openxr_genesis matches remote Teleop: Genesis +X=back,+Y=right,+Z=up."
        ),
    )
    parser.add_argument("--policy-input-axis-map", type=_axis_map, default=("x", "y", "z"))
    parser.add_argument("--policy-translation-scale", type=_vec3, default=DEFAULT_POLICY_TRANSLATION_SCALE)
    parser.add_argument("--policy-yaw-recenter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--policy-orientation-reference-mode",
        choices=("calibrated_tool_local", "tool_local_delta", "world_delta"),
        default="calibrated_tool_local",
    )
    parser.add_argument("--policy-orientation-axis-map", type=_axis_map, default=("x", "y", "z"))
    parser.add_argument("--policy-orientation-max-speed-rad-s", type=float, default=DEFAULT_POLICY_ORIENTATION_MAX_SPEED_RAD_S)
    parser.add_argument("--scene-support-collider-pos", type=_vec3, default=DEFAULT_SCENE_SUPPORT_COLLIDER_POS)
    parser.add_argument("--scene-support-collider-size", type=_vec3, default=DEFAULT_SCENE_SUPPORT_COLLIDER_SIZE)
    parser.add_argument(
        "--scene-mesh-collision",
        action="store_true",
        help="Use scene.glb itself as a collision mesh. Slower than the default support box proxy.",
    )
    parser.add_argument("--show-scene-support-collider", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-bottle-panel", action="store_true", help="Do not open the bottle Tk control window.")
    parser.add_argument(
        "--camera-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show ego_view and wrist_view model input windows while policy inference is running.",
    )
    parser.add_argument("--camera-preview-scale", type=int, default=2, help="Integer scale for camera preview windows.")
    parser.add_argument(
        "--eef-trajectory-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the current policy eef_9d xyz chunk trajectory in the Genesis viewer.",
    )
    parser.add_argument("--eef-trajectory-max-points", type=int, default=64)
    parser.add_argument("--eef-trajectory-line-radius", type=float, default=0.004)
    parser.add_argument("--eef-trajectory-point-radius", type=float, default=0.008)
    parser.add_argument(
        "--eef-orientation-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw rot6d orientation frames along the current eef_9d chunk trajectory.",
    )
    parser.add_argument("--eef-orientation-max-frames", type=int, default=16)
    parser.add_argument("--eef-orientation-axis-length", type=float, default=0.05)
    parser.add_argument("--eef-orientation-axis-radius", type=float, default=0.002)
    parser.add_argument("--start-policy", action="store_true", help="Start policy inference immediately.")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many policy action steps; 0 means no limit.")
    args = parser.parse_args()

    image_size = tuple(int(v) for v in args.image_size)
    if float(args.ego_roi_zoom) < 1.0:
        raise SystemExit(f"--ego-roi-zoom must be >= 1.0, got {args.ego_roi_zoom}")
    if not 0.0 <= float(args.ego_roi_center_x) <= 1.0:
        raise SystemExit(f"--ego-roi-center-x must be in [0, 1], got {args.ego_roi_center_x}")
    if not 0.0 <= float(args.ego_roi_center_y) <= 1.0:
        raise SystemExit(f"--ego-roi-center-y must be in [0, 1], got {args.ego_roi_center_y}")
    policy = _make_policy(args)
    modality_config = policy.get_modality_config()
    _validate_policy_video_schema(modality_config)
    print(
        "[policy] loaded "
        f"checkpoint={args.policy_checkpoint} dry_run={bool(args.dry_run_policy)} "
        f"video_keys={list(modality_config['video'].modality_keys)} "
        f"state_keys={list(modality_config['state'].modality_keys)} "
        f"action_keys={list(modality_config['action'].modality_keys)}",
        flush=True,
    )
    action_keys = list(modality_config["action"].modality_keys)
    action_horizon = len(modality_config["action"].delta_indices)
    action_dt_s = 1.0 / max(float(args.action_fps), 1.0e-6)

    print(f"[scene] creating add_scene_glb scene image_size={image_size}", flush=True)
    scene, _ = harness.create_scene(
        show_viewer=not bool(args.no_viewer),
        backend=args.backend,
        collision=bool(args.scene_mesh_collision),
        bottle_pos=args.bottle_pos,
        bottle_euler=args.bottle_euler,
        bottle_collision=True,
        add_table_collider=True,
        table_collider_pos=args.scene_support_collider_pos,
        table_collider_size=args.scene_support_collider_size,
        show_table_collider=bool(args.show_scene_support_collider),
        initial_base_pos=harness.DEFAULT_INITIAL_BASE_WORLD_POS,
        initial_base_euler=harness.DEFAULT_INITIAL_BASE_WORLD_EULER,
        d455_rgb_gui=False,
        d405_camera_gui=False,
        linker_hand_collision=True,
    )
    print(
        "[scene] base_world_pose "
        f"pos={tuple(round(float(v), 6) for v in harness.DEFAULT_INITIAL_BASE_WORLD_POS)} "
        f"euler_deg={tuple(round(float(v), 3) for v in harness.DEFAULT_INITIAL_BASE_WORLD_EULER)}",
        flush=True,
    )
    ego_camera, wrist_camera = _create_policy_cameras(scene, image_size=image_size)
    print(
        "[camera] ego_view_roi "
        f"zoom={float(args.ego_roi_zoom):.2f} "
        f"center=({float(args.ego_roi_center_x):.2f}, {float(args.ego_roi_center_y):.2f}) "
        "source=remote_realsense_collection",
        flush=True,
    )
    executor = RightArmPolicyExecutor(
        scene,
        max_joint_step=float(args.max_joint_step),
        ik_solver_max_joint_step=float(args.ik_solver_max_joint_step),
        min_joint_step=float(args.min_joint_step),
        pos_tol=float(args.ik_pos_tol),
        ik_j4_limit=bool(args.ik_j4_limit),
        ik_j4_limit_rad=args.ik_j4_limit_rad,
        workspace_min=args.workspace_min,
        workspace_max=args.workspace_max,
        print_hand_every=int(args.print_hand_every),
        max_hand_joint_delta=float(args.max_hand_joint_delta),
        initial_hand_q=args.initial_hand_q,
        initial_reference_eef_9d=args.initial_reference_eef_9d,
        initial_reference_hand_q=args.initial_reference_hand_q,
        initial_right_arm_q=args.initial_right_arm_q,
        initial_left_arm_q=args.initial_left_arm_q,
        policy_execution_mode=str(args.policy_execution_mode),
        policy_openxr_coordinate_adapter=str(args.policy_openxr_coordinate_adapter),
        policy_input_axis_map=args.policy_input_axis_map,
        policy_translation_scale=args.policy_translation_scale,
        policy_yaw_recenter=bool(args.policy_yaw_recenter),
        policy_orientation_reference_mode=str(args.policy_orientation_reference_mode),
        policy_orientation_axis_map=args.policy_orientation_axis_map,
        policy_orientation_max_speed_rad_s=float(args.policy_orientation_max_speed_rad_s),
        action_dt_s=action_dt_s,
    )
    obs_builder = Gr00tObservationBuilder(
        modality_config=modality_config,
        instruction=str(args.instruction),
        image_size=image_size,
    )
    print(
        "[policy] ready "
        f"instruction={args.instruction!r} image_size={image_size}",
        flush=True,
    )
    print(
        "[physics] enabled "
        f"scene_mesh_collision={bool(args.scene_mesh_collision)} "
        f"scene_support_collider=True pos={args.scene_support_collider_pos} size={args.scene_support_collider_size} "
        "bottle_collision=True linker_hand_collision=True",
        flush=True,
    )

    policy_running = multiprocessing.RawValue("b", bool(args.start_policy))
    panel = None if args.no_bottle_panel else _create_bottle_panel(args.bottle_pos, args.bottle_euler, policy_running)
    console = ConsoleController(policy_running)
    console.start()
    if console.policy_running:
        print("[policy] inference starts immediately because --start-policy was set", flush=True)
    camera_preview = CameraPreview(enabled=bool(args.camera_preview), scale=int(args.camera_preview_scale))
    eef_trajectory_overlay = EefChunkTrajectoryOverlay(
        scene,
        enabled=bool(args.eef_trajectory_overlay and not args.no_viewer),
        max_points=int(args.eef_trajectory_max_points),
        line_radius=float(args.eef_trajectory_line_radius),
        point_radius=float(args.eef_trajectory_point_radius),
        show_orientation=bool(args.eef_orientation_overlay),
        orientation_max_frames=int(args.eef_orientation_max_frames),
        orientation_axis_length=float(args.eef_orientation_axis_length),
        orientation_axis_radius=float(args.eef_orientation_axis_radius),
    )
    action_chunk: dict[str, np.ndarray] | None = None
    seed_manager = TeleopRtcSeedManager(
        action_keys=action_keys,
        action_dt_s=action_dt_s,
        horizon=action_horizon,
        max_chunks=4,
    )
    action_index = 0
    last_policy_time = 0.0
    last_panel_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    last_reset_counter = -1
    step_count = 0
    last_executor_reset_generation = int(executor.reset_generation)

    try:
        while bool(args.no_viewer) or scene.viewer.is_alive():
            console.update(executor)
            if console.quit_requested:
                break
            if int(executor.reset_generation) != last_executor_reset_generation:
                action_chunk = None
                action_index = 0
                seed_manager.clear()
                eef_trajectory_overlay.clear()
                last_executor_reset_generation = int(executor.reset_generation)
                print("[policy] cleared action/RTC history after executor reset", flush=True)
            sim_running = False
            if panel is not None:
                bottle_pos, bottle_euler, _, sim_running, reset_counter, stop_requested = _read_bottle_panel(panel)
                if stop_requested:
                    break
                panel_pose = (bottle_pos, bottle_euler)
                if panel_pose != last_panel_pose or reset_counter != last_reset_counter:
                    harness._apply_bottle_pose(scene.bottle_entity, bottle_pos, bottle_euler)  # noqa: SLF001
                    last_panel_pose = panel_pose
                    last_reset_counter = reset_counter
                    print(
                        f"[bottle] pos={tuple(round(v, 5) for v in bottle_pos)} "
                        f"euler_deg={tuple(round(v, 3) for v in bottle_euler)}",
                        flush=True,
                    )

            if console.policy_running:
                now = time.monotonic()
                should_replan = action_chunk is None or action_index >= int(args.replan_horizon)
                if bool(args.wall_clock_replan):
                    should_replan = should_replan or now - last_policy_time >= 1.0 / max(float(args.policy_hz), 1e-6)
                if should_replan:
                    last_policy_time = now
                    ego = _render_camera_rgb(
                        ego_camera,
                        image_size=image_size,
                        roi_zoom=float(args.ego_roi_zoom),
                        roi_center_x=float(args.ego_roi_center_x),
                        roi_center_y=float(args.ego_roi_center_y),
                    )
                    wrist = _render_camera_rgb(wrist_camera, image_size=image_size)
                    camera_preview.show(ego=ego, wrist=wrist)
                    obs_builder.append_frame(ego=ego, wrist=wrist)
                    reference_action = executor.current_reference_action()
                    observation_arm_q = executor.current_arm_q()
                    observation_hand_q = executor.current_observation_hand_q()
                    observation = obs_builder.build(
                        arm_q=observation_arm_q,
                        eef_pose=executor.current_eef_pose(),
                        policy_eef_9d=executor.current_policy_eef_9d(),
                        hand_q=observation_hand_q,
                        reference_action=reference_action,
                    )
                    if step_count == 0 and action_index == 0:
                        _print_first_observation_video_shapes(observation)
                    observation_ts_s = float(step_count) * action_dt_s
                    rtc_seed_action, rtc_seed_start_s, rtc_seed_metadata = seed_manager.seed_window(
                        anchor_start_monotonic_s=observation_ts_s,
                        anchor_start_frame_id=int(step_count),
                        horizon=action_horizon,
                    )
                    max_overlap_steps = (
                        int(args.rtc_overlap_steps)
                        if args.rtc_overlap_steps is not None
                        else int(args.rtc_max_overlap_steps)
                        if args.rtc_max_overlap_steps is not None
                        else None
                    )
                    rtc_options, rtc_metadata = _teleop_rtc_options(
                        enabled=bool(args.rtc),
                        rtc_mode=str(args.rtc_mode),
                        previous_action=rtc_seed_action,
                        previous_action_start_monotonic_s=rtc_seed_start_s,
                        current_observation_monotonic_s=observation_ts_s,
                        action_dt_s=action_dt_s,
                        fallback_replan_horizon=int(args.replan_horizon),
                        max_overlap_steps=max_overlap_steps,
                        frozen_steps=int(args.rtc_frozen_steps),
                        ramp_rate=float(args.rtc_ramp_rate),
                        guidance_beta=float(args.rtc_guidance_beta),
                    )
                    rtc_reference_action = _first_step_reference(rtc_seed_action) if rtc_options is not None else None
                    reference_action_source = "rtc_seed" if rtc_reference_action is not None else "executor_current"
                    if rtc_reference_action is not None:
                        reference_action = rtc_reference_action
                    tic = time.perf_counter()
                    policy_action_chunk, _ = _policy_get_action_cpu_processor(
                        policy,
                        observation,
                        reference_action=reference_action,
                        previous_action=rtc_seed_action if rtc_options is not None else None,
                        options=rtc_options,
                    )
                    clip_delta = executor.hand_policy_clip_delta(policy_action_chunk)
                    frozen_steps = 0 if rtc_options is None else int(rtc_options["rtc_frozen_steps"])
                    raw_stored_action, action_storage_metadata = _stored_rtc_action_chunk(
                        policy_action=policy_action_chunk,
                        rtc_seed_action=rtc_seed_action,
                        action_keys=action_keys,
                        frozen_steps=frozen_steps,
                    )
                    clipped_action_chunk = executor.probe_clip_action_chunk(raw_stored_action)
                    action_chunk = executor.guard_action_chunk_for_execution(clipped_action_chunk)
                    overlay_action_chunk = executor.preview_action_chunk_for_overlay(action_chunk)
                    eef_trajectory_overlay.draw_chunk(overlay_action_chunk)
                    hand_debug = executor.hand_debug_snapshot(
                        policy_action_chunk=policy_action_chunk,
                        clipped_action_chunk=clipped_action_chunk,
                        guarded_action_chunk=action_chunk,
                        reference_action=reference_action,
                        rtc_seed_action=rtc_seed_action,
                        reference_action_source=reference_action_source,
                        observation_hand_q=observation_hand_q,
                    )
                    rtc_store_action_chunk = executor.rtc_seed_action_from_command_chunk(
                        action_chunk,
                        observation_arm_q=observation_arm_q,
                    )
                    seed_manager.push(
                        rtc_store_action_chunk,
                        start_monotonic_s=observation_ts_s,
                        frame_id=int(step_count),
                    )
                    _append_jsonl(
                        args.policy_debug_jsonl,
                        {
                            "schema_version": "harness.groot_finetune_policy_debug.v1",
                            "step_count": int(step_count),
                            "action_index": int(action_index),
                            "observation_ts_s": float(observation_ts_s),
                            "rtc": {
                                "options": rtc_options,
                                "metadata": rtc_metadata,
                                "seed_metadata": rtc_seed_metadata,
                                "storage": action_storage_metadata,
                            },
                            "hand": hand_debug,
                            "eef_frame": executor.eef_frame_debug_snapshot(),
                            "teleop_bridge": executor.teleop_debug_snapshot(),
                            "raw_action": _jsonable_action(raw_stored_action),
                            "policy_raw_action": _jsonable_action(policy_action_chunk),
                            "rtc_stored_seed_action": _jsonable_action(rtc_store_action_chunk),
                            "rtc_seed_action": None
                            if rtc_seed_action is None
                            else _jsonable_action(rtc_seed_action),
                            "reference_action": _jsonable_action(reference_action),
                            "action_summary": _summarize_action_chunk(action_chunk),
                            "rtc_stored_seed_action_summary": _summarize_action_chunk(rtc_store_action_chunk),
                            "policy_raw_action_summary": _summarize_action_chunk(policy_action_chunk),
                            "overlay_action_summary": _summarize_action_chunk(overlay_action_chunk or {}),
                        },
                    )
                    if hand_debug.get("negative_floor_warning", {}).get("active"):
                        print(
                            "[policy-hand-debug] negative_floor "
                            f"reference_source={reference_action_source} "
                            f"reference={hand_debug.get('reference_hand_first')} "
                            f"observation={hand_debug.get('observation_hand_state')} "
                            f"rtc_seed={hand_debug.get('rtc_seed_hand_first')}",
                            flush=True,
                        )
                    print(
                        f"[policy] replan dt={time.perf_counter() - tic:.3f}s "
                        f"rtc={'off' if rtc_options is None else 'on'} "
                        f"rtc_reason={rtc_metadata.get('reason')} "
                        f"seed_reason={rtc_seed_metadata.get('reason')} "
                        f"reference={reference_action_source} "
                        f"stored_frozen={action_storage_metadata.get('frozen_seed_steps_applied', 0)} "
                        f"clip_delta={clip_delta:.4f} "
                        f"hand_debug={hand_debug.get('negative_floor_warning')} "
                        f"{_summarize_action_chunk(action_chunk)}",
                        flush=True,
                    )
                    action_index = 0
                if action_chunk is not None:
                    eef_trajectory_overlay.update_active(action_index)
                    executor.step_action(action_chunk, action_index)
                    teleop_debug = executor.teleop_debug_snapshot()
                    if teleop_debug is not None:
                        _append_jsonl(
                            args.policy_teleop_debug_jsonl,
                            {
                                "schema_version": "harness.groot_finetune_teleop_bridge.v1",
                                "step_count": int(step_count),
                                "action_index": int(action_index),
                                "debug": teleop_debug,
                            },
                        )
                    action_index += 1
                    step_count += 1
                    if int(args.max_steps) > 0 and step_count >= int(args.max_steps):
                        print(f"[done] reached --max-steps {args.max_steps}", flush=True)
                        break
            elif sim_running:
                harness._step_scene_with_attached_parts(scene)  # noqa: SLF001
            else:
                scene.visualizer.update(force=True)

            time.sleep(1.0 / 60.0)
    finally:
        eef_trajectory_overlay.clear()
        camera_preview.close()
        _shutdown_panel(panel)
    print(f"[done] steps={step_count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
