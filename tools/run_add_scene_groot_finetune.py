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
DEFAULT_BOTTLE_POS = (-0.016, 0.32889, 0.82667)
DEFAULT_BOTTLE_EULER = (0.0, 0.0, 0.0)
DEFAULT_SCENE_SUPPORT_COLLIDER_POS = (-0.107143, 0.455357, 0.683036)
DEFAULT_SCENE_SUPPORT_COLLIDER_SIZE = (0.732777, 0.772043, 0.108854)
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

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import add_scene_glb as harness  # noqa: E402


def _vec3(text: str) -> tuple[float, float, float]:
    parts = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected numeric x,y,z") from exc


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
    return rotation[:2, :].reshape(6).astype(np.float32)


def _rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    row0 = rot6d[0]
    row1 = rot6d[1]
    row0 = row0 / max(float(np.linalg.norm(row0)), 1e-12)
    row1 = row1 - float(np.dot(row0, row1)) * row0
    row1 = row1 / max(float(np.linalg.norm(row1)), 1e-12)
    row2 = np.cross(row0, row1)
    return np.stack([row0, row1, row2], axis=0)


def _rotation_to_quat_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    quat_wxyz = harness._quat_wxyz_from_rotation(rotation)  # noqa: SLF001
    return (float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]), float(quat_wxyz[0]))


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


def _as_hwc_uint8(value: Any, *, image_size: tuple[int, int]) -> np.ndarray:
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
    return _resize_with_pad(image.astype(np.uint8, copy=False), image_size[0], image_size[1])


def _render_camera_rgb(camera: object | None, *, image_size: tuple[int, int]) -> np.ndarray:
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
    return _as_hwc_uint8(rendered, image_size=image_size)


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
        eef_xyz = np.asarray(eef_pose[:3, 3], dtype=np.float32).reshape(3)
        eef_9d = np.concatenate([eef_xyz, _rotmat_to_rot6d(eef_pose[:3, :3])]).astype(np.float32)
        source = {
            "eef_9d": eef_9d,
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


class RightArmPolicyExecutor:
    def __init__(
        self,
        scene: object,
        *,
        max_joint_step: float,
        workspace_min: tuple[float, float, float],
        workspace_max: tuple[float, float, float],
    ) -> None:
        self.scene = scene
        assembly = getattr(scene, "nero_assembly_info", None)
        if not isinstance(assembly, dict):
            raise RuntimeError("Scene does not contain Nero assembly")
        self.assembly = assembly
        self.arm = assembly["right"]
        self.eef_link = self.arm.get_link(str(assembly.get("eef_link", harness.DEFAULT_EEF_LINK)))
        self.arm_dofs = harness._arm_dofs(self.arm)  # noqa: SLF001
        self.max_joint_step = float(max_joint_step)
        self.workspace_min = np.asarray(workspace_min, dtype=np.float64)
        self.workspace_max = np.asarray(workspace_max, dtype=np.float64)
        self.q_cmd = _tensor_to_np(self.arm.get_qpos()).reshape(-1)[self.arm_dofs].astype(np.float32)
        self.hand_q = np.zeros(10, dtype=np.float32)
        self.target_rotation = self.current_eef_pose()[:3, :3].copy()
        self.target_xyz = self.current_eef_pose()[:3, 3].copy()
        self._hand_target_print_count = 0
        print(f"[policy-hand] canonical_order={POLICY_HAND_JOINT_NAMES}", flush=True)

    def reset(self) -> None:
        harness._set_arm_initial_pose(self.arm, harness.INITIAL_RIGHT_ARM_Q)  # noqa: SLF001
        self.q_cmd = _tensor_to_np(self.arm.get_qpos()).reshape(-1)[self.arm_dofs].astype(np.float32)
        self.target_rotation = self.current_eef_pose()[:3, :3].copy()
        self.target_xyz = self.current_eef_pose()[:3, 3].copy()
        self.hand_q = np.zeros(10, dtype=np.float32)
        harness._step_scene_with_attached_parts(self.scene)  # noqa: SLF001

    def current_arm_q(self) -> np.ndarray:
        return _tensor_to_np(self.arm.get_qpos()).reshape(-1)[self.arm_dofs].astype(np.float32)

    def current_eef_pose(self) -> np.ndarray:
        pos = _tensor_to_np(self.eef_link.get_pos()).reshape(3).astype(np.float64)
        quat = _tensor_to_np(self.eef_link.get_quat()).reshape(4).astype(np.float64)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = harness._rotation_from_quat_wxyz(quat)  # noqa: SLF001
        pose[:3, 3] = pos
        return pose

    def current_reference_action(self) -> dict[str, np.ndarray]:
        eef_9d = np.concatenate([self.target_xyz.astype(np.float32), _rotmat_to_rot6d(self.target_rotation)])
        arm_joint_target = np.zeros(7, dtype=np.float32)
        arm_joint_target[: min(7, self.q_cmd.size)] = self.q_cmd[: min(7, self.q_cmd.size)]
        return {
            "eef_9d": eef_9d[None, :],
            "hand_joint_target": self.hand_q[:10][None, :].astype(np.float32),
            "arm_joint_target": arm_joint_target[None, :],
        }

    def step_action(self, action: dict[str, np.ndarray], action_index: int) -> None:
        applied_hand = False
        if "eef_9d" in action:
            eef = self._action_step(action["eef_9d"], action_index)
            if eef.size >= 3:
                self.target_xyz = np.clip(eef[:3], self.workspace_min, self.workspace_max)
            if eef.size >= 9:
                self.target_rotation = _rot6d_to_rotmat(eef[3:9])
            self._solve_and_apply_eef_target()
        elif "arm_joint_target" in action:
            self._apply_joint_target(self._action_step(action["arm_joint_target"], action_index))

        if "hand_joint_target" in action:
            self.hand_q = self._clip_hand_q(self._action_step(action["hand_joint_target"], action_index)[:10])
            self._apply_linker_hand_target()
            applied_hand = True
        harness._step_scene_with_attached_parts(self.scene)  # noqa: SLF001
        if applied_hand and (self._hand_target_print_count == 1 or self._hand_target_print_count % 10 == 0):
            print(f"[policy-hand] actual_after_step={self._current_linker_hand_positions()}", flush=True)

    def clip_action_chunk_for_execution(self, action: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
        clipped = {key: np.asarray(value, dtype=np.float32, copy=True) for key, value in action.items()}
        max_delta = 0.0
        if "hand_joint_target" in clipped:
            original = clipped["hand_joint_target"].copy()
            hand = clipped["hand_joint_target"]
            flat = hand.reshape(-1, hand.shape[-1])
            for row_idx in range(flat.shape[0]):
                flat[row_idx, :10] = self._clip_hand_q(flat[row_idx, :10])
            max_delta = max(max_delta, float(np.max(np.abs(clipped["hand_joint_target"] - original))))
        return clipped, max_delta

    def _clip_hand_q(self, hand_q: np.ndarray) -> np.ndarray:
        values = np.asarray(hand_q, dtype=np.float32).reshape(-1)[:10].copy()
        if values.size < 10:
            padded = np.zeros(10, dtype=np.float32)
            padded[: values.size] = values
            values = padded
        limits = self.assembly.get("linker_hand_joint_limits_by_name", {})
        for idx, name in enumerate(POLICY_HAND_JOINT_NAMES):
            lower, upper = (-0.6, 1.6)
            if isinstance(limits, dict) and name in limits:
                lower, upper = limits[name]
            values[idx] = float(np.clip(values[idx], float(lower), float(upper)))
        return values.astype(np.float32, copy=False)

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
                pos_tol=1e-3,
                max_step_size=self.max_joint_step,
                return_error=True,
            )
            solved = _tensor_to_np(qpos).reshape(-1)[self.arm_dofs].astype(np.float32)
            dq = np.clip(solved - self.q_cmd, -self.max_joint_step, self.max_joint_step)
            self.q_cmd = self.q_cmd + dq
            self.arm.set_dofs_position(self.q_cmd, self.arm_dofs, zero_velocity=True)
            self.arm.control_dofs_position(self.q_cmd, self.arm_dofs)
            err = tuple(round(float(v), 5) for v in _tensor_to_np(error).reshape(-1))
            print(
                f"[policy-step] eef_target={tuple(round(float(v), 4) for v in self.target_xyz)} ik_error={err}",
                flush=True,
            )
        except Exception as exc:
            print(f"[policy-step] IK failed: {exc}", flush=True)

    def _apply_joint_target(self, joint_target: np.ndarray) -> None:
        if joint_target.size < 7:
            return
        target_q = np.asarray(joint_target[:7], dtype=np.float32)
        dq = np.clip(target_q - self.q_cmd, -self.max_joint_step, self.max_joint_step)
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
        if self._hand_target_print_count == 1 or self._hand_target_print_count % 10 == 0:
            named = {
                name: round(float(value), 4)
                for name, value in zip(joint_names, self.hand_q[: len(joint_names)], strict=False)
            }
            print(f"[policy-hand] target={named}", flush=True)
        harness._set_linker_hand_target(self.assembly, "right", _HandTarget(joint_names, self.hand_q))  # noqa: SLF001

    def _current_linker_hand_positions(self) -> dict[str, float]:
        linker_hand = self.assembly.get("linker_hand")
        if linker_hand is None:
            return {}
        assembly_names = list(self.assembly.get("linker_hand_joint_names", ()))
        assembly_dofs = list(self.assembly.get("linker_hand_dofs", ()))
        if not assembly_names or not assembly_dofs:
            return {}
        try:
            qpos = _tensor_to_np(linker_hand.get_qpos()).reshape(-1)
        except Exception:
            return {}
        by_name = {
            name: float(qpos[int(dof)])
            for name, dof in zip(assembly_names, assembly_dofs, strict=False)
            if int(dof) < qpos.size
        }
        return {
            name: round(float(by_name[name]), 4)
            for name in POLICY_HAND_JOINT_NAMES
            if name in by_name
        }


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
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "enabled": bool(enabled),
        "mode": str(rtc_mode),
        "action_dt_s": float(action_dt_s),
        "fallback_replan_horizon": int(fallback_replan_horizon),
        "max_overlap_steps": max_overlap_steps,
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
    parser.add_argument("--policy-hz", type=float, default=2.0)
    parser.add_argument(
        "--wall-clock-replan",
        action="store_true",
        help="Also replan by wall-clock policy Hz. Disabled by default to match probe chunk execution.",
    )
    parser.add_argument("--replan-horizon", type=int, default=8)
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-fps", type=float, default=10.0, help="Policy action timeline FPS used for probe-style RTC seed windows.")
    parser.add_argument("--rtc-mode", choices=("seed_window", "off"), default="seed_window")
    parser.add_argument("--rtc-overlap-steps", type=int, default=None)
    parser.add_argument("--rtc-max-overlap-steps", type=int, default=None)
    parser.add_argument("--rtc-frozen-steps", type=int, default=4)
    parser.add_argument("--rtc-ramp-rate", type=float, default=20.0)
    parser.add_argument("--max-joint-step", type=float, default=0.035)
    parser.add_argument("--workspace-min", type=_vec3, default=(-0.75, -0.55, 0.30))
    parser.add_argument("--workspace-max", type=_vec3, default=(0.25, 0.55, 1.10))
    parser.add_argument("--bottle-pos", type=_vec3, default=DEFAULT_BOTTLE_POS)
    parser.add_argument("--bottle-euler", type=_vec3, default=DEFAULT_BOTTLE_EULER)
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
    parser.add_argument("--start-policy", action="store_true", help="Start policy inference immediately.")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many policy action steps; 0 means no limit.")
    args = parser.parse_args()

    image_size = tuple(int(v) for v in args.image_size)
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
        d455_rgb_gui=False,
        d405_camera_gui=False,
        linker_hand_collision=True,
    )
    ego_camera, wrist_camera = _create_policy_cameras(scene, image_size=image_size)
    executor = RightArmPolicyExecutor(
        scene,
        max_joint_step=float(args.max_joint_step),
        workspace_min=args.workspace_min,
        workspace_max=args.workspace_max,
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
    action_chunk: dict[str, np.ndarray] | None = None
    action_keys = list(modality_config["action"].modality_keys)
    action_horizon = len(modality_config["action"].delta_indices)
    action_dt_s = 1.0 / max(float(args.action_fps), 1.0e-6)
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

    try:
        while bool(args.no_viewer) or scene.viewer.is_alive():
            console.update(executor)
            if console.quit_requested:
                break
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
                    ego = _render_camera_rgb(ego_camera, image_size=image_size)
                    wrist = _render_camera_rgb(wrist_camera, image_size=image_size)
                    obs_builder.append_frame(ego=ego, wrist=wrist)
                    reference_action = executor.current_reference_action()
                    observation = obs_builder.build(
                        arm_q=executor.current_arm_q(),
                        eef_pose=executor.current_eef_pose(),
                        hand_q=executor.hand_q,
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
                    executable_policy_action, clip_delta = executor.clip_action_chunk_for_execution(policy_action_chunk)
                    frozen_steps = 0 if rtc_options is None else int(rtc_options["rtc_frozen_steps"])
                    action_chunk, action_storage_metadata = _stored_rtc_action_chunk(
                        policy_action=executable_policy_action,
                        rtc_seed_action=rtc_seed_action,
                        action_keys=action_keys,
                        frozen_steps=frozen_steps,
                    )
                    seed_manager.push(
                        action_chunk,
                        start_monotonic_s=observation_ts_s,
                        frame_id=int(step_count),
                    )
                    print(
                        f"[policy] replan dt={time.perf_counter() - tic:.3f}s "
                        f"rtc={'off' if rtc_options is None else 'on'} "
                        f"rtc_reason={rtc_metadata.get('reason')} "
                        f"seed_reason={rtc_seed_metadata.get('reason')} "
                        f"reference={reference_action_source} "
                        f"stored_frozen={action_storage_metadata.get('frozen_seed_steps_applied', 0)} "
                        f"clip_delta={clip_delta:.4f} "
                        f"{_summarize_action_chunk(action_chunk)}",
                        flush=True,
                    )
                    action_index = 0
                if action_chunk is not None:
                    executor.step_action(action_chunk, action_index)
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
        _shutdown_panel(panel)
    print(f"[done] steps={step_count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
