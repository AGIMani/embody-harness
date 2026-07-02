#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import add_scene_glb as harness  # noqa: E402
from tools import run_add_scene_groot_finetune as rollout  # noqa: E402
from tools import run_add_scene_smooth_playback as playback  # noqa: E402


SOURCE_LABELS = ("smooth", "sim")
DEFAULT_VIDEO_SOURCE = "smooth"
DEFAULT_STATE_SOURCE = "sim"
DEFAULT_TRACE_DIR = ROOT_DIR / "logs" / "groot_smooth_control_policy_trace"
DEFAULT_DEBUG_JSONL = ROOT_DIR / "logs" / "groot_smooth_control_policy_debug.jsonl"
DEFAULT_TELEOP_DEBUG_JSONL = ROOT_DIR / "logs" / "groot_smooth_control_teleop_bridge.jsonl"
DEFAULT_SMOOTH_VIDEO_BACKEND = "ffmpeg"
DEFAULT_POLICY_CHECKPOINT = (
    ROOT_DIR
    / "checkpoints"
    / "finetune"
    / "mission2-smooth-action-state-247-20260701-215311-checkpoint-126000"
)


@dataclass(frozen=True)
class SmoothPolicyEpisode:
    episode_index: int
    path: Path
    task: str
    state: np.ndarray
    action: np.ndarray | None
    timestamps: np.ndarray
    playback: playback.EpisodeData

    @property
    def length(self) -> int:
        return int(self.state.shape[0])


class SimVideoHistory:
    def __init__(self, modality_config: dict[str, Any], image_size: tuple[int, int]) -> None:
        deltas = [int(value) for value in modality_config["video"].delta_indices]
        self.delta_indices = deltas
        self.image_size = tuple(int(value) for value in image_size)
        self.frames: deque[dict[str, np.ndarray]] = deque(maxlen=max([abs(v) for v in deltas] + [0]) + 1)

    def clear(self) -> None:
        self.frames.clear()

    def append(self, *, ego: np.ndarray, wrist: np.ndarray) -> None:
        self.frames.append({"ego": np.asarray(ego, dtype=np.uint8), "wrist": np.asarray(wrist, dtype=np.uint8)})

    def build_video(self, modality_config: dict[str, Any]) -> dict[str, np.ndarray]:
        if not self.frames:
            blank = np.zeros((*self.image_size, 3), dtype=np.uint8)
            self.append(ego=blank, wrist=blank)
        buffer = list(self.frames)
        selected = []
        for delta in self.delta_indices:
            if int(delta) == 0:
                selected.append(buffer[-1])
            else:
                selected.append(buffer[max(int(delta), -len(buffer))])

        video: dict[str, np.ndarray] = {}
        for key in modality_config["video"].modality_keys:
            lowered = str(key).lower()
            source_key = "wrist" if "wrist" in lowered or "hand" in lowered else "ego"
            video[key] = np.stack([frame[source_key] for frame in selected], axis=0)[None, ...].astype(np.uint8)
        return video


def _source_to_index(source: str) -> int:
    try:
        return SOURCE_LABELS.index(str(source))
    except ValueError:
        return 0


def _index_to_source(index: int) -> str:
    if 0 <= int(index) < len(SOURCE_LABELS):
        return SOURCE_LABELS[int(index)]
    return SOURCE_LABELS[0]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_modality_ranges(smooth_dir: Path) -> dict[str, dict[str, tuple[int, int]]]:
    modality_path = smooth_dir / "meta" / "modality.json"
    ranges: dict[str, dict[str, tuple[int, int]]] = {
        "state": {
            "arm_joint_pos": (0, 7),
            "eef_9d": (7, 16),
            "arm_eef_pos": (7, 10),
            "arm_eef_rot6d": (10, 16),
            "hand_joint_pos": (16, 26),
        },
        "action": {
            "eef_9d": (0, 9),
            "arm_eef_pos_target": (0, 3),
            "arm_eef_rot6d_target": (3, 9),
            "hand_joint_target": (9, 19),
        },
    }
    if not modality_path.exists():
        return ranges
    payload = _load_json(modality_path)
    for group_name in ("state", "action"):
        group = payload.get(group_name) if isinstance(payload.get(group_name), dict) else {}
        for key, value in group.items():
            if isinstance(value, dict) and "start" in value and "end" in value:
                ranges[group_name][str(key)] = (int(value["start"]), int(value["end"]))
    return ranges


def _parquet_column_to_array(table: object, column_name: str) -> np.ndarray | None:
    try:
        column = table[column_name]
    except Exception:
        return None
    try:
        values = column.combine_chunks().to_pylist()
    except Exception:
        values = column.to_pylist()
    return np.asarray(values)


def _load_smooth_policy_episode(
    info: playback.EpisodeInfo,
    *,
    smooth_dir: Path,
    hand_source: str,
) -> SmoothPolicyEpisode:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("Reading smooth parquet files requires pyarrow in this environment.") from exc

    if not info.path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {info.path}")

    available = set(pq.ParquetFile(info.path).schema_arrow.names)
    columns = [column for column in ("observation.state", "action", "timestamp") if column in available]
    table = pq.read_table(info.path, columns=columns)
    state = _parquet_column_to_array(table, "observation.state")
    if state is None:
        raise ValueError(f"{info.path} does not contain observation.state")
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 2:
        state = state.reshape(state.shape[0], -1)

    action = _parquet_column_to_array(table, "action")
    if action is not None:
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 2:
            action = action.reshape(action.shape[0], -1)

    timestamps = _parquet_column_to_array(table, "timestamp")
    if timestamps is None:
        timestamps = np.arange(state.shape[0], dtype=np.float32) / 10.0
    timestamps = np.asarray(timestamps, dtype=np.float32).reshape(-1)
    if timestamps.size != state.shape[0]:
        timestamps = np.arange(state.shape[0], dtype=np.float32) / 10.0

    playback_episode = playback._load_episode_data(info, smooth_dir=smooth_dir, hand_source=hand_source)  # noqa: SLF001
    return SmoothPolicyEpisode(
        episode_index=int(info.episode_index),
        path=info.path,
        task=str(info.task),
        state=np.ascontiguousarray(state),
        action=None if action is None else np.ascontiguousarray(action),
        timestamps=np.ascontiguousarray(timestamps),
        playback=playback_episode,
    )


def _smooth_video_path(smooth_dir: Path, episode_index: int, key: str) -> Path:
    return smooth_dir / "videos" / f"chunk-{episode_index // 1000:03d}" / key / f"episode_{episode_index:06d}.mp4"


def _read_smooth_frame(
    path: Path,
    frame_index: int,
    image_size: tuple[int, int] | None,
    *,
    video_backend: str,
) -> np.ndarray:
    idx = int(max(frame_index, 0))
    if str(video_backend) != "opencv":
        if str(rollout.ISAAC_GROOT_ROOT) not in sys.path and rollout.ISAAC_GROOT_ROOT.exists():
            sys.path.insert(0, str(rollout.ISAAC_GROOT_ROOT))
        try:
            from gr00t.utils.video_utils import get_frames_by_indices
        except Exception as exc:
            raise RuntimeError("Reading smooth video frames with GR00T video backends requires Isaac-GR00T on PYTHONPATH.") from exc
        frames = get_frames_by_indices(
            str(path),
            np.asarray([idx], dtype=np.int64),
            video_backend=str(video_backend),
            video_backend_kwargs={},
        )
        frame = np.asarray(frames, dtype=np.uint8)[0]
        if image_size is None:
            return np.ascontiguousarray(frame).astype(np.uint8, copy=False)
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("Resizing smooth video frames requires cv2 in this environment.") from exc
        height, width = image_size
        return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA).astype(np.uint8)

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("Reading smooth video frames with --smooth-video-backend opencv requires cv2.") from exc
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame={frame_index} from {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if image_size is None:
        return frame.astype(np.uint8)
    height, width = image_size
    return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA).astype(np.uint8)


def _build_smooth_video_observation(
    *,
    smooth_dir: Path,
    episode: SmoothPolicyEpisode,
    frame_index: int,
    image_size: tuple[int, int] | None,
    video_backend: str,
    modality_config: dict[str, Any],
    ego_rotate_deg: int,
    ego_flip_horizontal: bool,
    ego_flip_vertical: bool,
    wrist_rotate_deg: int,
    wrist_flip_horizontal: bool,
    wrist_flip_vertical: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    deltas = [int(value) for value in modality_config["video"].delta_indices]
    ego_frames = []
    wrist_frames = []
    ego_path = _smooth_video_path(smooth_dir, int(episode.episode_index), "observation.images.ego_view")
    wrist_path = _smooth_video_path(smooth_dir, int(episode.episode_index), "observation.images.wrist_view")
    for delta in deltas:
        idx = int(np.clip(int(frame_index) + int(delta), 0, max(episode.length - 1, 0)))
        ego_frames.append(
            rollout._postprocess_policy_image(  # noqa: SLF001
                _read_smooth_frame(ego_path, idx, image_size, video_backend=str(video_backend)),
                rotate_deg=int(ego_rotate_deg),
                flip_horizontal=bool(ego_flip_horizontal),
                flip_vertical=bool(ego_flip_vertical),
            )
        )
        wrist_frames.append(
            rollout._postprocess_policy_image(  # noqa: SLF001
                _read_smooth_frame(wrist_path, idx, image_size, video_backend=str(video_backend)),
                rotate_deg=int(wrist_rotate_deg),
                flip_horizontal=bool(wrist_flip_horizontal),
                flip_vertical=bool(wrist_flip_vertical),
            )
        )

    source = {
        "ego": np.stack(ego_frames, axis=0)[None, ...].astype(np.uint8),
        "wrist": np.stack(wrist_frames, axis=0)[None, ...].astype(np.uint8),
    }
    video: dict[str, np.ndarray] = {}
    for key in modality_config["video"].modality_keys:
        lowered = str(key).lower()
        video[key] = source["wrist" if "wrist" in lowered or "hand" in lowered else "ego"]
    return video, {
        "source": "smooth",
        "smooth_dir": str(smooth_dir),
        "episode_index": int(episode.episode_index),
        "frame_index": int(frame_index),
        "delta_indices": deltas,
        "video_backend": str(video_backend),
        "ego_path": str(ego_path),
        "wrist_path": str(wrist_path),
    }


def _build_sim_video_observation(
    *,
    args: argparse.Namespace,
    ego_camera: object,
    wrist_camera: object,
    image_size: tuple[int, int],
    history: SimVideoHistory,
    modality_config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    ego, wrist, metadata = rollout._capture_policy_model_inputs(  # noqa: SLF001
        ego_camera=ego_camera,
        wrist_camera=wrist_camera,
        image_size=image_size,
        ego_roi_zoom=float(args.ego_roi_zoom),
        ego_roi_center_x=float(args.ego_roi_center_x),
        ego_roi_center_y=float(args.ego_roi_center_y),
        wrist_roi_zoom=float(args.wrist_roi_zoom),
        wrist_roi_center_x=float(args.wrist_roi_center_x),
        wrist_roi_center_y=float(args.wrist_roi_center_y),
        ego_rotate_deg=int(args.ego_image_rotate_deg),
        ego_flip_horizontal=bool(args.ego_image_flip_horizontal),
        ego_flip_vertical=bool(args.ego_image_flip_vertical),
        wrist_rotate_deg=int(args.wrist_image_rotate_deg),
        wrist_flip_horizontal=bool(args.wrist_image_flip_horizontal),
        wrist_flip_vertical=bool(args.wrist_image_flip_vertical),
        smooth_video_provider=None,
    )
    history.append(ego=ego, wrist=wrist)
    metadata["source"] = "sim"
    return history.build_video(modality_config), metadata


def _build_sim_state(executor: rollout.RightArmPolicyExecutor, modality_config: dict[str, Any]) -> dict[str, np.ndarray]:
    arm_q = executor.current_arm_q()
    hand_q = executor.current_observation_hand_q()
    eef_9d = executor.current_observation_eef_9d()
    source = {
        "eef_9d": eef_9d,
        "arm_eef_pos": eef_9d[:3],
        "arm_eef_rot6d": eef_9d[3:9],
        "hand_joint_pos": hand_q,
        "arm_joint_pos": arm_q,
        "hand_joint_target": hand_q,
        "arm_joint_target": arm_q,
    }
    state = {}
    for key in modality_config["state"].modality_keys:
        if key not in source:
            raise KeyError(f"Cannot build simulator state key {key!r}; available={sorted(source)}")
        state[key] = np.asarray(source[key], dtype=np.float32)[None, None, ...]
    return state


def _build_smooth_state(
    episode: SmoothPolicyEpisode,
    *,
    frame_index: int,
    modality_config: dict[str, Any],
    ranges: dict[str, dict[str, tuple[int, int]]],
) -> dict[str, np.ndarray]:
    idx = int(np.clip(int(frame_index), 0, max(episode.length - 1, 0)))
    row = np.asarray(episode.state[idx], dtype=np.float32).reshape(-1)
    out: dict[str, np.ndarray] = {}
    for key in modality_config["state"].modality_keys:
        if key in ranges["state"]:
            start, end = ranges["state"][str(key)]
            value = row[int(start) : int(end)]
        elif key == "arm_eef_pos" and "eef_9d" in ranges["state"]:
            start, _ = ranges["state"]["eef_9d"]
            value = row[int(start) : int(start) + 3]
        elif key == "arm_eef_rot6d" and "eef_9d" in ranges["state"]:
            start, _ = ranges["state"]["eef_9d"]
            value = row[int(start) + 3 : int(start) + 9]
        else:
            raise KeyError(f"Cannot build smooth state key {key!r}; available={sorted(ranges['state'])}")
        out[key] = np.asarray(value, dtype=np.float32)[None, None, ...]
    return out


def _pose_from_eef_9d(eef_9d: np.ndarray) -> np.ndarray:
    eef = np.asarray(eef_9d, dtype=np.float64).reshape(-1)
    if eef.size < 9:
        raise ValueError(f"Expected eef_9d with at least 9 values, got shape={eef.shape}")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = eef[:3]
    pose[:3, :3] = rollout._rot6d_to_rotmat(eef[3:9])  # noqa: SLF001
    return pose


def _eef_9d_from_observation(observation: dict[str, Any]) -> np.ndarray:
    state = observation.get("state", {})
    if not isinstance(state, dict) or "eef_9d" not in state:
        raise KeyError("Deployment bridge requires observation.state.eef_9d")
    arr = np.asarray(state["eef_9d"], dtype=np.float32)
    return arr.reshape(-1, arr.shape[-1])[-1, :9].astype(np.float32, copy=True)


def _reported_hand_value_to_command(joint_index: int, target_value: float) -> float:
    names = rollout.POLICY_HAND_JOINT_NAMES
    limits = rollout._l10_command_limits_by_name()  # noqa: SLF001
    name = names[int(joint_index)]
    lower, upper = limits[name]

    def reported_at(command_value: float) -> float:
        command = np.zeros(len(names), dtype=np.float32)
        command[int(joint_index)] = float(command_value)
        return float(rollout._l10_reported_hand_q_from_command(command)[int(joint_index)])  # noqa: SLF001

    low_reported = reported_at(float(lower))
    high_reported = reported_at(float(upper))
    increasing = high_reported >= low_reported
    lo, hi = (float(lower), float(upper)) if increasing else (float(upper), float(lower))
    r_lo, r_hi = (low_reported, high_reported) if increasing else (high_reported, low_reported)
    target = float(np.clip(float(target_value), min(r_lo, r_hi), max(r_lo, r_hi)))
    if target <= r_lo:
        return lo
    if target >= r_hi:
        return hi
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        r_mid = reported_at(mid)
        if r_mid < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _reported_hand_q_to_command(hand_q: np.ndarray) -> np.ndarray:
    values = np.asarray(hand_q, dtype=np.float32).reshape(-1)
    out = values.astype(np.float32, copy=True)
    count = min(out.size, len(rollout.POLICY_HAND_JOINT_NAMES))
    for idx in range(count):
        out[idx] = float(_reported_hand_value_to_command(idx, float(values[idx])))
    return out


class DeploymentActionBridge:
    """Keep model actions in training-state semantics until the executor boundary."""

    def __init__(
        self,
        executor: rollout.RightArmPolicyExecutor,
        *,
        mode: str,
        eef_update: str,
        hand_mode: str,
    ) -> None:
        self.executor = executor
        self.mode = str(mode)
        self.eef_update = str(eef_update)
        self.hand_mode = str(hand_mode)
        self.eef_calibrated = False
        self.last_metadata: dict[str, Any] = {}

    def reset(self) -> None:
        self.eef_calibrated = False
        self.last_metadata = {}

    def bridge_action_chunk(
        self,
        action: dict[str, np.ndarray],
        *,
        observation: dict[str, Any],
        reason: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        bridged = {key: np.array(value, dtype=np.float32, copy=True) for key, value in action.items()}
        metadata: dict[str, Any] = {
            "mode": self.mode,
            "eef_update": self.eef_update,
            "hand_mode": self.hand_mode,
            "input_action_semantics": "training_state",
            "executor_action_semantics": "state_eef_via_calibrated_frame_and_canonical_hand_command",
        }
        if self.mode == "state_to_robot":
            state_eef_9d = _eef_9d_from_observation(observation)
            metadata["eef"] = self._maybe_calibrate_eef_bridge(state_eef_9d, reason=reason)
        else:
            metadata["eef"] = {"enabled": False, "reason": "bridge_off"}
        if self.hand_mode == "reported_to_command" and "hand_joint_target" in bridged:
            bridged["hand_joint_target"], metadata["hand"] = self._bridge_hand_chunk(bridged["hand_joint_target"])
        else:
            metadata["hand"] = {"enabled": False, "reason": self.hand_mode}
        self.last_metadata = metadata
        return bridged, metadata

    def _maybe_calibrate_eef_bridge(self, state_eef_9d: np.ndarray, *, reason: str) -> dict[str, Any]:
        should_update = (not self.eef_calibrated) or self.eef_update == "replan"
        if not should_update:
            return {
                "enabled": True,
                "updated": False,
                "reason": "fixed_bridge_already_calibrated",
                "translation": np.asarray(self.executor.policy_action_frame_translation, dtype=np.float64).tolist(),
                "rotation_rot6d": rollout._rotmat_to_rot6d(self.executor.policy_action_frame_rotation).tolist(),  # noqa: SLF001
            }

        current_world = self.executor.current_eef_pose()
        current_uncalibrated = self.executor._world_pose_to_policy_pose_uncalibrated(current_world)  # noqa: SLF001
        desired_state = _pose_from_eef_9d(state_eef_9d)
        action_rotation = desired_state[:3, :3] @ current_uncalibrated[:3, :3].T
        action_translation = desired_state[:3, 3] - action_rotation @ current_uncalibrated[:3, 3]
        self.executor.policy_action_frame_rotation = action_rotation.astype(np.float64, copy=True)
        self.executor.policy_action_frame_translation = action_translation.astype(np.float64, copy=True)
        self.executor.reference_eef_9d = np.asarray(state_eef_9d, dtype=np.float32).reshape(9).copy()
        self.eef_calibrated = True

        check = self.executor.world_pose_to_policy_pose(current_world)
        residual = check[:3, 3] - desired_state[:3, 3]
        print(
            "[deployment-bridge] eef_state_to_robot_calibrated "
            f"reason={reason} update={self.eef_update} "
            f"state_xyz={tuple(round(float(v), 6) for v in desired_state[:3, 3])} "
            f"world_xyz={tuple(round(float(v), 6) for v in current_world[:3, 3])} "
            f"translation={tuple(round(float(v), 6) for v in action_translation)} "
            f"residual={tuple(round(float(v), 6) for v in residual)}",
            flush=True,
        )
        return {
            "enabled": True,
            "updated": True,
            "reason": reason,
            "state_xyz": desired_state[:3, 3].tolist(),
            "world_xyz": current_world[:3, 3].tolist(),
            "translation": action_translation.tolist(),
            "rotation_rot6d": rollout._rotmat_to_rot6d(action_rotation).tolist(),  # noqa: SLF001
            "residual_xyz": residual.tolist(),
        }

    def _bridge_hand_chunk(self, value: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        arr = np.array(value, dtype=np.float32, copy=True)
        if arr.ndim < 2 or arr.shape[-1] < len(rollout.POLICY_HAND_JOINT_NAMES):
            return arr, {"enabled": False, "reason": f"unexpected_shape={arr.shape}"}
        before = arr.copy()
        flat = arr.reshape(-1, arr.shape[-1])
        for row_idx in range(flat.shape[0]):
            flat[row_idx, : len(rollout.POLICY_HAND_JOINT_NAMES)] = _reported_hand_q_to_command(
                flat[row_idx, : len(rollout.POLICY_HAND_JOINT_NAMES)]
            )
        delta = arr - before
        return arr, {
            "enabled": True,
            "source": "l10_reported_state",
            "target": "l10_canonical_command",
            "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "first_reported": before.reshape(-1, before.shape[-1])[0, : len(rollout.POLICY_HAND_JOINT_NAMES)].tolist(),
            "first_command": arr.reshape(-1, arr.shape[-1])[0, : len(rollout.POLICY_HAND_JOINT_NAMES)].tolist(),
        }


def _build_observation(
    *,
    modality_config: dict[str, Any],
    video: dict[str, np.ndarray],
    state: dict[str, np.ndarray],
    instruction: str,
) -> dict[str, Any]:
    return {
        "video": {key: video[key] for key in modality_config["video"].modality_keys},
        "state": {key: state[key] for key in modality_config["state"].modality_keys},
        "language": {key: [[instruction]] for key in modality_config["language"].modality_keys},
    }


def _control_panel_main(
    labels: list[str],
    lengths: list[int],
    selected_episode: object,
    current_frame: object,
    selected_frame: object,
    load_counter: object,
    seek_counter: object,
    playing: object,
    speed: object,
    loop: object,
    policy_running: object,
    sim_running: object,
    video_source: object,
    state_source: object,
    apply_smooth_pose: object,
    reset_counter: object,
    stop_flag: object,
) -> None:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("GR00T Smooth Control")
    root.geometry("820x470")
    root.minsize(720, 420)

    selected_var = tk.StringVar(value=labels[int(selected_episode.value)] if labels else "")
    status_var = tk.StringVar(value="Paused")
    frame_var = tk.IntVar(value=0)
    speed_var = tk.DoubleVar(value=float(speed.value))
    loop_var = tk.BooleanVar(value=bool(loop.value))
    policy_var = tk.BooleanVar(value=bool(policy_running.value))
    sim_var = tk.BooleanVar(value=bool(sim_running.value))
    apply_pose_var = tk.BooleanVar(value=bool(apply_smooth_pose.value))
    video_var = tk.StringVar(value=_index_to_source(int(video_source.value)))
    state_var = tk.StringVar(value=_index_to_source(int(state_source.value)))
    internal_frame_update = {"active": False}

    def length_for_selection() -> int:
        index = int(selected_episode.value)
        if 0 <= index < len(lengths):
            return max(int(lengths[index]), 1)
        return 1

    def update_frame_scale_limit() -> None:
        frame_scale.configure(to=max(length_for_selection() - 1, 0))

    def select_episode(event: object | None = None) -> None:
        del event
        label = selected_var.get()
        try:
            index = labels.index(label)
        except ValueError:
            return
        selected_episode.value = int(index)
        selected_frame.value = 0
        current_frame.value = 0
        playing.value = False
        load_counter.value += 1
        update_frame_scale_limit()
        frame_var.set(0)

    def on_frame(value: str) -> None:
        if internal_frame_update["active"]:
            return
        try:
            frame = int(float(value))
        except ValueError:
            return
        selected_frame.value = int(np.clip(frame, 0, max(length_for_selection() - 1, 0)))
        current_frame.value = selected_frame.value
        seek_counter.value += 1

    def on_speed(value: str) -> None:
        try:
            speed.value = max(0.05, float(value))
        except ValueError:
            return

    def set_sources(event: object | None = None) -> None:
        del event
        video_source.value = _source_to_index(video_var.get())
        state_source.value = _source_to_index(state_var.get())

    def reset_robot() -> None:
        policy_running.value = False
        policy_var.set(False)
        reset_counter.value += 1

    def toggle_frames() -> None:
        playing.value = not bool(playing.value)

    def toggle_policy() -> None:
        policy_running.value = not bool(policy_running.value)
        policy_var.set(bool(policy_running.value))

    def toggle_sim() -> None:
        sim_running.value = not bool(sim_running.value)
        sim_var.set(bool(sim_running.value))

    def on_loop() -> None:
        loop.value = bool(loop_var.get())

    def on_apply_smooth_pose() -> None:
        apply_smooth_pose.value = bool(apply_pose_var.get())

    def close() -> None:
        stop_flag.value = True
        root.destroy()
        root.quit()

    def refresh() -> None:
        internal_frame_update["active"] = True
        frame_var.set(int(current_frame.value))
        internal_frame_update["active"] = False
        play_button.configure(text="Pause Frames" if bool(playing.value) else "Play Frames")
        policy_button.configure(text="Pause Policy" if bool(policy_running.value) else "Start Policy")
        sim_button.configure(text="Pause Physics" if bool(sim_running.value) else "Start Physics")
        state = "Policy" if bool(policy_running.value) else "Frames" if bool(playing.value) else "Paused"
        status_var.set(
            f"{state}  frame {int(current_frame.value)}/{max(length_for_selection() - 1, 0)}  "
            f"video={_index_to_source(int(video_source.value))} state={_index_to_source(int(state_source.value))} "
            f"speed {float(speed.value):.2f}x"
        )
        root.after(100, refresh)

    ttk.Label(root, text="Smooth episode", font=("Arial", 12, "bold")).pack(fill=tk.X, padx=12, pady=(12, 4))
    selector_row = ttk.Frame(root)
    selector_row.pack(fill=tk.X, padx=12, pady=6)
    ttk.Label(selector_row, text="Episode", width=10).pack(side=tk.LEFT)
    episode_box = ttk.Combobox(selector_row, textvariable=selected_var, values=labels, state="readonly")
    episode_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
    episode_box.bind("<<ComboboxSelected>>", select_episode)
    ttk.Button(selector_row, text="Load", command=select_episode).pack(side=tk.LEFT, padx=(8, 0))

    source_row = ttk.Frame(root)
    source_row.pack(fill=tk.X, padx=12, pady=6)
    ttk.Label(source_row, text="Video", width=10).pack(side=tk.LEFT)
    video_box = ttk.Combobox(source_row, textvariable=video_var, values=list(SOURCE_LABELS), state="readonly", width=12)
    video_box.pack(side=tk.LEFT)
    video_box.bind("<<ComboboxSelected>>", set_sources)
    ttk.Label(source_row, text="State", width=8).pack(side=tk.LEFT, padx=(18, 0))
    state_box = ttk.Combobox(source_row, textvariable=state_var, values=list(SOURCE_LABELS), state="readonly", width=12)
    state_box.pack(side=tk.LEFT)
    state_box.bind("<<ComboboxSelected>>", set_sources)
    ttk.Checkbutton(source_row, text="Apply smooth pose while paused", variable=apply_pose_var, command=on_apply_smooth_pose).pack(side=tk.LEFT, padx=(18, 0))

    frame_row = ttk.Frame(root)
    frame_row.pack(fill=tk.X, padx=12, pady=12)
    ttk.Label(frame_row, text="Frame", width=10).pack(side=tk.LEFT)
    frame_scale = ttk.Scale(frame_row, from_=0, to=max(length_for_selection() - 1, 0), orient=tk.HORIZONTAL, variable=frame_var, command=on_frame)
    frame_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

    speed_row = ttk.Frame(root)
    speed_row.pack(fill=tk.X, padx=12, pady=6)
    ttk.Label(speed_row, text="Speed", width=10).pack(side=tk.LEFT)
    speed_scale = ttk.Scale(speed_row, from_=0.05, to=4.0, orient=tk.HORIZONTAL, variable=speed_var, command=on_speed)
    speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Label(speed_row, textvariable=speed_var, width=8).pack(side=tk.LEFT, padx=(8, 0))

    buttons = ttk.Frame(root)
    buttons.pack(fill=tk.X, padx=12, pady=(14, 8))
    play_button = ttk.Button(buttons, text="Play Frames", command=toggle_frames)
    play_button.pack(side=tk.LEFT)
    policy_button = ttk.Button(buttons, text="Start Policy", command=toggle_policy)
    policy_button.pack(side=tk.LEFT, padx=8)
    sim_button = ttk.Button(buttons, text="Start Physics", command=toggle_sim)
    sim_button.pack(side=tk.LEFT)
    ttk.Checkbutton(buttons, text="Loop", variable=loop_var, command=on_loop).pack(side=tk.LEFT, padx=8)
    ttk.Button(buttons, text="Reset Robot", command=reset_robot).pack(side=tk.LEFT)
    ttk.Button(buttons, text="Close", command=close).pack(side=tk.RIGHT)

    ttk.Label(root, textvariable=status_var).pack(fill=tk.X, padx=12, pady=(4, 12))
    refresh()
    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()


def _create_control_panel(
    episodes: list[playback.EpisodeInfo],
    *,
    episode_index: int,
    start_frames: bool,
    start_policy: bool,
    speed_value: float,
    loop_value: bool,
    video_source_value: str,
    state_source_value: str,
    apply_smooth_pose_value: bool,
) -> dict[str, object]:
    selected_episode = multiprocessing.RawValue("i", int(episode_index))
    current_frame = multiprocessing.RawValue("i", 0)
    selected_frame = multiprocessing.RawValue("i", 0)
    load_counter = multiprocessing.RawValue("i", 0)
    seek_counter = multiprocessing.RawValue("i", 0)
    playing = multiprocessing.RawValue("b", bool(start_frames))
    speed = multiprocessing.RawValue("d", float(speed_value))
    loop = multiprocessing.RawValue("b", bool(loop_value))
    policy_running = multiprocessing.RawValue("b", bool(start_policy))
    sim_running = multiprocessing.RawValue("b", False)
    video_source = multiprocessing.RawValue("i", _source_to_index(video_source_value))
    state_source = multiprocessing.RawValue("i", _source_to_index(state_source_value))
    apply_smooth_pose = multiprocessing.RawValue("b", bool(apply_smooth_pose_value))
    reset_counter = multiprocessing.RawValue("i", 0)
    stop_flag = multiprocessing.RawValue("b", False)
    labels = [item.label for item in episodes]
    lengths = [int(item.length) for item in episodes]
    process = multiprocessing.Process(
        target=_control_panel_main,
        args=(
            labels,
            lengths,
            selected_episode,
            current_frame,
            selected_frame,
            load_counter,
            seek_counter,
            playing,
            speed,
            loop,
            policy_running,
            sim_running,
            video_source,
            state_source,
            apply_smooth_pose,
            reset_counter,
            stop_flag,
        ),
        daemon=True,
    )
    process.start()
    return {
        "selected_episode": selected_episode,
        "current_frame": current_frame,
        "selected_frame": selected_frame,
        "load_counter": load_counter,
        "seek_counter": seek_counter,
        "playing": playing,
        "speed": speed,
        "loop": loop,
        "policy_running": policy_running,
        "sim_running": sim_running,
        "video_source": video_source,
        "state_source": state_source,
        "apply_smooth_pose": apply_smooth_pose,
        "reset_counter": reset_counter,
        "stop_flag": stop_flag,
        "process": process,
    }


def _shutdown_control_panel(panel: dict[str, object] | None) -> None:
    if not panel:
        return
    panel["stop_flag"].value = True
    process = panel["process"]
    if process.is_alive():
        process.join(timeout=1.0)


def _create_scene(args: argparse.Namespace) -> object:
    d405_connector_rel_pos, d405_connector_rel_euler, d405_mount_source = harness.resolve_d405_mount_args(
        mount_json=args.d405_mount_json,
        rel_pos=args.d405_connector_rel_pos,
        rel_euler=args.d405_connector_rel_euler,
    )
    print(
        "[camera] d405_mount "
        f"source={d405_mount_source} "
        f"pos={tuple(round(float(v), 6) for v in d405_connector_rel_pos)} "
        f"euler_deg={tuple(round(float(v), 3) for v in d405_connector_rel_euler)}",
        flush=True,
    )
    scene, _ = harness.create_scene(
        show_viewer=not bool(args.no_viewer),
        backend=args.backend,
        pos=harness.DEFAULT_SCENE_WORLD_POS,
        euler=harness.DEFAULT_SCENE_WORLD_EULER,
        collision=bool(args.scene_mesh_collision),
        bottle_path=harness.DEFAULT_BOTTLE_GLB,
        bottle_pos=args.bottle_pos,
        bottle_euler=args.bottle_euler,
        bottle_collision=True,
        bottle_proxy_json=args.bottle_proxy_json,
        show_bottle_proxy=bool(args.show_bottle_proxy),
        add_table_collider=True,
        table_collider_pos=args.scene_support_collider_pos,
        table_collider_size=args.scene_support_collider_size,
        show_table_collider=bool(args.show_scene_support_collider),
        use_combined_urdf=True,
        combined_urdf=harness.DEFAULT_COMBINED_NERO_LINKER_URDF,
        initial_base_pos=(0.0, 0.0, 0.0),
        initial_base_euler=(0.0, 0.0, 0.0),
        d455_rgb_gui=False,
        d455_rgb_fov=args.d455_rgb_fov,
        d405_camera_gui=False,
        d405_fov=args.d405_fov,
        d405_connector_rel_pos=d405_connector_rel_pos,
        d405_connector_rel_euler=d405_connector_rel_euler,
        linker_hand_collision=bool(args.linker_hand_collision),
    )
    return scene


def _make_executor(scene: object, args: argparse.Namespace, *, action_dt_s: float) -> rollout.RightArmPolicyExecutor:
    return rollout.RightArmPolicyExecutor(
        scene,
        arm_ik_mode=str(args.arm_ik_mode),
        max_joint_step=float(args.max_joint_step),
        ik_solver_max_joint_step=float(args.ik_solver_max_joint_step),
        min_joint_step=float(args.min_joint_step),
        pos_tol=float(args.ik_pos_tol),
        ik_j4_limit=bool(args.ik_j4_limit),
        ik_j4_limit_rad=args.ik_j4_limit_rad,
        ik_command_hz=float(args.ik_command_hz),
        ik_differential_finite_difference_rad=float(args.ik_differential_finite_difference_rad),
        ik_differential_position_weight=float(args.ik_differential_position_weight),
        ik_differential_orientation_weight=float(args.ik_differential_orientation_weight),
        ik_differential_max_task_step_m=float(args.ik_differential_max_task_step_m),
        ik_differential_max_rotation_step_rad=float(args.ik_differential_max_rotation_step_rad),
        ik_differential_damping_lambda=float(args.ik_differential_damping_lambda),
        ik_differential_posture_bias_gain=float(args.ik_differential_posture_bias_gain),
        ik_differential_joint_limit_bias_gain=float(args.ik_differential_joint_limit_bias_gain),
        ik_differential_bias_weight=float(args.ik_differential_bias_weight),
        ik_differential_joint_limit_soft_margin_rad=float(args.ik_differential_joint_limit_soft_margin_rad),
        ik_differential_max_joint_acceleration_rad_s2=float(args.ik_differential_max_joint_acceleration_rad_s2),
        workspace_min=args.workspace_min,
        workspace_max=args.workspace_max,
        workspace_clamp=bool(args.workspace_clamp),
        print_hand_every=int(args.print_hand_every),
        max_hand_joint_delta=float(args.max_hand_joint_delta),
        initial_hand_q=args.initial_hand_q,
        initial_reference_eef_9d=args.initial_reference_eef_9d,
        initial_reference_hand_q=args.initial_reference_hand_q,
        initial_right_arm_q=args.initial_right_arm_q,
        initial_left_arm_q=args.initial_left_arm_q,
        policy_eef_link=str(args.policy_eef_link),
        policy_execution_mode=str(args.policy_execution_mode),
        policy_openxr_coordinate_adapter=str(args.policy_openxr_coordinate_adapter),
        policy_input_axis_map=args.policy_input_axis_map,
        policy_translation_scale=args.policy_translation_scale,
        policy_yaw_recenter=bool(args.policy_yaw_recenter),
        policy_orientation_reference_mode=str(args.policy_orientation_reference_mode),
        policy_orientation_axis_map=args.policy_orientation_axis_map,
        policy_orientation_max_speed_rad_s=float(args.policy_orientation_max_speed_rad_s),
        action_dt_s=float(action_dt_s),
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--smooth-dir", type=Path, default=playback.DEFAULT_SMOOTH_DIR)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--hand-source", choices=("state", "action"), default="action")
    parser.add_argument("--video-source", choices=SOURCE_LABELS, default=DEFAULT_VIDEO_SOURCE)
    parser.add_argument("--state-source", choices=SOURCE_LABELS, default=DEFAULT_STATE_SOURCE)
    parser.add_argument("--instruction", default="", help="Override smooth episode instruction. Empty uses the selected episode task.")
    parser.add_argument("--start-policy", action="store_true")
    parser.add_argument("--start-frames", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-smooth-pose", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-control-panel", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--policy-checkpoint", type=Path, default=DEFAULT_POLICY_CHECKPOINT)
    parser.add_argument("--cosmos-model", type=Path, default=rollout.DEFAULT_COSMOS_MODEL)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--dry-run-policy", action="store_true")
    parser.add_argument("--no-policy-strict", action="store_true")
    parser.add_argument("--image-size", type=rollout._image_size, default=rollout.DEFAULT_IMAGE_SIZE)  # noqa: SLF001
    parser.add_argument(
        "--smooth-video-resize",
        action="store_true",
        help="Resize smooth dataset videos to --image-size before GR00T preprocessing. Default preserves recorded resolution like the Isaac-GR00T probes.",
    )
    parser.add_argument(
        "--smooth-video-backend",
        choices=("ffmpeg", "torchcodec", "decord", "opencv"),
        default=DEFAULT_SMOOTH_VIDEO_BACKEND,
        help="Backend for reading smooth dataset videos. Default matches the Isaac-GR00T probes.",
    )
    parser.add_argument("--ego-roi-zoom", type=float, default=rollout.DEFAULT_EGO_ROI_ZOOM)
    parser.add_argument("--ego-roi-center-x", type=float, default=rollout.DEFAULT_EGO_ROI_CENTER_X)
    parser.add_argument("--ego-roi-center-y", type=float, default=rollout.DEFAULT_EGO_ROI_CENTER_Y)
    parser.add_argument("--wrist-roi-zoom", type=float, default=rollout.DEFAULT_WRIST_ROI_ZOOM)
    parser.add_argument("--wrist-roi-center-x", type=float, default=rollout.DEFAULT_WRIST_ROI_CENTER_X)
    parser.add_argument("--wrist-roi-center-y", type=float, default=rollout.DEFAULT_WRIST_ROI_CENTER_Y)
    parser.add_argument("--ego-image-rotate-deg", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--wrist-image-rotate-deg", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--ego-image-flip-horizontal", action="store_true")
    parser.add_argument("--ego-image-flip-vertical", action="store_true")
    parser.add_argument("--wrist-image-flip-horizontal", action="store_true")
    parser.add_argument("--wrist-image-flip-vertical", action="store_true")
    parser.add_argument("--d455-rgb-fov", type=float, default=None)
    parser.add_argument("--d405-fov", type=float, default=None)
    parser.add_argument("--d405-mount-json", type=Path, default=None)
    parser.add_argument("--d405-connector-rel-pos", type=rollout._vec3, default=None)  # noqa: SLF001
    parser.add_argument("--d405-connector-rel-euler", type=rollout._vec3, default=None)  # noqa: SLF001
    parser.add_argument("--policy-hz", type=float, default=2.0)
    parser.add_argument("--wall-clock-replan", action="store_true")
    parser.add_argument("--replan-horizon", type=int, default=8)
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-fps", type=float, default=10.0)
    parser.add_argument("--rtc-mode", choices=("compat", "seed_window", "off"), default="compat")
    parser.add_argument("--rtc-overlap-steps", type=int, default=None)
    parser.add_argument("--rtc-max-overlap-steps", type=int, default=5)
    parser.add_argument("--rtc-frozen-steps", type=int, default=2)
    parser.add_argument("--rtc-ramp-rate", type=float, default=3.0)
    parser.add_argument("--rtc-guidance-beta", type=float, default=0.5)
    parser.add_argument("--arm-ik-mode", choices=("genesis_pose", "differential_full_pose"), default="differential_full_pose")
    parser.add_argument("--max-joint-step", type=float, default=0.045)
    parser.add_argument("--ik-solver-max-joint-step", type=float, default=0.045)
    parser.add_argument("--min-joint-step", type=float, default=0.001)
    parser.add_argument("--ik-pos-tol", type=float, default=1e-3)
    parser.add_argument("--ik-j4-limit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ik-j4-limit-rad", type=rollout._vec2, default=rollout.DEFAULT_IK_J4_LIMIT_RAD)  # noqa: SLF001
    parser.add_argument("--ik-command-hz", type=float, default=10.0)
    parser.add_argument("--ik-differential-finite-difference-rad", type=float, default=1e-4)
    parser.add_argument("--ik-differential-position-weight", type=float, default=3.0)
    parser.add_argument("--ik-differential-orientation-weight", type=float, default=1.0)
    parser.add_argument("--ik-differential-max-task-step-m", type=float, default=0.03)
    parser.add_argument("--ik-differential-max-rotation-step-rad", type=float, default=math.radians(5.0))
    parser.add_argument("--ik-differential-damping-lambda", type=float, default=0.02)
    parser.add_argument("--ik-differential-posture-bias-gain", type=float, default=0.04)
    parser.add_argument("--ik-differential-joint-limit-bias-gain", type=float, default=0.35)
    parser.add_argument("--ik-differential-bias-weight", type=float, default=0.08)
    parser.add_argument("--ik-differential-joint-limit-soft-margin-rad", type=float, default=0.25)
    parser.add_argument("--ik-differential-max-joint-acceleration-rad-s2", type=float, default=0.0)
    parser.add_argument("--print-hand-every", type=int, default=1)
    parser.add_argument("--max-hand-joint-delta", type=float, default=rollout.DEFAULT_MAX_HAND_JOINT_DELTA)
    parser.add_argument("--workspace-min", type=rollout._vec3, default=(-0.85, -0.60, 0.50))  # noqa: SLF001
    parser.add_argument("--workspace-max", type=rollout._vec3, default=(-0.20, 0.60, 0.70))  # noqa: SLF001
    parser.add_argument("--workspace-clamp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bottle-pos", type=rollout._vec3, default=rollout.DEFAULT_BOTTLE_POS)  # noqa: SLF001
    parser.add_argument("--bottle-euler", type=rollout._vec3, default=rollout.DEFAULT_BOTTLE_EULER)  # noqa: SLF001
    parser.add_argument("--bottle-proxy-json", type=Path, default=harness.DEFAULT_BOTTLE_PROXY_JSON)
    parser.add_argument("--show-bottle-proxy", action="store_true")
    parser.add_argument("--initial-hand-q", type=rollout._vec10, default=rollout.DEFAULT_INITIAL_HAND_Q)  # noqa: SLF001
    parser.add_argument("--initial-reference-eef-9d", type=rollout._vec9, default=rollout.DEFAULT_INITIAL_REFERENCE_EEF_9D)  # noqa: SLF001
    parser.add_argument("--initial-reference-hand-q", type=rollout._vec10, default=rollout.DEFAULT_INITIAL_REFERENCE_HAND_Q)  # noqa: SLF001
    parser.add_argument("--initial-right-arm-q", type=rollout._vec7, default=rollout.DEFAULT_INITIAL_RIGHT_ARM_Q)  # noqa: SLF001
    parser.add_argument("--initial-left-arm-q", type=rollout._vec7, default=rollout.DEFAULT_INITIAL_LEFT_ARM_Q)  # noqa: SLF001
    parser.add_argument(
        "--deployment-action-bridge",
        choices=("state_to_robot", "off"),
        default="state_to_robot",
        help="Bridge decoded action-state checkpoint outputs to robot execution at the executor boundary.",
    )
    parser.add_argument(
        "--deployment-eef-bridge-update",
        choices=("once", "replan"),
        default="once",
        help="Calibrate the fixed state-frame to robot-frame EEF bridge once per policy/reset session or every replan.",
    )
    parser.add_argument(
        "--deployment-hand-bridge",
        choices=("reported_to_command", "identity"),
        default="reported_to_command",
        help="Map decoded L10 reported-state hand targets into canonical command targets before execution.",
    )
    parser.add_argument("--policy-eef-link", choices=("revo2_flange", "link7"), default=harness.DEFAULT_EEF_LINK)
    parser.add_argument("--policy-execution-mode", choices=("teleop_source", "robot_target"), default="robot_target")
    parser.add_argument("--policy-openxr-coordinate-adapter", choices=("openxr_genesis", "none"), default="openxr_genesis")
    parser.add_argument("--policy-input-axis-map", type=rollout._axis_map, default=("x", "y", "z"))  # noqa: SLF001
    parser.add_argument("--policy-translation-scale", type=rollout._vec3, default=rollout.DEFAULT_POLICY_TRANSLATION_SCALE)  # noqa: SLF001
    parser.add_argument("--policy-yaw-recenter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy-orientation-reference-mode", choices=("calibrated_tool_local", "tool_local_delta", "world_delta"), default="calibrated_tool_local")
    parser.add_argument("--policy-orientation-axis-map", type=rollout._axis_map, default=("x", "y", "z"))  # noqa: SLF001
    parser.add_argument("--policy-orientation-max-speed-rad-s", type=float, default=rollout.DEFAULT_POLICY_ORIENTATION_MAX_SPEED_RAD_S)
    parser.add_argument("--scene-support-collider-pos", type=rollout._vec3, default=rollout.DEFAULT_SCENE_SUPPORT_COLLIDER_POS)  # noqa: SLF001
    parser.add_argument("--scene-support-collider-size", type=rollout._vec3, default=rollout.DEFAULT_SCENE_SUPPORT_COLLIDER_SIZE)  # noqa: SLF001
    parser.add_argument("--scene-mesh-collision", action="store_true")
    parser.add_argument("--show-scene-support-collider", action="store_true")
    parser.add_argument("--linker-hand-collision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-preview-scale", type=int, default=2)
    parser.add_argument("--camera-preview-hz", type=float, default=10.0)
    parser.add_argument("--eef-trajectory-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eef-trajectory-max-points", type=int, default=64)
    parser.add_argument("--eef-trajectory-line-radius", type=float, default=0.004)
    parser.add_argument("--eef-trajectory-point-radius", type=float, default=0.008)
    parser.add_argument("--eef-orientation-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eef-orientation-max-frames", type=int, default=16)
    parser.add_argument("--eef-orientation-axis-length", type=float, default=0.05)
    parser.add_argument("--eef-orientation-axis-radius", type=float, default=0.002)
    parser.add_argument("--policy-debug-jsonl", type=Path, default=DEFAULT_DEBUG_JSONL)
    parser.add_argument("--policy-teleop-debug-jsonl", type=Path, default=DEFAULT_TELEOP_DEBUG_JSONL)
    parser.add_argument("--policy-trace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy-trace-dir", type=Path, default=DEFAULT_TRACE_DIR)


def _validate_args(args: argparse.Namespace) -> tuple[int, int]:
    image_size = tuple(int(v) for v in args.image_size)
    if float(args.ego_roi_zoom) < 1.0:
        raise SystemExit(f"--ego-roi-zoom must be >= 1.0, got {args.ego_roi_zoom}")
    if float(args.wrist_roi_zoom) < 1.0:
        raise SystemExit(f"--wrist-roi-zoom must be >= 1.0, got {args.wrist_roi_zoom}")
    for name in ("ego_roi_center_x", "ego_roi_center_y", "wrist_roi_center_x", "wrist_roi_center_y"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1], got {value}")
    return image_size


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run GR00T on the add_scene_glb finetune scene while interactively choosing smooth dataset "
            "or simulator video/state inputs."
        )
    )
    _add_common_args(parser)
    args = parser.parse_args()
    image_size = _validate_args(args)

    smooth_dir = args.smooth_dir.expanduser().resolve()
    episodes = playback._discover_episodes(smooth_dir)  # noqa: SLF001
    selected_list_index = playback._selected_episode_list_index(episodes, int(args.episode_index))  # noqa: SLF001
    ranges = _load_modality_ranges(smooth_dir)
    episode = _load_smooth_policy_episode(episodes[selected_list_index], smooth_dir=smooth_dir, hand_source=str(args.hand_source))

    policy = rollout._make_policy(args)  # noqa: SLF001
    modality_config = policy.get_modality_config()
    rollout._validate_policy_video_schema(modality_config)  # noqa: SLF001
    action_keys = list(modality_config["action"].modality_keys)
    action_horizon = len(modality_config["action"].delta_indices)
    action_dt_s = 1.0 / max(float(args.action_fps), 1.0e-6)
    print(
        "[policy] loaded "
        f"checkpoint={args.policy_checkpoint} dry_run={bool(args.dry_run_policy)} "
        f"video_keys={list(modality_config['video'].modality_keys)} "
        f"state_keys={list(modality_config['state'].modality_keys)} "
        f"action_keys={action_keys}",
        flush=True,
    )

    print(f"[scene] creating add_scene_glb finetune scene image_size={image_size}", flush=True)
    scene = _create_scene(args)
    ego_camera, wrist_camera = rollout._create_policy_cameras(scene, image_size=image_size)  # noqa: SLF001
    executor = _make_executor(scene, args, action_dt_s=action_dt_s)
    assembly = getattr(scene, "nero_assembly_info", None)
    if not isinstance(assembly, dict):
        raise RuntimeError("add_scene_glb scene did not create a Nero assembly")
    arm = assembly.get("right")
    if arm is None:
        raise RuntimeError("add_scene_glb scene does not contain a right Nero arm")
    arm_dofs = executor.arm_dofs
    deployment_bridge = DeploymentActionBridge(
        executor,
        mode=str(args.deployment_action_bridge),
        eef_update=str(args.deployment_eef_bridge_update),
        hand_mode=str(args.deployment_hand_bridge),
    )
    print(
        "[deployment-bridge] "
        f"mode={args.deployment_action_bridge} "
        f"eef_update={args.deployment_eef_bridge_update} "
        f"hand={args.deployment_hand_bridge}",
        flush=True,
    )

    panel = None
    if not bool(args.no_control_panel):
        panel = _create_control_panel(
            episodes,
            episode_index=selected_list_index,
            start_frames=bool(args.start_frames),
            start_policy=bool(args.start_policy),
            speed_value=float(args.speed),
            loop_value=bool(args.loop),
            video_source_value=str(args.video_source),
            state_source_value=str(args.state_source),
            apply_smooth_pose_value=bool(args.apply_smooth_pose),
        )

    policy_running_value = panel["policy_running"] if panel is not None else multiprocessing.RawValue("b", bool(args.start_policy))
    console = rollout.ConsoleController(policy_running_value)
    console.start()
    camera_preview = rollout.CameraPreview(enabled=bool(args.camera_preview), scale=int(args.camera_preview_scale))
    eef_trajectory_overlay = rollout.EefChunkTrajectoryOverlay(
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
    sim_video_history = SimVideoHistory(modality_config, image_size)
    seed_manager = rollout.TeleopRtcSeedManager(action_keys=action_keys, action_dt_s=action_dt_s, horizon=action_horizon, max_chunks=4)

    action_chunk: dict[str, np.ndarray] | None = None
    action_index = 0
    current_frame = 0
    last_load_counter = 0
    last_seek_counter = 0
    last_reset_counter = 0
    step_count = 0
    last_executor_reset_generation = int(executor.reset_generation)
    last_policy_running = False
    policy_session_start_step = 0
    last_policy_time = 0.0
    next_frame_time = time.monotonic()
    base_dt = 1.0 / max(float(args.fps), 1.0e-6)
    last_camera_preview_time = -float("inf")
    preview_period_s = 1.0 / max(float(args.camera_preview_hz), 1.0e-6)
    policy_trace_dir = args.policy_trace_dir.expanduser() if bool(args.policy_trace) else None
    trace_session_index = 0
    trace_session_id = ""
    trace_replan_index = 0

    if policy_trace_dir is not None:
        policy_trace_dir.mkdir(parents=True, exist_ok=True)
        print(f"[policy-trace] enabled dir={policy_trace_dir} index={rollout._trace_jsonl_path(policy_trace_dir)}", flush=True)  # noqa: SLF001
    print(
        "[smooth-control] ready "
        f"episode={episode.episode_index:06d} frames={episode.length} "
        f"video_source={args.video_source} state_source={args.state_source} "
        f"instruction={(args.instruction or episode.task or rollout.DEFAULT_TASK)!r}",
        flush=True,
    )

    def reset_policy_history() -> None:
        nonlocal action_chunk, action_index, trace_replan_index, policy_session_start_step
        action_chunk = None
        action_index = 0
        trace_replan_index = 0
        policy_session_start_step = int(step_count)
        seed_manager.clear()
        sim_video_history.clear()
        eef_trajectory_overlay.clear()
        executor.reset_policy_source_anchor()
        deployment_bridge.reset()

    def instruction_for_current_episode() -> str:
        return str(args.instruction or episode.task or rollout.DEFAULT_TASK)

    def active_video_source() -> str:
        if panel is None:
            return str(args.video_source)
        return _index_to_source(int(panel["video_source"].value))

    def active_state_source() -> str:
        if panel is None:
            return str(args.state_source)
        return _index_to_source(int(panel["state_source"].value))

    def build_current_observation() -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        frame = int(np.clip(current_frame, 0, max(episode.length - 1, 0)))
        video_source_name = active_video_source()
        state_source_name = active_state_source()
        if video_source_name == "smooth":
            video, video_metadata = _build_smooth_video_observation(
                smooth_dir=smooth_dir,
                episode=episode,
                frame_index=frame,
                image_size=image_size if bool(args.smooth_video_resize) else None,
                video_backend=str(args.smooth_video_backend),
                modality_config=modality_config,
                ego_rotate_deg=int(args.ego_image_rotate_deg),
                ego_flip_horizontal=bool(args.ego_image_flip_horizontal),
                ego_flip_vertical=bool(args.ego_image_flip_vertical),
                wrist_rotate_deg=int(args.wrist_image_rotate_deg),
                wrist_flip_horizontal=bool(args.wrist_image_flip_horizontal),
                wrist_flip_vertical=bool(args.wrist_image_flip_vertical),
            )
            video_metadata["resize"] = None if not bool(args.smooth_video_resize) else tuple(int(v) for v in image_size)
            preview_ego = np.asarray(video["ego_view"][0, -1], dtype=np.uint8) if "ego_view" in video else None
            preview_wrist = np.asarray(video["wrist_view"][0, -1], dtype=np.uint8) if "wrist_view" in video else None
            if preview_ego is not None and preview_wrist is not None:
                camera_preview.show(ego=preview_ego, wrist=preview_wrist)
        else:
            video, video_metadata = _build_sim_video_observation(
                args=args,
                ego_camera=ego_camera,
                wrist_camera=wrist_camera,
                image_size=image_size,
                history=sim_video_history,
                modality_config=modality_config,
            )
            if "ego_view" in video and "wrist_view" in video:
                camera_preview.show(ego=np.asarray(video["ego_view"][0, -1]), wrist=np.asarray(video["wrist_view"][0, -1]))

        if state_source_name == "smooth":
            state = _build_smooth_state(episode, frame_index=frame, modality_config=modality_config, ranges=ranges)
            state_metadata = {"source": "smooth", "episode_index": int(episode.episode_index), "frame_index": int(frame)}
        else:
            state = _build_sim_state(executor, modality_config)
            state_metadata = {"source": "sim"}

        observation = _build_observation(
            modality_config=modality_config,
            video=video,
            state=state,
            instruction=instruction_for_current_episode(),
        )
        metadata = {
            "video": video_metadata,
            "state": state_metadata,
            "instruction": instruction_for_current_episode(),
        }
        observation_arm_q = executor.current_arm_q()
        observation_hand_q = executor.current_observation_hand_q()
        observation_eef_pose = executor.current_observation_eef_pose()
        return observation, metadata, observation_arm_q, observation_hand_q, observation_eef_pose

    try:
        while bool(args.no_viewer) or scene.viewer.is_alive():
            console.update(executor)
            if console.quit_requested:
                break
            if panel is not None:
                if bool(panel["stop_flag"].value):
                    break
                panel_selected = int(panel["selected_episode"].value)
                panel_load_counter = int(panel["load_counter"].value)
                if panel_load_counter != last_load_counter:
                    selected_list_index = int(np.clip(panel_selected, 0, len(episodes) - 1))
                    episode = _load_smooth_policy_episode(episodes[selected_list_index], smooth_dir=smooth_dir, hand_source=str(args.hand_source))
                    current_frame = 0
                    panel["current_frame"].value = current_frame
                    last_load_counter = panel_load_counter
                    next_frame_time = time.monotonic()
                    reset_policy_history()
                    print(f"[smooth-control] loaded episode={episode.episode_index:06d} frames={episode.length} path={episode.path}", flush=True)

                panel_seek_counter = int(panel["seek_counter"].value)
                if panel_seek_counter != last_seek_counter:
                    current_frame = int(np.clip(int(panel["selected_frame"].value), 0, max(episode.length - 1, 0)))
                    panel["current_frame"].value = current_frame
                    last_seek_counter = panel_seek_counter
                    next_frame_time = time.monotonic()
                    reset_policy_history()

                panel_reset_counter = int(panel["reset_counter"].value)
                if panel_reset_counter != last_reset_counter:
                    executor.reset()
                    reset_policy_history()
                    last_reset_counter = panel_reset_counter
                    print("[smooth-control] robot reset", flush=True)

            current_policy_running = bool(console.policy_running)
            if current_policy_running and not last_policy_running:
                reset_policy_history()
                trace_session_index += 1
                trace_session_id = rollout._new_policy_trace_session_id(trace_session_index)  # noqa: SLF001
                rollout._append_policy_trace_event(  # noqa: SLF001
                    policy_trace_dir,
                    session_id=trace_session_id,
                    event="policy_start",
                    step_count=int(step_count),
                    action_index=int(action_index),
                    extra={
                        "instruction": instruction_for_current_episode(),
                        "checkpoint": str(args.policy_checkpoint),
                        "episode_index": int(episode.episode_index),
                        "frame_index": int(current_frame),
                        "video_source": active_video_source(),
                        "state_source": active_state_source(),
                    },
                )
                if policy_trace_dir is not None:
                    print(f"[policy-trace] session_start id={trace_session_id}", flush=True)
            elif (not current_policy_running) and last_policy_running:
                rollout._append_policy_trace_event(  # noqa: SLF001
                    policy_trace_dir,
                    session_id=trace_session_id,
                    event="policy_stop",
                    step_count=int(step_count),
                    action_index=int(action_index),
                )
                if policy_trace_dir is not None:
                    print(f"[policy-trace] session_stop id={trace_session_id}", flush=True)
            last_policy_running = current_policy_running

            if int(executor.reset_generation) != last_executor_reset_generation:
                reset_policy_history()
                last_executor_reset_generation = int(executor.reset_generation)

            if current_policy_running:
                now = time.monotonic()
                should_replan = action_chunk is None or action_index >= int(args.replan_horizon)
                if bool(args.wall_clock_replan):
                    should_replan = should_replan or now - last_policy_time >= 1.0 / max(float(args.policy_hz), 1.0e-6)
                if should_replan:
                    last_policy_time = now
                    observation, source_metadata, observation_arm_q, observation_hand_q, observation_eef_pose = build_current_observation()
                    if step_count == 0 and action_index == 0:
                        rollout._print_first_observation_video_shapes(observation)  # noqa: SLF001
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
                    rtc_options, rtc_metadata = rollout._teleop_rtc_options(  # noqa: SLF001
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
                    tic = time.perf_counter()
                    policy_action_chunk, policy_info = rollout._policy_get_action_cpu_processor(  # noqa: SLF001
                        policy,
                        observation,
                        previous_action=rtc_seed_action if rtc_options is not None else None,
                        options=rtc_options,
                    )
                    frozen_steps = 0 if rtc_options is None else int(rtc_options["rtc_frozen_steps"])
                    raw_stored_action, action_storage_metadata = rollout._stored_rtc_action_chunk(  # noqa: SLF001
                        policy_action=policy_action_chunk,
                        rtc_seed_action=rtc_seed_action,
                        action_keys=action_keys,
                        frozen_steps=frozen_steps,
                    )
                    executor_input_action, bridge_metadata = deployment_bridge.bridge_action_chunk(
                        raw_stored_action,
                        observation=observation,
                        reason=f"replan_{trace_replan_index}",
                    )
                    action_storage_metadata["deployment_bridge"] = bridge_metadata
                    clip_delta = executor.hand_policy_clip_delta(executor_input_action)
                    clipped_action_chunk = executor.probe_clip_action_chunk(executor_input_action)
                    action_chunk = executor.guard_action_chunk_for_execution(clipped_action_chunk)
                    overlay_action_chunk = executor.preview_action_chunk_for_overlay(action_chunk)
                    eef_trajectory_overlay.draw_chunk(overlay_action_chunk)
                    hand_debug = executor.hand_debug_snapshot(
                        policy_action_chunk=policy_action_chunk,
                        clipped_action_chunk=clipped_action_chunk,
                        guarded_action_chunk=action_chunk,
                        rtc_seed_action=rtc_seed_action,
                        observation_hand_q=observation_hand_q,
                    )
                    hand_debug["input_source"] = source_metadata
                    hand_debug["deployment_bridge"] = bridge_metadata.get("hand", {})
                    if str(args.deployment_action_bridge) == "state_to_robot":
                        rtc_store_action_chunk = {
                            key: np.array(value, dtype=np.float32, copy=True)
                            for key, value in raw_stored_action.items()
                            if key in action_keys
                        }
                    else:
                        rtc_store_action_chunk = executor.rtc_seed_action_from_command_chunk(action_chunk, observation_arm_q=observation_arm_q)
                    inference_dt_s = time.perf_counter() - tic
                    rollout._write_policy_replan_trace(  # noqa: SLF001
                        policy_trace_dir,
                        session_id=trace_session_id,
                        replan_index=int(trace_replan_index),
                        step_count=int(step_count),
                        action_index=int(action_index),
                        observation_ts_s=float(observation_ts_s),
                        observation=observation,
                        observation_arm_q=observation_arm_q,
                        observation_hand_q=observation_hand_q,
                        observation_eef_pose=observation_eef_pose,
                        rtc_seed_action=rtc_seed_action,
                        rtc_options=rtc_options,
                        rtc_metadata=rtc_metadata,
                        rtc_seed_metadata=rtc_seed_metadata,
                        policy_action_chunk=policy_action_chunk,
                        raw_stored_action=raw_stored_action,
                        clipped_action_chunk=clipped_action_chunk,
                        execution_action_chunk=action_chunk,
                        rtc_store_action_chunk=rtc_store_action_chunk,
                        policy_info=policy_info,
                        executor_debug=executor.eef_frame_debug_snapshot(),
                        hand_debug=hand_debug,
                        action_storage_metadata=action_storage_metadata,
                        inference_dt_s=float(inference_dt_s),
                    )
                    trace_replan_index += 1
                    seed_manager.push(rtc_store_action_chunk, start_monotonic_s=observation_ts_s, frame_id=int(step_count))
                    rollout._append_jsonl(  # noqa: SLF001
                        args.policy_debug_jsonl,
                        {
                            "schema_version": "harness.groot_smooth_control_policy_debug.v1",
                            "step_count": int(step_count),
                            "action_index": int(action_index),
                            "episode_index": int(episode.episode_index),
                            "frame_index": int(current_frame),
                            "input_source": source_metadata,
                            "rtc": {"options": rtc_options, "metadata": rtc_metadata, "seed_metadata": rtc_seed_metadata},
                            "deployment_bridge": bridge_metadata,
                            "hand": hand_debug,
                            "eef_frame": executor.eef_frame_debug_snapshot(),
                            "action_summary": rollout._summarize_action_chunk(action_chunk),  # noqa: SLF001
                            "model_state_action_summary": rollout._summarize_action_chunk(raw_stored_action),  # noqa: SLF001
                            "policy_raw_action_summary": rollout._summarize_action_chunk(policy_action_chunk),  # noqa: SLF001
                        },
                    )
                    print(
                        f"[policy] replan dt={inference_dt_s:.3f}s "
                        f"episode={episode.episode_index:06d} frame={current_frame} "
                        f"video={active_video_source()} state={active_state_source()} "
                        f"rtc={'off' if rtc_options is None else 'on'} rtc_reason={rtc_metadata.get('reason')} "
                        f"bridge={bridge_metadata.get('mode')} hand_bridge_delta={bridge_metadata.get('hand', {}).get('max_abs_delta', 0.0):.4f} "
                        f"clip_delta={clip_delta:.4f} {rollout._summarize_action_chunk(action_chunk)}",
                        flush=True,
                    )
                    action_index = 0
                if action_chunk is not None:
                    eef_trajectory_overlay.update_active(action_index)
                    executor.step_action(action_chunk, action_index)
                    teleop_debug = executor.teleop_debug_snapshot()
                    if teleop_debug is not None:
                        rollout._append_jsonl(  # noqa: SLF001
                            args.policy_teleop_debug_jsonl,
                            {
                                "schema_version": "harness.groot_smooth_control_teleop_bridge.v1",
                                "step_count": int(step_count),
                                "action_index": int(action_index),
                                "debug": teleop_debug,
                            },
                        )
                    action_index += 1
                    step_count += 1
                    current_frame += 1
                    if current_frame >= episode.length:
                        current_frame = 0 if (panel is not None and bool(panel["loop"].value)) or (panel is None and bool(args.loop)) else episode.length - 1
                    if panel is not None:
                        panel["current_frame"].value = current_frame
                    if int(args.max_steps) > 0 and step_count >= int(args.max_steps):
                        print(f"[done] reached --max-steps {args.max_steps}", flush=True)
                        break
            else:
                playing = bool(panel["playing"].value) if panel is not None else bool(args.start_frames)
                sim_running = bool(panel["sim_running"].value) if panel is not None else False
                apply_smooth_pose = bool(panel["apply_smooth_pose"].value) if panel is not None else bool(args.apply_smooth_pose)
                speed_value = float(panel["speed"].value) if panel is not None else float(args.speed)
                loop_value = bool(panel["loop"].value) if panel is not None else bool(args.loop)
                if playing and episode.length > 0:
                    now = time.monotonic()
                    if now >= next_frame_time:
                        if apply_smooth_pose:
                            playback._apply_episode_frame(scene, episode.playback, current_frame, arm, arm_dofs, assembly)  # noqa: SLF001
                        if panel is not None:
                            panel["current_frame"].value = current_frame
                        current_frame += 1
                        if current_frame >= episode.length:
                            current_frame = 0 if loop_value else episode.length - 1
                            if not loop_value and panel is not None:
                                panel["playing"].value = False
                        if episode.timestamps.size > current_frame > 0:
                            raw_dt = float(episode.timestamps[current_frame] - episode.timestamps[current_frame - 1])
                            dt = raw_dt if math.isfinite(raw_dt) and raw_dt > 1.0e-5 else base_dt
                        else:
                            dt = base_dt
                        next_frame_time = now + dt / max(speed_value, 1.0e-6)
                    else:
                        scene.visualizer.update(force=True)
                elif sim_running:
                    harness._step_scene_with_attached_parts(scene)  # noqa: SLF001
                else:
                    preview_now = time.monotonic()
                    if camera_preview.enabled and preview_now - last_camera_preview_time >= preview_period_s:
                        last_camera_preview_time = preview_now
                        if active_video_source() == "sim":
                            video, _ = _build_sim_video_observation(
                                args=args,
                                ego_camera=ego_camera,
                                wrist_camera=wrist_camera,
                                image_size=image_size,
                                history=sim_video_history,
                                modality_config=modality_config,
                            )
                            if "ego_view" in video and "wrist_view" in video:
                                camera_preview.show(ego=np.asarray(video["ego_view"][0, -1]), wrist=np.asarray(video["wrist_view"][0, -1]))
                    scene.visualizer.update(force=True)

            time.sleep(1.0 / 60.0)
    finally:
        if last_policy_running:
            rollout._append_policy_trace_event(  # noqa: SLF001
                policy_trace_dir,
                session_id=trace_session_id,
                event="policy_stop",
                step_count=int(step_count),
                action_index=int(action_index),
                extra={"reason": "shutdown"},
            )
        eef_trajectory_overlay.clear()
        camera_preview.close()
        _shutdown_control_panel(panel)

    print(f"[done] steps={step_count} episode={episode.episode_index:06d} frame={current_frame}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
