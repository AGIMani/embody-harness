#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SMOOTH_DIR = Path("/home/whf/Project/Isaac-GR00T/outputs/IsaacLab/nero/mission2/smooth")

HAND_NAMES = (
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


def _import_rollout_helpers():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import tools.run_add_scene_groot_finetune as rollout

    return rollout


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _latest_trace_npz(trace_dir: Path, *, which: str) -> Path:
    rows = [r for r in _read_jsonl(trace_dir / "trace.jsonl") if r.get("record_type") == "replan" and r.get("npz_path")]
    if not rows:
        raise RuntimeError(f"No replan rows with npz_path found in {trace_dir / 'trace.jsonl'}")
    if which == "first":
        return Path(str(rows[0]["npz_path"]))
    if which == "last":
        return Path(str(rows[-1]["npz_path"]))
    raise ValueError(f"unsupported trace sample {which!r}")


def _episode_path(smooth_dir: Path, episode_index: int) -> Path:
    return smooth_dir / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def _video_path(smooth_dir: Path, episode_index: int, key: str) -> Path:
    return smooth_dir / "videos" / f"chunk-{episode_index // 1000:03d}" / key / f"episode_{episode_index:06d}.mp4"


def _read_video_frame(path: Path, frame_index: int, image_size: tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(frame_index, 0)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame={frame_index} from {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = image_size
    return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA).astype(np.uint8)


def _load_episode(smooth_dir: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(_episode_path(smooth_dir, episode_index), columns=["observation.state", "action"])
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    return state, action


def _find_grasp_frame(action: np.ndarray) -> int:
    hand = action[:, 9:19]
    score = hand[:, [3, 4, 5]].mean(axis=1)
    return int(np.argmax(score))


def _smooth_video_observation(
    *,
    smooth_dir: Path,
    episode_index: int,
    frame_index: int,
    image_size: tuple[int, int],
    video_delta_indices: list[int],
) -> dict[str, np.ndarray]:
    ego_frames = []
    wrist_frames = []
    for delta in video_delta_indices:
        idx = max(0, int(frame_index) + int(delta))
        ego_frames.append(
            _read_video_frame(
                _video_path(smooth_dir, episode_index, "observation.images.ego_view"),
                idx,
                image_size,
            )
        )
        wrist_frames.append(
            _read_video_frame(
                _video_path(smooth_dir, episode_index, "observation.images.wrist_view"),
                idx,
                image_size,
            )
        )
    return {
        "ego_view": np.stack(ego_frames, axis=0)[None, ...].astype(np.uint8),
        "wrist_view": np.stack(wrist_frames, axis=0)[None, ...].astype(np.uint8),
    }


def _smooth_state_and_reference(state: np.ndarray, action: np.ndarray, frame_index: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    idx = int(frame_index)
    state_row = state[idx]
    action_row = action[idx]
    obs_state = {
        "arm_joint_pos": state_row[0:7][None, None, :].astype(np.float32),
        "eef_9d": state_row[7:16][None, None, :].astype(np.float32),
        "hand_joint_pos": state_row[16:26][None, None, :].astype(np.float32),
    }
    reference = {
        "eef_9d": action_row[0:9][None, :].astype(np.float32),
        "hand_joint_target": action_row[9:19][None, :].astype(np.float32),
        "arm_joint_target": state_row[0:7][None, :].astype(np.float32),
    }
    return obs_state, reference


def _trace_inputs(npz_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(npz_path) as z:
        video = {
            "ego_view": np.asarray(z["observation.video.ego_view"], dtype=np.uint8),
            "wrist_view": np.asarray(z["observation.video.wrist_view"], dtype=np.uint8),
        }
        state = {
            "arm_joint_pos": np.asarray(z["observation.state.arm_joint_pos"], dtype=np.float32),
            "eef_9d": np.asarray(z["observation.state.eef_9d"], dtype=np.float32),
            "hand_joint_pos": np.asarray(z["observation.state.hand_joint_pos"], dtype=np.float32),
        }
        reference = {
            "eef_9d": np.asarray(z["reference_action.eef_9d"], dtype=np.float32),
            "hand_joint_target": np.asarray(z["reference_action.hand_joint_target"], dtype=np.float32),
            "arm_joint_target": np.asarray(z["reference_action.arm_joint_target"], dtype=np.float32),
        }
    return video, state, reference


def _roi_crop_zoom_hwc(image: np.ndarray, *, zoom: float, center_x: float, center_y: float) -> np.ndarray:
    zoom = float(zoom)
    if zoom <= 1.0:
        return image
    h, w = image.shape[:2]
    crop_w = max(1, int(round(w / zoom)))
    crop_h = max(1, int(round(h / zoom)))
    cx = int(round(float(center_x) * (w - 1)))
    cy = int(round(float(center_y) * (h - 1)))
    x0 = min(max(cx - crop_w // 2, 0), max(w - crop_w, 0))
    y0 = min(max(cy - crop_h // 2, 0), max(h - crop_h, 0))
    cropped = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA).astype(image.dtype, copy=False)


def _apply_video_roi(video: np.ndarray, *, zoom: float, center_x: float, center_y: float) -> np.ndarray:
    if float(zoom) <= 1.0:
        return video
    out = np.asarray(video).copy()
    flat = out.reshape(-1, *out.shape[-3:])
    for idx in range(flat.shape[0]):
        flat[idx] = _roi_crop_zoom_hwc(flat[idx], zoom=zoom, center_x=center_x, center_y=center_y)
    return out


def _make_observation(
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


def _first_hand(action: dict[str, np.ndarray]) -> np.ndarray:
    hand = np.asarray(action["hand_joint_target"], dtype=np.float64)
    if hand.ndim == 3:
        return hand[0, 0, : len(HAND_NAMES)]
    if hand.ndim == 2:
        return hand[0, : len(HAND_NAMES)]
    return hand.reshape(-1)[: len(HAND_NAMES)]


def _hand_range(action: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    hand = np.asarray(action["hand_joint_target"], dtype=np.float64)
    if hand.ndim == 3:
        hand = hand[0]
    if hand.ndim == 1:
        hand = hand[None, :]
    hand = hand[:, : len(HAND_NAMES)]
    return np.min(hand, axis=0), np.max(hand, axis=0)


def _row_for_result(
    *,
    case: str,
    episode_index: int,
    frame_index: int,
    smooth_kind: str,
    trace_npz: Path,
    action: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> dict[str, Any]:
    first = _first_hand(action)
    low, high = _hand_range(action)
    ref = np.asarray(reference["hand_joint_target"], dtype=np.float64).reshape(-1)[: len(HAND_NAMES)]
    eef_first = np.asarray(action.get("eef_9d", []), dtype=np.float64).reshape(-1)[:9]
    eef_values = np.asarray(action.get("eef_9d", []), dtype=np.float64)
    if eef_values.size:
        if eef_values.ndim == 3:
            eef_values = eef_values[0]
        elif eef_values.ndim == 1:
            eef_values = eef_values[None, :]
        eef_low = np.min(eef_values[:, :9], axis=0)
        eef_high = np.max(eef_values[:, :9], axis=0)
    else:
        eef_low = np.asarray([], dtype=np.float64)
        eef_high = np.asarray([], dtype=np.float64)
    eef_ref = np.asarray(reference.get("eef_9d", []), dtype=np.float64).reshape(-1)[:9]
    row: dict[str, Any] = {
        "case": case,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "smooth_kind": smooth_kind,
        "trace_npz": trace_npz.name if trace_npz else "",
    }
    for idx, name in enumerate(HAND_NAMES):
        row[f"{name}.first"] = float(first[idx])
        row[f"{name}.delta_from_ref_first"] = float(first[idx] - ref[idx])
        row[f"{name}.min"] = float(low[idx])
        row[f"{name}.max"] = float(high[idx])
        row[f"{name}.ref"] = float(ref[idx])
    for idx in range(min(eef_first.size, 9)):
        row[f"eef_9d.{idx}.first"] = float(eef_first[idx])
        if idx < eef_low.size:
            row[f"eef_9d.{idx}.min"] = float(eef_low[idx])
            row[f"eef_9d.{idx}.max"] = float(eef_high[idx])
        if idx < eef_ref.size:
            row[f"eef_9d.{idx}.delta_from_ref_first"] = float(eef_first[idx] - eef_ref[idx])
            row[f"eef_9d.{idx}.ref"] = float(eef_ref[idx])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject smooth videos/states into the finetuned GR00T policy and compare against simulator inputs."
    )
    parser.add_argument("--smooth-dir", type=Path, default=SMOOTH_DIR)
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "logs" / "groot_finetune_policy_trace_current_180x320_no_rtc_160")
    parser.add_argument("--trace-sample", choices=("first", "last"), default="last")
    parser.add_argument("--episodes", default="0,1,2,3")
    parser.add_argument(
        "--frames",
        default="",
        help="Optional comma-separated smooth frame indices to test in addition to start/grasp.",
    )
    parser.add_argument("--image-size", default="180,320")
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--sim-ego-roi-zoom", type=float, default=1.0)
    parser.add_argument("--sim-ego-roi-center-x", type=float, default=0.5)
    parser.add_argument("--sim-ego-roi-center-y", type=float, default=0.5)
    parser.add_argument("--sim-wrist-roi-zoom", type=float, default=1.0)
    parser.add_argument("--sim-wrist-roi-center-x", type=float, default=0.5)
    parser.add_argument("--sim-wrist-roi-center-y", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=ROOT / "logs" / "groot_smooth_image_injection" / "results.csv")
    args = parser.parse_args()

    image_size = tuple(int(v.strip()) for v in str(args.image_size).split(","))
    if len(image_size) != 2:
        raise SystemExit("--image-size must be height,width")

    rollout = _import_rollout_helpers()
    policy_args = SimpleNamespace(
        policy_checkpoint=rollout.DEFAULT_POLICY_CHECKPOINT,
        cosmos_model=rollout.DEFAULT_COSMOS_MODEL,
        policy_device=str(args.policy_device),
        no_policy_strict=False,
    )
    policy = rollout._load_groot_policy(policy_args)  # noqa: SLF001
    modality_config = policy.get_modality_config()
    rollout._validate_policy_video_schema(modality_config)  # noqa: SLF001
    video_delta_indices = [int(v) for v in modality_config["video"].delta_indices]
    instruction = rollout.DEFAULT_TASK

    trace_npz = _latest_trace_npz(args.trace_dir.expanduser().resolve(), which=str(args.trace_sample))
    sim_video, sim_state, sim_reference = _trace_inputs(trace_npz)
    sim_video["ego_view"] = _apply_video_roi(
        sim_video["ego_view"],
        zoom=float(args.sim_ego_roi_zoom),
        center_x=float(args.sim_ego_roi_center_x),
        center_y=float(args.sim_ego_roi_center_y),
    )
    sim_video["wrist_view"] = _apply_video_roi(
        sim_video["wrist_view"],
        zoom=float(args.sim_wrist_roi_zoom),
        center_x=float(args.sim_wrist_roi_center_x),
        center_y=float(args.sim_wrist_roi_center_y),
    )

    rows: list[dict[str, Any]] = []
    episodes = [int(v.strip()) for v in str(args.episodes).split(",") if v.strip()]
    extra_frames = [int(v.strip()) for v in str(args.frames).split(",") if v.strip()]
    for episode_index in episodes:
        state, action = _load_episode(args.smooth_dir.expanduser().resolve(), episode_index)
        sample_frames = [("start", 0), ("grasp", _find_grasp_frame(action))]
        for frame_index in extra_frames:
            sample_frames.append((f"frame{int(frame_index):04d}", int(frame_index)))
        for smooth_kind, frame_index in sample_frames:
            smooth_video = _smooth_video_observation(
                smooth_dir=args.smooth_dir.expanduser().resolve(),
                episode_index=episode_index,
                frame_index=frame_index,
                image_size=image_size,
                video_delta_indices=video_delta_indices,
            )
            smooth_state, smooth_reference = _smooth_state_and_reference(state, action, frame_index)
            cases = (
                ("smooth_images+smooth_state", smooth_video, smooth_state, smooth_reference),
                ("smooth_images+sim_state", smooth_video, sim_state, sim_reference),
                ("smooth_images+smooth_state+sim_ref", smooth_video, smooth_state, sim_reference),
                ("smooth_images+sim_state+smooth_ref", smooth_video, sim_state, smooth_reference),
                ("sim_images+smooth_state", sim_video, smooth_state, smooth_reference),
                ("sim_images+sim_state", sim_video, sim_state, sim_reference),
                ("sim_images+smooth_state+sim_ref", sim_video, smooth_state, sim_reference),
                ("sim_images+sim_state+smooth_ref", sim_video, sim_state, smooth_reference),
            )
            for case, video_in, state_in, reference_in in cases:
                observation = _make_observation(
                    modality_config=modality_config,
                    video=video_in,
                    state=state_in,
                    instruction=instruction,
                )
                action_out, _ = rollout._policy_get_action_cpu_processor(  # noqa: SLF001
                    policy,
                    observation,
                    reference_action=reference_in,
                    previous_action=None,
                    options=None,
                )
                rows.append(
                    _row_for_result(
                        case=case,
                        episode_index=episode_index,
                        frame_index=frame_index,
                        smooth_kind=smooth_kind,
                        trace_npz=trace_npz,
                        action=action_out,
                        reference=reference_in,
                    )
                )
                first = _first_hand(action_out)
                print(
                    f"[case] {case:26s} ep={episode_index:02d} frame={frame_index:04d} {smooth_kind:5s} "
                    f"idx/mid/ring/pinky=({first[2]:+.4f},{first[3]:+.4f},{first[4]:+.4f},{first[5]:+.4f}) "
                    f"eef_xyz=({float(np.asarray(action_out.get('eef_9d')).reshape(-1)[0]):+.4f},"
                    f"{float(np.asarray(action_out.get('eef_9d')).reshape(-1)[1]):+.4f},"
                    f"{float(np.asarray(action_out.get('eef_9d')).reshape(-1)[2]):+.4f})",
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[output] {args.output}")


if __name__ == "__main__":
    main()
