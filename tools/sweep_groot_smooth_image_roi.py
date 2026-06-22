#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import analyze_groot_smooth_image_injection as inj


PITCH_NAMES = ("index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch")
PITCH_IDXS = tuple(inj.HAND_NAMES.index(name) for name in PITCH_NAMES)


def _float_list(text: str) -> list[float]:
    return [float(v.strip()) for v in str(text).split(",") if v.strip()]


def _int_list(text: str) -> list[int]:
    return [int(v.strip()) for v in str(text).split(",") if v.strip()]


def _hand(action: dict[str, np.ndarray]) -> np.ndarray:
    arr = np.asarray(action["hand_joint_target"], dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, : len(inj.HAND_NAMES)]


def _row(
    *,
    case: str,
    frame_index: int,
    smooth_kind: str,
    wrist_zoom: float,
    wrist_center_y: float,
    action: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> dict[str, Any]:
    hand = _hand(action)
    ref = np.asarray(reference["hand_joint_target"], dtype=np.float64).reshape(-1)[: len(inj.HAND_NAMES)]
    out: dict[str, Any] = {
        "case": case,
        "frame_index": int(frame_index),
        "smooth_kind": smooth_kind,
        "wrist_zoom": float(wrist_zoom),
        "wrist_center_y": float(wrist_center_y),
    }
    for name, idx in zip(PITCH_NAMES, PITCH_IDXS, strict=True):
        out[f"{name}.first"] = float(hand[0, idx])
        out[f"{name}.max"] = float(np.max(hand[:, idx]))
        out[f"{name}.delta_first"] = float(hand[0, idx] - ref[idx])
    out["pitch_first_mean"] = float(np.mean([out[f"{name}.first"] for name in PITCH_NAMES]))
    out["pitch_max_mean"] = float(np.mean([out[f"{name}.max"] for name in PITCH_NAMES]))
    out["non_index_pitch_max_mean"] = float(
        np.mean([out[f"{name}.max"] for name in ("middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch")])
    )
    return out


def _mixed_video(
    *,
    ego_view: np.ndarray,
    wrist_view: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "ego_view": np.asarray(ego_view, dtype=np.uint8),
        "wrist_view": np.asarray(wrist_view, dtype=np.uint8),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep digital ROI on traced simulator policy videos.")
    parser.add_argument("--smooth-dir", type=Path, default=inj.SMOOTH_DIR)
    parser.add_argument("--trace-dir", type=Path, default=inj.ROOT / "logs" / "groot_finetune_policy_trace_reset_aligned_no_rtc_64")
    parser.add_argument("--trace-sample", choices=("first", "last"), default="first")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", default="0,64")
    parser.add_argument("--image-size", default="180,320")
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--wrist-zooms", default="1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--wrist-center-ys", default="0.45,0.60,0.75,0.85")
    parser.add_argument("--wrist-center-x", type=float, default=0.5)
    parser.add_argument("--ego-zoom", type=float, default=1.0)
    parser.add_argument("--ego-center-x", type=float, default=0.5)
    parser.add_argument("--ego-center-y", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=inj.ROOT / "logs" / "groot_smooth_image_injection" / "roi_sweep.csv")
    args = parser.parse_args()

    image_size = tuple(int(v.strip()) for v in str(args.image_size).split(","))
    if len(image_size) != 2:
        raise SystemExit("--image-size must be height,width")

    rollout = inj._import_rollout_helpers()  # noqa: SLF001
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

    trace_npz = inj._latest_trace_npz(args.trace_dir.expanduser().resolve(), which=str(args.trace_sample))  # noqa: SLF001
    sim_video_base, sim_state, sim_reference = inj._trace_inputs(trace_npz)  # noqa: SLF001
    if float(args.ego_zoom) > 1.0:
        sim_video_base["ego_view"] = inj._apply_video_roi(  # noqa: SLF001
            sim_video_base["ego_view"],
            zoom=float(args.ego_zoom),
            center_x=float(args.ego_center_x),
            center_y=float(args.ego_center_y),
        )

    state, action = inj._load_episode(args.smooth_dir.expanduser().resolve(), int(args.episode))  # noqa: SLF001
    frames = _int_list(args.frames)
    if not frames:
        frames = [0, inj._find_grasp_frame(action)]  # noqa: SLF001

    rows: list[dict[str, Any]] = []
    for wrist_zoom in _float_list(args.wrist_zooms):
        for wrist_center_y in _float_list(args.wrist_center_ys):
            sim_video = {key: np.asarray(value).copy() for key, value in sim_video_base.items()}
            sim_video["wrist_view"] = inj._apply_video_roi(  # noqa: SLF001
                sim_video["wrist_view"],
                zoom=float(wrist_zoom),
                center_x=float(args.wrist_center_x),
                center_y=float(wrist_center_y),
            )
            for frame_index in frames:
                smooth_video = inj._smooth_video_observation(  # noqa: SLF001
                    smooth_dir=args.smooth_dir.expanduser().resolve(),
                    episode_index=int(args.episode),
                    frame_index=int(frame_index),
                    image_size=image_size,
                    video_delta_indices=video_delta_indices,
                )
                smooth_state, smooth_reference = inj._smooth_state_and_reference(state, action, int(frame_index))  # noqa: SLF001
                smooth_ego_sim_wrist = _mixed_video(
                    ego_view=smooth_video["ego_view"],
                    wrist_view=sim_video["wrist_view"],
                )
                sim_ego_smooth_wrist = _mixed_video(
                    ego_view=sim_video["ego_view"],
                    wrist_view=smooth_video["wrist_view"],
                )
                cases = (
                    ("smooth_images+smooth_state", smooth_video, smooth_state, smooth_reference),
                    ("smooth_ego+sim_wrist+smooth_state", smooth_ego_sim_wrist, smooth_state, smooth_reference),
                    ("sim_ego+smooth_wrist+smooth_state", sim_ego_smooth_wrist, smooth_state, smooth_reference),
                    ("sim_images+smooth_state", sim_video, smooth_state, smooth_reference),
                    ("smooth_images+sim_state+sim_ref", smooth_video, sim_state, sim_reference),
                    ("smooth_ego+sim_wrist+sim_state+sim_ref", smooth_ego_sim_wrist, sim_state, sim_reference),
                    ("sim_ego+smooth_wrist+sim_state+sim_ref", sim_ego_smooth_wrist, sim_state, sim_reference),
                    ("sim_images+sim_state", sim_video, sim_state, sim_reference),
                    ("sim_images+sim_state+smooth_ref", sim_video, sim_state, smooth_reference),
                )
                for case, video_in, state_in, reference_in in cases:
                    observation = inj._make_observation(  # noqa: SLF001
                        modality_config=modality_config,
                        video=video_in,
                        state=state_in,
                        instruction=rollout.DEFAULT_TASK,
                    )
                    action_out, _ = rollout._policy_get_action_cpu_processor(  # noqa: SLF001
                        policy,
                        observation,
                        reference_action=reference_in,
                        previous_action=None,
                        options=None,
                    )
                    row = _row(
                        case=case,
                        frame_index=int(frame_index),
                        smooth_kind=f"frame{int(frame_index):04d}",
                        wrist_zoom=float(wrist_zoom),
                        wrist_center_y=float(wrist_center_y),
                        action=action_out,
                        reference=reference_in,
                    )
                    rows.append(row)
                    print(
                        f"[roi] zoom={wrist_zoom:.2f} y={wrist_center_y:.2f} "
                        f"frame={frame_index:04d} {case:28s} "
                        f"non_index_max={row['non_index_pitch_max_mean']:+.4f}",
                        flush=True,
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[output] {args.output}")


if __name__ == "__main__":
    main()
