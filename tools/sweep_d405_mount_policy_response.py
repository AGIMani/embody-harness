#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import add_scene_glb as harness  # noqa: E402
from debug import debug_d405_smooth_wrist_calibration as calib  # noqa: E402
from tools import analyze_groot_smooth_image_injection as inj  # noqa: E402
from tools import run_add_scene_smooth_playback as playback  # noqa: E402


PITCH_NAMES = ("index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch")
PITCH_IDXS = tuple(inj.HAND_NAMES.index(name) for name in PITCH_NAMES)


def _vec3(text: str) -> tuple[float, float, float]:
    parts = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected numeric x,y,z") from exc


def _vec2_int(text: str) -> tuple[int, int]:
    parts = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected height,width")
    try:
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected integer height,width") from exc


def _float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated number")
    return values


def _int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def _hand_chunk(action: dict[str, np.ndarray]) -> np.ndarray:
    arr = np.asarray(action["hand_joint_target"], dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, : len(inj.HAND_NAMES)]


def _score_action(action: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, float]:
    hand = _hand_chunk(action)
    ref = np.asarray(reference["hand_joint_target"], dtype=np.float64).reshape(-1)[: len(inj.HAND_NAMES)]
    out: dict[str, float] = {}
    for name, idx in zip(PITCH_NAMES, PITCH_IDXS, strict=True):
        out[f"{name}.first"] = float(hand[0, idx])
        out[f"{name}.max"] = float(np.max(hand[:, idx]))
        out[f"{name}.delta_first"] = float(hand[0, idx] - ref[idx])
    out["non_index_pitch_max_mean"] = float(
        np.mean([out[f"{name}.max"] for name in ("middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch")])
    )
    out["all_pitch_max_mean"] = float(np.mean([out[f"{name}.max"] for name in PITCH_NAMES]))
    return out


def _smooth_modality_video_delta_indices(smooth_dir: Path) -> list[int]:
    modality_path = smooth_dir / "meta" / "modality.json"
    if not modality_path.exists():
        return [0]
    data = json.loads(modality_path.read_text(encoding="utf-8"))
    video = data.get("video", {})
    delta_indices = video.get("delta_indices", [0]) if isinstance(video, dict) else [0]
    return [int(v) for v in delta_indices]


def _image_similarity(sim: np.ndarray, smooth: np.ndarray) -> dict[str, float]:
    import cv2

    sim_f = np.asarray(sim, dtype=np.float32)
    smooth_f = np.asarray(smooth, dtype=np.float32)
    diff = sim_f - smooth_f
    mad = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    sim_flat = sim_f.reshape(-1)
    smooth_flat = smooth_f.reshape(-1)
    if float(np.std(sim_flat)) < 1.0e-6 or float(np.std(smooth_flat)) < 1.0e-6:
        rgb_corr = 0.0
    else:
        rgb_corr = float(np.corrcoef(sim_flat, smooth_flat)[0, 1])

    sim_gray = cv2.cvtColor(np.asarray(sim, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    smooth_gray = cv2.cvtColor(np.asarray(smooth, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    sim_edges = cv2.Canny(sim_gray, 40, 120).astype(np.float32)
    smooth_edges = cv2.Canny(smooth_gray, 40, 120).astype(np.float32)
    if float(np.std(sim_edges)) < 1.0e-6 or float(np.std(smooth_edges)) < 1.0e-6:
        edge_corr = 0.0
    else:
        edge_corr = float(np.corrcoef(sim_edges.reshape(-1), smooth_edges.reshape(-1))[0, 1])
    score = float(rgb_corr + 0.5 * edge_corr - 0.01 * mad)
    return {
        "image_score": score,
        "image_rgb_corr": rgb_corr,
        "image_edge_corr": edge_corr,
        "image_mad": mad,
        "image_rmse": rmse,
    }


def _make_scene(args: argparse.Namespace) -> object:
    return calib._create_scene(args)  # noqa: SLF001


def _load_smooth_episode(args: argparse.Namespace, *, episode_index: int):
    smooth_dir = args.smooth_dir.expanduser().resolve()
    episodes = playback._discover_episodes(smooth_dir)  # noqa: SLF001
    selected_index = playback._selected_episode_list_index(episodes, int(episode_index))  # noqa: SLF001
    episode = playback._load_episode_data(  # noqa: SLF001
        episodes[selected_index],
        smooth_dir=smooth_dir,
        hand_source=str(args.hand_source),
    )
    return smooth_dir, episode


def _render_candidate_wrist(scene: object, *, values: tuple[float, ...], image_size: tuple[int, int]) -> np.ndarray:
    calib._apply_d405_values(scene, values)  # noqa: SLF001
    return calib._render_wrist(scene, image_size)  # noqa: SLF001


def _candidate_values(
    *,
    base_offset: tuple[float, float, float],
    base_euler: tuple[float, float, float],
    x_offsets: tuple[float, ...],
    y_offsets: tuple[float, ...],
    z_offsets: tuple[float, ...],
    roll_offsets: tuple[float, ...],
    pitch_offsets: tuple[float, ...],
    yaw_offsets: tuple[float, ...],
) -> list[tuple[float, ...]]:
    values: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for dx in x_offsets:
        for dy in y_offsets:
            for dz in z_offsets:
                for dr in roll_offsets:
                    for dp in pitch_offsets:
                        for dyaw in yaw_offsets:
                            item = (
                                float(base_offset[0] + dx),
                                float(base_offset[1] + dy),
                                float(base_offset[2] + dz),
                                float(base_euler[0] + dr),
                                float(base_euler[1] + dp),
                                float(base_euler[2] + dyaw),
                            )
                            key = tuple(round(v, 9) for v in item)
                            if key not in seen:
                                seen.add(key)
                                values.append(item)
    return values


def _label_image(image: np.ndarray, text: str) -> np.ndarray:
    import cv2

    out = np.asarray(image, dtype=np.uint8).copy()
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1] - 1, 24), (0, 0, 0), -1)
    cv2.putText(bgr, text[:80], (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _save_montage(
    *,
    output: Path,
    smooth_wrist_images: list[np.ndarray],
    rows: list[dict[str, Any]],
    images: dict[int, np.ndarray],
    top_k: int,
) -> None:
    import cv2

    panels = []
    for frame_i, smooth_wrist in enumerate(smooth_wrist_images[:3]):
        panels.append(_label_image(smooth_wrist, f"smooth wrist target {frame_i}"))
    for row in rows[: max(0, int(top_k))]:
        idx = int(row["candidate_index"])
        image = images.get(idx)
        if image is None:
            continue
        score_value = float(row.get("non_index_pitch_max_mean", row.get("image_score", 0.0)))
        label = f"#{idx} score={score_value:.3f} euler=({row['roll_deg']:.1f},{row['pitch_deg']:.1f},{row['yaw_deg']:.1f})"
        panels.append(_label_image(image, label))
    if not panels:
        return
    height = panels[0].shape[0]
    width = panels[0].shape[1]
    cols = min(3, len(panels))
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    grid_rows = []
    for start in range(0, len(panels), cols):
        chunk = panels[start : start + cols]
        while len(chunk) < cols:
            chunk.append(blank.copy())
        grid_rows.append(np.concatenate(chunk, axis=1))
    montage = np.concatenate(grid_rows, axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    print(f"[output] montage={output}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score D405 connector-relative mount candidates by feeding rendered Genesis wrist images "
            "through the GR00T policy with smooth ego/state/reference."
        )
    )
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--smooth-dir", type=Path, default=inj.SMOOTH_DIR)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=56)
    parser.add_argument(
        "--episode-indices",
        type=_int_list,
        default=None,
        help="Comma-separated smooth episode indices for multi-frame scoring. Overrides --episode-index when set.",
    )
    parser.add_argument(
        "--frame-indices",
        type=_int_list,
        default=None,
        help="Comma-separated smooth frame indices for multi-frame scoring. Overrides --frame-index when set.",
    )
    parser.add_argument("--hand-source", choices=("state", "action"), default="action")
    parser.add_argument("--image-size", type=_vec2_int, default=(180, 320), help="height,width")
    parser.add_argument(
        "--score-mode",
        choices=("policy", "image"),
        default="policy",
        help="policy runs GR00T action response scoring; image only scores rendered wrist similarity to smooth wrist.",
    )
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--initial-offset", type=_vec3, default=harness.RIGHT_D405_CONNECTOR_REL_POS_M)
    parser.add_argument("--initial-euler", type=_vec3, default=harness.RIGHT_D405_CONNECTOR_REL_EULER_DEG)
    parser.add_argument("--x-offsets", type=_float_list, default=(0.0,))
    parser.add_argument("--y-offsets", type=_float_list, default=(0.0,))
    parser.add_argument("--z-offsets", type=_float_list, default=(0.0,))
    parser.add_argument("--roll-offsets", type=_float_list, default=(0.0,))
    parser.add_argument("--pitch-offsets", type=_float_list, default=(-90.0, -45.0, 0.0, 45.0, 90.0))
    parser.add_argument("--yaw-offsets", type=_float_list, default=(-180.0, -90.0, 0.0, 90.0, 180.0))
    parser.add_argument("--max-candidates", type=int, default=0, help="Optional cap after candidate generation; 0 means all.")
    parser.add_argument("--output", type=Path, default=ROOT / "logs" / "groot_smooth_image_injection" / "d405_mount_policy_sweep.csv")
    parser.add_argument("--output-montage", type=Path, default=ROOT / "logs" / "groot_smooth_image_injection" / "d405_mount_policy_sweep_top.png")
    parser.add_argument("--top-k", type=int, default=8)

    # Scene arguments reused by debug_d405_smooth_wrist_calibration._create_scene.
    parser.add_argument("--combined-urdf", type=Path, default=harness.DEFAULT_COMBINED_NERO_LINKER_URDF)
    parser.add_argument("--bottle-pos", type=_vec3, default=playback.DEFAULT_BOTTLE_POS)
    parser.add_argument("--bottle-euler", type=_vec3, default=playback.DEFAULT_BOTTLE_EULER)
    parser.add_argument("--bottle-proxy-json", type=Path, default=harness.DEFAULT_BOTTLE_PROXY_JSON)
    parser.add_argument("--show-bottle-proxy", action="store_true")
    parser.add_argument("--scene-support-collider-pos", type=_vec3, default=playback.DEFAULT_SCENE_SUPPORT_COLLIDER_POS)
    parser.add_argument("--scene-support-collider-size", type=_vec3, default=playback.DEFAULT_SCENE_SUPPORT_COLLIDER_SIZE)
    parser.add_argument("--show-scene-support-collider", action="store_true")
    parser.add_argument("--linker-hand-collision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-viewer", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    image_size = tuple(int(v) for v in args.image_size)
    episode_indices = tuple(int(v) for v in (args.episode_indices if args.episode_indices is not None else (args.episode_index,)))
    requested_frame_indices = tuple(
        int(v) for v in (args.frame_indices if args.frame_indices is not None else (args.frame_index,))
    )
    smooth_dir = args.smooth_dir.expanduser().resolve()
    episodes_by_index: dict[int, Any] = {}
    state_action_by_episode: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    samples: list[dict[str, Any]] = []
    for episode_index in episode_indices:
        smooth_dir, episode = _load_smooth_episode(args, episode_index=episode_index)
        episodes_by_index[int(episode.episode_index)] = episode
        state, action = inj._load_episode(smooth_dir, int(episode.episode_index))  # noqa: SLF001
        state_action_by_episode[int(episode.episode_index)] = (state, action)
        for frame_index_raw in requested_frame_indices:
            frame_index = int(np.clip(int(frame_index_raw), 0, max(episode.length - 1, 0)))
            samples.append(
                {
                    "episode": episode,
                    "episode_index": int(episode.episode_index),
                    "frame_index": frame_index,
                    "state": state,
                    "action": action,
                }
            )
    if not samples:
        raise RuntimeError("no smooth samples selected")

    rollout = None
    policy = None
    modality_config = None
    if str(args.score_mode) == "policy":
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
    else:
        video_delta_indices = _smooth_modality_video_delta_indices(smooth_dir)

    prepared_samples: list[dict[str, Any]] = []
    smooth_wrist_images: list[np.ndarray] = []
    for sample in samples:
        smooth_state, smooth_reference = inj._smooth_state_and_reference(  # noqa: SLF001
            sample["state"],
            sample["action"],
            int(sample["frame_index"]),
        )
        smooth_video = inj._smooth_video_observation(  # noqa: SLF001
            smooth_dir=smooth_dir,
            episode_index=int(sample["episode_index"]),
            frame_index=int(sample["frame_index"]),
            image_size=image_size,
            video_delta_indices=video_delta_indices,
        )
        smooth_wrist = np.asarray(smooth_video["wrist_view"][0, -1], dtype=np.uint8)
        smooth_wrist_images.append(smooth_wrist)
        prepared_samples.append(
            {
                **sample,
                "smooth_state": smooth_state,
                "smooth_reference": smooth_reference,
                "smooth_video": smooth_video,
                "smooth_wrist": smooth_wrist,
            }
        )

    scene = _make_scene(args)
    assembly = getattr(scene, "nero_assembly_info", None)
    if not isinstance(assembly, dict):
        raise RuntimeError("add_scene_glb scene did not create Nero assembly info")
    arm = assembly.get("right")
    if arm is None:
        raise RuntimeError("add_scene_glb scene does not contain right arm")
    prefixes = assembly.get("arm_joint_prefixes", {})
    right_prefix = str(prefixes.get("right", "")) if isinstance(prefixes, dict) else ""
    arm_dofs = harness._arm_dofs(arm, joint_prefix=right_prefix)  # noqa: SLF001
    first_sample = prepared_samples[0]
    playback._apply_episode_frame(  # noqa: SLF001
        scene,
        first_sample["episode"],
        int(first_sample["frame_index"]),
        arm,
        arm_dofs,
        assembly,
    )

    candidates = _candidate_values(
        base_offset=tuple(float(v) for v in args.initial_offset),
        base_euler=tuple(float(v) for v in args.initial_euler),
        x_offsets=tuple(float(v) for v in args.x_offsets),
        y_offsets=tuple(float(v) for v in args.y_offsets),
        z_offsets=tuple(float(v) for v in args.z_offsets),
        roll_offsets=tuple(float(v) for v in args.roll_offsets),
        pitch_offsets=tuple(float(v) for v in args.pitch_offsets),
        yaw_offsets=tuple(float(v) for v in args.yaw_offsets),
    )
    if int(args.max_candidates) > 0:
        candidates = candidates[: int(args.max_candidates)]
    print(
        f"[sweep] samples={[(s['episode_index'], s['frame_index']) for s in prepared_samples]} "
        f"candidates={len(candidates)} image_size={image_size}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    candidate_images: dict[int, np.ndarray] = {}

    baseline_non_index_mean = float("nan")
    baseline_non_index_values: list[float] = []
    if str(args.score_mode) == "policy":
        assert rollout is not None and policy is not None and modality_config is not None
        baseline_scores: list[dict[str, float]] = []
        for sample in prepared_samples:
            baseline_observation = inj._make_observation(  # noqa: SLF001
                modality_config=modality_config,
                video=sample["smooth_video"],
                state=sample["smooth_state"],
                instruction=rollout.DEFAULT_TASK,
            )
            baseline_action, _ = rollout._policy_get_action_cpu_processor(  # noqa: SLF001
                policy,
                baseline_observation,
                reference_action=sample["smooth_reference"],
                previous_action=None,
                options=None,
            )
            baseline_scores.append(_score_action(baseline_action, sample["smooth_reference"]))
        baseline_non_index_values = [score["non_index_pitch_max_mean"] for score in baseline_scores]
        baseline_non_index_mean = float(np.mean(baseline_non_index_values))
        print(
            f"[baseline] smooth_wrist non_index_max_mean={baseline_non_index_mean:.4f} "
            f"per_sample={[round(v, 4) for v in baseline_non_index_values]}",
            flush=True,
        )
    else:
        print("[baseline] image mode skips GR00T policy baseline", flush=True)

    for idx, values in enumerate(candidates):
        sample_scores: list[dict[str, float]] = []
        image_for_montage: np.ndarray | None = None
        for sample_i, sample in enumerate(prepared_samples):
            playback._apply_episode_frame(  # noqa: SLF001
                scene,
                sample["episode"],
                int(sample["frame_index"]),
                arm,
                arm_dofs,
                assembly,
            )
            wrist = _render_candidate_wrist(scene, values=values, image_size=image_size)
            if sample_i == 0:
                image_for_montage = wrist
            if str(args.score_mode) == "policy":
                assert rollout is not None and policy is not None and modality_config is not None
                video = {
                    "ego_view": np.asarray(sample["smooth_video"]["ego_view"], dtype=np.uint8),
                    "wrist_view": wrist[None, None, ...].astype(np.uint8),
                }
                observation = inj._make_observation(  # noqa: SLF001
                    modality_config=modality_config,
                    video=video,
                    state=sample["smooth_state"],
                    instruction=rollout.DEFAULT_TASK,
                )
                action_out, _ = rollout._policy_get_action_cpu_processor(  # noqa: SLF001
                    policy,
                    observation,
                    reference_action=sample["smooth_reference"],
                    previous_action=None,
                    options=None,
                )
                sample_scores.append(_score_action(action_out, sample["smooth_reference"]))
            else:
                sample_scores.append(_image_similarity(wrist, sample["smooth_wrist"]))
        if image_for_montage is not None:
            candidate_images[idx] = image_for_montage
        score: dict[str, float] = {}
        score_keys = sorted(sample_scores[0].keys())
        for key in score_keys:
            values_for_key = [float(item[key]) for item in sample_scores]
            score[key] = float(np.mean(values_for_key))
            score[f"{key}.min_sample"] = float(np.min(values_for_key))
            score[f"{key}.max_sample"] = float(np.max(values_for_key))
        row: dict[str, Any] = {
            "candidate_index": int(idx),
            "x_m": float(values[0]),
            "y_m": float(values[1]),
            "z_m": float(values[2]),
            "roll_deg": float(values[3]),
            "pitch_deg": float(values[4]),
            "yaw_deg": float(values[5]),
            "sample_count": int(len(prepared_samples)),
            "sample_keys": ";".join(f"ep{int(s['episode_index']):06d}:f{int(s['frame_index']):03d}" for s in prepared_samples),
            "score_mode": str(args.score_mode),
        }
        if baseline_non_index_values:
            row.update(
                {
                    "smooth_baseline_non_index_pitch_max_mean": baseline_non_index_mean,
                    "smooth_baseline_non_index_pitch_max_min_sample": float(np.min(baseline_non_index_values)),
                    "smooth_baseline_non_index_pitch_max_max_sample": float(np.max(baseline_non_index_values)),
                }
            )
        row.update(score)
        rows.append(row)
        if str(args.score_mode) == "policy":
            print(
                f"[candidate] #{idx:03d} non_index_max_mean={score['non_index_pitch_max_mean']:.4f} "
                f"min={score['non_index_pitch_max_mean.min_sample']:.4f} "
                f"pos=({values[0]:+.4f},{values[1]:+.4f},{values[2]:+.4f}) "
                f"euler=({values[3]:+.1f},{values[4]:+.1f},{values[5]:+.1f})",
                flush=True,
            )
        else:
            print(
                f"[candidate] #{idx:03d} image_score={score['image_score']:.4f} "
                f"rgb_corr={score['image_rgb_corr']:.4f} edge_corr={score['image_edge_corr']:.4f} "
                f"mad={score['image_mad']:.2f} "
                f"pos=({values[0]:+.4f},{values[1]:+.4f},{values[2]:+.4f}) "
                f"euler=({values[3]:+.1f},{values[4]:+.1f},{values[5]:+.1f})",
                flush=True,
            )

    sort_key = "non_index_pitch_max_mean" if str(args.score_mode) == "policy" else "image_score"
    rows.sort(key=lambda item: float(item[sort_key]), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[output] csv={args.output}", flush=True)
    _save_montage(
        output=args.output_montage,
        smooth_wrist_images=smooth_wrist_images,
        rows=rows,
        images=candidate_images,
        top_k=int(args.top_k),
    )
    if rows:
        best = rows[0]
        print(
            "[best] "
            f"candidate_index={best['candidate_index']} "
            f"{sort_key}={float(best[sort_key]):.4f} "
            f"pos=({float(best['x_m']):.6f},{float(best['y_m']):.6f},{float(best['z_m']):.6f}) "
            f"euler=({float(best['roll_deg']):.3f},{float(best['pitch_deg']):.3f},{float(best['yaw_deg']):.3f})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
