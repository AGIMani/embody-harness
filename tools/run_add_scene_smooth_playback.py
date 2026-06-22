#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
DEFAULT_SMOOTH_DIR = ROOT_DIR.parent / "Isaac-GR00T" / "outputs" / "IsaacLab" / "nero" / "mission2" / "smooth"
DEFAULT_BOTTLE_POS = (-0.395556, -0.093333, 0.794444)
DEFAULT_BOTTLE_EULER = (0.0, 0.0, 37.448)
DEFAULT_SCENE_SUPPORT_COLLIDER_POS = (-0.616071, -0.064286, 0.620536)
DEFAULT_SCENE_SUPPORT_COLLIDER_SIZE = (0.700000, 0.700000, 0.040000)
HAND_JOINT_NAMES = (
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


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    length: int
    path: Path
    task: str

    @property
    def label(self) -> str:
        return f"{self.episode_index:06d}  {self.length:4d} frames  {self.task}"


@dataclass(frozen=True)
class EpisodeData:
    episode_index: int
    arm_q: np.ndarray
    hand_q: np.ndarray | None
    timestamps: np.ndarray

    @property
    def length(self) -> int:
        return int(self.arm_q.shape[0])


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_path_from_index(smooth_dir: Path, episode_index: int, data_path_pattern: str | None) -> Path:
    if data_path_pattern:
        episode_chunk = int(episode_index) // 1000
        rel = data_path_pattern.format(episode_chunk=episode_chunk, episode_index=int(episode_index))
        return smooth_dir / rel
    episode_chunk = int(episode_index) // 1000
    return smooth_dir / "data" / f"chunk-{episode_chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _discover_episodes(smooth_dir: Path) -> list[EpisodeInfo]:
    smooth_dir = smooth_dir.expanduser().resolve()
    info_path = smooth_dir / "meta" / "info.json"
    episodes_path = smooth_dir / "meta" / "episodes.jsonl"
    data_path_pattern = None
    if info_path.exists():
        data_path_pattern = str(_load_json(info_path).get("data_path") or "")

    episodes: list[EpisodeInfo] = []
    if episodes_path.exists():
        for raw_line in episodes_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            item = json.loads(raw_line)
            episode_index = int(item["episode_index"])
            metadata = item.get("teleop_stack_metadata") if isinstance(item.get("teleop_stack_metadata"), dict) else {}
            rel_path = metadata.get("data_path") if isinstance(metadata, dict) else None
            path = smooth_dir / str(rel_path) if rel_path else _episode_path_from_index(smooth_dir, episode_index, data_path_pattern)
            tasks = item.get("tasks") if isinstance(item.get("tasks"), list) else []
            task = str(tasks[0]) if tasks else ""
            episodes.append(
                EpisodeInfo(
                    episode_index=episode_index,
                    length=int(item.get("length", 0)),
                    path=path.expanduser().resolve(),
                    task=task,
                )
            )

    if not episodes:
        for path in sorted((smooth_dir / "data").glob("chunk-*/episode_*.parquet")):
            stem = path.stem.removeprefix("episode_")
            try:
                episode_index = int(stem)
            except ValueError:
                continue
            episodes.append(EpisodeInfo(episode_index=episode_index, length=0, path=path.resolve(), task=""))

    episodes.sort(key=lambda item: item.episode_index)
    if not episodes:
        raise FileNotFoundError(f"No smooth episodes found under {smooth_dir}")
    return episodes


def _load_modality_ranges(smooth_dir: Path) -> dict[str, tuple[int, int]]:
    modality_path = smooth_dir / "meta" / "modality.json"
    if not modality_path.exists():
        return {
            "arm_joint_pos": (0, 7),
            "hand_joint_pos": (16, 26),
            "hand_joint_target": (9, 19),
        }
    payload = _load_json(modality_path)
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    action = payload.get("action") if isinstance(payload.get("action"), dict) else {}

    def get_range(group: dict[str, Any], key: str, default: tuple[int, int]) -> tuple[int, int]:
        item = group.get(key) if isinstance(group.get(key), dict) else {}
        return int(item.get("start", default[0])), int(item.get("end", default[1]))

    return {
        "arm_joint_pos": get_range(state, "arm_joint_pos", (0, 7)),
        "hand_joint_pos": get_range(state, "hand_joint_pos", (16, 26)),
        "hand_joint_target": get_range(action, "hand_joint_target", (9, 19)),
    }


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


def _load_episode_data(
    info: EpisodeInfo,
    *,
    smooth_dir: Path,
    hand_source: str,
) -> EpisodeData:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Reading smooth parquet files requires pyarrow. Install it in this environment, for example: "
            "python3 -m pip install pyarrow"
        ) from exc

    if not info.path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {info.path}")

    ranges = _load_modality_ranges(smooth_dir)
    columns = ["observation.state", "timestamp"]
    if hand_source == "action":
        columns.append("action")
    available_columns = set(pq.ParquetFile(info.path).schema_arrow.names)
    columns = [column for column in columns if column in available_columns]
    table = pq.read_table(info.path, columns=columns)
    state = _parquet_column_to_array(table, "observation.state")
    if state is None:
        raise ValueError(f"{info.path} does not contain observation.state")
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 2:
        state = state.reshape(state.shape[0], -1)

    arm_start, arm_end = ranges["arm_joint_pos"]
    hand_start, hand_end = ranges["hand_joint_pos"]
    arm_q = state[:, arm_start:arm_end].astype(np.float32, copy=False)
    if arm_q.shape[1] != 7:
        raise ValueError(f"Expected 7 arm joints in observation.state, got shape {arm_q.shape}")

    hand_q: np.ndarray | None = None
    if hand_source == "state":
        if state.shape[1] >= hand_end:
            hand_q = state[:, hand_start:hand_end].astype(np.float32, copy=False)
    else:
        action = _parquet_column_to_array(table, "action")
        if action is not None:
            action = np.asarray(action, dtype=np.float32)
            if action.ndim != 2:
                action = action.reshape(action.shape[0], -1)
            target_start, target_end = ranges["hand_joint_target"]
            if action.shape[1] >= target_end:
                hand_q = action[:, target_start:target_end].astype(np.float32, copy=False)
    if hand_q is not None and hand_q.shape[1] != len(HAND_JOINT_NAMES):
        print(f"[smooth] ignoring hand playback with unexpected shape {hand_q.shape}", flush=True)
        hand_q = None

    timestamps = _parquet_column_to_array(table, "timestamp")
    if timestamps is None:
        timestamps = np.arange(arm_q.shape[0], dtype=np.float32) / 10.0
    timestamps = np.asarray(timestamps, dtype=np.float32).reshape(-1)
    if timestamps.size != arm_q.shape[0]:
        timestamps = np.arange(arm_q.shape[0], dtype=np.float32) / 10.0

    return EpisodeData(
        episode_index=info.episode_index,
        arm_q=np.ascontiguousarray(arm_q),
        hand_q=None if hand_q is None else np.ascontiguousarray(hand_q),
        timestamps=np.ascontiguousarray(timestamps),
    )


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
    stop_flag: object,
) -> None:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Smooth Episode Playback")
    root.geometry("780x360")
    root.minsize(680, 320)

    selected_var = tk.StringVar(value=labels[int(selected_episode.value)] if labels else "")
    status_var = tk.StringVar(value="Paused")
    frame_var = tk.IntVar(value=0)
    speed_var = tk.DoubleVar(value=float(speed.value))
    loop_var = tk.BooleanVar(value=bool(loop.value))
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

    def toggle_play() -> None:
        playing.value = not bool(playing.value)
        refresh()

    def reset() -> None:
        selected_frame.value = 0
        current_frame.value = 0
        seek_counter.value += 1
        frame_var.set(0)

    def close() -> None:
        stop_flag.value = True
        root.destroy()
        root.quit()

    def on_frame(value: str) -> None:
        if internal_frame_update["active"]:
            return
        try:
            frame = int(float(value))
        except ValueError:
            return
        selected_frame.value = frame
        current_frame.value = frame
        seek_counter.value += 1

    def on_speed(value: str) -> None:
        try:
            speed.value = max(0.05, float(value))
        except ValueError:
            return

    def on_loop() -> None:
        loop.value = bool(loop_var.get())

    def refresh() -> None:
        internal_frame_update["active"] = True
        frame_var.set(int(current_frame.value))
        internal_frame_update["active"] = False
        play_button.configure(text="Pause" if bool(playing.value) else "Play")
        state = "Playing" if bool(playing.value) else "Paused"
        status_var.set(
            f"{state}  frame {int(current_frame.value)}/{max(length_for_selection() - 1, 0)}  "
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

    frame_row = ttk.Frame(root)
    frame_row.pack(fill=tk.X, padx=12, pady=12)
    ttk.Label(frame_row, text="Frame", width=10).pack(side=tk.LEFT)
    frame_scale = ttk.Scale(
        frame_row,
        from_=0,
        to=max(length_for_selection() - 1, 0),
        orient=tk.HORIZONTAL,
        variable=frame_var,
        command=on_frame,
    )
    frame_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

    speed_row = ttk.Frame(root)
    speed_row.pack(fill=tk.X, padx=12, pady=6)
    ttk.Label(speed_row, text="Speed", width=10).pack(side=tk.LEFT)
    speed_scale = ttk.Scale(speed_row, from_=0.05, to=4.0, orient=tk.HORIZONTAL, variable=speed_var, command=on_speed)
    speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Label(speed_row, textvariable=speed_var, width=8).pack(side=tk.LEFT, padx=(8, 0))

    buttons = ttk.Frame(root)
    buttons.pack(fill=tk.X, padx=12, pady=(14, 8))
    play_button = ttk.Button(buttons, text="Play", command=toggle_play)
    play_button.pack(side=tk.LEFT)
    ttk.Button(buttons, text="Reset", command=reset).pack(side=tk.LEFT, padx=8)
    ttk.Checkbutton(buttons, text="Loop", variable=loop_var, command=on_loop).pack(side=tk.LEFT)
    ttk.Button(buttons, text="Close", command=close).pack(side=tk.RIGHT)

    ttk.Label(root, textvariable=status_var).pack(fill=tk.X, padx=12, pady=(4, 12))

    refresh()
    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()


def _create_control_panel(episodes: list[EpisodeInfo], *, episode_index: int, start: bool, speed_value: float, loop_value: bool) -> dict[str, object]:
    selected_episode = multiprocessing.RawValue("i", int(episode_index))
    current_frame = multiprocessing.RawValue("i", 0)
    selected_frame = multiprocessing.RawValue("i", 0)
    load_counter = multiprocessing.RawValue("i", 0)
    seek_counter = multiprocessing.RawValue("i", 0)
    playing = multiprocessing.RawValue("b", bool(start))
    speed = multiprocessing.RawValue("d", float(speed_value))
    loop = multiprocessing.RawValue("b", bool(loop_value))
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


def _apply_arm_frame(arm: object, dofs: list[int], q: np.ndarray) -> None:
    values = np.asarray(q, dtype=np.float32).reshape(7)
    arm.set_dofs_position(values, dofs, zero_velocity=True)
    arm.control_dofs_position(values, dofs)


def _apply_hand_frame(assembly: dict[str, object], q: np.ndarray | None) -> None:
    if q is None:
        return
    linker_hand = assembly.get("linker_hand")
    if linker_hand is None:
        return
    joint_names = list(assembly.get("linker_hand_joint_names", ()))
    dofs = list(assembly.get("linker_hand_dofs", ()))
    if not joint_names or not dofs:
        return

    values_by_name = {name: float(value) for name, value in zip(HAND_JOINT_NAMES, q, strict=False)}
    mimic_by_name = assembly.get("linker_hand_mimic_by_name", {})
    if isinstance(mimic_by_name, dict):
        for mimic_name, spec in mimic_by_name.items():
            try:
                source_name, multiplier, offset = spec
            except (TypeError, ValueError):
                continue
            if str(source_name) in values_by_name:
                values_by_name[str(mimic_name)] = float(multiplier) * float(values_by_name[str(source_name)]) + float(offset)

    values = np.asarray([float(values_by_name.get(str(name), 0.0)) for name in joint_names], dtype=np.float32)
    linker_hand.set_dofs_position(values, dofs, zero_velocity=True)
    linker_hand.control_dofs_position(values, dofs)


def _apply_episode_frame(scene: object, episode: EpisodeData, frame_index: int, arm: object, arm_dofs: list[int], assembly: dict[str, object]) -> None:
    frame_index = int(np.clip(frame_index, 0, max(episode.length - 1, 0)))
    _apply_arm_frame(arm, arm_dofs, episode.arm_q[frame_index])
    hand_q = None if episode.hand_q is None else episode.hand_q[frame_index]
    _apply_hand_frame(assembly, hand_q)
    harness._step_scene_with_attached_parts(scene)  # noqa: SLF001


def _selected_episode_list_index(episodes: list[EpisodeInfo], episode_index: int) -> int:
    for idx, item in enumerate(episodes):
        if item.episode_index == int(episode_index):
            return idx
    if 0 <= int(episode_index) < len(episodes):
        return int(episode_index)
    return 0


def _create_scene(args: argparse.Namespace) -> object:
    print("[scene] creating add_scene_glb scene for smooth playback", flush=True)
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
        combined_urdf=args.combined_urdf,
        initial_base_pos=(0.0, 0.0, 0.0),
        initial_base_euler=(0.0, 0.0, 0.0),
        d455_rgb_gui=False,
        d405_camera_gui=False,
        linker_hand_collision=bool(args.linker_hand_collision),
    )
    return scene


def _smooth_video_path(smooth_dir: Path, episode_index: int, key: str) -> Path:
    return smooth_dir / "videos" / f"chunk-{episode_index // 1000:03d}" / key / f"episode_{episode_index:06d}.mp4"


def _read_video_frame(path: Path, frame_index: int, image_size: tuple[int, int]) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("Dumping camera comparisons requires cv2 in this environment.") from exc
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(frame_index, 0)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame={frame_index} from {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = image_size
    return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA).astype(np.uint8)


def _label_panel(image: np.ndarray, label: str) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("Dumping camera comparisons requires cv2 in this environment.") from exc
    panel = np.asarray(image, dtype=np.uint8).copy()
    bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1] - 1, 24), (0, 0, 0), -1)
    cv2.putText(bgr, str(label), (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _dump_camera_comparison(
    scene: object,
    *,
    smooth_dir: Path,
    episode_index: int,
    frame_index: int,
    image_size: tuple[int, int],
    output: Path,
) -> None:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("Dumping camera comparisons requires cv2 in this environment.") from exc
    sim_ego = harness._render_camera_rgb_model_input(  # noqa: SLF001
        getattr(scene, "d455_rgb_camera", None),
        image_size=image_size,
    )
    sim_wrist = harness._render_camera_rgb_model_input(  # noqa: SLF001
        getattr(scene, "right_d405_camera", None),
        image_size=image_size,
    )
    smooth_ego = _read_video_frame(
        _smooth_video_path(smooth_dir, episode_index, "observation.images.ego_view"),
        frame_index,
        image_size,
    )
    smooth_wrist = _read_video_frame(
        _smooth_video_path(smooth_dir, episode_index, "observation.images.wrist_view"),
        frame_index,
        image_size,
    )
    row1 = np.concatenate(
        (
            _label_panel(sim_ego, f"Genesis ego playback frame{frame_index}"),
            _label_panel(smooth_ego, f"smooth ego frame{frame_index}"),
        ),
        axis=1,
    )
    row2 = np.concatenate(
        (
            _label_panel(sim_wrist, f"Genesis wrist playback frame{frame_index}"),
            _label_panel(smooth_wrist, f"smooth wrist frame{frame_index}"),
        ),
        axis=1,
    )
    canvas = np.concatenate((row1, row2), axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"[smooth-camera-dump] output={output}", flush=True)
    for label, image in (
        ("genesis_ego", sim_ego),
        ("smooth_ego", smooth_ego),
        ("genesis_wrist", sim_wrist),
        ("smooth_wrist", smooth_wrist),
    ):
        arr = np.asarray(image, dtype=np.float32).reshape(-1, 3)
        print(
            f"[smooth-camera-dump] {label} mean_rgb={np.round(arr.mean(axis=0), 2).tolist()} "
            f"std_rgb={np.round(arr.std(axis=0), 2).tolist()}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Play one smooth dataset episode on the add_scene_glb scene and combined Nero/L10 URDF."
    )
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--smooth-dir", type=Path, default=DEFAULT_SMOOTH_DIR)
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index to select initially.")
    parser.add_argument("--start", action="store_true", help="Start playback immediately.")
    parser.add_argument("--fps", type=float, default=10.0, help="Playback FPS when timestamps are unavailable.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hand-source", choices=("state", "action"), default="action")
    parser.add_argument("--no-control-panel", action="store_true", help="Run without the Tk episode control window.")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--combined-urdf", type=Path, default=harness.DEFAULT_COMBINED_NERO_LINKER_URDF)
    parser.add_argument("--bottle-pos", type=_vec3, default=DEFAULT_BOTTLE_POS)
    parser.add_argument("--bottle-euler", type=_vec3, default=DEFAULT_BOTTLE_EULER)
    parser.add_argument("--bottle-proxy-json", type=Path, default=harness.DEFAULT_BOTTLE_PROXY_JSON)
    parser.add_argument("--show-bottle-proxy", action="store_true")
    parser.add_argument("--scene-support-collider-pos", type=_vec3, default=DEFAULT_SCENE_SUPPORT_COLLIDER_POS)
    parser.add_argument("--scene-support-collider-size", type=_vec3, default=DEFAULT_SCENE_SUPPORT_COLLIDER_SIZE)
    parser.add_argument("--scene-mesh-collision", action="store_true")
    parser.add_argument("--show-scene-support-collider", action="store_true")
    parser.add_argument("--linker-hand-collision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dump-camera-frame",
        type=int,
        default=None,
        help="Apply this smooth frame, dump Genesis ego/wrist cameras next to dataset videos, then optionally exit.",
    )
    parser.add_argument(
        "--dump-camera-image-size",
        type=_vec2_int,
        default=(180, 320),
        help="Camera dump model input size as height,width.",
    )
    parser.add_argument(
        "--dump-camera-output",
        type=Path,
        default=ROOT_DIR / "logs" / "smooth_playback_camera_dump.png",
    )
    parser.add_argument("--dump-camera-exit", action="store_true", help="Exit after --dump-camera-frame is written.")
    args = parser.parse_args()

    smooth_dir = args.smooth_dir.expanduser().resolve()
    episodes = _discover_episodes(smooth_dir)
    selected_list_index = _selected_episode_list_index(episodes, int(args.episode_index))
    episode = _load_episode_data(episodes[selected_list_index], smooth_dir=smooth_dir, hand_source=str(args.hand_source))
    panel = None
    try:
        if not bool(args.no_control_panel):
            panel = _create_control_panel(
                episodes,
                episode_index=selected_list_index,
                start=bool(args.start),
                speed_value=float(args.speed),
                loop_value=bool(args.loop),
            )

        scene = _create_scene(args)
        assembly = getattr(scene, "nero_assembly_info", None)
        if not isinstance(assembly, dict):
            raise RuntimeError("add_scene_glb scene did not create a Nero assembly")
        arm = assembly.get("right")
        if arm is None:
            raise RuntimeError("add_scene_glb scene does not contain a right Nero arm")
        arm_prefixes = assembly.get("arm_joint_prefixes", {})
        right_prefix = str(arm_prefixes.get("right", "")) if isinstance(arm_prefixes, dict) else ""
        arm_dofs = harness._arm_dofs(arm, joint_prefix=right_prefix)  # noqa: SLF001

        current_frame = 0
        last_load_counter = 0
        last_seek_counter = 0
        playing = bool(args.start)
        speed_value = float(args.speed)
        loop_value = bool(args.loop)
        next_frame_time = time.monotonic()
        base_dt = 1.0 / max(float(args.fps), 1.0e-6)
        _apply_episode_frame(scene, episode, current_frame, arm, arm_dofs, assembly)
        print(
            f"[smooth] loaded episode={episode.episode_index:06d} frames={episode.length} "
            f"hand_source={args.hand_source} path={episodes[selected_list_index].path}",
            flush=True,
        )
        if args.dump_camera_frame is not None:
            dump_frame = int(np.clip(int(args.dump_camera_frame), 0, max(episode.length - 1, 0)))
            _apply_episode_frame(scene, episode, dump_frame, arm, arm_dofs, assembly)
            _dump_camera_comparison(
                scene,
                smooth_dir=smooth_dir,
                episode_index=int(episode.episode_index),
                frame_index=dump_frame,
                image_size=tuple(args.dump_camera_image_size),
                output=args.dump_camera_output.expanduser().resolve(),
            )
            if bool(args.dump_camera_exit):
                print(f"[done] episode={episode.episode_index:06d} frame={dump_frame}", flush=True)
                return 0
        if panel is not None:
            print("[smooth] controls: use the Tk window to select episode, seek, play/pause, speed, loop", flush=True)

        while bool(args.no_viewer) or scene.viewer.is_alive():
            if panel is not None:
                if bool(panel["stop_flag"].value):
                    break
                panel_selected = int(panel["selected_episode"].value)
                panel_load_counter = int(panel["load_counter"].value)
                if panel_load_counter != last_load_counter:
                    selected_list_index = int(np.clip(panel_selected, 0, len(episodes) - 1))
                    episode = _load_episode_data(
                        episodes[selected_list_index],
                        smooth_dir=smooth_dir,
                        hand_source=str(args.hand_source),
                    )
                    current_frame = 0
                    panel["current_frame"].value = current_frame
                    last_load_counter = panel_load_counter
                    next_frame_time = time.monotonic()
                    _apply_episode_frame(scene, episode, current_frame, arm, arm_dofs, assembly)
                    print(
                        f"[smooth] loaded episode={episode.episode_index:06d} frames={episode.length} "
                        f"path={episodes[selected_list_index].path}",
                        flush=True,
                    )

                panel_seek_counter = int(panel["seek_counter"].value)
                if panel_seek_counter != last_seek_counter:
                    current_frame = int(np.clip(int(panel["selected_frame"].value), 0, max(episode.length - 1, 0)))
                    panel["current_frame"].value = current_frame
                    last_seek_counter = panel_seek_counter
                    next_frame_time = time.monotonic()
                    _apply_episode_frame(scene, episode, current_frame, arm, arm_dofs, assembly)

                playing = bool(panel["playing"].value)
                speed_value = float(panel["speed"].value)
                loop_value = bool(panel["loop"].value)

            if playing and episode.length > 0:
                now = time.monotonic()
                if now >= next_frame_time:
                    _apply_episode_frame(scene, episode, current_frame, arm, arm_dofs, assembly)
                    if panel is not None:
                        panel["current_frame"].value = current_frame
                    current_frame += 1
                    if current_frame >= episode.length:
                        if loop_value:
                            current_frame = 0
                        else:
                            current_frame = episode.length - 1
                            playing = False
                            if panel is not None:
                                panel["playing"].value = False
                    if episode.timestamps.size > current_frame > 0:
                        raw_dt = float(episode.timestamps[current_frame] - episode.timestamps[current_frame - 1])
                        dt = raw_dt if math.isfinite(raw_dt) and raw_dt > 1.0e-5 else base_dt
                    else:
                        dt = base_dt
                    next_frame_time = now + dt / max(speed_value, 1.0e-6)
                else:
                    scene.visualizer.update(force=True)
            else:
                scene.visualizer.update(force=True)
            time.sleep(1.0 / 120.0)
    finally:
        _shutdown_control_panel(panel)

    print(f"[done] episode={episode.episode_index:06d} frame={current_frame}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
