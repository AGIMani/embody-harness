from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import add_scene_glb as harness  # noqa: E402
from tools import run_add_scene_smooth_playback as playback  # noqa: E402


DEFAULT_SMOOTH_DIR = ROOT_DIR.parent / "Isaac-GR00T" / "outputs" / "IsaacLab" / "nero" / "mission2" / "smooth"
DEFAULT_OUTPUT_JSON = ROOT_DIR / "assets" / "d405_smooth_wrist_mount_debug.json"
DEFAULT_OUTPUT_IMAGE = ROOT_DIR / "logs" / "groot_smooth_image_injection" / "d405_smooth_wrist_calibration.png"

COARSE_TRANSLATION_RANGE_M = 0.20
COARSE_TRANSLATION_SLIDER_STEP_M = 0.01
FINE_TRANSLATION_STEP_M = 0.001
COARSE_ROTATION_RANGE_DEG = 180.0
FINE_ROTATION_STEP_DEG = 0.1


def _format_tuple(values: tuple[float, ...], digits: int = 6) -> str:
    return "(" + ", ".join(f"{float(v):.{digits}f}" for v in values) + ")"


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


def _payload_from_values(values: tuple[float, ...]) -> dict[str, object]:
    offset_xyz = tuple(float(v) for v in values[:3])
    euler_deg = tuple(float(v) for v in values[3:])
    quat_wxyz = harness._quat_wxyz_from_rotation(harness._rotation_from_euler_deg(euler_deg))  # noqa: SLF001
    return {
        "description": "D405 body pose relative to right_connector in dual_nero_linker_l10_combined.urdf.",
        "connector_side": "right",
        "connector_entity": "right_connector",
        "offset_xyz_in_connector_frame_m": [float(v) for v in offset_xyz],
        "euler_xyz_in_connector_frame_deg": [float(v) for v in euler_deg],
        "quat_wxyz_in_connector_frame": [float(v) for v in quat_wxyz],
        "python_constants": {
            "RIGHT_D405_CONNECTOR_REL_POS_M": [float(v) for v in offset_xyz],
            "RIGHT_D405_CONNECTOR_REL_EULER_DEG": [float(v) for v in euler_deg],
        },
    }


def _print_payload(values: tuple[float, ...]) -> None:
    payload = _payload_from_values(values)
    offset_xyz = tuple(float(v) for v in payload["offset_xyz_in_connector_frame_m"])
    euler_deg = tuple(float(v) for v in payload["euler_xyz_in_connector_frame_deg"])
    quat_wxyz = tuple(float(v) for v in payload["quat_wxyz_in_connector_frame"])
    print("[d405-smooth-wrist-debug] relative_to_right_connector", flush=True)
    print(
        f"  offset_xyz={_format_tuple(offset_xyz)} "
        f"euler_deg={_format_tuple(euler_deg, 3)} "
        f"quat_wxyz={_format_tuple(quat_wxyz)}",
        flush=True,
    )
    print(
        "  python_constants:\n"
        f"    RIGHT_D405_CONNECTOR_REL_POS_M = {_format_tuple(offset_xyz)}\n"
        f"    RIGHT_D405_CONNECTOR_REL_EULER_DEG = {_format_tuple(euler_deg, 3)}",
        flush=True,
    )


def _write_payload(path: Path, values: tuple[float, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_payload_from_values(values), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[d405-smooth-wrist-debug] saved_json={path}", flush=True)


def _apply_d405_values(scene: object, values: tuple[float, ...]) -> None:
    assembly = getattr(scene, "nero_assembly_info", None)
    if not isinstance(assembly, dict):
        raise RuntimeError("scene does not contain nero_assembly_info")
    assembly["d405_connector_rel_pos"] = tuple(float(v) for v in values[:3])
    assembly["d405_connector_rel_euler"] = tuple(float(v) for v in values[3:])
    harness._mount_d405_to_right_connector(assembly)  # noqa: SLF001


def _render_wrist(scene: object, image_size: tuple[int, int]) -> np.ndarray:
    return harness._render_camera_rgb_model_input(  # noqa: SLF001
        getattr(scene, "right_d405_camera", None),
        image_size=image_size,
    )


def _label_panel(image: np.ndarray, label: str) -> np.ndarray:
    import cv2

    panel = np.asarray(image, dtype=np.uint8).copy()
    bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1] - 1, 24), (0, 0, 0), -1)
    cv2.putText(bgr, str(label), (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _comparison_image(
    *,
    scene: object,
    smooth_wrist: np.ndarray,
    image_size: tuple[int, int],
    frame_index: int,
    values: tuple[float, ...],
) -> np.ndarray:
    sim_wrist = _render_wrist(scene, image_size)
    title = "Genesis D405 " + _format_tuple(values[:3], 4) + " " + _format_tuple(values[3:], 1)
    left = _label_panel(sim_wrist, title)
    right = _label_panel(smooth_wrist, f"smooth wrist frame{int(frame_index)}")
    return np.concatenate((left, right), axis=1)


def _save_comparison(path: Path, image: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    print(f"[d405-smooth-wrist-debug] saved_image={path}", flush=True)


def _show_preview(window_name: str, image: np.ndarray, *, scale: int) -> bool:
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        import cv2
    except Exception:
        return False
    frame = np.asarray(image, dtype=np.uint8)
    if int(scale) > 1:
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (width * int(scale), height * int(scale)), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(window_name, frame[..., ::-1])
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27)


def _panel_main(initial_values, values, print_counter, save_counter, dump_counter, reset_counter, stop_flag) -> None:
    import tkinter as tk
    from tkinter import ttk

    specs = (
        ("x", -COARSE_TRANSLATION_RANGE_M, COARSE_TRANSLATION_RANGE_M, COARSE_TRANSLATION_SLIDER_STEP_M, FINE_TRANSLATION_STEP_M, "m"),
        ("y", -COARSE_TRANSLATION_RANGE_M, COARSE_TRANSLATION_RANGE_M, COARSE_TRANSLATION_SLIDER_STEP_M, FINE_TRANSLATION_STEP_M, "m"),
        ("z", -COARSE_TRANSLATION_RANGE_M, COARSE_TRANSLATION_RANGE_M, COARSE_TRANSLATION_SLIDER_STEP_M, FINE_TRANSLATION_STEP_M, "m"),
        ("roll", -COARSE_ROTATION_RANGE_DEG, COARSE_ROTATION_RANGE_DEG, FINE_ROTATION_STEP_DEG, FINE_ROTATION_STEP_DEG, "deg"),
        ("pitch", -COARSE_ROTATION_RANGE_DEG, COARSE_ROTATION_RANGE_DEG, FINE_ROTATION_STEP_DEG, FINE_ROTATION_STEP_DEG, "deg"),
        ("yaw", -COARSE_ROTATION_RANGE_DEG, COARSE_ROTATION_RANGE_DEG, FINE_ROTATION_STEP_DEG, FINE_ROTATION_STEP_DEG, "deg"),
    )
    programmatic_slider_update = [False]

    def set_value(idx: int, value: float | str, *, snap_to_slider: bool = False) -> None:
        lower = float(specs[idx][1])
        upper = float(specs[idx][2])
        slider_step = float(specs[idx][3])
        clamped = max(lower, min(upper, float(value)))
        if snap_to_slider and not programmatic_slider_update[0]:
            clamped = round(clamped / slider_step) * slider_step
            clamped = max(lower, min(upper, clamped))
            if abs(float(sliders[idx].get()) - clamped) > 1.0e-9:
                sliders[idx].set(clamped)
        values[idx] = clamped
        value_labels[idx].config(text=f"{clamped: .5f}")

    def step_value(idx: int, direction: int) -> None:
        current = float(sliders[idx].get())
        step = float(specs[idx][4])
        lower = float(specs[idx][1])
        upper = float(specs[idx][2])
        next_value = max(lower, min(upper, current + float(direction) * step))
        programmatic_slider_update[0] = True
        try:
            sliders[idx].set(next_value)
        finally:
            programmatic_slider_update[0] = False
        set_value(idx, next_value, snap_to_slider=False)

    def reset() -> None:
        for idx, slider in enumerate(sliders):
            slider.set(float(initial_values[idx]))
            set_value(idx, float(initial_values[idx]))
        reset_counter.value += 1

    def close() -> None:
        stop_flag.value = True
        root.destroy()
        root.quit()

    root = tk.Tk()
    root.title("D405 Smooth Wrist Calibration")
    root.geometry("920x450")
    root.minsize(780, 380)

    title = ttk.Label(root, text="D405 pose relative to right_connector; match Genesis wrist to smooth wrist", font=("Arial", 12, "bold"))
    title.pack(fill=tk.X, padx=12, pady=(12, 4))

    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

    sliders = []
    value_labels = []
    for idx, (label, lower, upper, _slider_step, _button_step, unit) in enumerate(specs):
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text=f"{label} ({unit})", width=12).pack(side=tk.LEFT)
        slider = ttk.Scale(
            row,
            from_=float(lower),
            to=float(upper),
            orient=tk.HORIZONTAL,
            command=lambda value, i=idx: set_value(i, value, snap_to_slider=True),
        )
        slider.set(float(initial_values[idx]))
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(row, text="-", width=3, command=lambda i=idx: step_value(i, -1)).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(row, text="+", width=3, command=lambda i=idx: step_value(i, 1)).pack(side=tk.LEFT)
        value_label = ttk.Label(row, text=f"{float(initial_values[idx]): .5f}", width=12)
        value_label.pack(side=tk.RIGHT, padx=(8, 0))
        sliders.append(slider)
        value_labels.append(value_label)

    buttons = ttk.Frame(root)
    buttons.pack(fill=tk.X, padx=12, pady=(4, 12))
    ttk.Button(buttons, text="Print", command=lambda: setattr(print_counter, "value", print_counter.value + 1)).pack(side=tk.LEFT)
    ttk.Button(buttons, text="Save JSON", command=lambda: setattr(save_counter, "value", save_counter.value + 1)).pack(side=tk.LEFT, padx=8)
    ttk.Button(buttons, text="Dump Image", command=lambda: setattr(dump_counter, "value", dump_counter.value + 1)).pack(side=tk.LEFT)
    ttk.Button(buttons, text="Reset", command=reset).pack(side=tk.LEFT, padx=8)
    ttk.Button(buttons, text="Close", command=close).pack(side=tk.RIGHT)

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()


def _create_panel(initial_values: tuple[float, ...]) -> dict[str, object]:
    values = multiprocessing.RawArray("d", initial_values)
    print_counter = multiprocessing.RawValue("i", 0)
    save_counter = multiprocessing.RawValue("i", 0)
    dump_counter = multiprocessing.RawValue("i", 0)
    reset_counter = multiprocessing.RawValue("i", 0)
    stop_flag = multiprocessing.RawValue("b", False)
    process = multiprocessing.Process(
        target=_panel_main,
        args=(initial_values, values, print_counter, save_counter, dump_counter, reset_counter, stop_flag),
        daemon=True,
    )
    process.start()
    return {
        "values": values,
        "print_counter": print_counter,
        "save_counter": save_counter,
        "dump_counter": dump_counter,
        "reset_counter": reset_counter,
        "stop_flag": stop_flag,
        "process": process,
    }


def _shutdown_panel(panel: dict[str, object] | None) -> None:
    if not panel:
        return
    panel["stop_flag"].value = True
    process = panel["process"]
    if process.is_alive():
        process.join(timeout=1.0)


def _read_values(panel: dict[str, object]) -> tuple[float, ...]:
    values = panel["values"]
    return tuple(float(values[idx]) for idx in range(6))


def _create_scene(args: argparse.Namespace) -> object:
    return harness.create_scene(
        show_viewer=not bool(args.no_viewer),
        backend=str(args.backend),
        pos=harness.DEFAULT_SCENE_WORLD_POS,
        euler=harness.DEFAULT_SCENE_WORLD_EULER,
        collision=False,
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
        d405_connector_rel_pos=args.initial_offset,
        d405_connector_rel_euler=args.initial_euler,
        linker_hand_collision=bool(args.linker_hand_collision),
    )[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the right D405 wrist camera against smooth dataset wrist_view frames."
    )
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--smooth-dir", type=Path, default=DEFAULT_SMOOTH_DIR)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=56)
    parser.add_argument("--hand-source", choices=("state", "action"), default="action")
    parser.add_argument("--image-size", type=_vec2_int, default=(180, 320), help="height,width")
    parser.add_argument("--initial-offset", type=_vec3, default=harness.RIGHT_D405_CONNECTOR_REL_POS_M)
    parser.add_argument("--initial-euler", type=_vec3, default=harness.RIGHT_D405_CONNECTOR_REL_EULER_DEG)
    parser.add_argument("--combined-urdf", type=Path, default=harness.DEFAULT_COMBINED_NERO_LINKER_URDF)
    parser.add_argument("--bottle-pos", type=_vec3, default=playback.DEFAULT_BOTTLE_POS)
    parser.add_argument("--bottle-euler", type=_vec3, default=playback.DEFAULT_BOTTLE_EULER)
    parser.add_argument("--bottle-proxy-json", type=Path, default=harness.DEFAULT_BOTTLE_PROXY_JSON)
    parser.add_argument("--show-bottle-proxy", action="store_true")
    parser.add_argument("--scene-support-collider-pos", type=_vec3, default=playback.DEFAULT_SCENE_SUPPORT_COLLIDER_POS)
    parser.add_argument("--scene-support-collider-size", type=_vec3, default=playback.DEFAULT_SCENE_SUPPORT_COLLIDER_SIZE)
    parser.add_argument("--show-scene-support-collider", action="store_true")
    parser.add_argument("--linker-hand-collision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-image", type=Path, default=DEFAULT_OUTPUT_IMAGE)
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview-scale", type=int, default=2)
    parser.add_argument("--no-panel", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--dump-once", action="store_true", help="Dump one comparison image and exit.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    smooth_dir = args.smooth_dir.expanduser().resolve()
    episodes = playback._discover_episodes(smooth_dir)  # noqa: SLF001
    selected_index = playback._selected_episode_list_index(episodes, int(args.episode_index))  # noqa: SLF001
    episode = playback._load_episode_data(  # noqa: SLF001
        episodes[selected_index],
        smooth_dir=smooth_dir,
        hand_source=str(args.hand_source),
    )
    frame_index = int(np.clip(int(args.frame_index), 0, max(episode.length - 1, 0)))
    smooth_wrist = playback._read_video_frame(  # noqa: SLF001
        playback._smooth_video_path(smooth_dir, int(episode.episode_index), "observation.images.wrist_view"),  # noqa: SLF001
        frame_index,
        tuple(args.image_size),
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
    playback._apply_episode_frame(scene, episode, frame_index, arm, arm_dofs, assembly)  # noqa: SLF001

    initial_values = tuple(float(v) for v in (*args.initial_offset, *args.initial_euler))
    _apply_d405_values(scene, initial_values)
    print(
        f"[d405-smooth-wrist-debug] smooth_episode={episode.episode_index:06d} frame={frame_index} "
        f"image_size={tuple(args.image_size)}",
        flush=True,
    )
    _print_payload(initial_values)

    comparison = _comparison_image(
        scene=scene,
        smooth_wrist=smooth_wrist,
        image_size=tuple(args.image_size),
        frame_index=frame_index,
        values=initial_values,
    )
    if bool(args.dump_once):
        _save_comparison(args.output_image.expanduser().resolve(), comparison)
        return 0

    panel = None if bool(args.no_panel) else _create_panel(initial_values)
    last_values: tuple[float, ...] | None = None
    last_print_counter = 0
    last_save_counter = 0
    last_dump_counter = 0
    preview_enabled = bool(args.preview)
    try:
        while bool(args.no_viewer) or scene.viewer.is_alive():
            if panel is not None and panel["stop_flag"].value:
                break
            values = initial_values if panel is None else _read_values(panel)
            if values != last_values:
                _apply_d405_values(scene, values)
                last_values = values
            comparison = _comparison_image(
                scene=scene,
                smooth_wrist=smooth_wrist,
                image_size=tuple(args.image_size),
                frame_index=frame_index,
                values=values,
            )
            if panel is not None:
                if int(panel["print_counter"].value) != last_print_counter:
                    last_print_counter = int(panel["print_counter"].value)
                    _print_payload(values)
                if int(panel["save_counter"].value) != last_save_counter:
                    last_save_counter = int(panel["save_counter"].value)
                    _write_payload(args.output_json.expanduser().resolve(), values)
                if int(panel["dump_counter"].value) != last_dump_counter:
                    last_dump_counter = int(panel["dump_counter"].value)
                    _save_comparison(args.output_image.expanduser().resolve(), comparison)
            if preview_enabled:
                preview_enabled = _show_preview("D405 Genesis wrist vs smooth wrist", comparison, scale=int(args.preview_scale))
            if not bool(args.no_viewer):
                scene.visualizer.update(force=True)
            time.sleep(1.0 / 30.0)
    finally:
        _shutdown_panel(panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
