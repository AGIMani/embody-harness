from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

import genesis as gs
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import add_scene_glb as harness


DEFAULT_D455_JSON = ROOT_DIR / "assets" / "d455json.json"
DEFAULT_OUTPUT = ROOT_DIR / "assets" / "d455_base_mount_debug.json"

DEFAULT_D455_REL_POS_M = (-0.327778, 0.252000, 1.288889)
DEFAULT_D455_REL_EULER_DEG = (180.0, 140.0, 0.0)

COARSE_TRANSLATION_RANGE_M = 1.5
FINE_TRANSLATION_STEP_M = 0.001
COARSE_ROTATION_RANGE_DEG = 180.0
FINE_ROTATION_STEP_DEG = 0.1


def _format_tuple(values: tuple[float, ...], digits: int = 6) -> str:
    return "(" + ", ".join(f"{float(v):.{digits}f}" for v in values) + ")"


def _load_d455_body_size(path: Path) -> tuple[float, float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        values = data["body"]["body_size_m_xyz"]
    except KeyError as exc:
        raise KeyError(f"{path} missing body.body_size_m_xyz") from exc
    if len(values) != 3:
        raise ValueError(f"{path} body.body_size_m_xyz must contain three numbers")
    return tuple(float(v) for v in values)  # type: ignore[return-value]


def _pose_world_from_base_relative(
    *,
    base_pos: np.ndarray,
    base_rotation: np.ndarray,
    rel_pos: tuple[float, float, float],
    rel_euler: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    rel_rotation = harness._rotation_from_euler_deg(rel_euler)
    world_pos = np.asarray(base_pos, dtype=np.float64) + np.asarray(base_rotation, dtype=np.float64) @ np.asarray(
        rel_pos, dtype=np.float64
    )
    world_rotation = np.asarray(base_rotation, dtype=np.float64) @ rel_rotation
    return world_pos, world_rotation


def _apply_d455_pose(
    *,
    body: object,
    front_marker: object,
    base_pos: np.ndarray,
    base_rotation: np.ndarray,
    body_size: tuple[float, float, float],
    values: tuple[float, ...],
) -> None:
    d455_pos, d455_rotation = _pose_world_from_base_relative(
        base_pos=base_pos,
        base_rotation=base_rotation,
        rel_pos=values[:3],
        rel_euler=values[3:],
    )
    harness._set_entity_pose(body, d455_pos, d455_rotation)

    marker_local_offset = np.asarray((body_size[0] * 0.5 + 0.003, 0.0, 0.0), dtype=np.float64)
    marker_world_pos = d455_pos + d455_rotation @ marker_local_offset
    harness._set_entity_pose(front_marker, marker_world_pos, d455_rotation)


def _payload_from_values(
    *,
    values: tuple[float, ...],
    d455_json: Path,
    body_size: tuple[float, float, float],
) -> dict[str, object]:
    rel_pos = values[:3]
    rel_euler = values[3:]
    rel_quat = harness._quat_wxyz_from_rotation(harness._rotation_from_euler_deg(rel_euler))
    return {
        "description": "D455 body pose relative to the loaded base.STL entity frame.",
        "d455_json": str(d455_json),
        "body_size_m_xyz": [float(v) for v in body_size],
        "pos_in_base_frame_m": [float(v) for v in rel_pos],
        "euler_xyz_in_base_frame_deg": [float(v) for v in rel_euler],
        "quat_wxyz_in_base_frame": [float(v) for v in rel_quat],
    }


def _print_payload(payload: dict[str, object]) -> None:
    pos = tuple(float(v) for v in payload["pos_in_base_frame_m"])
    euler = tuple(float(v) for v in payload["euler_xyz_in_base_frame_deg"])
    quat = tuple(float(v) for v in payload["quat_wxyz_in_base_frame"])
    print("[d455-base-debug] relative_to_base_stl", flush=True)
    print(
        f"  pos={_format_tuple(pos)} "
        f"euler_deg={_format_tuple(euler, 3)} "
        f"quat_wxyz={_format_tuple(quat)}",
        flush=True,
    )
    print(
        "  python_constants:\n"
        f"    D455_BASE_REL_POS_M = {_format_tuple(pos)}\n"
        f"    D455_BASE_REL_EULER_DEG = {_format_tuple(euler, 3)}\n"
        f"    D455_BASE_REL_QUAT_WXYZ = {_format_tuple(quat)}",
        flush=True,
    )


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[d455-base-debug] saved {path}", flush=True)


def _panel_main(initial_values, values, print_counter, save_counter, reset_counter, stop_flag) -> None:
    import tkinter as tk
    from tkinter import ttk

    specs = (
        ("x", -COARSE_TRANSLATION_RANGE_M, COARSE_TRANSLATION_RANGE_M, FINE_TRANSLATION_STEP_M, "m"),
        ("y", -COARSE_TRANSLATION_RANGE_M, COARSE_TRANSLATION_RANGE_M, FINE_TRANSLATION_STEP_M, "m"),
        ("z", -COARSE_TRANSLATION_RANGE_M, COARSE_TRANSLATION_RANGE_M, FINE_TRANSLATION_STEP_M, "m"),
        ("roll", -COARSE_ROTATION_RANGE_DEG, COARSE_ROTATION_RANGE_DEG, FINE_ROTATION_STEP_DEG, "deg"),
        ("pitch", -COARSE_ROTATION_RANGE_DEG, COARSE_ROTATION_RANGE_DEG, FINE_ROTATION_STEP_DEG, "deg"),
        ("yaw", -COARSE_ROTATION_RANGE_DEG, COARSE_ROTATION_RANGE_DEG, FINE_ROTATION_STEP_DEG, "deg"),
    )

    def set_value(idx: int, value: float | str) -> None:
        lower = float(specs[idx][1])
        upper = float(specs[idx][2])
        clamped = max(lower, min(upper, float(value)))
        values[idx] = clamped
        value_labels[idx].config(text=f"{clamped: .5f}")

    def step_value(idx: int, direction: int) -> None:
        current = float(sliders[idx].get())
        step = float(specs[idx][3])
        lower = float(specs[idx][1])
        upper = float(specs[idx][2])
        next_value = max(lower, min(upper, current + float(direction) * step))
        sliders[idx].set(next_value)
        set_value(idx, next_value)

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
    root.title("D455 Base Mount Calibration")
    root.geometry("860x430")
    root.minsize(760, 360)

    title = ttk.Label(root, text="D455 pose relative to base.STL", font=("Arial", 12, "bold"))
    title.pack(fill=tk.X, padx=12, pady=(12, 4))

    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

    sliders = []
    value_labels = []
    for idx, (label, lower, upper, step, unit) in enumerate(specs):
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text=f"{label} ({unit})", width=12).pack(side=tk.LEFT)
        slider = ttk.Scale(
            row,
            from_=float(lower),
            to=float(upper),
            orient=tk.HORIZONTAL,
            command=lambda value, i=idx: set_value(i, value),
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
    ttk.Button(buttons, text="Print Pose", command=lambda: setattr(print_counter, "value", print_counter.value + 1)).pack(side=tk.LEFT)
    ttk.Button(buttons, text="Save JSON", command=lambda: setattr(save_counter, "value", save_counter.value + 1)).pack(side=tk.LEFT, padx=8)
    ttk.Button(buttons, text="Reset", command=reset).pack(side=tk.LEFT)
    ttk.Button(buttons, text="Close", command=close).pack(side=tk.RIGHT)

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()


def _create_panel(initial_values: tuple[float, ...]) -> dict[str, object]:
    values = multiprocessing.RawArray("d", initial_values)
    print_counter = multiprocessing.RawValue("i", 0)
    save_counter = multiprocessing.RawValue("i", 0)
    reset_counter = multiprocessing.RawValue("i", 0)
    stop_flag = multiprocessing.RawValue("b", False)
    process = multiprocessing.Process(
        target=_panel_main,
        args=(initial_values, values, print_counter, save_counter, reset_counter, stop_flag),
        daemon=True,
    )
    process.start()
    return {
        "values": values,
        "print_counter": print_counter,
        "save_counter": save_counter,
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


def _vec3(value: str) -> tuple[float, float, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers, e.g. 0,0,0")
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune a D455 body pose relative to base.STL.")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--d455-json", type=Path, default=DEFAULT_D455_JSON)
    parser.add_argument("--initial-pos", type=_vec3, default=DEFAULT_D455_REL_POS_M)
    parser.add_argument("--initial-euler", type=_vec3, default=DEFAULT_D455_REL_EULER_DEG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-mesh", type=Path, default=harness.DEFAULT_BASE_MESH)
    parser.add_argument("--nero-urdf", type=Path, default=harness.DEFAULT_NERO_URDF)
    parser.add_argument("--package-root", type=Path, default=harness.DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--base-scale", type=float, default=harness.DEFAULT_BASE_SCALE)
    parser.add_argument("--base-pos", type=harness._vec3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--base-euler", type=harness._vec3, default=(0.0, 0.0, 0.0))
    parser.add_argument(
        "--use-base-foot-anchor",
        action="store_true",
        help="Use --base-foot-center-mm as an anchor at world origin instead of --base-pos.",
    )
    parser.add_argument("--base-foot-center-mm", type=harness._vec3, default=harness.DEFAULT_BASE_FOOT_CENTER_MM)
    parser.add_argument("--assembly-origin", type=harness._vec3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--connector-mesh", type=Path, default=harness.DEFAULT_CONNECTOR_MESH)
    parser.add_argument("--connector-scale", type=float, default=harness.DEFAULT_CONNECTOR_SCALE)
    parser.add_argument("--show-connectors", action="store_true", help="Show EEF connectors while tuning D455.")
    parser.add_argument("--no-revo2-flange", action="store_true")
    parser.add_argument("--no-viewer", action="store_true", help="Build once and print the initial D455 pose.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    d455_json = args.d455_json.expanduser().resolve()
    if not d455_json.exists():
        raise FileNotFoundError(f"D455 JSON not found: {d455_json}")
    body_size = _load_d455_body_size(d455_json)
    initial_values = tuple(float(v) for v in (*args.initial_pos, *args.initial_euler))
    base_world_pos = (
        harness._pose_from_local_anchor(
            tuple(float(v) for v in args.base_foot_center_mm),
            tuple(float(v) for v in args.base_euler),
            float(args.base_scale),
            tuple(float(v) for v in args.assembly_origin),
        )
        if args.use_base_foot_anchor
        else tuple(float(v) for v in args.base_pos)
    )
    base_world_euler = tuple(float(v) for v in args.base_euler)

    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.4, -2.0, 1.2),
            camera_lookat=(0.0, 0.0, 0.45),
            camera_fov=35,
            res=(1280, 720),
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(dt=0.01, enable_self_collision=False, enable_adjacent_collision=False),
        show_viewer=not args.no_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    assembly = harness._add_dual_nero_arm_assembly(
        scene,
        base_mesh=args.base_mesh,
        nero_urdf=args.nero_urdf,
        package_root=args.package_root,
        linker_hand_urdf=None,
        connector_mesh=args.connector_mesh if args.show_connectors else None,
        connector_scale=float(args.connector_scale),
        origin=args.assembly_origin,
        base_scale=float(args.base_scale),
        base_euler=tuple(float(v) for v in args.base_euler),
        base_foot_center_mm=tuple(float(v) for v in args.base_foot_center_mm),
        add_revo2_flange=not args.no_revo2_flange,
    )

    d455_body = scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0.0, 0.0),
            size=body_size,
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Plastic(color=(0.08, 0.08, 0.08, 1.0), roughness=0.55),
        name="d455_body_debug",
    )
    front_marker = scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0.0, 0.0),
            size=(0.003, body_size[1] * 0.75, body_size[2] * 0.55),
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Plastic(color=(0.05, 0.55, 1.0, 1.0), roughness=0.35),
        name="d455_front_marker_debug",
    )

    scene.build()
    harness._apply_base_world_pose(assembly, base_world_pos, base_world_euler)
    scene.step()
    if args.show_connectors:
        left_arm = assembly["left"]
        right_arm = assembly["right"]
        harness._mount_connectors_to_arms(assembly, left_arm, right_arm)

    base = assembly["base"]
    base_pos = harness._tensor_to_np(base.get_pos()).reshape(3).astype(np.float64)
    base_quat = harness._tensor_to_np(base.get_quat()).reshape(4).astype(np.float64)
    base_rotation = harness._rotation_from_quat_wxyz(base_quat)
    _apply_d455_pose(
        body=d455_body,
        front_marker=front_marker,
        base_pos=base_pos,
        base_rotation=base_rotation,
        body_size=body_size,
        values=initial_values,
    )

    if args.no_viewer:
        _print_payload(_payload_from_values(values=initial_values, d455_json=d455_json, body_size=body_size))
        return

    panel = _create_panel(initial_values)
    last_values: tuple[float, ...] | None = initial_values
    last_print_counter = 0
    last_save_counter = 0

    try:
        while scene.viewer.is_alive():
            if panel["stop_flag"].value:
                break
            values = _read_values(panel)
            if values != last_values:
                _apply_d455_pose(
                    body=d455_body,
                    front_marker=front_marker,
                    base_pos=base_pos,
                    base_rotation=base_rotation,
                    body_size=body_size,
                    values=values,
                )
                last_values = values
            if int(panel["print_counter"].value) != last_print_counter:
                last_print_counter = int(panel["print_counter"].value)
                _print_payload(_payload_from_values(values=values, d455_json=d455_json, body_size=body_size))
            if int(panel["save_counter"].value) != last_save_counter:
                last_save_counter = int(panel["save_counter"].value)
                payload = _payload_from_values(values=values, d455_json=d455_json, body_size=body_size)
                _write_payload(args.output.expanduser().resolve(), payload)
            scene.visualizer.update(force=True)
            time.sleep(1.0 / 60.0)
    finally:
        _shutdown_panel(panel)


if __name__ == "__main__":
    main()
