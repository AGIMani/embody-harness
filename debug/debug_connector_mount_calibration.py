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


DEFAULT_CONNECTOR_MESH = ROOT_DIR / "assets" / "connector.STL"
DEFAULT_OUTPUT = ROOT_DIR / "assets" / "connector_mount_debug.json"

DEFAULT_SIDE = "left"

RIGHT_CONNECTOR_MOUNT_OFFSET_XYZ = (0.022963, 0.089630, 0.037778)
RIGHT_CONNECTOR_MOUNT_EULER_DEG = (-90.0, 0.0, 180.0)

LEFT_CONNECTOR_MOUNT_OFFSET_XYZ = (-0.021481, -0.088889, 0.037778)
LEFT_CONNECTOR_MOUNT_EULER_DEG = (-90.0, 0.0, 0.0)

COARSE_TRANSLATION_RANGE_M = 0.20
FINE_TRANSLATION_STEP_M = 0.001
COARSE_ROTATION_RANGE_DEG = 180.0
FINE_ROTATION_STEP_DEG = 0.1


def _format_tuple(values: tuple[float, ...], digits: int = 6) -> str:
    return "(" + ", ".join(f"{float(v):.{digits}f}" for v in values) + ")"


def _payload_from_values(
    *,
    values: tuple[float, ...],
    connector_mesh: Path,
    connector_scale: float,
    side: str,
    eef_link: str,
) -> dict[str, object]:
    offset_xyz = values[:3]
    euler_deg = values[3:]
    quat_wxyz = harness._quat_wxyz_from_rotation(harness._rotation_from_euler_deg(euler_deg))
    return {
        "description": "connector.STL pose relative to the selected Nero end-effector link frame.",
        "connector_mesh": str(connector_mesh),
        "connector_scale": float(connector_scale),
        "arm_side": side,
        "eef_link": eef_link,
        "offset_xyz_in_eef_frame_m": [float(v) for v in offset_xyz],
        "euler_xyz_in_eef_frame_deg": [float(v) for v in euler_deg],
        "quat_wxyz_in_eef_frame": [float(v) for v in quat_wxyz],
    }


def _print_payload(payload: dict[str, object]) -> None:
    offset_xyz = tuple(float(v) for v in payload["offset_xyz_in_eef_frame_m"])
    euler_deg = tuple(float(v) for v in payload["euler_xyz_in_eef_frame_deg"])
    quat_wxyz = tuple(float(v) for v in payload["quat_wxyz_in_eef_frame"])
    print("[connector-debug] relative_to_eef", flush=True)
    print(
        f"  side={payload['arm_side']} eef_link={payload['eef_link']} "
        f"offset_xyz={_format_tuple(offset_xyz)} "
        f"euler_deg={_format_tuple(euler_deg, 3)} "
        f"quat_wxyz={_format_tuple(quat_wxyz)}",
        flush=True,
    )
    print(
        "  python_constants:\n"
        f"    CONNECTOR_MOUNT_OFFSET_XYZ = {_format_tuple(offset_xyz)}\n"
        f"    CONNECTOR_MOUNT_EULER_DEG = {_format_tuple(euler_deg, 3)}\n"
        f"    CONNECTOR_MOUNT_QUAT_WXYZ = {_format_tuple(quat_wxyz)}",
        flush=True,
    )


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[connector-debug] saved {path}", flush=True)


def _mount_connector_pose(
    arm: object,
    *,
    eef_link_name: str,
    offset_xyz: tuple[float, float, float],
    euler_deg: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    eef_link = arm.get_link(eef_link_name)
    eef_pos = harness._tensor_to_np(eef_link.get_pos()).reshape(3).astype(np.float64)
    eef_quat = harness._tensor_to_np(eef_link.get_quat()).reshape(4).astype(np.float64)
    eef_rotation = harness._rotation_from_quat_wxyz(eef_quat)
    mount_rotation = harness._rotation_from_euler_deg(euler_deg)
    connector_pos = eef_pos + eef_rotation @ np.asarray(offset_xyz, dtype=np.float64)
    connector_rotation = eef_rotation @ mount_rotation
    return connector_pos, connector_rotation


def _apply_connector_pose(
    connector: object,
    arm: object,
    *,
    eef_link_name: str,
    values: tuple[float, ...],
) -> None:
    connector_pos, connector_rotation = _mount_connector_pose(
        arm,
        eef_link_name=eef_link_name,
        offset_xyz=values[:3],
        euler_deg=values[3:],
    )
    harness._set_entity_pose(connector, connector_pos, connector_rotation)


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
    root.title("Connector Mount Calibration")
    root.geometry("860x430")
    root.minsize(760, 360)

    title = ttk.Label(root, text="connector.STL pose relative to Nero end-effector", font=("Arial", 12, "bold"))
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
    ttk.Button(buttons, text="Print Mount", command=lambda: setattr(print_counter, "value", print_counter.value + 1)).pack(side=tk.LEFT)
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


def _default_mount_for_side(side: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if side == "right":
        return RIGHT_CONNECTOR_MOUNT_OFFSET_XYZ, RIGHT_CONNECTOR_MOUNT_EULER_DEG
    return LEFT_CONNECTOR_MOUNT_OFFSET_XYZ, LEFT_CONNECTOR_MOUNT_EULER_DEG


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune connector.STL pose relative to a Nero arm end-effector frame.")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--side", choices=("left", "right"), default=DEFAULT_SIDE)
    parser.add_argument("--eef-link", type=str, default=harness.DEFAULT_EEF_LINK)
    parser.add_argument("--connector-mesh", type=Path, default=DEFAULT_CONNECTOR_MESH)
    parser.add_argument("--connector-scale", type=float, default=0.001)
    parser.add_argument("--initial-offset", type=_vec3, default=None)
    parser.add_argument("--initial-euler", type=_vec3, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-mesh", type=Path, default=harness.DEFAULT_BASE_MESH)
    parser.add_argument("--nero-urdf", type=Path, default=harness.DEFAULT_NERO_URDF)
    parser.add_argument("--package-root", type=Path, default=harness.DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--base-scale", type=float, default=harness.DEFAULT_BASE_SCALE)
    parser.add_argument("--base-euler", type=harness._vec3, default=harness.DEFAULT_BASE_EULER)
    parser.add_argument("--base-foot-center-mm", type=harness._vec3, default=harness.DEFAULT_BASE_FOOT_CENTER_MM)
    parser.add_argument("--assembly-origin", type=harness._vec3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--no-revo2-flange", action="store_true")
    parser.add_argument("--no-viewer", action="store_true", help="Build once and print the initial connector mount.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    connector_mesh = args.connector_mesh.expanduser().resolve()
    if not connector_mesh.exists():
        raise FileNotFoundError(f"Connector mesh not found: {connector_mesh}")

    default_offset, default_euler = _default_mount_for_side(str(args.side))
    initial_offset = args.initial_offset if args.initial_offset is not None else default_offset
    initial_euler = args.initial_euler if args.initial_euler is not None else default_euler
    initial_values = tuple(float(v) for v in (*initial_offset, *initial_euler))

    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.4, -2.0, 1.2),
            camera_lookat=(0.0, 0.0, 0.35),
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
        origin=args.assembly_origin,
        base_scale=float(args.base_scale),
        base_euler=tuple(float(v) for v in args.base_euler),
        base_foot_center_mm=tuple(float(v) for v in args.base_foot_center_mm),
        add_revo2_flange=not args.no_revo2_flange,
    )
    connector = scene.add_entity(
        gs.morphs.Mesh(
            file=str(connector_mesh),
            pos=(0.0, 0.0, 0.0),
            euler=(0.0, 0.0, 0.0),
            scale=float(args.connector_scale),
            fixed=True,
            collision=False,
            convexify=False,
        ),
        name=f"{args.side}_connector_debug",
    )

    scene.build()
    mount_arm = assembly["left"] if args.side == "left" else assembly["right"]
    _apply_connector_pose(connector, mount_arm, eef_link_name=str(args.eef_link), values=initial_values)

    if args.no_viewer:
        _print_payload(
            _payload_from_values(
                values=initial_values,
                connector_mesh=connector_mesh,
                connector_scale=float(args.connector_scale),
                side=str(args.side),
                eef_link=str(args.eef_link),
            )
        )
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
                _apply_connector_pose(connector, mount_arm, eef_link_name=str(args.eef_link), values=values)
                last_values = values
            if int(panel["print_counter"].value) != last_print_counter:
                last_print_counter = int(panel["print_counter"].value)
                _print_payload(
                    _payload_from_values(
                        values=values,
                        connector_mesh=connector_mesh,
                        connector_scale=float(args.connector_scale),
                        side=str(args.side),
                        eef_link=str(args.eef_link),
                    )
                )
            if int(panel["save_counter"].value) != last_save_counter:
                last_save_counter = int(panel["save_counter"].value)
                payload = _payload_from_values(
                    values=values,
                    connector_mesh=connector_mesh,
                    connector_scale=float(args.connector_scale),
                    side=str(args.side),
                    eef_link=str(args.eef_link),
                )
                _write_payload(args.output.expanduser().resolve(), payload)
            scene.visualizer.update(force=True)
            time.sleep(1.0 / 60.0)
    finally:
        _shutdown_panel(panel)


if __name__ == "__main__":
    main()
