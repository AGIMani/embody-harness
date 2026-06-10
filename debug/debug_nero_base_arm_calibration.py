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


DEFAULT_OUTPUT = ROOT_DIR / "assets" / "nero_twin" / "arm_base_relative_pose_debug.json"

DEFAULT_LEFT_ARM_REL_POS_M = (0.252915, 1.078233, 0.193274)
DEFAULT_LEFT_ARM_REL_EULER_DEG = (180.0, 0.0, 90.0)
DEFAULT_RIGHT_ARM_REL_POS_M = (0.252915, 1.078472, 0.311659)
DEFAULT_RIGHT_ARM_REL_EULER_DEG = (0.0, 0.0, 90.0)

FINE_TUNE_TRANSLATION_RANGE_M = 0.10
FINE_TUNE_TRANSLATION_STEP_M = 0.001
FINE_TUNE_ROTATION_RANGE_DEG = 15.0
FINE_TUNE_ROTATION_STEP_DEG = 0.1


def _rotation_to_euler_deg(rotation: np.ndarray) -> tuple[float, float, float]:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sy = -float(rotation[2, 0])
    sy = max(-1.0, min(1.0, sy))
    y = np.arcsin(sy)
    cy = np.cos(y)
    if abs(cy) > 1e-8:
        x = np.arctan2(rotation[2, 1], rotation[2, 2])
        z = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        x = 0.0
        z = np.arctan2(-rotation[0, 1], rotation[1, 1])
    return tuple(float(v) for v in np.rad2deg((x, y, z)))


def _pose_world_from_base_relative(
    *,
    base_pos: tuple[float, float, float],
    base_euler: tuple[float, float, float],
    rel_pos: tuple[float, float, float],
    rel_euler: tuple[float, float, float],
) -> tuple[tuple[float, float, float], np.ndarray]:
    base_rotation = harness._rotation_from_euler_deg(base_euler)
    rel_rotation = harness._rotation_from_euler_deg(rel_euler)
    world_pos = np.asarray(base_pos, dtype=np.float64) + base_rotation @ np.asarray(rel_pos, dtype=np.float64)
    world_rotation = base_rotation @ rel_rotation
    return tuple(float(v) for v in world_pos), world_rotation


def _relative_pose_from_world(
    *,
    base_pos: tuple[float, float, float],
    base_euler: tuple[float, float, float],
    arm_pos: tuple[float, float, float],
    arm_rotation: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    base_rotation = harness._rotation_from_euler_deg(base_euler)
    rel_pos = base_rotation.T @ (np.asarray(arm_pos, dtype=np.float64) - np.asarray(base_pos, dtype=np.float64))
    rel_rotation = base_rotation.T @ np.asarray(arm_rotation, dtype=np.float64)
    return tuple(float(v) for v in rel_pos), _rotation_to_euler_deg(rel_rotation)


def _format_tuple(values: tuple[float, ...], digits: int = 6) -> str:
    return "(" + ", ".join(f"{float(v):.{digits}f}" for v in values) + ")"


def _payload_from_values(
    *,
    values: tuple[float, ...],
    base_mesh: Path,
    base_scale: float,
    base_pos: tuple[float, float, float],
    base_euler: tuple[float, float, float],
) -> dict[str, object]:
    left_rel_pos = values[0:3]
    left_rel_euler = values[3:6]
    right_rel_pos = values[6:9]
    right_rel_euler = values[9:12]
    left_rel_quat = harness._quat_wxyz_from_rotation(harness._rotation_from_euler_deg(left_rel_euler))
    right_rel_quat = harness._quat_wxyz_from_rotation(harness._rotation_from_euler_deg(right_rel_euler))
    return {
        "description": "Nero arm poses relative to the loaded base.STL entity frame.",
        "base_mesh": str(base_mesh),
        "base_scale": float(base_scale),
        "base_world_pos_m": [float(v) for v in base_pos],
        "base_world_euler_deg": [float(v) for v in base_euler],
        "left": {
            "pos_in_base_frame_m": [float(v) for v in left_rel_pos],
            "pos_in_base_stl_units": [float(v) / float(base_scale) for v in left_rel_pos],
            "euler_xyz_in_base_frame_deg": [float(v) for v in left_rel_euler],
            "quat_wxyz_in_base_frame": [float(v) for v in left_rel_quat],
        },
        "right": {
            "pos_in_base_frame_m": [float(v) for v in right_rel_pos],
            "pos_in_base_stl_units": [float(v) / float(base_scale) for v in right_rel_pos],
            "euler_xyz_in_base_frame_deg": [float(v) for v in right_rel_euler],
            "quat_wxyz_in_base_frame": [float(v) for v in right_rel_quat],
        },
    }


def _print_payload(payload: dict[str, object]) -> None:
    left = payload["left"]
    right = payload["right"]
    print("[arm-base-calibration] relative_to_base_stl", flush=True)
    for side, data in (("left", left), ("right", right)):
        pos_m = tuple(float(v) for v in data["pos_in_base_frame_m"])
        euler = tuple(float(v) for v in data["euler_xyz_in_base_frame_deg"])
        quat = tuple(float(v) for v in data["quat_wxyz_in_base_frame"])
        pos_units = tuple(float(v) for v in data["pos_in_base_stl_units"])
        print(
            f"  {side}.pos_in_base_frame_m={_format_tuple(pos_m)} "
            f"euler_deg={_format_tuple(euler, 3)} "
            f"quat_wxyz={_format_tuple(quat)} "
            f"base_stl_units={_format_tuple(pos_units, 3)}",
            flush=True,
        )
    print(
        "  python_constants:\n"
        f"    LEFT_ARM_REL_POS_M = {_format_tuple(tuple(float(v) for v in left['pos_in_base_frame_m']))}\n"
        f"    LEFT_ARM_REL_EULER_DEG = {_format_tuple(tuple(float(v) for v in left['euler_xyz_in_base_frame_deg']), 3)}\n"
        f"    RIGHT_ARM_REL_POS_M = {_format_tuple(tuple(float(v) for v in right['pos_in_base_frame_m']))}\n"
        f"    RIGHT_ARM_REL_EULER_DEG = {_format_tuple(tuple(float(v) for v in right['euler_xyz_in_base_frame_deg']), 3)}",
        flush=True,
    )


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[arm-base-calibration] saved {path}", flush=True)


def _panel_main(initial_values, values, print_counter, save_counter, reset_counter, stop_flag) -> None:
    import tkinter as tk
    from tkinter import ttk

    specs = (
        ("left x", FINE_TUNE_TRANSLATION_RANGE_M, FINE_TUNE_TRANSLATION_STEP_M, "m"),
        ("left y", FINE_TUNE_TRANSLATION_RANGE_M, FINE_TUNE_TRANSLATION_STEP_M, "m"),
        ("left z", FINE_TUNE_TRANSLATION_RANGE_M, FINE_TUNE_TRANSLATION_STEP_M, "m"),
        ("left roll", FINE_TUNE_ROTATION_RANGE_DEG, FINE_TUNE_ROTATION_STEP_DEG, "deg"),
        ("left pitch", FINE_TUNE_ROTATION_RANGE_DEG, FINE_TUNE_ROTATION_STEP_DEG, "deg"),
        ("left yaw", FINE_TUNE_ROTATION_RANGE_DEG, FINE_TUNE_ROTATION_STEP_DEG, "deg"),
        ("right x", FINE_TUNE_TRANSLATION_RANGE_M, FINE_TUNE_TRANSLATION_STEP_M, "m"),
        ("right y", FINE_TUNE_TRANSLATION_RANGE_M, FINE_TUNE_TRANSLATION_STEP_M, "m"),
        ("right z", FINE_TUNE_TRANSLATION_RANGE_M, FINE_TUNE_TRANSLATION_STEP_M, "m"),
        ("right roll", FINE_TUNE_ROTATION_RANGE_DEG, FINE_TUNE_ROTATION_STEP_DEG, "deg"),
        ("right pitch", FINE_TUNE_ROTATION_RANGE_DEG, FINE_TUNE_ROTATION_STEP_DEG, "deg"),
        ("right yaw", FINE_TUNE_ROTATION_RANGE_DEG, FINE_TUNE_ROTATION_STEP_DEG, "deg"),
    )

    def set_delta(idx: int, delta: float | str) -> None:
        span = float(specs[idx][1])
        delta_value = max(-span, min(span, float(delta)))
        absolute_value = float(initial_values[idx]) + delta_value
        values[idx] = absolute_value
        delta_labels[idx].config(text=f"{delta_value:+.5f}")
        value_labels[idx].config(text=f"{absolute_value: .5f}")

    def step_delta(idx: int, direction: int) -> None:
        step = float(specs[idx][2])
        delta_value = float(sliders[idx].get()) + float(direction) * step
        span = float(specs[idx][1])
        delta_value = max(-span, min(span, delta_value))
        sliders[idx].set(delta_value)
        set_delta(idx, delta_value)

    def reset() -> None:
        for idx, slider in enumerate(sliders):
            slider.set(0.0)
            set_delta(idx, 0.0)
        reset_counter.value += 1

    def close() -> None:
        stop_flag.value = True
        root.destroy()
        root.quit()

    root = tk.Tk()
    root.title("Nero Arm Base Calibration")
    root.geometry("980x730")
    root.minsize(820, 620)

    title = ttk.Label(root, text="Arm pose fine tuning relative to base.STL", font=("Arial", 12, "bold"))
    title.pack(fill=tk.X, padx=12, pady=(12, 4))

    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

    sliders = []
    delta_labels = []
    value_labels = []
    for idx, (label, span, step, unit) in enumerate(specs):
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=f"{label} ({unit})", width=14).pack(side=tk.LEFT)
        ttk.Button(row, text="-", width=3, command=lambda i=idx: step_delta(i, -1)).pack(side=tk.LEFT)
        slider = ttk.Scale(
            row,
            from_=-float(span),
            to=float(span),
            orient=tk.HORIZONTAL,
            command=lambda value, i=idx: set_delta(i, value),
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(row, text="+", width=3, command=lambda i=idx: step_delta(i, 1)).pack(side=tk.LEFT)
        delta_label = ttk.Label(row, text="+0.00000", width=11)
        delta_label.pack(side=tk.LEFT, padx=(10, 2))
        value_label = ttk.Label(row, text=f"{float(initial_values[idx]): .5f}", width=12)
        value_label.pack(side=tk.RIGHT)
        sliders.append(slider)
        delta_labels.append(delta_label)
        value_labels.append(value_label)
        slider.set(0.0)
        set_delta(idx, 0.0)

    buttons = ttk.Frame(root)
    buttons.pack(fill=tk.X, padx=12, pady=(4, 12))
    ttk.Button(buttons, text="Print Relative Pose", command=lambda: setattr(print_counter, "value", print_counter.value + 1)).pack(side=tk.LEFT)
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
    return tuple(float(values[idx]) for idx in range(12))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Nero arm poses relative to base.STL and record base-local transforms.")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--base-mesh", type=Path, default=harness.DEFAULT_BASE_MESH)
    parser.add_argument("--nero-urdf", type=Path, default=harness.DEFAULT_NERO_URDF)
    parser.add_argument("--package-root", type=Path, default=harness.DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--base-scale", type=float, default=harness.DEFAULT_BASE_SCALE)
    parser.add_argument(
        "--base-pos",
        type=harness._vec3,
        default=None,
        help="Base world position as x,y,z. Default uses --base-foot-center-mm as an anchor at world origin.",
    )
    parser.add_argument("--base-euler", type=harness._vec3, default=harness.DEFAULT_BASE_EULER)
    parser.add_argument(
        "--base-foot-center-mm",
        type=harness._vec3,
        default=harness.DEFAULT_BASE_FOOT_CENTER_MM,
        help="base.STL local XYZ in millimeters for the bottom-foot center anchor.",
    )
    parser.add_argument("--right-support-hole-z-mm", type=float, default=harness.RIGHT_SUPPORT_HOLE_Z_MM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-viewer", action="store_true", help="Build once and print the initial relative poses without opening UI.")
    parser.add_argument("--no-revo2-flange", action="store_true")
    parser.add_argument("--show-hole-markers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_mesh = args.base_mesh.expanduser().resolve()
    nero_urdf = args.nero_urdf.expanduser().resolve()
    package_root = args.package_root.expanduser().resolve()
    if not base_mesh.exists():
        raise FileNotFoundError(f"Base mesh not found: {base_mesh}")
    if not nero_urdf.exists():
        raise FileNotFoundError(f"Nero URDF not found: {nero_urdf}")

    base_pos = (
        tuple(float(v) for v in args.base_pos)
        if args.base_pos is not None
        else harness._pose_from_local_anchor(
            tuple(float(v) for v in args.base_foot_center_mm),
            tuple(float(v) for v in args.base_euler),
            float(args.base_scale),
        )
    )
    base_euler = tuple(float(v) for v in args.base_euler)
    initial_values = tuple(
        float(v)
        for v in (
            *DEFAULT_LEFT_ARM_REL_POS_M,
            *DEFAULT_LEFT_ARM_REL_EULER_DEG,
            *DEFAULT_RIGHT_ARM_REL_POS_M,
            *DEFAULT_RIGHT_ARM_REL_EULER_DEG,
        )
    )
    left_pos, left_rotation = _pose_world_from_base_relative(
        base_pos=base_pos,
        base_euler=base_euler,
        rel_pos=initial_values[0:3],
        rel_euler=initial_values[3:6],
    )
    right_pos, right_rotation = _pose_world_from_base_relative(
        base_pos=base_pos,
        base_euler=base_euler,
        rel_pos=initial_values[6:9],
        rel_euler=initial_values[9:12],
    )
    left_quat = harness._quat_wxyz_from_rotation(left_rotation)
    right_quat = harness._quat_wxyz_from_rotation(right_rotation)

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
    base = scene.add_entity(
        gs.morphs.Mesh(
            file=str(base_mesh),
            pos=base_pos,
            euler=base_euler,
            scale=float(args.base_scale),
            fixed=True,
            collision=False,
            convexify=False,
        ),
        name="dual_nero_base_debug",
    )
    urdf_for_genesis = harness._make_revo2_flange_urdf(nero_urdf) if not args.no_revo2_flange else nero_urdf
    urdf_for_genesis = harness._sanitize_urdf_for_genesis(urdf_for_genesis, package_root)
    arm_kwargs = {
        "file": str(urdf_for_genesis),
        "fixed": True,
        "collision": False,
        "convexify": False,
        "merge_fixed_links": False,
        "prioritize_urdf_material": True,
    }
    if not args.no_revo2_flange:
        arm_kwargs["links_to_keep"] = (harness.DEFAULT_EEF_LINK,)
    left_arm = scene.add_entity(gs.morphs.URDF(**arm_kwargs), name="left_nero_arm_debug")
    right_arm = scene.add_entity(gs.morphs.URDF(**arm_kwargs), name="right_nero_arm_debug")

    markers: list[dict[str, object]] = []
    if args.show_hole_markers:
        markers = harness._add_hole_markers(
            scene,
            base_pos=base_pos,
            base_euler=base_euler,
            right_support_hole_z_mm=float(args.right_support_hole_z_mm),
            left_pos=left_pos,
            left_quat=left_quat,
            right_pos=right_pos,
            right_quat=right_quat,
        )

    def apply_values(values: tuple[float, ...]) -> None:
        left_world_pos, left_world_rotation = _pose_world_from_base_relative(
            base_pos=base_pos,
            base_euler=base_euler,
            rel_pos=values[0:3],
            rel_euler=values[3:6],
        )
        right_world_pos, right_world_rotation = _pose_world_from_base_relative(
            base_pos=base_pos,
            base_euler=base_euler,
            rel_pos=values[6:9],
            rel_euler=values[9:12],
        )
        harness._set_entity_pose(left_arm, np.asarray(left_world_pos, dtype=np.float64), left_world_rotation)
        harness._set_entity_pose(right_arm, np.asarray(right_world_pos, dtype=np.float64), right_world_rotation)

    scene.build()
    apply_values(initial_values)
    if args.no_viewer:
        _print_payload(
            _payload_from_values(
                values=initial_values,
                base_mesh=base_mesh,
                base_scale=float(args.base_scale),
                base_pos=base_pos,
                base_euler=base_euler,
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
                apply_values(values)
                last_values = values
            if int(panel["print_counter"].value) != last_print_counter:
                last_print_counter = int(panel["print_counter"].value)
                _print_payload(
                    _payload_from_values(
                        values=values,
                        base_mesh=base_mesh,
                        base_scale=float(args.base_scale),
                        base_pos=base_pos,
                        base_euler=base_euler,
                    )
                )
            if int(panel["save_counter"].value) != last_save_counter:
                last_save_counter = int(panel["save_counter"].value)
                payload = _payload_from_values(
                    values=values,
                    base_mesh=base_mesh,
                    base_scale=float(args.base_scale),
                    base_pos=base_pos,
                    base_euler=base_euler,
                )
                _write_payload(args.output.expanduser().resolve(), payload)
            scene.visualizer.update(force=True)
            time.sleep(1.0 / 60.0)
    finally:
        _shutdown_panel(panel)


if __name__ == "__main__":
    main()
