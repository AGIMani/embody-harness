#!/usr/bin/env python3
"""Build a Genesis assembly scene for a base frame and two arms.

Start with the frame only, then add the left/right arms once the frame pose is set.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


NERO_TWIN_TMP_ROOT = Path(os.environ.get("NERO_TWIN_TMPDIR", f"/tmp/teleop_nero_{os.environ.get('USER', 'user')}"))
NERO_TWIN_TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NERO_TWIN_TMP_ROOT / "genesis_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(NERO_TWIN_TMP_ROOT / "genesis_mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(NERO_TWIN_TMP_ROOT / "genesis_xdg_cache"))
os.environ.setdefault("GS_CACHE_FILE_PATH", str(NERO_TWIN_TMP_ROOT / "genesis_cache"))
os.environ.setdefault("QD_OFFLINE_CACHE_FILE_PATH", str(NERO_TWIN_TMP_ROOT / "genesis_qd_cache"))

import genesis as gs


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
ASSET_ROOT = PROJECT_ROOT / "assets"
WORKSPACE_ROOT = ASSET_ROOT
DEFAULT_BASE_MESH = ASSET_ROOT / "mesh/base.STL"
DEFAULT_NERO_URDF = ASSET_ROOT / "agx_arm_urdf/nero/urdf/nero_description.urdf"
DEFAULT_BASE_FOOT_CENTER_MM = (-51.439, -842.036, -50.0)
SUPPORT_HOLES_MM = np.asarray(
    (
        (-86.439, 105.964, 9.0),
        (-16.439, 105.964, 9.0),
        (-86.439, 35.964, 9.0),
        (-16.439, 35.964, 9.0),
    ),
    dtype=np.float64,
)
RIGHT_SUPPORT_HOLE_Z_MM = -109.0
ARM_HOLES_MM = np.asarray(
    (
        (-35.0, -35.0, 0.0),
        (-35.0, 35.0, 0.0),
        (35.0, -35.0, 0.0),
        (35.0, 35.0, 0.0),
    ),
    dtype=np.float64,
)


def _parse_vec3(text: str, *, name: str) -> tuple[float, float, float]:
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(values) != 3:
        raise ValueError(f"{name} must contain 3 comma-separated floats, got {len(values)}")
    return tuple(values)


def _rotation_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)), dtype=np.float64)


def _rotation_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.asarray(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), dtype=np.float64)


def _rotation_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)


def _rotation_from_euler_deg(euler_deg: tuple[float, float, float]) -> np.ndarray:
    x, y, z = (np.deg2rad(v) for v in euler_deg)
    return _rotation_z(z) @ _rotation_y(y) @ _rotation_x(x)


def _quat_wxyz_from_rotation(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array((0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s))
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array(((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s))
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array(((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s))
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array(((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s))


def _quat_from_euler_deg(euler_deg: tuple[float, float, float]) -> np.ndarray:
    quat = _quat_wxyz_from_rotation(_rotation_from_euler_deg(euler_deg))
    return (quat / np.linalg.norm(quat)).astype(np.float32)


def _mesh_bounds(path: Path, scale: float, euler_deg: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import trimesh

        mesh = trimesh.load_mesh(path, force="mesh")
        vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
        vertices = vertices @ _rotation_from_euler_deg(euler_deg).T
        return vertices.min(axis=0), vertices.max(axis=0)
    except Exception as exc:
        print(f"[warn] failed to inspect mesh bounds with trimesh: {exc}", flush=True)
        return None


def _resolve_mesh_path(source_urdf: Path, package_root: Path, mesh_filename: str) -> str:
    if mesh_filename.startswith("package://"):
        package_path = Path(mesh_filename.removeprefix("package://"))
        candidates = (
            package_root / package_path,
            package_root / Path(*package_path.parts[1:]) if len(package_path.parts) > 1 else package_root / package_path,
            source_urdf.parents[1] / Path(*package_path.parts[2:]) if len(package_path.parts) > 2 else source_urdf.parent / package_path,
        )
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return mesh_filename

    path = Path(mesh_filename)
    if path.is_absolute():
        return str(path)
    return str((source_urdf.parent / path).resolve())


def _sanitize_urdf_for_genesis(source_urdf: Path, package_root: Path) -> Path:
    tree = ET.parse(source_urdf)
    root = tree.getroot()

    removed_transmissions = 0
    for transmission in list(root.findall("transmission")):
        root.remove(transmission)
        removed_transmissions += 1

    rewritten_meshes = 0
    unresolved: list[str] = []
    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        resolved = _resolve_mesh_path(source_urdf, package_root, filename)
        if resolved.startswith("package://"):
            unresolved.append(filename)
            continue
        if resolved != filename:
            mesh.set("filename", resolved)
            rewritten_meshes += 1

    out = NERO_TWIN_TMP_ROOT / f"{source_urdf.stem}_genesis_sanitized.urdf"
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print(
        "[genesis] sanitized arm URDF: "
        f"{source_urdf} -> {out} "
        f"(removed_transmissions={removed_transmissions}, rewritten_meshes={rewritten_meshes})",
        flush=True,
    )
    if unresolved:
        print("[warn] unresolved arm mesh paths:", *unresolved, sep="\n  ", flush=True)
    return out


def _default_centered_pos(path: Path, scale: float, euler_deg: tuple[float, float, float]) -> tuple[float, float, float]:
    bounds = _mesh_bounds(path, scale, euler_deg)
    if bounds is None:
        return (0.0, 0.0, 0.0)
    lower, upper = bounds
    center_xy = 0.5 * (lower[:2] + upper[:2])
    # Genesis applies scale and euler before translation. Move rotated mesh center to
    # world XY origin and its rotated bottom to z=0.
    pos = np.array((-center_xy[0], -center_xy[1], -lower[2]), dtype=np.float64)
    extents = upper - lower
    print(
        "[base] mesh bounds "
        f"lower={np.round(lower, 4).tolist()} upper={np.round(upper, 4).tolist()} "
        f"scale={scale:g} euler_deg={euler_deg} extents_m={np.round(extents, 4).tolist()} "
        f"default_pos={np.round(pos, 4).tolist()}",
        flush=True,
    )
    return tuple(float(v) for v in pos)


def _pose_from_local_anchor(
    anchor_mm: tuple[float, float, float],
    euler_deg: tuple[float, float, float],
    scale: float,
    world_anchor: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    anchor_local_m = np.asarray(anchor_mm, dtype=np.float64) * scale
    world_anchor_m = np.asarray(world_anchor, dtype=np.float64)
    pos = world_anchor_m - _rotation_from_euler_deg(euler_deg) @ anchor_local_m
    print(
        "[base] stand-up anchor "
        f"local_mm={tuple(round(v, 4) for v in anchor_mm)} "
        f"world_m={tuple(round(v, 5) for v in world_anchor)} "
        f"euler_deg={euler_deg} pos={np.round(pos, 5).tolist()}",
        flush=True,
    )
    return tuple(float(v) for v in pos)


def _default_arm_positions(
    base_mesh: Path,
    base_scale: float,
    base_euler: tuple[float, float, float],
    base_pos: tuple[float, float, float],
    *,
    side_axis: str,
    side_offset: float,
    center_offset: float,
    top_clearance: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    bounds = _mesh_bounds(base_mesh, base_scale, base_euler)
    top_z = base_pos[2] if bounds is None else base_pos[2] + float(bounds[1][2])

    left = np.array((0.0, 0.0, top_z + top_clearance), dtype=np.float64)
    right = left.copy()
    axis_idx = 0 if side_axis == "x" else 1
    center_idx = 1 if side_axis == "x" else 0
    left[axis_idx] = -abs(side_offset)
    right[axis_idx] = abs(side_offset)
    left[center_idx] = center_offset
    right[center_idx] = center_offset
    return tuple(float(v) for v in left), tuple(float(v) for v in right)


def _transform_point(
    point: tuple[float, float, float],
    pos: tuple[float, float, float],
    euler_deg: tuple[float, float, float],
) -> np.ndarray:
    return np.asarray(pos, dtype=np.float64) + _rotation_from_euler_deg(euler_deg) @ np.asarray(point, dtype=np.float64)


def _hole_aligned_arm_pose(
    base_pos: tuple[float, float, float],
    base_euler: tuple[float, float, float],
    support_hole_offset: tuple[float, float, float],
    arm_hole_offset: tuple[float, float, float],
    arm_hole_euler: tuple[float, float, float],
    *,
    support_holes_mm: np.ndarray = SUPPORT_HOLES_MM,
    arm_euler: tuple[float, float, float] | None = None,
    label: str = "left",
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    support_holes_m = np.asarray(support_holes_mm, dtype=np.float64) * 0.001 + np.asarray(
        support_hole_offset, dtype=np.float64
    )
    arm_holes_m = (
        ARM_HOLES_MM * 0.001 @ _rotation_from_euler_deg(arm_hole_euler).T
        + np.asarray(arm_hole_offset, dtype=np.float64)
    )
    support_center_m = support_holes_m.mean(axis=0)
    arm_center_m = arm_holes_m.mean(axis=0)

    # Both four-hole patterns are 70 mm squares in their local XY plane. Start
    # with no relative in-plane rotation; the slider can be used to test 90 deg
    # alternatives if the arm faces the wrong way.
    arm_euler = base_euler if arm_euler is None else arm_euler
    arm_world_pos = _transform_point(support_center_m, base_pos, base_euler) - (
        _rotation_from_euler_deg(arm_euler) @ arm_center_m
    )
    print(
        f"[align:{label}] support hole center in base mesh frame mm="
        f"{np.round(support_center_m * 1000.0, 4).tolist()} "
        f"arm hole center in URDF root/base_link frame mm={np.round(arm_center_m * 1000.0, 4).tolist()} "
        f"arm_world_pos={np.round(arm_world_pos, 5).tolist()} arm_euler={arm_euler}",
        flush=True,
    )
    print(
        "[align] derivation: arm_pos = base_pos + R_base * support_center - R_arm * arm_center; "
        f"support_hole_offset_m={tuple(round(v, 6) for v in support_hole_offset)} "
        f"arm_hole_offset_m={tuple(round(v, 6) for v in arm_hole_offset)} "
        f"arm_hole_euler_deg={arm_hole_euler}",
        flush=True,
    )
    return tuple(float(v) for v in arm_world_pos), arm_euler


def _transform_points(
    points_m: np.ndarray,
    pos: tuple[float, float, float],
    euler_deg: tuple[float, float, float],
) -> np.ndarray:
    return np.asarray(pos, dtype=np.float64) + points_m @ _rotation_from_euler_deg(euler_deg).T


def _add_hole_markers(
    scene,
    *,
    base_pos: tuple[float, float, float],
    base_euler: tuple[float, float, float],
    arm_pos: tuple[float, float, float],
    arm_euler: tuple[float, float, float],
    support_hole_offset: tuple[float, float, float],
    arm_hole_offset: tuple[float, float, float],
    arm_hole_euler: tuple[float, float, float],
) -> None:
    support_holes_m = SUPPORT_HOLES_MM * 0.001 + np.asarray(support_hole_offset, dtype=np.float64)
    arm_holes_m = (
        ARM_HOLES_MM * 0.001 @ _rotation_from_euler_deg(arm_hole_euler).T
        + np.asarray(arm_hole_offset, dtype=np.float64)
    )
    support_world = _transform_points(support_holes_m, base_pos, base_euler)
    arm_world = _transform_points(arm_holes_m, arm_pos, arm_euler)
    for pos in support_world:
        scene.add_entity(
            gs.morphs.Sphere(
                pos=tuple(float(v) for v in pos),
                radius=0.006,
                fixed=True,
                collision=False,
            )
        )
    for pos in arm_world:
        scene.add_entity(
            gs.morphs.Sphere(
                pos=tuple(float(v) for v in pos),
                radius=0.0035,
                fixed=True,
                collision=False,
            )
        )
    print("[debug] support_holes_world_m=", np.round(support_world, 5).tolist(), flush=True)
    print("[debug] arm_holes_world_m=", np.round(arm_world, 5).tolist(), flush=True)


def _mirror_arm_pose(
    side_offset: float,
    center_offset: float,
    z: float,
    left_euler: tuple[float, float, float],
    *,
    side_axis: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float], np.ndarray, np.ndarray]:
    left_pos = np.array((0.0, 0.0, z), dtype=np.float64)
    right_pos = left_pos.copy()
    axis_idx = 0 if side_axis == "x" else 1
    center_idx = 1 if side_axis == "x" else 0
    left_pos[axis_idx] = -abs(side_offset)
    right_pos[axis_idx] = abs(side_offset)
    left_pos[center_idx] = center_offset
    right_pos[center_idx] = center_offset

    left_quat = _quat_from_euler_deg(left_euler)
    right_quat = _quat_from_euler_deg((left_euler[0], left_euler[1], -left_euler[2]))
    return (
        tuple(float(v) for v in left_pos),
        tuple(float(v) for v in right_pos),
        left_quat,
        right_quat,
    )


def _start_arm_pose_gui(initial_values, side_axis: str, values, stop_flag) -> None:
    import tkinter as tk
    from tkinter import ttk

    specs = (
        ("side offset", 0.0, 0.6, 0.001, "m"),
        ("center offset", -0.6, 0.6, 0.001, "m"),
        ("height z", 0.0, 1.8, 0.001, "m"),
        ("roll", -180.0, 180.0, 0.1, "deg"),
        ("pitch", -180.0, 180.0, 0.1, "deg"),
        ("yaw", -180.0, 180.0, 0.1, "deg"),
    )

    def set_value(idx: int, value: float | str) -> None:
        values[idx] = float(value)
        value_labels[idx].config(text=f"{float(value): .3f}")

    def close() -> None:
        stop_flag.value = True
        root.destroy()
        root.quit()

    root = tk.Tk()
    root.title("Nero Arm Pose")
    root.geometry("700x360")

    title = ttk.Label(
        root,
        text=f"Arm pose controls. In two-arm mode, the right arm is mirrored across {side_axis.upper()}.",
        font=("Arial", 11, "bold"),
    )
    title.pack(fill=tk.X, padx=12, pady=(12, 4))

    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

    value_labels = []
    for idx, (label, lower, upper, _resolution, unit) in enumerate(specs):
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text=f"{label} ({unit})", width=18).pack(side=tk.LEFT)
        value_label = ttk.Label(row, text=f"{float(initial_values[idx]): .3f}", width=10)
        value_label.pack(side=tk.RIGHT)
        value_labels.append(value_label)
        slider = ttk.Scale(
            row,
            from_=lower,
            to=upper,
            orient=tk.HORIZONTAL,
            command=lambda value, i=idx: set_value(i, value),
        )
        slider.set(float(initial_values[idx]))
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

    ttk.Button(root, text="Close", command=close).pack(padx=12, pady=(0, 12), anchor=tk.E)
    root.protocol("WM_DELETE_WINDOW", close)

    def poll_stop() -> None:
        if stop_flag.value:
            close()
        else:
            root.after(100, poll_stop)

    root.after(100, poll_stop)
    root.mainloop()


def _create_arm_pose_window(
    enabled: bool,
    initial_left_pos: tuple[float, float, float],
    initial_left_euler: tuple[float, float, float],
    *,
    side_axis: str,
) -> dict[str, object] | None:
    if not enabled:
        return None
    axis_idx = 0 if side_axis == "x" else 1
    center_idx = 1 if side_axis == "x" else 0
    initial_values = (
        abs(initial_left_pos[axis_idx]),
        initial_left_pos[center_idx],
        initial_left_pos[2],
        initial_left_euler[0],
        initial_left_euler[1],
        initial_left_euler[2],
    )
    values = multiprocessing.RawArray("d", initial_values)
    stop_flag = multiprocessing.RawValue("b", False)
    process = multiprocessing.Process(
        target=_start_arm_pose_gui,
        args=(initial_values, side_axis, values, stop_flag),
        daemon=True,
    )
    process.start()
    print("[control] started arm pose slider window.", flush=True)
    return {"values": values, "stop_flag": stop_flag, "process": process}


def _shutdown_arm_pose_window(pose_window: dict[str, object] | None) -> None:
    if not pose_window:
        return
    pose_window["stop_flag"].value = True
    process = pose_window["process"]
    if process.is_alive():
        process.join(timeout=1.0)


def _set_entity_pose(entity, pos: tuple[float, float, float], quat: np.ndarray) -> None:
    entity.set_pos(np.asarray(pos, dtype=np.float32), zero_velocity=True)
    entity.set_quat(np.asarray(quat, dtype=np.float32), zero_velocity=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--max-fps", type=int, default=60)
    parser.add_argument("--base-mesh", type=Path, default=DEFAULT_BASE_MESH)
    parser.add_argument("--base-scale", type=float, default=0.001, help="STL appears to be in millimeters.")
    parser.add_argument(
        "--base-pos",
        type=str,
        default=None,
        help="World XYZ in meters. Default stands the assembly on --base-foot-center-mm.",
    )
    parser.add_argument(
        "--base-lift",
        type=float,
        default=0.0,
        help="Extra Z offset in meters applied to the auto-centered base pose.",
    )
    parser.add_argument(
        "--base-euler",
        type=str,
        default="90,0,0",
        help="World XYZ Euler angles in degrees. Default stands the STEP/STL -Y bottom face upright.",
    )
    parser.add_argument(
        "--base-foot-center-mm",
        type=str,
        default=",".join(str(v) for v in DEFAULT_BASE_FOOT_CENTER_MM),
        help="STEP/STL local XYZ in millimeters for the base bottom-face center used as the stand-up anchor.",
    )
    parser.add_argument(
        "--keep-base-raw-pose",
        action="store_true",
        help="Keep the base mesh at pos=(0,0,0), euler=(0,0,0) unless --base-pos/--base-euler are provided.",
    )
    parser.add_argument(
        "--auto-center-base",
        action="store_true",
        help="Center base XY and place its bottom at z=0 before applying --base-lift.",
    )
    parser.add_argument("--base-collision", action="store_true")
    parser.add_argument("--base-convexify", action="store_true")
    parser.add_argument("--no-plane", action="store_true")
    parser.add_argument("--no-arms", action="store_true", help="Only load the support base.")
    parser.add_argument("--nero-urdf", type=Path, default=DEFAULT_NERO_URDF)
    parser.add_argument("--package-root", type=Path, default=WORKSPACE_ROOT)
    parser.add_argument("--arm-side-axis", choices=("x", "y"), default="x")
    parser.add_argument("--arm-side-offset", type=float, default=0.19)
    parser.add_argument("--arm-center-offset", type=float, default=0.0)
    parser.add_argument("--arm-top-clearance", type=float, default=0.015)
    parser.add_argument("--left-arm-pos", type=str, default=None)
    parser.add_argument("--right-arm-pos", type=str, default=None)
    parser.add_argument("--left-arm-euler", type=str, default="0,0,90")
    parser.add_argument("--right-arm-euler", type=str, default="0,0,-90")
    parser.add_argument("--one-arm", action="store_true", help="Only load the left/original Nero arm.")
    parser.add_argument(
        "--align-one-arm-to-holes",
        action="store_true",
        help="Place original Nero arm(s) by aligning their 70 mm base-hole squares to the provided support holes.",
    )
    parser.add_argument(
        "--right-support-hole-z-mm",
        type=float,
        default=RIGHT_SUPPORT_HOLE_Z_MM,
        help="STEP/STL local Z coordinate in millimeters for the opposite-side support screw holes.",
    )
    parser.add_argument(
        "--support-hole-offset",
        type=str,
        default="0,0,0",
        help=(
            "XYZ offset in meters from the provided support-hole CAD frame to the loaded base.STL frame. "
            "Keep this at zero when the STL was exported in the same frame used for STEP/CAD measurements."
        ),
    )
    parser.add_argument(
        "--arm-hole-offset",
        type=str,
        default="0,0,0",
        help="XYZ offset in meters from the provided arm-hole frame to the Nero URDF root/base_link frame.",
    )
    parser.add_argument(
        "--arm-hole-euler",
        type=str,
        default="0,0,0",
        help="XYZ Euler angles in degrees from the provided arm-hole frame to the Nero URDF root/base_link frame.",
    )
    parser.add_argument(
        "--show-hole-markers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show support-hole and arm-hole marker spheres for alignment debugging.",
    )
    parser.add_argument(
        "--arm-pose-window",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show a slider window to tune mirrored arm pose. Default: enabled when viewer is enabled.",
    )
    parser.add_argument("--arm-collision", action="store_true")
    parser.add_argument("--arm-convexify", action="store_true")
    parser.add_argument("--merge-arm-fixed-links", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_mesh = args.base_mesh.expanduser().resolve()
    if not base_mesh.exists():
        raise FileNotFoundError(f"Base mesh not found: {base_mesh}")

    base_euler = _parse_vec3(args.base_euler, name="--base-euler")
    if args.keep_base_raw_pose:
        base_euler = (0.0, 0.0, 0.0)
    base_foot_center_mm = _parse_vec3(args.base_foot_center_mm, name="--base-foot-center-mm")
    if args.base_pos:
        base_pos = _parse_vec3(args.base_pos, name="--base-pos")
    elif args.auto_center_base:
        base_pos = _default_centered_pos(base_mesh, args.base_scale, base_euler)
    elif args.keep_base_raw_pose:
        base_pos = (0.0, 0.0, 0.0)
    else:
        base_pos = _pose_from_local_anchor(base_foot_center_mm, base_euler, args.base_scale)
    if args.base_pos is None and args.auto_center_base and args.base_lift:
        base_pos = (base_pos[0], base_pos[1], base_pos[2] + float(args.base_lift))
    left_arm_euler = _parse_vec3(args.left_arm_euler, name="--left-arm-euler")
    right_arm_euler = _parse_vec3(args.right_arm_euler, name="--right-arm-euler")
    support_hole_offset = _parse_vec3(args.support_hole_offset, name="--support-hole-offset")
    arm_hole_offset = _parse_vec3(args.arm_hole_offset, name="--arm-hole-offset")
    arm_hole_euler = _parse_vec3(args.arm_hole_euler, name="--arm-hole-euler")
    nero_urdf = args.nero_urdf.expanduser().resolve()
    package_root = args.package_root.expanduser().resolve()
    if not args.no_arms and not nero_urdf.exists():
        raise FileNotFoundError(f"Nero URDF not found: {nero_urdf}")

    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -2.2, 1.4),
            camera_lookat=(0.0, 0.0, 0.45),
            camera_fov=35,
            res=(1280, 720),
            max_FPS=args.max_fps,
        ),
        sim_options=gs.options.SimOptions(dt=args.dt),
        rigid_options=gs.options.RigidOptions(dt=args.dt),
        show_viewer=not args.no_viewer,
    )

    if not args.no_plane:
        scene.add_entity(gs.morphs.Plane())

    scene.add_entity(
        gs.morphs.Mesh(
            file=str(base_mesh),
            pos=base_pos,
            euler=base_euler,
            scale=args.base_scale,
            fixed=True,
            collision=args.base_collision,
            convexify=args.base_convexify,
        )
    )
    print(
        "[base] added "
        f"mesh={base_mesh} pos={tuple(round(v, 5) for v in base_pos)} "
        f"euler_deg={base_euler} scale={args.base_scale:g} collision={args.base_collision}",
        flush=True,
    )

    arm_entities: dict[str, object] = {}
    initial_left_arm_pos: tuple[float, float, float] | None = None
    initial_left_arm_euler: tuple[float, float, float] | None = None
    if not args.no_arms:
        if args.align_one_arm_to_holes:
            right_support_holes_mm = SUPPORT_HOLES_MM.copy()
            right_support_holes_mm[:, 2] = float(args.right_support_hole_z_mm)
            default_left_pos, left_arm_euler = _hole_aligned_arm_pose(
                base_pos,
                base_euler,
                support_hole_offset,
                arm_hole_offset,
                arm_hole_euler,
                support_holes_mm=SUPPORT_HOLES_MM,
                arm_euler=base_euler,
                label="left",
            )
            right_arm_euler = (base_euler[0] + 180.0, base_euler[1], base_euler[2])
            default_right_pos, right_arm_euler = _hole_aligned_arm_pose(
                base_pos,
                base_euler,
                support_hole_offset,
                arm_hole_offset,
                arm_hole_euler,
                support_holes_mm=right_support_holes_mm,
                arm_euler=right_arm_euler,
                label="right",
            )
        else:
            default_left_pos, default_right_pos = _default_arm_positions(
                base_mesh,
                args.base_scale,
                base_euler,
                base_pos,
                side_axis=args.arm_side_axis,
                side_offset=args.arm_side_offset,
                center_offset=args.arm_center_offset,
                top_clearance=args.arm_top_clearance,
            )
        left_arm_pos = (
            _parse_vec3(args.left_arm_pos, name="--left-arm-pos")
            if args.left_arm_pos
            else default_left_pos
        )
        right_arm_pos = (
            _parse_vec3(args.right_arm_pos, name="--right-arm-pos")
            if args.right_arm_pos
            else default_right_pos
        )
        left_arm_quat = _quat_from_euler_deg(left_arm_euler)
        right_arm_quat = _quat_from_euler_deg(right_arm_euler)
        if not args.align_one_arm_to_holes and not args.left_arm_pos and not args.right_arm_pos:
            left_arm_pos, right_arm_pos, left_arm_quat, right_arm_quat = _mirror_arm_pose(
                args.arm_side_offset,
                args.arm_center_offset,
                left_arm_pos[2],
                left_arm_euler,
                side_axis=args.arm_side_axis,
            )
        genesis_nero_urdf = _sanitize_urdf_for_genesis(nero_urdf, package_root)

        arm_specs = [("left", genesis_nero_urdf, left_arm_pos, left_arm_quat)]
        if not args.one_arm:
            arm_specs.append(("right", genesis_nero_urdf, right_arm_pos, right_arm_quat))

        for label, urdf_file, pos, euler in arm_specs:
            arm_entities[label] = scene.add_entity(
                gs.morphs.URDF(
                    file=str(urdf_file),
                    pos=pos,
                    quat=tuple(float(v) for v in euler),
                    fixed=True,
                    collision=args.arm_collision,
                    convexify=args.arm_convexify,
                    merge_fixed_links=args.merge_arm_fixed_links,
                    prioritize_urdf_material=True,
                )
            )
            print(
                f"[arm:{label}] added urdf={urdf_file} "
                f"pos={tuple(round(v, 5) for v in pos)} quat_wxyz={tuple(round(float(v), 5) for v in euler)} "
                f"collision={args.arm_collision}",
                flush=True,
            )
        if args.show_hole_markers and args.align_one_arm_to_holes:
            _add_hole_markers(
                scene,
                base_pos=base_pos,
                base_euler=base_euler,
                arm_pos=left_arm_pos,
                arm_euler=left_arm_euler,
                support_hole_offset=support_hole_offset,
                arm_hole_offset=arm_hole_offset,
                arm_hole_euler=arm_hole_euler,
            )
        initial_left_arm_pos = left_arm_pos
        initial_left_arm_euler = left_arm_euler

    scene.build()
    enable_arm_pose_window = (
        (not args.no_viewer and not args.no_arms and not args.align_one_arm_to_holes)
        if args.arm_pose_window is None
        else bool(args.arm_pose_window)
    )
    arm_pose_window = _create_arm_pose_window(
        enable_arm_pose_window,
        initial_left_arm_pos or (0.0, 0.0, 0.0),
        initial_left_arm_euler or (0.0, 0.0, 0.0),
        side_axis=args.arm_side_axis,
    )
    step = 0
    try:
        while True:
            if arm_pose_window and arm_entities:
                values = arm_pose_window["values"]
                left_pos, right_pos, left_quat_live, right_quat_live = _mirror_arm_pose(
                    float(values[0]),
                    float(values[1]),
                    float(values[2]),
                    (float(values[3]), float(values[4]), float(values[5])),
                    side_axis=args.arm_side_axis,
                )
                _set_entity_pose(arm_entities["left"], left_pos, left_quat_live)
                if "right" in arm_entities:
                    _set_entity_pose(arm_entities["right"], right_pos, right_quat_live)

            scene.step()
            step += 1
            if arm_pose_window and arm_pose_window["stop_flag"].value:
                break
            if args.steps >= 0 and step >= args.steps:
                break
            if args.steps < 0 and args.no_viewer:
                break
            if not args.no_viewer and not scene.viewer.is_alive():
                break
    finally:
        _shutdown_arm_pose_window(arm_pose_window)


if __name__ == "__main__":
    main()
