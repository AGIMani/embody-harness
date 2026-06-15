#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import genesis as gs
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import add_scene_glb as harness  # noqa: E402


DEFAULT_OUTPUT = ROOT_DIR / "assets" / "assembly_landmark_debug.json"
PAIR_COLORS = (
    (1.0, 0.12, 0.12, 1.0),
    (0.12, 0.85, 0.12, 1.0),
    (0.12, 0.35, 1.0, 1.0),
    (1.0, 0.85, 0.12, 1.0),
    (0.9, 0.25, 1.0, 1.0),
    (0.0, 0.9, 0.9, 1.0),
)


def _vec3(text: str) -> tuple[float, float, float]:
    parts = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected numeric x,y,z") from exc


def _parse_points(text: str, *, unit_scale: float) -> np.ndarray:
    rows = []
    for raw_row in str(text).split(";"):
        row = raw_row.strip()
        if not row:
            continue
        rows.append(_vec3(row))
    points = np.asarray(rows, dtype=np.float64) * float(unit_scale)
    if points.ndim != 2 or points.shape[1] != 3:
        raise argparse.ArgumentTypeError("expected point list as x,y,z;x,y,z;...")
    return points


def _format_tuple(values: tuple[float, ...] | np.ndarray, digits: int = 6) -> str:
    return "(" + ", ".join(f"{float(v):.{digits}f}" for v in values) + ")"


def _pose_matrix(pos: tuple[float, float, float], euler_deg: tuple[float, float, float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = harness._rotation_from_euler_deg(euler_deg)  # noqa: SLF001
    pose[:3, 3] = np.asarray(pos, dtype=np.float64)
    return pose


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def _rigid_transform_child_to_parent(child_points: np.ndarray, parent_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    child_points = np.asarray(child_points, dtype=np.float64)
    parent_points = np.asarray(parent_points, dtype=np.float64)
    if child_points.shape != parent_points.shape:
        raise ValueError(f"point shapes must match: child={child_points.shape} parent={parent_points.shape}")
    if child_points.ndim != 2 or child_points.shape[1] != 3 or child_points.shape[0] < 3:
        raise ValueError("need at least three 3D landmark pairs")

    child_center = child_points.mean(axis=0)
    parent_center = parent_points.mean(axis=0)
    child_centered = child_points - child_center
    parent_centered = parent_points - parent_center
    covariance = child_centered.T @ parent_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = parent_center - rotation @ child_center
    return rotation, translation


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _load_points_json(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "base_points_m" in data:
        base_points = np.asarray(data["base_points_m"], dtype=np.float64)
    elif "base_points_mm" in data:
        base_points = np.asarray(data["base_points_mm"], dtype=np.float64) * 0.001
    else:
        raise ValueError(f"{path} must contain base_points_m or base_points_mm")

    if "child_points_m" in data:
        child_points = np.asarray(data["child_points_m"], dtype=np.float64)
    elif "arm_points_m" in data:
        child_points = np.asarray(data["arm_points_m"], dtype=np.float64)
    elif "child_points_mm" in data:
        child_points = np.asarray(data["child_points_mm"], dtype=np.float64) * 0.001
    elif "arm_points_mm" in data:
        child_points = np.asarray(data["arm_points_mm"], dtype=np.float64) * 0.001
    else:
        raise ValueError(f"{path} must contain child_points_m/mm or arm_points_m/mm")
    return base_points, child_points


def _write_points_template(path: Path, *, side: str) -> None:
    payload = {
        "description": (
            "Edit corresponding landmark pairs. base_points_mm are points measured in the base local frame; "
            "child_points_mm are the matching points measured in the child local frame. Row order must match."
        ),
        "base_points_mm": _default_base_points(side).astype(float).tolist(),
        "child_points_mm": _default_child_points().astype(float).tolist(),
    }
    payload["base_points_mm"] = (np.asarray(payload["base_points_mm"], dtype=np.float64) * 1000.0).astype(float).tolist()
    payload["child_points_mm"] = (np.asarray(payload["child_points_mm"], dtype=np.float64) * 1000.0).astype(float).tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[landmark-assembly] template saved {path}", flush=True)


def _default_base_points(side: str) -> np.ndarray:
    points = np.asarray(harness.SUPPORT_HOLES_MM, dtype=np.float64).copy()
    if side == "right":
        points[:, 2] = float(harness.RIGHT_SUPPORT_HOLE_Z_MM)
    return points * 0.001


def _default_child_points() -> np.ndarray:
    return np.asarray(harness.ARM_HOLES_MM, dtype=np.float64) * 0.001


def _write_output(
    path: Path,
    *,
    child_name: str,
    base_points: np.ndarray,
    child_points: np.ndarray,
    child_in_base: np.ndarray,
    residuals: np.ndarray,
) -> None:
    euler = harness._euler_deg_from_rotation(child_in_base[:3, :3])  # noqa: SLF001
    payload = {
        "description": "Rigid landmark assembly result. child_in_base maps child local points onto base local points.",
        "child_name": child_name,
        "base_points_m": base_points.astype(float).tolist(),
        "child_points_m": child_points.astype(float).tolist(),
        "child_in_base": {
            "pos_m": child_in_base[:3, 3].astype(float).tolist(),
            "euler_xyz_deg": [float(v) for v in euler],
            "quat_wxyz": harness._quat_wxyz_from_rotation(child_in_base[:3, :3]).astype(float).tolist(),  # noqa: SLF001
            "matrix_row_major": child_in_base.astype(float).tolist(),
        },
        "residuals_m": residuals.astype(float).tolist(),
        "rms_error_m": float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1)))),
        "max_error_m": float(np.max(np.linalg.norm(residuals, axis=1))),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[landmark-assembly] saved {path}", flush=True)


def _print_result(child_name: str, child_in_base: np.ndarray, residuals: np.ndarray) -> None:
    pos = child_in_base[:3, 3]
    euler = harness._euler_deg_from_rotation(child_in_base[:3, :3])  # noqa: SLF001
    quat = harness._quat_wxyz_from_rotation(child_in_base[:3, :3])  # noqa: SLF001
    errors = np.linalg.norm(residuals, axis=1)
    print("[landmark-assembly]", flush=True)
    print(f"  child={child_name}", flush=True)
    print(f"  child_in_base.pos_m={_format_tuple(pos)}", flush=True)
    print(f"  child_in_base.euler_xyz_deg={_format_tuple(np.asarray(euler), 3)}", flush=True)
    print(f"  child_in_base.quat_wxyz={_format_tuple(quat, 8)}", flush=True)
    print(f"  rms_error_m={float(np.sqrt(np.mean(errors * errors))):.8f} max_error_m={float(np.max(errors)):.8f}", flush=True)
    print("  python_constants:", flush=True)
    prefix = child_name.upper().replace("-", "_").replace(" ", "_")
    print(f"    {prefix}_REL_POS_M = {_format_tuple(pos)}", flush=True)
    print(f"    {prefix}_REL_EULER_DEG = {_format_tuple(np.asarray(euler), 3)}", flush=True)


def _draw_frame_axes(scene: gs.Scene, pose: np.ndarray, *, axis_length: float, axis_radius: float) -> list[object]:
    origin = pose[:3, 3].astype(np.float32)
    rotation = pose[:3, :3].astype(np.float32)
    colors = ((1.0, 0.05, 0.05, 0.95), (0.1, 0.9, 0.1, 0.95), (0.1, 0.25, 1.0, 0.95))
    objects: list[object] = []
    for idx, color in enumerate(colors):
        points = np.stack([origin, origin + rotation[:, idx] * float(axis_length)], axis=0)
        objects.append(scene.draw_debug_trajectory(points, radius=float(axis_radius), color=color))
        objects.append(scene.draw_debug_sphere(points[1], radius=float(axis_radius) * 2.8, color=color))
    objects.append(scene.draw_debug_sphere(origin, radius=float(axis_radius) * 3.2, color=(1.0, 1.0, 1.0, 0.95)))
    return objects


def _draw_landmarks(
    scene: gs.Scene,
    *,
    base_points_world: np.ndarray,
    child_points_world: np.ndarray,
    radius: float,
) -> list[object]:
    objects: list[object] = []
    for idx, (base_point, child_point) in enumerate(zip(base_points_world, child_points_world, strict=True)):
        color = PAIR_COLORS[idx % len(PAIR_COLORS)]
        objects.append(scene.draw_debug_sphere(base_point.astype(np.float32), radius=float(radius), color=color))
        objects.append(scene.draw_debug_sphere(child_point.astype(np.float32), radius=float(radius) * 0.72, color=color))
        line = np.stack([base_point, child_point], axis=0).astype(np.float32)
        objects.append(scene.draw_debug_trajectory(line, radius=float(radius) * 0.18, color=(1.0, 1.0, 1.0, 0.7)))
    return objects


def _add_child_entity(
    scene: gs.Scene,
    *,
    child_kind: str,
    child_file: Path,
    child_world: np.ndarray,
    package_root: Path,
    scale: float,
) -> object:
    quat = harness._quat_wxyz_from_rotation(child_world[:3, :3])  # noqa: SLF001
    pos = tuple(float(v) for v in child_world[:3, 3])
    if child_kind == "urdf":
        urdf = harness._sanitize_urdf_for_genesis(child_file, package_root)  # noqa: SLF001
        return scene.add_entity(
            gs.morphs.URDF(
                file=str(urdf),
                pos=pos,
                quat=tuple(float(v) for v in quat),
                fixed=True,
                collision=False,
                merge_fixed_links=False,
                prioritize_urdf_material=True,
            ),
            name="landmark_child_urdf",
        )
    return scene.add_entity(
        gs.morphs.Mesh(
            file=str(child_file),
            pos=pos,
            quat=tuple(float(v) for v in quat),
            scale=float(scale),
            fixed=True,
            collision=False,
            convexify=False,
        ),
        name="landmark_child_mesh",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve child-in-base assembly pose from corresponding landmarks. "
            "Default landmarks are Nero support holes on base and arm mounting holes."
        )
    )
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--side", choices=("left", "right"), default="right", help="Default Nero base hole set.")
    parser.add_argument("--points-json", type=Path, default=None, help="JSON containing base_points_m/mm and child_points_m/mm.")
    parser.add_argument(
        "--write-template",
        type=Path,
        default=None,
        help="Write an editable landmark JSON template and exit.",
    )
    parser.add_argument("--base-points", default=None, help="Base points as x,y,z;x,y,z;... in --point-unit.")
    parser.add_argument("--child-points", default=None, help="Child points as x,y,z;x,y,z;... in --point-unit.")
    parser.add_argument("--point-unit", choices=("m", "mm"), default="m")
    parser.add_argument("--base-mesh", type=Path, default=harness.DEFAULT_BASE_MESH)
    parser.add_argument("--base-scale", type=float, default=harness.DEFAULT_BASE_SCALE)
    parser.add_argument("--base-pos", type=_vec3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--base-euler", type=_vec3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--child-name", default=None)
    parser.add_argument("--child-kind", choices=("urdf", "mesh"), default="urdf")
    parser.add_argument("--child-file", type=Path, default=harness.DEFAULT_NERO_URDF)
    parser.add_argument("--child-scale", type=float, default=1.0)
    parser.add_argument("--package-root", type=Path, default=harness.DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--scene-glb", type=Path, default=harness.DEFAULT_GLB)
    parser.add_argument("--show-scene", action="store_true")
    parser.add_argument("--show-viewer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--axis-length", type=float, default=0.18)
    parser.add_argument("--axis-radius", type=float, default=0.004)
    parser.add_argument("--landmark-radius", type=float, default=0.008)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.write_template is not None:
        _write_points_template(args.write_template.expanduser().resolve(), side=args.side)
        return

    unit_scale = 0.001 if args.point_unit == "mm" else 1.0
    if args.points_json is not None:
        base_points, child_points = _load_points_json(args.points_json.expanduser().resolve())
    elif args.base_points is not None or args.child_points is not None:
        if args.base_points is None or args.child_points is None:
            raise SystemExit("--base-points and --child-points must be provided together")
        base_points = _parse_points(args.base_points, unit_scale=unit_scale)
        child_points = _parse_points(args.child_points, unit_scale=unit_scale)
    else:
        base_points = _default_base_points(args.side)
        child_points = _default_child_points()

    rotation, translation = _rigid_transform_child_to_parent(child_points, base_points)
    child_in_base = _make_transform(rotation, translation)
    fitted_child_points = _transform_points(child_in_base, child_points)
    residuals = fitted_child_points - base_points
    child_name = args.child_name or f"{args.side}_arm"

    _print_result(child_name, child_in_base, residuals)
    if not args.no_save:
        _write_output(
            args.output.expanduser().resolve(),
            child_name=child_name,
            base_points=base_points,
            child_points=child_points,
            child_in_base=child_in_base,
            residuals=residuals,
        )
    if not args.show_viewer:
        return

    base_mesh = args.base_mesh.expanduser().resolve()
    child_file = args.child_file.expanduser().resolve()
    scene_glb = args.scene_glb.expanduser().resolve()
    package_root = args.package_root.expanduser().resolve()
    if not base_mesh.exists():
        raise FileNotFoundError(f"base mesh not found: {base_mesh}")
    if not child_file.exists():
        raise FileNotFoundError(f"child file not found: {child_file}")

    base_world = _pose_matrix(args.base_pos, args.base_euler)
    child_world = base_world @ child_in_base
    base_points_world = _transform_points(base_world, base_points)
    child_points_world = _transform_points(child_world, child_points)

    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.8, 1.1),
            camera_lookat=(0.0, 0.0, 0.25),
            camera_fov=42,
            res=(1280, 720),
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(dt=0.01, enable_self_collision=False, enable_adjacent_collision=False),
        vis_options=gs.options.VisOptions(show_world_frame=True),
        show_viewer=True,
    )
    scene.add_entity(gs.morphs.Plane(), name="world_xy_plane")
    if args.show_scene and scene_glb.exists():
        scene.add_entity(
            gs.morphs.Mesh(file=str(scene_glb), fixed=True, collision=False, convexify=False),
            surface=gs.surfaces.Default(vis_mode="visual"),
            name="landmark_scene_glb",
        )
    scene.add_entity(
        gs.morphs.Mesh(
            file=str(base_mesh),
            pos=args.base_pos,
            euler=args.base_euler,
            scale=float(args.base_scale),
            fixed=True,
            collision=False,
            convexify=False,
        ),
        name="landmark_base_mesh",
    )
    _add_child_entity(
        scene,
        child_kind=args.child_kind,
        child_file=child_file,
        child_world=child_world,
        package_root=package_root,
        scale=float(args.child_scale),
    )
    scene.build()
    _draw_frame_axes(scene, np.eye(4, dtype=np.float64), axis_length=args.axis_length, axis_radius=args.axis_radius)
    _draw_frame_axes(scene, base_world, axis_length=args.axis_length, axis_radius=args.axis_radius)
    _draw_frame_axes(scene, child_world, axis_length=args.axis_length * 0.8, axis_radius=args.axis_radius * 0.8)
    _draw_landmarks(
        scene,
        base_points_world=base_points_world,
        child_points_world=child_points_world,
        radius=float(args.landmark_radius),
    )

    while scene.viewer.is_alive():
        scene.visualizer.update(force=True)
        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
