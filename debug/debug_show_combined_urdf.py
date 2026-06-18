#!/usr/bin/env python3
"""Load the generated dual Nero + Linker L10 combined URDF in Genesis."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import genesis as gs


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_COMBINED_URDF = ROOT_DIR / "assets" / "generated" / "dual_nero_linker_l10_combined.urdf"


def _parse_vec3(text: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values, e.g. 0,0,0")
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid vec3: {text!r}") from exc


def _draw_frame_axes(
    scene: gs.Scene,
    pose: np.ndarray,
    *,
    axis_length: float,
    axis_radius: float,
) -> list[Any]:
    origin = pose[:3, 3].astype(np.float64)
    rotation = pose[:3, :3].astype(np.float64)
    colors = (
        (1.0, 0.05, 0.05, 0.95),
        (0.05, 0.85, 0.05, 0.95),
        (0.1, 0.25, 1.0, 0.95),
    )
    objects: list[Any] = []
    for idx, color in enumerate(colors):
        tip = origin + rotation[:, idx] * float(axis_length)
        points = np.stack([origin, tip], axis=0)
        objects.append(scene.draw_debug_trajectory(points, radius=float(axis_radius), color=color))
        objects.append(scene.draw_debug_sphere(tip, radius=float(axis_radius) * 2.8, color=color))
    objects.append(scene.draw_debug_sphere(origin, radius=float(axis_radius) * 3.2, color=(1.0, 1.0, 1.0, 0.95)))
    return objects


def _entity_names(entity: Any) -> tuple[list[str], list[str]]:
    links: list[str] = []
    dofs: list[str] = []
    for attr in ("links", "_links"):
        value = getattr(entity, attr, None)
        if value is not None:
            try:
                links = [str(getattr(item, "name", item)) for item in value]
                break
            except Exception:
                pass
    for attr in ("dofs", "_dofs"):
        value = getattr(entity, attr, None)
        if value is not None:
            try:
                dofs = [str(getattr(item, "name", item)) for item in value]
                break
            except Exception:
                pass
    if not dofs and hasattr(entity, "get_dofs"):
        try:
            dofs = [str(getattr(item, "name", item)) for item in entity.get_dofs()]
        except Exception:
            pass
    if not links and hasattr(entity, "get_links"):
        try:
            links = [str(getattr(item, "name", item)) for item in entity.get_links()]
        except Exception:
            pass
    return links, dofs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_COMBINED_URDF)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--pos", type=_parse_vec3, default=(0.0, 0.0, 0.0), help="URDF world position x,y,z.")
    parser.add_argument(
        "--euler",
        type=_parse_vec3,
        default=(0.0, 0.0, 0.0),
        help="URDF world Euler angles in degrees.",
    )
    parser.add_argument("--fixed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--collision", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--convexify", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--merge-fixed-links", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-plane", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-axes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--axis-length", type=float, default=0.35)
    parser.add_argument("--axis-radius", type=float, default=0.006)
    parser.add_argument("--camera-pos", type=_parse_vec3, default=(1.6, -2.2, 1.4))
    parser.add_argument("--camera-lookat", type=_parse_vec3, default=(0.0, 0.0, 0.55))
    parser.add_argument("--camera-fov", type=float, default=35.0)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many steps; 0 means until viewer closes.")
    args = parser.parse_args()

    urdf = args.urdf.expanduser().resolve()
    if not urdf.exists():
        raise FileNotFoundError(f"combined URDF not found: {urdf}")

    print(
        "[combined-urdf-debug] loading "
        f"urdf={urdf} pos={tuple(float(v) for v in args.pos)} "
        f"euler_deg={tuple(float(v) for v in args.euler)} "
        f"fixed={bool(args.fixed)} collision={bool(args.collision)}",
        flush=True,
    )

    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=args.camera_pos,
            camera_lookat=args.camera_lookat,
            camera_fov=float(args.camera_fov),
            res=(1280, 720),
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0.0, 0.0, 0.0),
            enable_self_collision=False,
            enable_adjacent_collision=False,
        ),
        vis_options=gs.options.VisOptions(show_world_frame=True),
        show_viewer=not bool(args.no_viewer),
    )

    if args.show_plane:
        scene.add_entity(gs.morphs.Plane(), name="world_xy_plane")

    combined = scene.add_entity(
        gs.morphs.URDF(
            file=str(urdf),
            pos=args.pos,
            euler=args.euler,
            fixed=bool(args.fixed),
            collision=bool(args.collision),
            convexify=bool(args.convexify),
            merge_fixed_links=bool(args.merge_fixed_links),
            prioritize_urdf_material=True,
            requires_jac_and_IK=True,
        ),
        material=gs.materials.Rigid(rho=1000.0, friction=1.0),
        name="dual_nero_linker_l10_combined_debug",
    )
    scene.build()

    links, dofs = _entity_names(combined)
    print(
        "[combined-urdf-debug] built "
        f"n_links={len(links)} n_dofs={len(dofs)} "
        f"first_links={links[:12]} first_dofs={dofs[:20]}",
        flush=True,
    )

    if args.show_axes:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.asarray(args.pos, dtype=np.float64)
        _draw_frame_axes(scene, pose, axis_length=float(args.axis_length), axis_radius=float(args.axis_radius))

    step = 0
    while True:
        if args.max_steps > 0 and step >= args.max_steps:
            break
        if not args.no_viewer and not scene.viewer.is_alive():
            break
        scene.step()
        step += 1
        if args.no_viewer:
            time.sleep(0.001)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
