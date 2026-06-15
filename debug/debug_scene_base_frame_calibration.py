#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import multiprocessing
import shutil
import sys
import time
from pathlib import Path

import genesis as gs
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import add_scene_glb as harness  # noqa: E402


DEFAULT_OUTPUT = ROOT_DIR / "assets" / "scene_base_frame_debug.json"
DEFAULT_SCENE_POS = (0.0, 0.0, 0.0)
DEFAULT_SCENE_EULER = (0.0, 0.0, 0.0)
DEFAULT_BASE_POS = (0.0, 0.0, 0.0)
DEFAULT_BASE_EULER = (0.0, 0.0, 0.0)
DEFAULT_AXIS_LENGTH = 0.30
DEFAULT_AXIS_RADIUS = 0.006

POS_RANGE_M = 3.0
ROT_RANGE_DEG = 180.0
FINE_POS_STEP_M = 0.001
FINE_ROT_STEP_DEG = 0.1

# Genesis imports GLB/GLTF as glTF-standard Y-up assets and converts mesh
# coordinates into Genesis' Z-up world with (x, y, z) -> (x, -z, y).  The
# calibration sliders operate in the Genesis world, so baking a slider transform
# back into raw GLB vertex data must be conjugated by this conversion.
GENESIS_GLB_IMPORT_YUP_TO_ZUP = np.asarray(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)


def _vec3(text: str) -> tuple[float, float, float]:
    parts = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected numeric x,y,z") from exc


def _format_tuple(values: tuple[float, ...], digits: int = 6) -> str:
    return "(" + ", ".join(f"{float(v):.{digits}f}" for v in values) + ")"


def _pose_matrix(pos: tuple[float, float, float], euler_deg: tuple[float, float, float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = harness._rotation_from_euler_deg(euler_deg)  # noqa: SLF001
    pose[:3, 3] = np.asarray(pos, dtype=np.float64)
    return pose


def _set_mesh_pose(entity: object, pos: tuple[float, float, float], euler_deg: tuple[float, float, float]) -> None:
    harness._set_entity_pose(  # noqa: SLF001
        entity,
        np.asarray(pos, dtype=np.float64),
        harness._rotation_from_euler_deg(euler_deg),  # noqa: SLF001
    )


def _payload_from_values(
    *,
    target: str,
    values: tuple[float, ...],
    scene_glb: Path,
    base_mesh: Path,
    fixed_scene_pos: tuple[float, float, float],
    fixed_scene_euler: tuple[float, float, float],
    fixed_base_pos: tuple[float, float, float],
    fixed_base_euler: tuple[float, float, float],
) -> dict[str, object]:
    target_pos = tuple(float(v) for v in values[:3])
    target_euler = tuple(float(v) for v in values[3:6])
    scene_pos = target_pos if target == "scene" else fixed_scene_pos
    scene_euler = target_euler if target == "scene" else fixed_scene_euler
    base_pos = target_pos if target == "base" else fixed_base_pos
    base_euler = target_euler if target == "base" else fixed_base_euler
    return {
        "description": (
            "Single-target visual frame calibration. Sliders are fixed world XYZ/RPY; "
            "Bake To File writes the current target transform into the selected asset."
        ),
        "target": target,
        "scene_glb": str(scene_glb),
        "base_mesh": str(base_mesh),
        "scene": {
            "pos_m": list(scene_pos),
            "euler_xyz_deg": list(scene_euler),
            "matrix_row_major": _pose_matrix(scene_pos, scene_euler).astype(float).tolist(),
        },
        "base": {
            "pos_m": list(base_pos),
            "euler_xyz_deg": list(base_euler),
            "matrix_row_major": _pose_matrix(base_pos, base_euler).astype(float).tolist(),
        },
    }


def _print_payload(payload: dict[str, object]) -> None:
    target = str(payload["target"])
    scene = payload["scene"]
    base = payload["base"]
    scene_pos = tuple(float(v) for v in scene["pos_m"])
    scene_euler = tuple(float(v) for v in scene["euler_xyz_deg"])
    base_pos = tuple(float(v) for v in base["pos_m"])
    base_euler = tuple(float(v) for v in base["euler_xyz_deg"])
    target_data = scene if target == "scene" else base
    target_pos = tuple(float(v) for v in target_data["pos_m"])
    target_euler = tuple(float(v) for v in target_data["euler_xyz_deg"])
    print("[scene-base-frame-debug]", flush=True)
    print(f"  target={target} pos={_format_tuple(target_pos)} euler_deg={_format_tuple(target_euler, 3)}", flush=True)
    print(f"  scene.glb pos={_format_tuple(scene_pos)} euler_deg={_format_tuple(scene_euler, 3)}", flush=True)
    print(f"  base.STL  pos={_format_tuple(base_pos)} euler_deg={_format_tuple(base_euler, 3)}", flush=True)
    print(
        "  workflow:\n"
        "    1. Run with --target scene, adjust scene, click Bake To File.\n"
        "    2. Restart with --target base, adjust base against the baked scene, click Bake To File.",
        flush=True,
    )


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[scene-base-frame-debug] saved {path}", flush=True)


def _backup_asset(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.bak_{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    print(f"[scene-base-frame-debug] backup={backup}", flush=True)
    return backup


def _gltf_node_matrix(node) -> np.ndarray:
    if node.matrix is not None:
        return np.asarray(node.matrix, dtype=np.float64).reshape(4, 4, order="F")
    translation = np.asarray(node.translation or (0.0, 0.0, 0.0), dtype=np.float64)
    scale = np.asarray(node.scale or (1.0, 1.0, 1.0), dtype=np.float64)
    rotation = np.asarray(node.rotation or (0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    x, y, z, w = rotation / max(float(np.linalg.norm(rotation)), 1.0e-12)
    rot = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rot @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def _gltf_scene_roots(gltf) -> list[int]:
    if gltf.scenes:
        scene_index = int(gltf.scene or 0)
        scene_index = max(0, min(scene_index, len(gltf.scenes) - 1))
        roots = gltf.scenes[scene_index].nodes or []
        if roots:
            return [int(root) for root in roots]
    child_nodes: set[int] = set()
    for node in gltf.nodes or []:
        child_nodes.update(int(child) for child in (node.children or []))
    roots = [idx for idx in range(len(gltf.nodes or [])) if idx not in child_nodes]
    return roots or list(range(len(gltf.nodes or [])))


def _gltf_mesh_node_transforms(gltf) -> dict[int, list[np.ndarray]]:
    transforms: dict[int, list[np.ndarray]] = {}

    def visit(node_index: int, parent_transform: np.ndarray) -> None:
        node = gltf.nodes[node_index]
        node_transform = parent_transform @ _gltf_node_matrix(node)
        if node.mesh is not None:
            transforms.setdefault(int(node.mesh), []).append(node_transform)
        for child in node.children or []:
            visit(int(child), node_transform)

    for root in _gltf_scene_roots(gltf):
        visit(root, np.eye(4, dtype=np.float64))
    return transforms


def _bake_scene_glb(path: Path, transform_m: np.ndarray) -> Path:
    from pygltflib import GLTF2

    backup = _backup_asset(path)
    tmp = path.with_suffix(".frame_bake.tmp.glb")
    gltf = GLTF2().load_binary(str(path))
    blob = bytearray(gltf.binary_blob())
    slider_transform = np.asarray(transform_m, dtype=np.float64).reshape(4, 4)
    genesis_import_transform = GENESIS_GLB_IMPORT_YUP_TO_ZUP
    file_space_slider_transform = np.linalg.inv(genesis_import_transform) @ slider_transform @ genesis_import_transform
    mesh_node_transforms = _gltf_mesh_node_transforms(gltf)
    transformed_positions: dict[int, np.ndarray] = {}
    transformed_normals: dict[int, np.ndarray] = {}
    for mesh_index, mesh in enumerate(gltf.meshes or []):
        node_transforms = mesh_node_transforms.get(mesh_index) or [np.eye(4, dtype=np.float64)]
        if len(node_transforms) > 1:
            first = node_transforms[0]
            if any(not np.allclose(first, other, atol=1.0e-9, rtol=1.0e-9) for other in node_transforms[1:]):
                raise RuntimeError(
                    f"{path} reuses mesh {mesh_index} under multiple different node transforms; "
                    "baking this safely would require duplicating mesh accessors first"
                )
        bake_transform = file_space_slider_transform @ node_transforms[0]
        for primitive in mesh.primitives or []:
            position_accessor = getattr(primitive.attributes, "POSITION", None)
            if position_accessor is not None:
                existing = transformed_positions.get(int(position_accessor))
                if existing is not None and not np.allclose(existing, bake_transform, atol=1.0e-9, rtol=1.0e-9):
                    raise RuntimeError(
                        f"{path} reuses POSITION accessor {position_accessor} with incompatible transforms"
                    )
            if position_accessor is not None and int(position_accessor) not in transformed_positions:
                _transform_gltf_accessor_vec3(
                    gltf,
                    blob,
                    int(position_accessor),
                    bake_transform,
                    is_position=True,
                )
                transformed_positions[int(position_accessor)] = bake_transform
            normal_accessor = getattr(primitive.attributes, "NORMAL", None)
            if normal_accessor is not None:
                existing = transformed_normals.get(int(normal_accessor))
                if existing is not None and not np.allclose(existing, bake_transform, atol=1.0e-9, rtol=1.0e-9):
                    raise RuntimeError(f"{path} reuses NORMAL accessor {normal_accessor} with incompatible transforms")
            if normal_accessor is not None and int(normal_accessor) not in transformed_normals:
                _transform_gltf_accessor_vec3(
                    gltf,
                    blob,
                    int(normal_accessor),
                    bake_transform,
                    is_position=False,
                )
                transformed_normals[int(normal_accessor)] = bake_transform
    if not transformed_positions:
        raise RuntimeError(f"{path} has no POSITION accessors to bake")
    for node in gltf.nodes or []:
        node.matrix = None
        node.translation = None
        node.rotation = None
        node.scale = None
    gltf.set_binary_blob(bytes(blob))
    if gltf.buffers:
        gltf.buffers[0].byteLength = len(blob)
    gltf.save_binary(str(tmp))
    os_replace(tmp, path)
    print(
        "[scene-base-frame-debug] scene GLB baked into vertex data "
        f"positions={len(transformed_positions)} normals={len(transformed_normals)}",
        flush=True,
    )
    return backup


def _transform_gltf_accessor_vec3(gltf, blob: bytearray, accessor_index: int, transform: np.ndarray, *, is_position: bool) -> None:
    accessor = gltf.accessors[accessor_index]
    if accessor.componentType != 5126 or accessor.type != "VEC3":
        raise RuntimeError(
            f"accessor {accessor_index} must be FLOAT VEC3, got componentType={accessor.componentType} type={accessor.type}"
        )
    if accessor.bufferView is None:
        raise RuntimeError(f"accessor {accessor_index} has no bufferView")
    view = gltf.bufferViews[accessor.bufferView]
    if int(view.buffer or 0) != 0:
        raise RuntimeError(f"only single-buffer GLB assets are supported, accessor={accessor_index}")
    count = int(accessor.count)
    start = int(view.byteOffset or 0) + int(accessor.byteOffset or 0)
    stride = int(view.byteStride or 12)
    if stride < 12:
        raise RuntimeError(f"accessor {accessor_index} has invalid stride={stride}")
    arr = np.ndarray(
        shape=(count, 3),
        dtype="<f4",
        buffer=blob,
        offset=start,
        strides=(stride, 4),
    )
    values = np.asarray(arr, dtype=np.float64)
    if is_position:
        homo = np.concatenate([values, np.ones((count, 1), dtype=np.float64)], axis=1)
        transformed = (transform @ homo.T).T[:, :3]
        accessor.min = [float(v) for v in np.min(transformed, axis=0)]
        accessor.max = [float(v) for v in np.max(transformed, axis=0)]
    else:
        linear = transform[:3, :3]
        normal_matrix = np.linalg.inv(linear).T
        transformed = (normal_matrix @ values.T).T
        norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        transformed = transformed / np.maximum(norms, 1.0e-12)
        accessor.min = [float(v) for v in np.min(transformed, axis=0)]
        accessor.max = [float(v) for v in np.max(transformed, axis=0)]
    arr[:] = transformed.astype("<f4")


def _bake_base_stl(path: Path, transform_m: np.ndarray, *, base_scale: float) -> Path:
    import trimesh

    backup = _backup_asset(path)
    mesh = trimesh.load(path, force="mesh")
    if mesh.is_empty:
        raise RuntimeError(f"base mesh is empty: {path}")
    transform_units = np.asarray(transform_m, dtype=np.float64).reshape(4, 4).copy()
    scale = float(base_scale)
    if abs(scale) <= 1.0e-12:
        raise RuntimeError("--base-scale must be non-zero when baking STL")
    transform_units[:3, 3] = transform_units[:3, 3] / scale
    mesh.apply_transform(transform_units)
    tmp = path.with_suffix(".frame_bake.tmp.stl")
    mesh.export(tmp)
    os_replace(tmp, path)
    return backup


def os_replace(src: Path, dst: Path) -> None:
    src.replace(dst)


def _bake_target(
    *,
    target: str,
    scene_glb: Path,
    base_mesh: Path,
    base_scale: float,
    pos: tuple[float, float, float],
    euler: tuple[float, float, float],
) -> Path:
    transform = _pose_matrix(pos, euler)
    if target == "scene":
        backup = _bake_scene_glb(scene_glb, transform)
    elif target == "base":
        backup = _bake_base_stl(base_mesh, transform, base_scale=base_scale)
    else:
        raise ValueError(f"unsupported target: {target}")
    print(
        f"[scene-base-frame-debug] baked target={target} "
        f"pos={_format_tuple(pos)} euler_deg={_format_tuple(euler, 3)}",
        flush=True,
    )
    return backup


def _panel_main(target, initial_values, values, running, print_counter, save_counter, bake_counter, reset_counter, stop_flag):
    import tkinter as tk
    from tkinter import ttk

    specs = (
        ("x", -POS_RANGE_M, POS_RANGE_M, FINE_POS_STEP_M, "m"),
        ("y", -POS_RANGE_M, POS_RANGE_M, FINE_POS_STEP_M, "m"),
        ("z", -POS_RANGE_M, POS_RANGE_M, FINE_POS_STEP_M, "m"),
        ("roll", -ROT_RANGE_DEG, ROT_RANGE_DEG, FINE_ROT_STEP_DEG, "deg"),
        ("pitch", -ROT_RANGE_DEG, ROT_RANGE_DEG, FINE_ROT_STEP_DEG, "deg"),
        ("yaw", -ROT_RANGE_DEG, ROT_RANGE_DEG, FINE_ROT_STEP_DEG, "deg"),
    )
    sliders = []
    value_labels = []

    def set_value(idx: int, value: float | str) -> None:
        _, lower, upper, _, unit = specs[idx]
        current = max(float(lower), min(float(upper), float(value)))
        values[idx] = current
        precision = 5 if unit == "m" else 3
        value_labels[idx].config(text=f"{current: .{precision}f}")

    def step_value(idx: int, direction: int) -> None:
        current = float(sliders[idx].get())
        step = float(specs[idx][3])
        next_value = current + float(direction) * step
        sliders[idx].set(next_value)
        set_value(idx, next_value)

    def set_running(enabled: bool) -> None:
        running.value = bool(enabled)
        start_button.config(text="Pause Step" if running.value else "Start Step")
        status_label.config(text="Stepping" if running.value else "Paused")

    def reset() -> None:
        set_running(False)
        for idx, slider in enumerate(sliders):
            slider.set(float(initial_values[idx]))
            set_value(idx, float(initial_values[idx]))
        reset_counter.value += 1

    def close() -> None:
        stop_flag.value = True
        root.destroy()
        root.quit()

    root = tk.Tk()
    root.title(f"{target.capitalize()} Frame Calibration")
    root.geometry("860x430")
    root.minsize(760, 380)

    ttk.Label(root, text=f"Calibrate {target} frame", font=("Arial", 12, "bold")).pack(fill=tk.X, padx=12, pady=(12, 4))
    ttk.Label(
        root,
        text="Sliders are fixed world XYZ / XYZ Euler. RGB axes: X=red, Y=green, Z=blue.",
    ).pack(fill=tk.X, padx=12, pady=(0, 6))

    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

    for idx, (label, lower, upper, _, unit) in enumerate(specs):
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text=f"{target} {label} ({unit})", width=16).pack(side=tk.LEFT)
        slider = ttk.Scale(row, from_=float(lower), to=float(upper), orient=tk.HORIZONTAL, command=lambda value, i=idx: set_value(i, value))
        slider.set(float(initial_values[idx]))
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(row, text="-", width=3, command=lambda i=idx: step_value(i, -1)).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(row, text="+", width=3, command=lambda i=idx: step_value(i, 1)).pack(side=tk.LEFT)
        value_label = ttk.Label(row, text="", width=12)
        value_label.pack(side=tk.RIGHT, padx=(8, 0))
        sliders.append(slider)
        value_labels.append(value_label)
        set_value(idx, float(initial_values[idx]))

    buttons = ttk.Frame(root)
    buttons.pack(fill=tk.X, padx=12, pady=(4, 12))
    start_button = ttk.Button(buttons, text="Start Step", command=lambda: set_running(not bool(running.value)))
    start_button.pack(side=tk.LEFT)
    ttk.Button(buttons, text="Reset", command=reset).pack(side=tk.LEFT, padx=8)
    ttk.Button(buttons, text="Print", command=lambda: setattr(print_counter, "value", print_counter.value + 1)).pack(side=tk.LEFT)
    ttk.Button(buttons, text="Save JSON", command=lambda: setattr(save_counter, "value", save_counter.value + 1)).pack(side=tk.LEFT, padx=8)
    ttk.Button(buttons, text="Bake To File", command=lambda: setattr(bake_counter, "value", bake_counter.value + 1)).pack(side=tk.LEFT)
    ttk.Button(buttons, text="Close", command=close).pack(side=tk.RIGHT)
    status_label = ttk.Label(buttons, text="Paused")
    status_label.pack(side=tk.RIGHT, padx=12)

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()


def _create_panel(target: str, initial_values: tuple[float, ...]) -> dict[str, object]:
    values = multiprocessing.RawArray("d", initial_values)
    running = multiprocessing.RawValue("b", False)
    print_counter = multiprocessing.RawValue("i", 0)
    save_counter = multiprocessing.RawValue("i", 0)
    bake_counter = multiprocessing.RawValue("i", 0)
    reset_counter = multiprocessing.RawValue("i", 0)
    stop_flag = multiprocessing.RawValue("b", False)
    process = multiprocessing.Process(
        target=_panel_main,
        args=(target, initial_values, values, running, print_counter, save_counter, bake_counter, reset_counter, stop_flag),
        daemon=True,
    )
    process.start()
    return {
        "values": values,
        "running": running,
        "print_counter": print_counter,
        "save_counter": save_counter,
        "bake_counter": bake_counter,
        "reset_counter": reset_counter,
        "stop_flag": stop_flag,
        "process": process,
    }


def _read_values(panel: dict[str, object]) -> tuple[float, ...]:
    return tuple(float(panel["values"][idx]) for idx in range(6))


def _shutdown_panel(panel: dict[str, object] | None) -> None:
    if panel is None:
        return
    panel["stop_flag"].value = True
    process = panel["process"]
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)


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


def _refresh_axes(
    scene: gs.Scene,
    objects: list[object],
    *,
    target_pose: np.ndarray,
    fixed_pose: np.ndarray,
    axis_length: float,
    axis_radius: float,
) -> list[object]:
    for obj in tuple(objects):
        try:
            scene.clear_debug_object(obj)
        except Exception:
            try:
                scene.clear_debug_objects()
            except Exception:
                pass
            break
    new_objects: list[object] = []
    new_objects.extend(_draw_frame_axes(scene, np.eye(4, dtype=np.float64), axis_length=axis_length * 0.9, axis_radius=axis_radius))
    new_objects.extend(_draw_frame_axes(scene, target_pose, axis_length=axis_length, axis_radius=axis_radius))
    new_objects.extend(_draw_frame_axes(scene, fixed_pose, axis_length=axis_length * 0.6, axis_radius=axis_radius * 0.7))
    return new_objects


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-target visual calibration for scene.glb or base.STL frame alignment.")
    parser.add_argument("--target", choices=("scene", "base"), required=True, help="Asset to adjust and bake.")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--scene-glb", type=Path, default=harness.DEFAULT_GLB)
    parser.add_argument("--base-mesh", type=Path, default=harness.DEFAULT_BASE_MESH)
    parser.add_argument("--base-scale", type=float, default=harness.DEFAULT_BASE_SCALE)
    parser.add_argument("--scene-pos", type=_vec3, default=DEFAULT_SCENE_POS, help="Fixed/initial scene pose in world meters.")
    parser.add_argument("--scene-euler", type=_vec3, default=DEFAULT_SCENE_EULER, help="Fixed/initial scene XYZ Euler degrees.")
    parser.add_argument("--base-pos", type=_vec3, default=DEFAULT_BASE_POS, help="Fixed/initial base pose in world meters.")
    parser.add_argument("--base-euler", type=_vec3, default=DEFAULT_BASE_EULER, help="Fixed/initial base XYZ Euler degrees.")
    parser.add_argument("--axis-length", type=float, default=DEFAULT_AXIS_LENGTH)
    parser.add_argument("--axis-radius", type=float, default=DEFAULT_AXIS_RADIUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-viewer", action="store_true", help="Print initial values without opening the panel.")
    parser.add_argument("--start-step", action="store_true", help="Start stepping immediately.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scene_glb = args.scene_glb.expanduser().resolve()
    base_mesh = args.base_mesh.expanduser().resolve()
    if not scene_glb.exists():
        raise FileNotFoundError(f"scene GLB not found: {scene_glb}")
    if not base_mesh.exists():
        raise FileNotFoundError(f"base mesh not found: {base_mesh}")

    target_values = tuple(float(v) for v in ((*args.scene_pos, *args.scene_euler) if args.target == "scene" else (*args.base_pos, *args.base_euler)))
    payload = _payload_from_values(
        target=args.target,
        values=target_values,
        scene_glb=scene_glb,
        base_mesh=base_mesh,
        fixed_scene_pos=args.scene_pos,
        fixed_scene_euler=args.scene_euler,
        fixed_base_pos=args.base_pos,
        fixed_base_euler=args.base_euler,
    )
    _print_payload(payload)
    if args.no_viewer:
        return

    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.4, -2.0, 1.2),
            camera_lookat=(0.0, 0.0, 0.25),
            camera_fov=40,
            res=(1280, 720),
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(dt=0.01, enable_self_collision=False, enable_adjacent_collision=False),
        show_viewer=True,
    )
    scene.add_entity(gs.morphs.Plane(), name="world_xy_plane")
    scene_entity = scene.add_entity(
        gs.morphs.Mesh(
            file=str(scene_glb),
            pos=args.scene_pos,
            euler=args.scene_euler,
            fixed=True,
            collision=False,
            convexify=False,
        ),
        name="scene_glb_frame_debug",
    )
    base_entity = scene.add_entity(
        gs.morphs.Mesh(
            file=str(base_mesh),
            pos=args.base_pos,
            euler=args.base_euler,
            scale=float(args.base_scale),
            fixed=True,
            collision=False,
            convexify=False,
        ),
        name="base_stl_frame_debug",
    )
    scene.build()

    panel = _create_panel(str(args.target), target_values)
    panel["running"].value = bool(args.start_step)
    last_values: tuple[float, ...] | None = None
    last_print_counter = 0
    last_save_counter = 0
    last_bake_counter = 0
    last_reset_counter = 0
    debug_objects: list[object] = []
    try:
        while scene.viewer.is_alive():
            if panel["stop_flag"].value:
                break
            values = _read_values(panel)
            target_pos = tuple(values[0:3])
            target_euler = tuple(values[3:6])
            scene_pos = target_pos if args.target == "scene" else args.scene_pos
            scene_euler = target_euler if args.target == "scene" else args.scene_euler
            base_pos = target_pos if args.target == "base" else args.base_pos
            base_euler = target_euler if args.target == "base" else args.base_euler

            if values != last_values:
                _set_mesh_pose(scene_entity, scene_pos, scene_euler)
                _set_mesh_pose(base_entity, base_pos, base_euler)
                target_pose = _pose_matrix(target_pos, target_euler)
                fixed_pose = _pose_matrix(args.base_pos, args.base_euler) if args.target == "scene" else _pose_matrix(args.scene_pos, args.scene_euler)
                debug_objects = _refresh_axes(
                    scene,
                    debug_objects,
                    target_pose=target_pose,
                    fixed_pose=fixed_pose,
                    axis_length=float(args.axis_length),
                    axis_radius=float(args.axis_radius),
                )
                scene.visualizer.update(force=True)
                last_values = values

            payload = _payload_from_values(
                target=args.target,
                values=values,
                scene_glb=scene_glb,
                base_mesh=base_mesh,
                fixed_scene_pos=args.scene_pos,
                fixed_scene_euler=args.scene_euler,
                fixed_base_pos=args.base_pos,
                fixed_base_euler=args.base_euler,
            )
            if int(panel["print_counter"].value) != last_print_counter:
                last_print_counter = int(panel["print_counter"].value)
                _print_payload(payload)
            if int(panel["save_counter"].value) != last_save_counter:
                last_save_counter = int(panel["save_counter"].value)
                _write_payload(args.output.expanduser().resolve(), payload)
            if int(panel["bake_counter"].value) != last_bake_counter:
                last_bake_counter = int(panel["bake_counter"].value)
                _print_payload(payload)
                backup = _bake_target(
                    target=args.target,
                    scene_glb=scene_glb,
                    base_mesh=base_mesh,
                    base_scale=float(args.base_scale),
                    pos=target_pos,
                    euler=target_euler,
                )
                print(
                    "[scene-base-frame-debug] bake complete. "
                    f"Restart this script with zero {args.target} pose to view baked asset. backup={backup}",
                    flush=True,
                )
                print(
                    "[scene-base-frame-debug] exiting after bake so the next view reloads the asset from disk",
                    flush=True,
                )
                panel["stop_flag"].value = True
                break
            if int(panel["reset_counter"].value) != last_reset_counter:
                last_reset_counter = int(panel["reset_counter"].value)
                print("[scene-base-frame-debug] reset to initial values", flush=True)

            if bool(panel["running"].value):
                scene.step()
            else:
                scene.visualizer.update(force=True)
                time.sleep(1.0 / 60.0)
    finally:
        _shutdown_panel(panel)


if __name__ == "__main__":
    main()
