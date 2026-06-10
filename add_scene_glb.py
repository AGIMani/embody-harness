from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import genesis as gs
import numpy as np

from assets.nero_arm_linker_l10_genesis_config import (
    ACTIVE_LINKER_L10_JOINTS,
    INITIAL_LEFT_ARM_Q,
    INITIAL_RIGHT_ARM_Q,
    make_runtime_config,
)


ROOT_DIR = Path(__file__).resolve().parent
NERO_LINKER_CONFIG = make_runtime_config(backend="cpu", show_viewer=False)
DEFAULT_GLB = ROOT_DIR / "scene" / "scene.glb"
DEFAULT_BOTTLE_GLB = ROOT_DIR / "scene" / "bottle.glb"
DEFAULT_CONNECTOR_MESH = ROOT_DIR / "assets" / "connector.STL"
DEFAULT_D455_JSON = ROOT_DIR / "assets" / "d455json.json"
DEFAULT_D405_JSON = ROOT_DIR / "assets" / "d405json.json"
DEFAULT_BASE_MESH = NERO_LINKER_CONFIG.base_mesh
DEFAULT_NERO_URDF = NERO_LINKER_CONFIG.nero_urdf
DEFAULT_PACKAGE_ROOT = NERO_LINKER_CONFIG.package_root
DEFAULT_LINKER_HAND_URDF = NERO_LINKER_CONFIG.linker_hand_urdf
TMP_ROOT = Path(os.environ.get("HARNESS_GENESIS_TMPDIR", f"/tmp/harness_genesis_{os.environ.get('USER', 'user')}"))

DEFAULT_BASE_SCALE = 0.001
DEFAULT_BASE_EULER = (90.0, 0.0, 0.0)
DEFAULT_BASE_FOOT_CENTER_MM = (-51.439, -842.036, -50.0)
DEFAULT_MOUNT_HOLE_YAW_DEG = 90.0
DEFAULT_ARM_LIFT_M = 0.005
FIXED_ASSEMBLY_TRANSLATION = (-0.235556, -0.486667, -0.805556)
FIXED_ASSEMBLY_EULER = (0.0, 0.0, 96.0)
LEFT_ARM_REL_POS_M = (0.252915, 1.078233, 0.193274)
LEFT_ARM_REL_EULER_DEG = (180.0, 0.0, 90.0)
RIGHT_ARM_REL_POS_M = (0.252915, 1.078472, 0.311659)
RIGHT_ARM_REL_EULER_DEG = (0.0, 0.0, 90.0)
DEFAULT_CONNECTOR_SCALE = 0.001
LEFT_CONNECTOR_MOUNT_OFFSET_XYZ = (-0.021481, -0.088889, 0.037778)
LEFT_CONNECTOR_MOUNT_EULER_DEG = (-90.0, 0.0, 0.0)
RIGHT_CONNECTOR_MOUNT_OFFSET_XYZ = (0.022963, 0.089630, 0.037778)
RIGHT_CONNECTOR_MOUNT_EULER_DEG = (-90.0, 0.0, 180.0)
D455_BASE_REL_POS_M = (0.327778, 1.288889, 0.252556)
D455_BASE_REL_EULER_DEG = (-90.0, 0.0, -40.0)
D455_BODY_SIZE_FALLBACK = (0.026, 0.124, 0.029)
D455_RGB_LOCAL_POS_RATIO = (0.5, 0.0, 0.0)
DEFAULT_D455_RGB_GUI = True
D405_BODY_SIZE_FALLBACK = (0.042, 0.042, 0.023)
RIGHT_D405_CONNECTOR_REL_POS_M = (0.022759, -0.004138, 0.013103)
RIGHT_D405_CONNECTOR_REL_EULER_DEG = (79.969, 0.0, 0.0)
D405_CAMERA_LOCAL_POS_RATIO = (0.0, 0.0, 0.5)
D405_CAMERA_FAR_M = 1.0e6
DEFAULT_D405_CAMERA_GUI = True
SILVER_WHITE_METAL_COLOR = (0.86, 0.88, 0.88, 1.0)
SILVER_WHITE_METAL_ROUGHNESS = 0.28
CEILING_AREA_LIGHT_POS = (0.0, 0.0, 2.6)
CEILING_AREA_LIGHT_SIZE = (0.58, 0.58)
CEILING_AREA_LIGHT_COLOR = (1.0, 0.97, 0.90)
CEILING_AREA_LIGHT_EMISSIVE = (4.0, 3.8, 3.4)
CEILING_AREA_LIGHT_POINT_INTENSITY = 18.0
CEILING_AREA_LIGHT_DIRECTIONAL_INTENSITY = 3.0
BOTTLE_X_RANGE = (-0.17112, 0.03209)
BOTTLE_Y_RANGE = (0.19251, 0.41711)
BOTTLE_Z = 0.70
BOTTLE_YAW_RANGE_DEG = (0.0, 360.0)
DEFAULT_GRAVITY = (0.0, 0.0, -9.81)
DEFAULT_TABLE_COLLIDER_POS = (-0.07, 0.305, 0.02)
DEFAULT_TABLE_COLLIDER_SIZE = (0.36, 0.36, 0.04)
RIGHT_SUPPORT_HOLE_Z_MM = -109.0
SUPPORT_HOLES_MM = np.asarray(
    (
        (-86.439, 105.964, 9.0),
        (-16.439, 105.964, 9.0),
        (-86.439, 35.964, 9.0),
        (-16.439, 35.964, 9.0),
    ),
    dtype=np.float64,
)
ARM_HOLES_MM = np.asarray(
    (
        (-35.0, -35.0, 0.0),
        (-35.0, 35.0, 0.0),
        (35.0, -35.0, 0.0),
        (35.0, 35.0, 0.0),
    ),
    dtype=np.float64,
)
DEFAULT_EEF_LINK = "revo2_flange"
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
REVO2_FLANGE_VISUAL_MESH = "package://agx_arm_description/agx_arm_urdf/nero/meshes/dae/revo2_flange.dae"
REVO2_FLANGE_COLLISION_MESH = "package://agx_arm_description/agx_arm_urdf/nero/meshes/revo2_flange.stl"
REVO2_FLANGE_JOINT_XYZ = "0.032 0 -0.0235"
REVO2_FLANGE_JOINT_RPY = "-1.5708 0 -1.5708"


def _vec3(value: str) -> tuple[float, float, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers, e.g. 0,0,0")
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers") from exc


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


def _rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError("rotation axis must be non-zero")
    x, y, z = axis / norm
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    one_c = 1.0 - c
    return np.asarray(
        (
            (c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s),
            (y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s),
            (z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c),
        ),
        dtype=np.float64,
    )


def _quat_wxyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            (
                0.25 * s,
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            )
        )
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        quat = np.array(
            (
                (rotation[2, 1] - rotation[1, 2]) / s,
                0.25 * s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
            )
        )
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        quat = np.array(
            (
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                0.25 * s,
                (rotation[1, 2] + rotation[2, 1]) / s,
            )
        )
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quat = np.array(
            (
                (rotation[1, 0] - rotation[0, 1]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
                (rotation[1, 2] + rotation[2, 1]) / s,
                0.25 * s,
            )
        )
    return (quat / np.linalg.norm(quat)).astype(np.float32)


def _rotation_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm((w, x, y, z)))
    if norm <= 0.0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _quat_multiply_wxyz(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    quat = (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )
    norm = math.sqrt(sum(float(v) * float(v) for v in quat))
    if norm <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(float(v) / norm for v in quat)


def _tensor_to_np(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _pose_from_local_anchor(
    anchor_mm: tuple[float, float, float],
    euler_deg: tuple[float, float, float],
    scale: float,
    world_anchor: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    anchor_local_m = np.asarray(anchor_mm, dtype=np.float64) * scale
    world_anchor_m = np.asarray(world_anchor, dtype=np.float64)
    pos = world_anchor_m - _rotation_from_euler_deg(euler_deg) @ anchor_local_m
    return tuple(float(v) for v in pos)


def _transform_point(
    point: np.ndarray,
    pos: tuple[float, float, float],
    euler_deg: tuple[float, float, float],
) -> np.ndarray:
    return np.asarray(pos, dtype=np.float64) + _rotation_from_euler_deg(euler_deg) @ np.asarray(point, dtype=np.float64)


def _transform_points(
    points_m: np.ndarray,
    pos: tuple[float, float, float],
    euler_deg: tuple[float, float, float],
) -> np.ndarray:
    return np.asarray(pos, dtype=np.float64) + points_m @ _rotation_from_euler_deg(euler_deg).T


def _add_origin(parent: ET.Element, xyz: str, rpy: str) -> ET.Element:
    return ET.SubElement(parent, "origin", {"xyz": xyz, "rpy": rpy})


def _add_mesh_geometry(parent: ET.Element, filename: str) -> None:
    geometry = ET.SubElement(parent, "geometry")
    ET.SubElement(geometry, "mesh", {"filename": filename})


def _make_revo2_flange_urdf(source_urdf: Path) -> Path:
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    if root.find("./link[@name='revo2_flange']") is not None:
        return source_urdf

    flange_link = ET.SubElement(root, "link", {"name": "revo2_flange"})
    inertial = ET.SubElement(flange_link, "inertial")
    _add_origin(inertial, "0.0 0.0 -0.00032", "0 0 0")
    ET.SubElement(inertial, "mass", {"value": "0.04771096"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "2.697e-05",
            "ixy": "0",
            "ixz": "0",
            "iyy": "4.311e-05",
            "iyz": "0",
            "izz": "2.479e-05",
        },
    )

    visual = ET.SubElement(flange_link, "visual")
    _add_origin(visual, "0 0 0", "0 0 0")
    _add_mesh_geometry(visual, REVO2_FLANGE_VISUAL_MESH)

    collision = ET.SubElement(flange_link, "collision")
    _add_origin(collision, "0 0 0", "0 0 0")
    _add_mesh_geometry(collision, REVO2_FLANGE_COLLISION_MESH)

    joint = ET.SubElement(root, "joint", {"name": "revo2_flange_joint", "type": "fixed"})
    _add_origin(joint, REVO2_FLANGE_JOINT_XYZ, REVO2_FLANGE_JOINT_RPY)
    ET.SubElement(joint, "parent", {"link": "link7"})
    ET.SubElement(joint, "child", {"link": "revo2_flange"})

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    out = TMP_ROOT / f"{source_urdf.stem}_with_revo2_flange.urdf"
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


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

    for transmission in list(root.findall("transmission")):
        root.remove(transmission)

    unresolved: list[str] = []
    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        resolved = _resolve_mesh_path(source_urdf, package_root, filename)
        if resolved.startswith("package://"):
            unresolved.append(filename)
        elif resolved != filename:
            mesh.set("filename", resolved)

    if unresolved:
        raise FileNotFoundError("Unresolved URDF mesh paths: " + ", ".join(unresolved))

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    out = TMP_ROOT / f"{source_urdf.stem}_genesis_sanitized.urdf"
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def _sanitize_relative_mesh_urdf(source_urdf: Path) -> Path:
    source_urdf = source_urdf.expanduser().resolve()
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    rewritten = 0
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        if not filename or filename.startswith("package://"):
            continue
        mesh_path = Path(filename)
        if mesh_path.is_absolute():
            continue
        mesh.set("filename", str((source_urdf.parent / mesh_path).resolve()))
        rewritten += 1

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    out = TMP_ROOT / f"{source_urdf.stem}_genesis_abs_mesh.urdf"
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print(
        f"[linker-hand] sanitized URDF relative meshes: {source_urdf} -> {out} rewritten_meshes={rewritten}",
        flush=True,
    )
    return out


def _load_d455_config(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    body_size = tuple(float(v) for v in data.get("body", {}).get("body_size_m_xyz", D455_BODY_SIZE_FALLBACK))
    if len(body_size) != 3:
        raise ValueError(f"{path} body.body_size_m_xyz must contain three numbers")
    rgb_preset = data.get("genesis_presets", {}).get("rgb_native_1280x800_30fps", {})
    return {
        "body_size": body_size,
        "rgb_res": tuple(int(v) for v in rgb_preset.get("res", (1280, 800))),
        "rgb_fov": float(rgb_preset.get("fov", 65.0)),
        "rgb_near": float(rgb_preset.get("near", 0.05)),
        "rgb_far": float(rgb_preset.get("far", 100.0)),
    }


def _load_d405_config(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    body_size = tuple(float(v) for v in data.get("body", {}).get("body_size_m_xyz", D405_BODY_SIZE_FALLBACK))
    if len(body_size) != 3:
        raise ValueError(f"{path} body.body_size_m_xyz must contain three numbers")
    resolution = data.get("resolution", {})
    fov_degrees = data.get("fov_degrees", {})
    clipping_range = data.get("clipping_range_m", {})
    return {
        "body_size": body_size,
        "res": (
            int(resolution.get("width", 1280)),
            int(resolution.get("height", 720)),
        ),
        "fov": float(fov_degrees.get("vertical", 58.0)),
        "near": float(clipping_range.get("near", 0.07)),
        "far": D405_CAMERA_FAR_M,
    }


def _pose_world_from_base_relative(
    *,
    base_pos: np.ndarray,
    base_rotation: np.ndarray,
    rel_pos: tuple[float, float, float],
    rel_euler: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    rel_rotation = _rotation_from_euler_deg(rel_euler)
    world_pos = np.asarray(base_pos, dtype=np.float64) + np.asarray(base_rotation, dtype=np.float64) @ np.asarray(
        rel_pos, dtype=np.float64
    )
    world_rotation = np.asarray(base_rotation, dtype=np.float64) @ rel_rotation
    return world_pos, world_rotation


def _hole_aligned_arm_pose_from_rotation(
    base_pos: tuple[float, float, float],
    base_euler: tuple[float, float, float],
    support_holes_mm: np.ndarray,
    arm_rotation: np.ndarray,
) -> tuple[tuple[float, float, float], np.ndarray]:
    support_holes_m = np.asarray(support_holes_mm, dtype=np.float64) * 0.001
    arm_holes_m = np.asarray(ARM_HOLES_MM, dtype=np.float64) * 0.001
    support_center_m = support_holes_m.mean(axis=0)
    arm_center_m = arm_holes_m.mean(axis=0)
    support_world = _transform_point(support_center_m, base_pos, base_euler)
    arm_world_pos = support_world - np.asarray(arm_rotation, dtype=np.float64) @ arm_center_m
    return tuple(float(v) for v in arm_world_pos), np.asarray(arm_rotation, dtype=np.float64)


def _arm_pose_from_base_relative(
    *,
    base_pos: tuple[float, float, float],
    base_euler: tuple[float, float, float],
    rel_pos: tuple[float, float, float],
    rel_euler: tuple[float, float, float],
) -> tuple[tuple[float, float, float], np.ndarray]:
    base_rotation = _rotation_from_euler_deg(base_euler)
    rel_rotation = _rotation_from_euler_deg(rel_euler)
    world_pos = np.asarray(base_pos, dtype=np.float64) + base_rotation @ np.asarray(rel_pos, dtype=np.float64)
    world_rotation = base_rotation @ rel_rotation
    return tuple(float(v) for v in world_pos), _quat_wxyz_from_rotation(world_rotation)


def _dual_arm_poses(
    *,
    base_pos: tuple[float, float, float],
    base_euler: tuple[float, float, float],
    right_support_hole_z_mm: float,
) -> tuple[tuple[float, float, float], np.ndarray, tuple[float, float, float], np.ndarray]:
    del right_support_hole_z_mm
    left_pos, left_quat = _arm_pose_from_base_relative(
        base_pos=base_pos,
        base_euler=base_euler,
        rel_pos=LEFT_ARM_REL_POS_M,
        rel_euler=LEFT_ARM_REL_EULER_DEG,
    )
    right_pos, right_quat = _arm_pose_from_base_relative(
        base_pos=base_pos,
        base_euler=base_euler,
        rel_pos=RIGHT_ARM_REL_POS_M,
        rel_euler=RIGHT_ARM_REL_EULER_DEG,
    )
    return left_pos, left_quat, right_pos, right_quat


def _add_hole_markers(
    scene: gs.Scene,
    *,
    base_pos: tuple[float, float, float],
    base_euler: tuple[float, float, float],
    right_support_hole_z_mm: float,
    left_pos: tuple[float, float, float],
    left_quat: np.ndarray,
    right_pos: tuple[float, float, float],
    right_quat: np.ndarray,
) -> list[dict[str, object]]:
    marker_items: list[dict[str, object]] = []
    right_support_holes_mm = SUPPORT_HOLES_MM.copy()
    right_support_holes_mm[:, 2] = float(right_support_hole_z_mm)
    specs = (
        (SUPPORT_HOLES_MM, left_pos, _rotation_from_quat_wxyz(left_quat)),
        (right_support_holes_mm, right_pos, _rotation_from_quat_wxyz(right_quat)),
    )
    for support_holes_mm, arm_pos, arm_rotation in specs:
        support_world = _transform_points(np.asarray(support_holes_mm, dtype=np.float64) * 0.001, base_pos, base_euler)
        arm_world = np.asarray(arm_pos, dtype=np.float64) + (ARM_HOLES_MM * 0.001) @ arm_rotation.T
        for pos in support_world:
            marker = scene.add_entity(
                gs.morphs.Sphere(pos=tuple(float(v) for v in pos), radius=0.007, fixed=True, collision=False),
                surface=gs.surfaces.Plastic(color=(1.0, 0.85, 0.05, 1.0)),
            )
            marker_items.append({"entity": marker, "pos": np.asarray(pos, dtype=np.float64), "rotation": np.eye(3)})
        for pos in arm_world:
            marker = scene.add_entity(
                gs.morphs.Sphere(pos=tuple(float(v) for v in pos), radius=0.0045, fixed=True, collision=False),
                surface=gs.surfaces.Plastic(color=(0.0, 1.0, 1.0, 1.0)),
            )
            marker_items.append({"entity": marker, "pos": np.asarray(pos, dtype=np.float64), "rotation": np.eye(3)})
    return marker_items


def _pose_item(entity: object, pos: tuple[float, float, float], rotation: np.ndarray) -> dict[str, object]:
    return {
        "entity": entity,
        "pos": np.asarray(pos, dtype=np.float64),
        "rotation": np.asarray(rotation, dtype=np.float64),
    }


def _set_entity_pose(entity: object, pos: np.ndarray, rotation: np.ndarray) -> None:
    entity.set_pos(np.asarray(pos, dtype=np.float32), zero_velocity=True)
    entity.set_quat(_quat_wxyz_from_rotation(rotation), zero_velocity=True, relative=False)


def _apply_assembly_transform(
    assembly: dict[str, object] | None,
    translation: tuple[float, float, float],
    euler_deg: tuple[float, float, float],
) -> None:
    if not assembly:
        return
    origin = np.asarray(assembly["origin"], dtype=np.float64)
    translation_vec = np.asarray(translation, dtype=np.float64)
    rotation = _rotation_from_euler_deg(euler_deg)
    for item in assembly["pose_items"]:
        original_pos = np.asarray(item["pos"], dtype=np.float64)
        original_rotation = np.asarray(item["rotation"], dtype=np.float64)
        pos = origin + translation_vec + rotation @ (original_pos - origin)
        _set_entity_pose(item["entity"], pos, rotation @ original_rotation)


def _get_joint_dofs(robot: object, joint_name: str) -> list[int]:
    try:
        joint = robot.get_joint(joint_name)
    except Exception:
        return []
    return [int(idx) for idx in getattr(joint, "dofs_idx_local", ())]


def _arm_dofs(robot: object) -> list[int]:
    dofs = [idx for name in ARM_JOINT_NAMES for idx in _get_joint_dofs(robot, name)]
    if len(dofs) != 7:
        raise RuntimeError(f"Expected 7 Nero arm DOFs, got {dofs}")
    return dofs


def _set_arm_initial_pose(robot: object, joint_values: tuple[float, ...]) -> None:
    dofs = _arm_dofs(robot)
    values = np.asarray(joint_values, dtype=np.float32).reshape(7)
    robot.set_dofs_position(values, dofs, zero_velocity=True)
    robot.control_dofs_position(values, dofs)


def _initialize_linker_hand_open_pose(linker_hand: object | None) -> None:
    if linker_hand is None:
        return
    dofs: list[int] = []
    for joint_name in ACTIVE_LINKER_L10_JOINTS:
        dofs.extend(_get_joint_dofs(linker_hand, joint_name))
    if not dofs:
        return
    open_pose = np.zeros(len(dofs), dtype=np.float32)
    linker_hand.set_dofs_position(open_pose, dofs, zero_velocity=True)
    linker_hand.control_dofs_position(open_pose, dofs)


def _mount_linker_hand_to_arm(
    linker_hand: object | None,
    arm: object,
    *,
    eef_link_name: str,
    mount_offset_xyz: tuple[float, float, float],
    mount_quat_wxyz: tuple[float, float, float, float],
) -> None:
    if linker_hand is None:
        return
    eef_link = arm.get_link(eef_link_name)
    eef_pos = _tensor_to_np(eef_link.get_pos()).reshape(3).astype(np.float64)
    eef_quat = tuple(float(v) for v in _tensor_to_np(eef_link.get_quat()).reshape(4))
    mounted_pos = eef_pos + _rotation_from_quat_wxyz(np.asarray(eef_quat)) @ np.asarray(
        mount_offset_xyz, dtype=np.float64
    )
    mounted_quat = _quat_multiply_wxyz(eef_quat, mount_quat_wxyz)
    linker_hand.set_pos(mounted_pos.astype(np.float32), zero_velocity=True)
    linker_hand.set_quat(np.asarray(mounted_quat, dtype=np.float32), zero_velocity=True)


def _mount_entity_to_arm_eef(
    entity: object | None,
    arm: object,
    *,
    eef_link_name: str,
    mount_offset_xyz: tuple[float, float, float],
    mount_euler_deg: tuple[float, float, float],
) -> None:
    if entity is None:
        return
    eef_link = arm.get_link(eef_link_name)
    eef_pos = _tensor_to_np(eef_link.get_pos()).reshape(3).astype(np.float64)
    eef_quat = _tensor_to_np(eef_link.get_quat()).reshape(4).astype(np.float64)
    eef_rotation = _rotation_from_quat_wxyz(eef_quat)
    mounted_pos = eef_pos + eef_rotation @ np.asarray(mount_offset_xyz, dtype=np.float64)
    mounted_rotation = eef_rotation @ _rotation_from_euler_deg(mount_euler_deg)
    _set_entity_pose(entity, mounted_pos, mounted_rotation)


def _mount_connectors_to_arms(assembly: dict[str, object], left_arm: object, right_arm: object) -> None:
    connectors = assembly.get("connectors")
    if not isinstance(connectors, dict):
        return
    eef_link_name = str(assembly.get("eef_link", DEFAULT_EEF_LINK))
    _mount_entity_to_arm_eef(
        connectors.get("left"),
        left_arm,
        eef_link_name=eef_link_name,
        mount_offset_xyz=LEFT_CONNECTOR_MOUNT_OFFSET_XYZ,
        mount_euler_deg=LEFT_CONNECTOR_MOUNT_EULER_DEG,
    )
    _mount_entity_to_arm_eef(
        connectors.get("right"),
        right_arm,
        eef_link_name=eef_link_name,
        mount_offset_xyz=RIGHT_CONNECTOR_MOUNT_OFFSET_XYZ,
        mount_euler_deg=RIGHT_CONNECTOR_MOUNT_EULER_DEG,
    )


def _mount_assembly_attached_parts(assembly: dict[str, object], *, print_linker_status: bool = False) -> None:
    left_arm = assembly["left"]
    right_arm = assembly["right"]
    _mount_connectors_to_arms(assembly, left_arm, right_arm)
    _mount_d455_to_base(assembly)
    _mount_d405_to_right_connector(assembly)

    hand_side = str(assembly.get("linker_hand_side", "right"))
    mount_arm = left_arm if hand_side == "left" else right_arm
    _mount_linker_hand_to_arm(
        assembly.get("linker_hand"),
        mount_arm,
        eef_link_name=str(assembly.get("eef_link", DEFAULT_EEF_LINK)),
        mount_offset_xyz=tuple(float(v) for v in assembly.get("linker_hand_mount_offset_xyz", (0.0, 0.0, 0.0))),
        mount_quat_wxyz=tuple(float(v) for v in assembly.get("linker_hand_mount_quat_wxyz", (1.0, 0.0, 0.0, 0.0))),
    )
    linker_hand = assembly.get("linker_hand")
    if print_linker_status and linker_hand is not None:
        hand_pos = _tensor_to_np(linker_hand.get_pos()).reshape(3)
        print(
            "[nero-linker] mounted "
            f"side={hand_side} "
            f"pos={tuple(round(float(v), 5) for v in hand_pos)}",
            flush=True,
        )


def _mount_d455_to_base(assembly: dict[str, object]) -> None:
    d455 = assembly.get("d455")
    if not isinstance(d455, dict):
        return
    body = d455.get("body")
    camera = d455.get("rgb_camera")
    if body is None and camera is None:
        return

    base = assembly["base"]
    base_pos = _tensor_to_np(base.get_pos()).reshape(3).astype(np.float64)
    base_quat = _tensor_to_np(base.get_quat()).reshape(4).astype(np.float64)
    base_rotation = _rotation_from_quat_wxyz(base_quat)
    d455_pos, d455_rotation = _pose_world_from_base_relative(
        base_pos=base_pos,
        base_rotation=base_rotation,
        rel_pos=D455_BASE_REL_POS_M,
        rel_euler=D455_BASE_REL_EULER_DEG,
    )

    if body is not None:
        _set_entity_pose(body, d455_pos, d455_rotation)

    if camera is not None:
        body_size = tuple(float(v) for v in d455.get("body_size", D455_BODY_SIZE_FALLBACK))
        local_camera_pos = np.asarray(body_size, dtype=np.float64) * np.asarray(D455_RGB_LOCAL_POS_RATIO, dtype=np.float64)
        camera_pos = d455_pos + d455_rotation @ local_camera_pos
        camera_forward = d455_rotation @ np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        camera_up = d455_rotation @ np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        camera.set_pose(
            pos=tuple(float(v) for v in camera_pos),
            lookat=tuple(float(v) for v in camera_pos + camera_forward),
            up=tuple(float(v) for v in camera_up),
        )


def _mount_d405_to_right_connector(assembly: dict[str, object]) -> None:
    d405 = assembly.get("d405")
    connectors = assembly.get("connectors")
    if not isinstance(d405, dict) or not isinstance(connectors, dict):
        return
    right_connector = connectors.get("right")
    body = d405.get("body")
    camera = d405.get("camera")
    if right_connector is None or (body is None and camera is None):
        return

    connector_pos = _tensor_to_np(right_connector.get_pos()).reshape(3).astype(np.float64)
    connector_quat = _tensor_to_np(right_connector.get_quat()).reshape(4).astype(np.float64)
    connector_rotation = _rotation_from_quat_wxyz(connector_quat)
    d405_rotation = connector_rotation @ _rotation_from_euler_deg(RIGHT_D405_CONNECTOR_REL_EULER_DEG)
    d405_pos = connector_pos + connector_rotation @ np.asarray(RIGHT_D405_CONNECTOR_REL_POS_M, dtype=np.float64)

    if body is not None:
        _set_entity_pose(body, d405_pos, d405_rotation)

    if camera is not None:
        body_size = tuple(float(v) for v in d405.get("body_size", D405_BODY_SIZE_FALLBACK))
        camera_local_pos = np.asarray(body_size, dtype=np.float64) * np.asarray(
            D405_CAMERA_LOCAL_POS_RATIO,
            dtype=np.float64,
        )
        camera_pos = d405_pos + d405_rotation @ camera_local_pos
        camera_forward = d405_rotation @ np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        camera_up = d405_rotation @ np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        camera.set_pose(
            pos=tuple(float(v) for v in camera_pos),
            lookat=tuple(float(v) for v in camera_pos + camera_forward),
            up=tuple(float(v) for v in camera_up),
        )


def _render_ego_view(scene: gs.Scene, enabled: bool = True) -> None:
    if not enabled:
        return
    camera = getattr(scene, "d455_rgb_camera", None)
    if camera is None:
        return
    camera.render(rgb=True, depth=False, segmentation=False, normal=False, force_render=True)


def _render_d405_view(scene: gs.Scene, enabled: bool = True) -> None:
    if not enabled:
        return
    camera = getattr(scene, "right_d405_camera", None)
    if camera is None:
        return
    camera.render(rgb=True, depth=False, segmentation=False, normal=False, force_render=True)


def _refresh_scene_attached_parts(scene: gs.Scene) -> None:
    assembly = getattr(scene, "nero_assembly_info", None)
    if isinstance(assembly, dict):
        _mount_assembly_attached_parts(assembly)


def _install_scene_step_attachment_hook(scene: gs.Scene) -> None:
    if getattr(scene, "_harness_step_hook_installed", False):
        return
    raw_step = scene.step

    def step_with_attached_parts(*args, **kwargs):
        result = raw_step(*args, **kwargs)
        _refresh_scene_attached_parts(scene)
        return result

    scene.raw_step_without_attached_parts = raw_step
    scene.step_with_attached_parts = step_with_attached_parts
    scene.refresh_attached_parts = lambda: _refresh_scene_attached_parts(scene)
    scene.step = step_with_attached_parts
    scene._harness_step_hook_installed = True


def _step_scene_with_attached_parts(scene: gs.Scene) -> None:
    step_with_attached_parts = getattr(scene, "step_with_attached_parts", None)
    if callable(step_with_attached_parts):
        step_with_attached_parts()
    else:
        scene.step()
        _refresh_scene_attached_parts(scene)


def _initialize_nero_linker_assembly(scene: gs.Scene, assembly: dict[str, object] | None) -> None:
    if not assembly:
        return
    _apply_assembly_transform(assembly, FIXED_ASSEMBLY_TRANSLATION, FIXED_ASSEMBLY_EULER)
    left_arm = assembly["left"]
    right_arm = assembly["right"]
    _set_arm_initial_pose(left_arm, INITIAL_LEFT_ARM_Q)
    _set_arm_initial_pose(right_arm, INITIAL_RIGHT_ARM_Q)
    _initialize_linker_hand_open_pose(assembly.get("linker_hand"))

    scene.step()
    _mount_assembly_attached_parts(assembly, print_linker_status=True)


def _apply_assembly_debug_pose(
    scene: gs.Scene,
    assembly: dict[str, object] | None,
    translation: tuple[float, float, float],
    euler_deg: tuple[float, float, float],
) -> None:
    if not assembly:
        return
    _apply_assembly_transform(assembly, translation, euler_deg)
    scene.step()
    _mount_assembly_attached_parts(assembly)


def _random_bottle_pose(
    rng: np.random.Generator,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    pos = (
        float(rng.uniform(*BOTTLE_X_RANGE)),
        float(rng.uniform(*BOTTLE_Y_RANGE)),
        BOTTLE_Z,
    )
    euler_deg = (
        0.0,
        0.0,
        float(rng.uniform(*BOTTLE_YAW_RANGE_DEG)),
    )
    return pos, euler_deg


def _apply_bottle_pose(
    bottle_entity: object | None,
    pos: tuple[float, float, float],
    euler_deg: tuple[float, float, float],
) -> None:
    if bottle_entity is None:
        return
    _set_entity_pose(bottle_entity, np.asarray(pos, dtype=np.float64), _rotation_from_euler_deg(euler_deg))


def _base_pose_panel_main(initial_values, values, running, reset_counter, stop_flag) -> None:
    import tkinter as tk
    from tkinter import ttk

    specs = (
        ("x", -1.0, 1.0, "m", 0.001),
        ("y", -1.5, 1.5, "m", 0.001),
        ("z", -1.0, 1.5, "m", 0.001),
        ("roll", -180.0, 180.0, "deg", 0.1),
        ("pitch", -180.0, 180.0, "deg", 0.1),
        ("yaw", -180.0, 180.0, "deg", 0.1),
    )
    sliders = []
    value_labels = []

    def set_value(idx: int, value: float | str, *, update_slider: bool = False) -> None:
        _, lower, upper, unit, _ = specs[idx]
        current = max(float(lower), min(float(upper), float(value)))
        values[idx] = current
        precision = 5 if unit == "m" else 3
        value_labels[idx].config(text=f"{current: .{precision}f}")
        if update_slider:
            sliders[idx].set(current)

    def nudge(idx: int, delta: float) -> None:
        set_value(idx, float(values[idx]) + delta, update_slider=True)

    def set_running(is_running: bool) -> None:
        running.value = bool(is_running)
        start_button.config(text="Pause Step" if running.value else "Start Step")
        status_label.config(text="Running" if running.value else "Paused")

    def toggle_running() -> None:
        set_running(not bool(running.value))

    def reset() -> None:
        set_running(False)
        for idx, value in enumerate(initial_values):
            set_value(idx, value, update_slider=True)
        reset_counter.value += 1

    def print_pose() -> None:
        current = [float(values[idx]) for idx in range(6)]
        print(
            "[base-debug] relative_to_assembly_origin\n"
            f"  translation={tuple(round(v, 6) for v in current[:3])} "
            f"euler_deg={tuple(round(v, 3) for v in current[3:])}\n"
            "  python_constants:\n"
            f"    FIXED_ASSEMBLY_TRANSLATION = ({current[0]:.6f}, {current[1]:.6f}, {current[2]:.6f})\n"
            f"    FIXED_ASSEMBLY_EULER = ({current[3]:.3f}, {current[4]:.3f}, {current[5]:.3f})",
            flush=True,
        )

    def close() -> None:
        stop_flag.value = True
        root.destroy()
        root.quit()

    root = tk.Tk()
    root.title("Base Pose Debug")
    root.geometry("760x430")
    root.minsize(660, 380)

    title = ttk.Label(root, text="Base assembly pose", font=("Arial", 12, "bold"))
    title.pack(fill=tk.X, padx=12, pady=(12, 4))

    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

    for idx, (label, lower, upper, unit, step) in enumerate(specs):
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text=f"{label} ({unit})", width=12).pack(side=tk.LEFT)
        ttk.Button(row, text="-", width=3, command=lambda i=idx, s=step: nudge(i, -s)).pack(side=tk.LEFT)
        slider = ttk.Scale(
            row,
            from_=lower,
            to=upper,
            orient=tk.HORIZONTAL,
            command=lambda value, i=idx: set_value(i, value),
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(row, text="+", width=3, command=lambda i=idx, s=step: nudge(i, s)).pack(side=tk.LEFT)
        value_label = ttk.Label(row, text=f"{float(initial_values[idx]): .5f}", width=12)
        value_label.pack(side=tk.RIGHT)
        sliders.append(slider)
        value_labels.append(value_label)
        set_value(idx, float(initial_values[idx]), update_slider=True)

    buttons = ttk.Frame(root)
    buttons.pack(fill=tk.X, padx=12, pady=(4, 12))
    start_button = ttk.Button(buttons, text="Start Step", command=toggle_running)
    start_button.pack(side=tk.LEFT)
    ttk.Button(buttons, text="Reset Defaults", command=reset).pack(side=tk.LEFT, padx=8)
    ttk.Button(buttons, text="Print Pose", command=print_pose).pack(side=tk.LEFT)
    ttk.Button(buttons, text="Close", command=close).pack(side=tk.RIGHT)
    status_label = ttk.Label(buttons, text="Paused")
    status_label.pack(side=tk.RIGHT, padx=12)

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()


def _create_base_pose_panel(
    enabled: bool,
    initial_translation: tuple[float, float, float],
    initial_euler: tuple[float, float, float],
) -> dict[str, object] | None:
    if not enabled:
        return None
    initial_values = tuple(float(v) for v in (*initial_translation, *initial_euler))
    values = multiprocessing.RawArray("d", initial_values)
    running = multiprocessing.RawValue("b", False)
    reset_counter = multiprocessing.RawValue("i", 0)
    stop_flag = multiprocessing.RawValue("b", False)
    process = multiprocessing.Process(
        target=_base_pose_panel_main,
        args=(initial_values, values, running, reset_counter, stop_flag),
        daemon=True,
    )
    process.start()
    return {
        "values": values,
        "running": running,
        "reset_counter": reset_counter,
        "stop_flag": stop_flag,
        "process": process,
    }


def _shutdown_base_pose_panel(panel: dict[str, object] | None) -> None:
    if not panel:
        return
    panel["stop_flag"].value = True
    process = panel["process"]
    if process.is_alive():
        process.join(timeout=1.0)


def _read_base_pose_panel(
    panel: dict[str, object],
) -> tuple[tuple[float, float, float], tuple[float, float, float], bool, int, bool]:
    values = panel["values"]
    current = tuple(float(values[idx]) for idx in range(6))
    return (
        current[:3],
        current[3:],
        bool(panel["running"].value),
        int(panel["reset_counter"].value),
        bool(panel["stop_flag"].value),
    )


def _add_dual_nero_arm_assembly(
    scene: gs.Scene,
    *,
    base_mesh: Path = DEFAULT_BASE_MESH,
    nero_urdf: Path = DEFAULT_NERO_URDF,
    package_root: Path = DEFAULT_PACKAGE_ROOT,
    linker_hand_urdf: Path | None = DEFAULT_LINKER_HAND_URDF,
    connector_mesh: Path | None = DEFAULT_CONNECTOR_MESH,
    connector_scale: float = DEFAULT_CONNECTOR_SCALE,
    d455_json: Path | None = None,
    d455_rgb_gui: bool = DEFAULT_D455_RGB_GUI,
    d405_json: Path | None = None,
    d405_camera_gui: bool = DEFAULT_D405_CAMERA_GUI,
    linker_hand_side: str = NERO_LINKER_CONFIG.linker_hand_side,
    linker_hand_mount_offset_xyz: tuple[float, float, float] = NERO_LINKER_CONFIG.linker_hand_mount_offset_xyz,
    linker_hand_mount_quat_wxyz: tuple[float, float, float, float] = NERO_LINKER_CONFIG.linker_hand_mount_quat_wxyz,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    base_scale: float = DEFAULT_BASE_SCALE,
    base_euler: tuple[float, float, float] = DEFAULT_BASE_EULER,
    base_foot_center_mm: tuple[float, float, float] = DEFAULT_BASE_FOOT_CENTER_MM,
    right_support_hole_z_mm: float = RIGHT_SUPPORT_HOLE_Z_MM,
    base_collision: bool = False,
    arm_collision: bool = False,
    add_revo2_flange: bool = True,
    show_hole_markers: bool = False,
) -> dict[str, object]:
    base_mesh = base_mesh.expanduser().resolve()
    nero_urdf = nero_urdf.expanduser().resolve()
    package_root = package_root.expanduser().resolve()
    linker_hand_urdf = linker_hand_urdf.expanduser().resolve() if linker_hand_urdf is not None else None
    connector_mesh = connector_mesh.expanduser().resolve() if connector_mesh is not None else None
    d455_json = d455_json.expanduser().resolve() if d455_json is not None else None
    d405_json = d405_json.expanduser().resolve() if d405_json is not None else None
    if not base_mesh.exists():
        raise FileNotFoundError(f"Base mesh not found: {base_mesh}")
    if not nero_urdf.exists():
        raise FileNotFoundError(f"Nero URDF not found: {nero_urdf}")
    if linker_hand_urdf is not None and not linker_hand_urdf.exists():
        raise FileNotFoundError(f"Linker Hand URDF not found: {linker_hand_urdf}")
    if connector_mesh is not None and not connector_mesh.exists():
        raise FileNotFoundError(f"Connector mesh not found: {connector_mesh}")
    d455_config = None
    if d455_json is not None:
        if not d455_json.exists():
            raise FileNotFoundError(f"D455 JSON not found: {d455_json}")
        d455_config = _load_d455_config(d455_json)
    d405_config = None
    if d405_json is not None and connector_mesh is not None:
        if not d405_json.exists():
            raise FileNotFoundError(f"D405 JSON not found: {d405_json}")
        d405_config = _load_d405_config(d405_json)

    base_pos = _pose_from_local_anchor(base_foot_center_mm, base_euler, base_scale, origin)
    urdf_for_genesis = _make_revo2_flange_urdf(nero_urdf) if add_revo2_flange else nero_urdf
    urdf_for_genesis = _sanitize_urdf_for_genesis(urdf_for_genesis, package_root)
    left_pos, left_quat, right_pos, right_quat = _dual_arm_poses(
        base_pos=base_pos,
        base_euler=base_euler,
        right_support_hole_z_mm=right_support_hole_z_mm,
    )

    base = scene.add_entity(
        gs.morphs.Mesh(
            file=str(base_mesh),
            pos=base_pos,
            euler=base_euler,
            scale=base_scale,
            fixed=True,
            collision=base_collision,
            convexify=False,
        ),
        name="dual_nero_base",
    )
    arm_kwargs = {
        "file": str(urdf_for_genesis),
        "fixed": True,
        "collision": arm_collision,
        "convexify": False,
        "merge_fixed_links": False,
        "prioritize_urdf_material": True,
        "requires_jac_and_IK": True,
    }
    if add_revo2_flange:
        arm_kwargs["links_to_keep"] = (DEFAULT_EEF_LINK,)
    left_arm = scene.add_entity(
        gs.morphs.URDF(pos=left_pos, quat=tuple(float(v) for v in left_quat), **arm_kwargs),
        name="left_nero_arm",
    )
    right_arm = scene.add_entity(
        gs.morphs.URDF(pos=right_pos, quat=tuple(float(v) for v in right_quat), **arm_kwargs),
        name="right_nero_arm",
    )
    connectors: dict[str, object] = {}
    if connector_mesh is not None:
        connector_kwargs = {
            "file": str(connector_mesh),
            "pos": (0.0, 0.0, 0.0),
            "euler": (0.0, 0.0, 0.0),
            "scale": float(connector_scale),
            "fixed": True,
            "collision": False,
            "convexify": False,
        }
        connectors["left"] = scene.add_entity(
            gs.morphs.Mesh(**connector_kwargs),
            name="left_connector",
        )
        connectors["right"] = scene.add_entity(
            gs.morphs.Mesh(**connector_kwargs),
            name="right_connector",
        )
    d455: dict[str, object] = {}
    if d455_config is not None:
        d455_body_size = tuple(float(v) for v in d455_config["body_size"])
        d455["body_size"] = d455_body_size
        d455["body"] = scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, 0.0, 0.0),
                size=d455_body_size,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Plastic(color=(0.08, 0.08, 0.08, 1.0), roughness=0.55),
            name="d455_body",
        )
        d455["rgb_camera"] = scene.add_camera(
            model="pinhole",
            res=tuple(int(v) for v in d455_config["rgb_res"]),
            pos=(0.0, 0.0, 0.0),
            lookat=(1.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov=float(d455_config["rgb_fov"]),
            GUI=bool(d455_rgb_gui),
            spp=64,
            near=float(d455_config["rgb_near"]),
            far=float(d455_config["rgb_far"]),
        )
    d405: dict[str, object] = {}
    if d405_config is not None:
        d405_body_size = tuple(float(v) for v in d405_config["body_size"])
        d405["body_size"] = d405_body_size
        d405["body"] = scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, 0.0, 0.0),
                size=d405_body_size,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Aluminium(
                color=SILVER_WHITE_METAL_COLOR,
                roughness=SILVER_WHITE_METAL_ROUGHNESS,
            ),
            name="right_d405_body",
        )
        d405["camera"] = scene.add_camera(
            model="pinhole",
            res=tuple(int(v) for v in d405_config["res"]),
            pos=(0.0, 0.0, 0.0),
            lookat=(0.0, 0.0, 1.0),
            up=(0.0, 1.0, 0.0),
            fov=float(d405_config["fov"]),
            GUI=bool(d405_camera_gui),
            spp=64,
            near=float(d405_config["near"]),
            far=float(d405_config["far"]),
        )
    linker_hand = None
    if linker_hand_urdf is not None:
        linker_hand = scene.add_entity(
            gs.morphs.URDF(
                file=str(_sanitize_relative_mesh_urdf(linker_hand_urdf)),
                fixed=True,
                collision=False,
                convexify=False,
                merge_fixed_links=False,
                prioritize_urdf_material=False,
            ),
            surface=gs.surfaces.Aluminium(
                color=SILVER_WHITE_METAL_COLOR,
                roughness=SILVER_WHITE_METAL_ROUGHNESS,
            ),
            name=f"linkerhand_l10_{linker_hand_side}",
        )

    marker_items: list[dict[str, object]] = []
    if show_hole_markers:
        marker_items = _add_hole_markers(
            scene,
            base_pos=base_pos,
            base_euler=base_euler,
            right_support_hole_z_mm=right_support_hole_z_mm,
            left_pos=left_pos,
            left_quat=left_quat,
            right_pos=right_pos,
            right_quat=right_quat,
        )

    pose_items = [
        _pose_item(base, base_pos, _rotation_from_euler_deg(base_euler)),
        _pose_item(left_arm, left_pos, _rotation_from_quat_wxyz(left_quat)),
        _pose_item(right_arm, right_pos, _rotation_from_quat_wxyz(right_quat)),
        *marker_items,
    ]
    return {
        "base": base,
        "left": left_arm,
        "right": right_arm,
        "connectors": connectors,
        "d455": d455,
        "d405": d405,
        "linker_hand": linker_hand,
        "linker_hand_side": "left" if str(linker_hand_side) == "left" else "right",
        "linker_hand_mount_offset_xyz": tuple(float(v) for v in linker_hand_mount_offset_xyz),
        "linker_hand_mount_quat_wxyz": tuple(float(v) for v in linker_hand_mount_quat_wxyz),
        "eef_link": DEFAULT_EEF_LINK,
        "origin": np.asarray(origin, dtype=np.float64),
        "pose_items": pose_items,
    }


def create_scene(
    glb_path: str | Path = DEFAULT_GLB,
    *,
    show_viewer: bool = True,
    backend: str = "cpu",
    dt: float = 0.01,
    gravity: tuple[float, float, float] = DEFAULT_GRAVITY,
    scale: float | tuple[float, float, float] = 1.0,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
    collision: bool = True,
    fixed: bool = True,
    add_bottle: bool = True,
    bottle_path: str | Path = DEFAULT_BOTTLE_GLB,
    bottle_pos: tuple[float, float, float] | None = None,
    bottle_euler: tuple[float, float, float] | None = None,
    bottle_scale: float | tuple[float, float, float] = 1.0,
    bottle_collision: bool = True,
    seed: int | None = None,
    add_table_collider: bool = True,
    table_collider_pos: tuple[float, float, float] = DEFAULT_TABLE_COLLIDER_POS,
    table_collider_size: tuple[float, float, float] = DEFAULT_TABLE_COLLIDER_SIZE,
    show_table_collider: bool = False,
    add_arm_assembly: bool = True,
    assembly_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    base_mesh: str | Path = DEFAULT_BASE_MESH,
    nero_urdf: str | Path = DEFAULT_NERO_URDF,
    package_root: str | Path = DEFAULT_PACKAGE_ROOT,
    add_linker_hand: bool = True,
    linker_hand_urdf: str | Path | None = DEFAULT_LINKER_HAND_URDF,
    linker_hand_side: str = NERO_LINKER_CONFIG.linker_hand_side,
    add_connectors: bool = True,
    connector_mesh: str | Path | None = DEFAULT_CONNECTOR_MESH,
    connector_scale: float = DEFAULT_CONNECTOR_SCALE,
    add_d455: bool = True,
    d455_json: str | Path | None = DEFAULT_D455_JSON,
    d455_rgb_gui: bool = DEFAULT_D455_RGB_GUI,
    add_d405: bool = True,
    d405_json: str | Path | None = DEFAULT_D405_JSON,
    d405_camera_gui: bool = DEFAULT_D405_CAMERA_GUI,
    base_collision: bool = False,
    arm_collision: bool = False,
    add_revo2_flange: bool = True,
    show_hole_markers: bool = False,
) -> tuple[gs.Scene, gs.Entity]:
    """Create a Genesis scene, add the GLB, and optionally add the dual Nero assembly."""
    glb_path = Path(glb_path).expanduser().resolve()
    if not glb_path.exists():
        raise FileNotFoundError(f"GLB file not found: {glb_path}")
    bottle_path = Path(bottle_path).expanduser().resolve()
    if add_bottle and not bottle_path.exists():
        raise FileNotFoundError(f"Bottle GLB file not found: {bottle_path}")
    if add_bottle and (bottle_pos is None or bottle_euler is None):
        random_pos, random_euler = _random_bottle_pose(np.random.default_rng(seed))
        bottle_pos = random_pos if bottle_pos is None else bottle_pos
        bottle_euler = random_euler if bottle_euler is None else bottle_euler
        print(
            "[bottle-random] "
            f"pos={tuple(round(v, 5) for v in bottle_pos)} "
            f"euler_deg={tuple(round(v, 3) for v in bottle_euler)}",
            flush=True,
        )

    gs.init(backend=gs.gpu if backend == "gpu" else gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -2.2, 1.4),
            camera_lookat=(0.0, 0.0, 0.45),
            camera_fov=35,
            res=(1280, 720),
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=dt, gravity=gravity),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            gravity=gravity,
            enable_self_collision=False,
            enable_adjacent_collision=False,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            ambient_light=(0.16, 0.16, 0.16),
            lights=(
                gs.options.vis.DirectionalLight(
                    dir=(0.0, 0.0, -1.0),
                    color=CEILING_AREA_LIGHT_COLOR,
                    intensity=CEILING_AREA_LIGHT_DIRECTIONAL_INTENSITY,
                ),
                gs.options.vis.PointLight(
                    pos=CEILING_AREA_LIGHT_POS,
                    color=CEILING_AREA_LIGHT_COLOR,
                    intensity=CEILING_AREA_LIGHT_POINT_INTENSITY,
                ),
            ),
        ),
        show_viewer=show_viewer,
    )

    ceiling_area_light_entity = scene.add_entity(
        morph=gs.morphs.Plane(
            pos=CEILING_AREA_LIGHT_POS,
            normal=(0.0, 0.0, -1.0),
            plane_size=CEILING_AREA_LIGHT_SIZE,
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Emission(
            color=CEILING_AREA_LIGHT_EMISSIVE,
        ),
        name="ceiling_area_light_580mm",
    )

    scene_entity = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(glb_path),
            scale=scale,
            pos=pos,
            euler=euler,
            fixed=fixed,
            collision=False,
            convexify=False,
            decimate=False,
        ),
        surface=gs.surfaces.Default(vis_mode="visual"),
        name="scene_glb",
    )

    table_collider_entity = None
    if add_table_collider:
        table_collider_entity = scene.add_entity(
            morph=gs.morphs.Box(
                pos=table_collider_pos,
                size=table_collider_size,
                fixed=True,
                collision=True,
                visualization=show_table_collider,
            ),
            material=gs.materials.Rigid(friction=0.8, coup_restitution=0.0),
            surface=gs.surfaces.Default(color=(0.0, 0.8, 1.0, 0.25), vis_mode="visual"),
            name="table_collider",
        )

    bottle_entity = None
    if add_bottle:
        bottle_entity = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(bottle_path),
                scale=bottle_scale,
                pos=bottle_pos,
                euler=bottle_euler,
                fixed=False,
                collision=bottle_collision,
                convexify=True,
            ),
            material=gs.materials.Rigid(
                rho=950.0,
                friction=0.45,
                coup_friction=0.35,
                coup_restitution=0.0,
            ),
            surface=gs.surfaces.Plastic(
                roughness=0.65,
                metallic=0.0,
            ),
            name="bottle_glb",
        )

    assembly_info = None
    if add_arm_assembly:
        assembly_info = _add_dual_nero_arm_assembly(
            scene,
            base_mesh=Path(base_mesh),
            nero_urdf=Path(nero_urdf),
            package_root=Path(package_root),
            linker_hand_urdf=Path(linker_hand_urdf) if add_linker_hand and linker_hand_urdf is not None else None,
            connector_mesh=Path(connector_mesh) if add_connectors and connector_mesh is not None else None,
            connector_scale=connector_scale,
            d455_json=Path(d455_json) if add_d455 and d455_json is not None else None,
            d455_rgb_gui=d455_rgb_gui,
            d405_json=Path(d405_json) if add_d405 and d405_json is not None else None,
            d405_camera_gui=d405_camera_gui,
            linker_hand_side=linker_hand_side,
            origin=assembly_origin,
            base_collision=base_collision,
            arm_collision=arm_collision,
            add_revo2_flange=add_revo2_flange,
            show_hole_markers=show_hole_markers,
        )

    scene.build()
    _initialize_nero_linker_assembly(scene, assembly_info)
    _apply_bottle_pose(bottle_entity, bottle_pos, bottle_euler)
    scene.nero_assembly_info = assembly_info
    scene.d455_info = assembly_info.get("d455") if isinstance(assembly_info, dict) else None
    scene.d455_rgb_camera = scene.d455_info.get("rgb_camera") if isinstance(scene.d455_info, dict) else None
    scene.d405_info = assembly_info.get("d405") if isinstance(assembly_info, dict) else None
    scene.right_d405_camera = scene.d405_info.get("camera") if isinstance(scene.d405_info, dict) else None
    _install_scene_step_attachment_hook(scene)
    scene.ceiling_area_light_entity = ceiling_area_light_entity
    scene.bottle_entity = bottle_entity
    scene.table_collider_entity = table_collider_entity
    scene.bottle_initial_pos = bottle_pos
    scene.bottle_initial_euler = bottle_euler
    return scene, scene_entity


def main() -> None:
    parser = argparse.ArgumentParser(description="Add scene/scene.glb to a Genesis scene.")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--glb", default=str(DEFAULT_GLB), help="Path to the GLB file.")
    parser.add_argument("--gravity", type=_vec3, default=DEFAULT_GRAVITY, help="Gravity vector as x,y,z in m/s^2.")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform scale for the GLB.")
    parser.add_argument("--pos", type=_vec3, default=(0.0, 0.0, 0.0), help="Position as x,y,z.")
    parser.add_argument("--euler", type=_vec3, default=(0.0, 0.0, 0.0), help="Euler angles in degrees as x,y,z.")
    parser.add_argument(
        "--no-collision",
        action="store_true",
        help="Deprecated compatibility flag; scene.glb is visual-only and table collider is controlled separately.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Deprecated compatibility flag; scene.glb stays fixed.",
    )
    parser.add_argument("--no-bottle", action="store_true", help="Do not load scene/bottle.glb.")
    parser.add_argument("--bottle-glb", type=Path, default=DEFAULT_BOTTLE_GLB)
    parser.add_argument("--bottle-pos", type=_vec3, default=None, help="Override bottle position as x,y,z in meters.")
    parser.add_argument("--bottle-euler", type=_vec3, default=None, help="Override bottle Euler angles in degrees.")
    parser.add_argument("--bottle-scale", type=float, default=1.0, help="Uniform scale for the bottle.")
    parser.add_argument(
        "--bottle-collision",
        dest="bottle_collision",
        action="store_true",
        default=True,
        help="Enable bottle collision (default).",
    )
    parser.add_argument(
        "--no-bottle-collision",
        dest="bottle_collision",
        action="store_false",
        help="Load the bottle as visual-only.",
    )
    parser.add_argument(
        "--no-base-pose-panel",
        action="store_true",
        help="Do not open the base pose debug panel in viewer mode.",
    )
    parser.add_argument(
        "--no-bottle-release-panel",
        dest="no_base_pose_panel",
        action="store_true",
        help="Deprecated alias for --no-base-pose-panel.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for bottle placement.")
    parser.add_argument("--no-arm-assembly", action="store_true", help="Only load the GLB scene.")
    parser.add_argument("--no-table-collider", action="store_true", help="Do not add the table Box collision proxy.")
    parser.add_argument(
        "--table-collider-pos",
        type=_vec3,
        default=DEFAULT_TABLE_COLLIDER_POS,
        help="Table collision box center as x,y,z in meters.",
    )
    parser.add_argument(
        "--table-collider-size",
        type=_vec3,
        default=DEFAULT_TABLE_COLLIDER_SIZE,
        help="Table collision box size as x,y,z in meters.",
    )
    parser.add_argument(
        "--show-table-collider",
        action="store_true",
        help="Render the table collision proxy for alignment debugging.",
    )
    parser.add_argument("--assembly-origin", type=_vec3, default=(0.0, 0.0, 0.0), help="Dual-arm assembly origin as x,y,z.")
    parser.add_argument("--base-mesh", type=Path, default=DEFAULT_BASE_MESH)
    parser.add_argument("--nero-urdf", type=Path, default=DEFAULT_NERO_URDF)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--no-linker-hand", action="store_true", help="Do not mount the Linker Hand L10.")
    parser.add_argument("--linker-hand-urdf", type=Path, default=DEFAULT_LINKER_HAND_URDF)
    parser.add_argument("--linker-hand-side", choices=("left", "right"), default=NERO_LINKER_CONFIG.linker_hand_side)
    parser.add_argument("--no-connectors", action="store_true", help="Do not mount connector.STL on the Nero end-effectors.")
    parser.add_argument("--connector-mesh", type=Path, default=DEFAULT_CONNECTOR_MESH)
    parser.add_argument("--connector-scale", type=float, default=DEFAULT_CONNECTOR_SCALE)
    parser.add_argument("--no-d455", action="store_true", help="Do not mount the fixed D455 body and RGB camera.")
    parser.add_argument("--d455-json", type=Path, default=DEFAULT_D455_JSON)
    parser.add_argument(
        "--d455-rgb-gui",
        dest="d455_rgb_gui",
        action="store_true",
        default=DEFAULT_D455_RGB_GUI,
        help="Open the built-in Genesis GUI window for the D455 RGB ego camera (default).",
    )
    parser.add_argument(
        "--no-d455-rgb-gui",
        dest="d455_rgb_gui",
        action="store_false",
        help="Disable the built-in Genesis GUI window for the D455 RGB ego camera.",
    )
    parser.add_argument("--no-d405", action="store_true", help="Do not mount the D405 camera on the right connector.")
    parser.add_argument("--d405-json", type=Path, default=DEFAULT_D405_JSON)
    parser.add_argument(
        "--d405-camera-gui",
        dest="d405_camera_gui",
        action="store_true",
        default=DEFAULT_D405_CAMERA_GUI,
        help="Open the built-in Genesis GUI window for the right wrist D405 camera (default).",
    )
    parser.add_argument(
        "--no-d405-camera-gui",
        dest="d405_camera_gui",
        action="store_false",
        help="Disable the built-in Genesis GUI window for the right wrist D405 camera.",
    )
    parser.add_argument("--base-collision", action="store_true")
    parser.add_argument("--arm-collision", action="store_true")
    parser.add_argument("--no-revo2-flange", action="store_true")
    parser.add_argument("--show-hole-markers", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Build the scene without opening the viewer.")
    parser.add_argument("--steps", type=int, default=0, help="Simulation steps to run in headless mode.")
    args = parser.parse_args()
    if args.no_collision:
        print("[scene] --no-collision ignored: scene.glb is visual-only; table collision uses --no-table-collider.", flush=True)
    if args.dynamic:
        print("[scene] --dynamic ignored: scene.glb stays fixed as the support surface.", flush=True)

    scene, _ = create_scene(
        args.glb,
        show_viewer=not args.headless,
        backend=args.backend,
        gravity=args.gravity,
        scale=args.scale,
        pos=args.pos,
        euler=args.euler,
        collision=False,
        fixed=True,
        add_bottle=not args.no_bottle,
        bottle_path=args.bottle_glb,
        bottle_pos=args.bottle_pos,
        bottle_euler=args.bottle_euler,
        bottle_scale=args.bottle_scale,
        bottle_collision=args.bottle_collision,
        seed=args.seed,
        add_table_collider=not args.no_table_collider,
        table_collider_pos=args.table_collider_pos,
        table_collider_size=args.table_collider_size,
        show_table_collider=args.show_table_collider,
        add_arm_assembly=not args.no_arm_assembly,
        assembly_origin=args.assembly_origin,
        base_mesh=args.base_mesh,
        nero_urdf=args.nero_urdf,
        package_root=args.package_root,
        add_linker_hand=not args.no_linker_hand,
        linker_hand_urdf=args.linker_hand_urdf,
        linker_hand_side=args.linker_hand_side,
        add_connectors=not args.no_connectors,
        connector_mesh=args.connector_mesh,
        connector_scale=args.connector_scale,
        add_d455=not args.no_d455,
        d455_json=args.d455_json,
        d455_rgb_gui=args.d455_rgb_gui,
        add_d405=not args.no_d405,
        d405_json=args.d405_json,
        d405_camera_gui=args.d405_camera_gui,
        base_collision=args.base_collision,
        arm_collision=args.arm_collision,
        add_revo2_flange=not args.no_revo2_flange,
        show_hole_markers=args.show_hole_markers,
    )

    if args.headless:
        for _ in range(args.steps):
            _step_scene_with_attached_parts(scene)
        return

    base_pose_panel = _create_base_pose_panel(
        not args.no_arm_assembly and not args.no_base_pose_panel,
        FIXED_ASSEMBLY_TRANSLATION,
        FIXED_ASSEMBLY_EULER,
    )
    last_panel_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = (
        (FIXED_ASSEMBLY_TRANSLATION, FIXED_ASSEMBLY_EULER) if base_pose_panel else None
    )
    last_reset_counter = 0
    ego_view_enabled = bool(args.d455_rgb_gui and not args.no_d455)
    d405_view_enabled = bool(args.d405_camera_gui and not args.no_d405)
    try:
        _render_ego_view(scene, ego_view_enabled)
        _render_d405_view(scene, d405_view_enabled)
        while scene.viewer.is_alive():
            if base_pose_panel:
                panel_translation, panel_euler, running, reset_counter, stop_requested = _read_base_pose_panel(
                    base_pose_panel
                )
                panel_pose = (panel_translation, panel_euler)
                if stop_requested:
                    _shutdown_base_pose_panel(base_pose_panel)
                    base_pose_panel = None
                    _step_scene_with_attached_parts(scene)
                elif reset_counter != last_reset_counter or panel_pose != last_panel_pose:
                    reset_requested = reset_counter != last_reset_counter
                    _apply_assembly_debug_pose(
                        scene,
                        scene.nero_assembly_info,
                        panel_translation,
                        panel_euler,
                    )
                    if reset_requested:
                        print(
                            "[base-reset] "
                            f"translation={tuple(round(v, 6) for v in panel_translation)} "
                            f"euler_deg={tuple(round(v, 3) for v in panel_euler)}",
                            flush=True,
                        )
                    last_panel_pose = panel_pose
                    last_reset_counter = reset_counter
                    scene.visualizer.update(force=True)
                elif running:
                    _step_scene_with_attached_parts(scene)
                else:
                    scene.visualizer.update(force=True)
            else:
                _step_scene_with_attached_parts(scene)
            _render_ego_view(scene, ego_view_enabled)
            _render_d405_view(scene, d405_view_enabled)
            time.sleep(1.0 / 60.0)
    finally:
        _shutdown_base_pose_panel(base_pose_panel)


if __name__ == "__main__":
    main()
