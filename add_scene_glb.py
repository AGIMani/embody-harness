from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import socket
import time
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import genesis as gs
import numpy as np

from assets.nero_arm_linker_l10_genesis_config import (
    ACTIVE_LINKER_L10_JOINTS,
    INITIAL_LEFT_ARM_Q,
    INITIAL_RIGHT_ARM_Q,
    make_runtime_config,
)
from teleop_stack.session.xr_status import XrTeleopStatusPublisher
from teleop_stack.session.voice_controls import VoiceTeleopControlConfig, VoiceTeleopControlPolicy
from teleop_stack.teleop.openxr_genesis_adapter import (
    adapt_openxr_hand_frame_to_genesis_parent,
    adapt_openxr_hand_frame_to_genesis_wrist_frame,
    map_openxr_quaternion_to_genesis_parent,
    map_openxr_vector_to_genesis,
)
from teleop_stack.teleop.orientation_tracker import OrientationTargetTracker, OrientationTrackerConfig
from teleop_stack.teleop.spatial_frames import (
    BeavrHandFrameSmoother,
    FrameAxes,
    HandAnatomicalFrame,
    hand_anatomical_frame_from_debug,
    hand_beavr_anatomical_frame_from_debug,
    matrix_from_axes,
    matrix_to_quat_xyzw,
    quat_xyzw_to_matrix,
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
D405_CAMERA_NEAR_M = 1.0e-4
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
DEFAULT_OVERLAY_HAND_TRACE_PATH = ROOT_DIR / "logs" / "xr_debug" / "camera_overlay_hand.jsonl"
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
DEFAULT_NERO_ORIENTATION_AXIS_MAP = ("x", "y", "z")
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
POLICY_LINKER_L10_JOINT_NAMES = tuple(ACTIVE_LINKER_L10_JOINTS)
LINKER_L10_HAND_GAIN_TEMPLATES: dict[str, tuple[float, float]] = {
    "thumb_cmc_roll": (3000.0, 300.0),
    "thumb_cmc_yaw": (4000.0, 400.0),
    "thumb_cmc_pitch": (5000.0, 500.0),
    "thumb_mcp": (3500.0, 350.0),
    "thumb_ip": (2500.0, 250.0),
    "mcp_roll": (2500.0, 250.0),
    "mcp_pitch": (5000.0, 500.0),
    "pip": (3000.0, 300.0),
    "dip": (1800.0, 180.0),
}
LINKER_L10_HAND_FORCE_RANGE = 1.0e6
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
    return {
        "body_size": body_size,
        "res": (
            int(resolution.get("width", 1280)),
            int(resolution.get("height", 720)),
        ),
        "fov": float(fov_degrees.get("vertical", 58.0)),
        "near": D405_CAMERA_NEAR_M,
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


def _linker_l10_hand_gains_by_joint() -> dict[str, tuple[float, float]]:
    gains = {
        "thumb_cmc_roll": LINKER_L10_HAND_GAIN_TEMPLATES["thumb_cmc_roll"],
        "thumb_cmc_yaw": LINKER_L10_HAND_GAIN_TEMPLATES["thumb_cmc_yaw"],
        "thumb_cmc_pitch": LINKER_L10_HAND_GAIN_TEMPLATES["thumb_cmc_pitch"],
        "thumb_mcp": LINKER_L10_HAND_GAIN_TEMPLATES["thumb_mcp"],
        "thumb_ip": LINKER_L10_HAND_GAIN_TEMPLATES["thumb_ip"],
    }
    for finger in ("index", "middle", "ring", "pinky"):
        gains[f"{finger}_mcp_roll"] = LINKER_L10_HAND_GAIN_TEMPLATES["mcp_roll"]
        gains[f"{finger}_mcp_pitch"] = LINKER_L10_HAND_GAIN_TEMPLATES["mcp_pitch"]
        gains[f"{finger}_pip"] = LINKER_L10_HAND_GAIN_TEMPLATES["pip"]
        gains[f"{finger}_dip"] = LINKER_L10_HAND_GAIN_TEMPLATES["dip"]
    return gains


def _set_named_dof_gains(robot: object, gains: dict[str, tuple[float, float]], force_range: float) -> None:
    dofs: list[int] = []
    kp: list[float] = []
    kv: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    for joint_name, (joint_kp, joint_kv) in gains.items():
        joint_dofs = _get_joint_dofs(robot, joint_name)
        dofs.extend(joint_dofs)
        kp.extend([float(joint_kp)] * len(joint_dofs))
        kv.extend([float(joint_kv)] * len(joint_dofs))
        lower.extend([-float(force_range)] * len(joint_dofs))
        upper.extend([float(force_range)] * len(joint_dofs))

    if not dofs:
        return
    robot.set_dofs_kp(np.asarray(kp, dtype=np.float32), dofs)
    robot.set_dofs_kv(np.asarray(kv, dtype=np.float32), dofs)
    robot.set_dofs_force_range(np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32), dofs)


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


def _load_linker_hand_mimic_specs(urdf_path: Path | None) -> dict[str, tuple[str, float, float]]:
    if urdf_path is None:
        return {}
    try:
        root = ET.parse(urdf_path.expanduser().resolve()).getroot()
    except Exception:
        return {}
    mimic_by_name: dict[str, tuple[str, float, float]] = {}
    for joint in root.findall("joint"):
        joint_name = joint.attrib.get("name")
        mimic = joint.find("mimic")
        if not joint_name or mimic is None:
            continue
        source_name = mimic.attrib.get("joint")
        if not source_name:
            continue
        mimic_by_name[joint_name] = (
            source_name,
            float(mimic.attrib.get("multiplier", "1.0")),
            float(mimic.attrib.get("offset", "0.0")),
        )
    return mimic_by_name


def _load_linker_hand_joint_limits(urdf_path: Path | None) -> dict[str, tuple[float, float]]:
    if urdf_path is None:
        return {}
    try:
        root = ET.parse(urdf_path.expanduser().resolve()).getroot()
    except Exception:
        return {}
    limits_by_name: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        joint_name = joint.attrib.get("name")
        limit = joint.find("limit")
        if not joint_name or limit is None:
            continue
        limits_by_name[joint_name] = (
            float(limit.attrib.get("lower", "0.0")),
            float(limit.attrib.get("upper", "0.0")),
        )
    return limits_by_name


def _initialize_linker_hand_control_info(assembly: dict[str, object]) -> None:
    linker_hand = assembly.get("linker_hand")
    if linker_hand is None:
        assembly["linker_hand_joint_names"] = []
        assembly["linker_hand_dofs"] = []
        assembly["linker_hand_mimic_by_name"] = {}
        assembly["linker_hand_joint_limits_by_name"] = {}
        assembly["pending_linker_hand_target"] = None
        return

    urdf_value = assembly.get("linker_hand_urdf")
    urdf_path = Path(urdf_value) if urdf_value is not None else None
    mimic_by_name = _load_linker_hand_mimic_specs(urdf_path)
    limits_by_name = _load_linker_hand_joint_limits(urdf_path)
    joint_names = list(ACTIVE_LINKER_L10_JOINTS)
    for mimic_joint_name in mimic_by_name:
        if mimic_joint_name not in joint_names:
            joint_names.append(mimic_joint_name)

    names: list[str] = []
    dofs: list[int] = []
    for joint_name in joint_names:
        joint_dofs = _get_joint_dofs(linker_hand, joint_name)
        if not joint_dofs:
            continue
        names.append(str(joint_name))
        dofs.append(int(joint_dofs[0]))

    assembly["linker_hand_joint_names"] = names
    assembly["linker_hand_dofs"] = dofs
    assembly["linker_hand_mimic_by_name"] = mimic_by_name
    assembly["linker_hand_joint_limits_by_name"] = limits_by_name
    assembly["pending_linker_hand_target"] = None

    if dofs:
        open_pose = np.zeros(len(dofs), dtype=np.float32)
        linker_hand.set_dofs_position(open_pose, dofs, zero_velocity=True)
        linker_hand.control_dofs_position(open_pose, dofs)
        _set_named_dof_gains(linker_hand, _linker_l10_hand_gains_by_joint(), LINKER_L10_HAND_FORCE_RANGE)
    print(
        "[add-scene-linker] hand control ready "
        f"side={assembly.get('linker_hand_side', 'right')} "
        f"active_dofs={len(dofs)}",
        flush=True,
    )


def _set_linker_hand_target(assembly: dict[str, object], side: str, joint_values: object | None) -> None:
    if side != str(assembly.get("linker_hand_side", "right")):
        return
    assembly["pending_linker_hand_target"] = joint_values


def _apply_linker_hand_target(assembly: dict[str, object]) -> None:
    linker_hand = assembly.get("linker_hand")
    if linker_hand is None:
        return
    target = assembly.get("pending_linker_hand_target")
    joint_names = list(assembly.get("linker_hand_joint_names", ()))
    dofs = list(assembly.get("linker_hand_dofs", ()))
    if target is None or not joint_names or not dofs:
        return

    target_names = tuple(getattr(target, "joint_names", ()))
    target_positions = tuple(getattr(target, "joint_positions", ()))
    values_by_name = dict(zip(target_names, target_positions, strict=False))
    mimic_by_name = assembly.get("linker_hand_mimic_by_name", {})
    if isinstance(mimic_by_name, dict):
        for mimic_name, spec in mimic_by_name.items():
            source_name, multiplier, offset = spec
            if mimic_name not in values_by_name and source_name in values_by_name:
                values_by_name[str(mimic_name)] = multiplier * float(values_by_name[source_name]) + offset

    limits_by_name = assembly.get("linker_hand_joint_limits_by_name", {})
    default_limits = (-np.inf, np.inf)
    values = np.asarray(
        [
            float(
                np.clip(
                    values_by_name.get(name, 0.0),
                    *(limits_by_name.get(name, default_limits) if isinstance(limits_by_name, dict) else default_limits),
                )
            )
            for name in joint_names
        ],
        dtype=np.float32,
    )
    linker_hand.control_dofs_position(values, dofs)


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
    _apply_linker_hand_target(assembly)
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
    _initialize_linker_hand_control_info(assembly)

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


def _load_export_env_file(path: Path) -> bool:
    if not path.is_file():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return True


def _check_cloudxr_runtime() -> tuple[bool, str]:
    runtime_dir = os.environ.get("NV_CXR_RUNTIME_DIR")
    if not runtime_dir:
        return False, "NV_CXR_RUNTIME_DIR is not set. Run: source ~/.cloudxr/run/cloudxr.env"
    socket_path = Path(runtime_dir) / "ipc_cloudxr"
    if not socket_path.exists():
        return False, f"CloudXR IPC socket does not exist: {socket_path}"
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.settimeout(1.0)
        client.connect(str(socket_path))
    except OSError as exc:
        return False, f"CloudXR IPC socket is not accepting connections: {socket_path} ({exc})"
    finally:
        client.close()
    return True, f"CloudXR IPC socket is ready: {socket_path}"


def _axis_value(values_xyz: tuple[float, float, float], token: str) -> float:
    token = token.strip().lower()
    sign = -1.0 if token.startswith("-") else 1.0
    axis = token[-1]
    return sign * float(values_xyz[{"x": 0, "y": 1, "z": 2}[axis]])


def _map_vec3_axes(values_xyz: tuple[float, float, float], axis_map: tuple[str, str, str]) -> tuple[float, float, float]:
    return tuple(_axis_value(values_xyz, token) for token in axis_map)  # type: ignore[return-value]


def _xyzw_to_wxyz(quat_xyzw: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = quat_xyzw
    return (float(w), float(x), float(y), float(z))


def _normalize_quat_xyzw(quat_xyzw: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in quat_xyzw))
    if norm <= 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(float(v) / norm for v in quat_xyzw)  # type: ignore[return-value]


def _parse_axis_map(text: str) -> tuple[str, str, str]:
    values = tuple(part.strip().lower() for part in text.split(",") if part.strip())
    if len(values) != 3:
        raise argparse.ArgumentTypeError("axis map must contain 3 comma-separated tokens")
    valid = {"x", "y", "z", "+x", "+y", "+z", "-x", "-y", "-z"}
    invalid = [value for value in values if value not in valid]
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid axis map token(s): {', '.join(invalid)}")
    return values  # type: ignore[return-value]


def _parse_quat4(text: str) -> tuple[float, float, float, float]:
    parts = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("quaternion must contain 4 comma-separated numbers")
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("quaternion must contain 4 comma-separated numbers") from exc


def _vr_arm_pose_command_mode(*, pose_input_mode: str, use_teleop_orientation: bool) -> str:
    if pose_input_mode != "hand_abs":
        return "legacy_retargeted_ee"
    return "raw_wrist_position_full_orientation" if use_teleop_orientation else "raw_wrist_position_fixed_orientation"


def _default_voice_control_port() -> int:
    for name in ("TELEOP_QUEST_VOICE_UDP_PORT", "TELEOP_VOICE_UDP_PORT"):
        raw_value = os.environ.get(name)
        if raw_value:
            return int(raw_value)
    return VoiceTeleopControlConfig().port


def _apply_voice_control_events(robot: _AddSceneNeroTeleopRobot, events) -> bool:
    if not events.commands_seen:
        return False
    if events.estop_requested:
        robot.estop_teleop()
    if events.recenter_requested:
        robot.recenter_teleop()
    if events.clutch_requested:
        robot.enter_clutch()
    if events.resume_requested:
        robot.resume_teleop()
    if events.engage_requested:
        robot.engage_teleop()
    if events.stop_requested:
        robot.disengage_teleop()
    return bool(events.exit_requested)


def _overlay_hand_trace_is_fresh(path: Path, *, max_age_s: float = 2.0) -> bool:
    try:
        return path.is_file() and (time.time() - path.stat().st_mtime) <= float(max_age_s)
    except OSError:
        return False


class _OverlayHandLogTeleopSession:
    def __init__(
        self,
        robot: _AddSceneNeroTeleopRobot,
        *,
        arm_side: str,
        trace_path: Path,
        hand_side: str,
        use_teleop_orientation: bool,
        print_every_n: int,
        stale_after_s: float = 1.0,
    ) -> None:
        self.robot = robot
        self.arm_side = "left" if arm_side == "left" else "right"
        self.trace_path = trace_path.expanduser()
        self.hand_side = str(hand_side)
        self.use_teleop_orientation = bool(use_teleop_orientation)
        self.print_every_n = max(1, int(print_every_n))
        self.stale_after_s = max(0.1, float(stale_after_s))
        self.frame_count = 0
        self._handle = None
        self._latest_sample: dict[str, object] | None = None
        self._last_warn_time_s = 0.0

    def __enter__(self) -> "_OverlayHandLogTeleopSession":
        from teleop_stack.models import GripperCommand, Pose7, SingleArmTeleopCommand  # noqa: F401

        self.robot.connect()
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.trace_path.open("r", encoding="utf-8")
        self._handle.seek(0, os.SEEK_END)
        print(
            "[add-scene-vr] using camera overlay hand log "
            f"path={self.trace_path} hand={self.hand_side} arm={self.arm_side}",
            flush=True,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if self._handle is not None:
                self._handle.close()
        finally:
            self._handle = None
            self.robot.stop()
            self.robot.disconnect()

    def _accept_sample(self, sample: dict[str, object]) -> bool:
        if sample.get("event") != "frame":
            return False
        if self.hand_side != "auto" and str(sample.get("hand")) != self.hand_side:
            return False
        try:
            positions = np.asarray(sample.get("raw_hand_positions_xyz"), dtype=np.float32)
            valid = np.asarray(sample.get("joint_valid"), dtype=np.uint8)
        except Exception:
            return False
        return positions.shape == (26, 3) and valid.shape == (26,) and int(valid.sum()) >= 10

    def _read_latest_sample(self) -> dict[str, object] | None:
        if self._handle is None:
            return None
        for raw_line in self._handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(sample, dict) and self._accept_sample(sample):
                self._latest_sample = sample
        return self._latest_sample

    def _sample_age_s(self, sample: dict[str, object]) -> float | None:
        raw_monotonic = sample.get("monotonic_time_s")
        if isinstance(raw_monotonic, (int, float)):
            return max(0.0, time.monotonic() - float(raw_monotonic))
        raw_time = sample.get("time_s")
        if isinstance(raw_time, (int, float)):
            return max(0.0, time.time() - float(raw_time))
        return None

    @staticmethod
    def _hand_debug_from_sample(sample: dict[str, object]) -> dict[str, object]:
        names = sample.get("raw_hand_joint_names")
        positions = sample.get("raw_hand_positions_xyz")
        orientations = sample.get("raw_hand_orientations_xyzw")
        valid = sample.get("joint_valid")
        return {
            "joint_valid_count": int(sample.get("valid_joint_count", 0)),
            "joint_positions_xyz": positions if isinstance(positions, list) else [],
            "joint_quaternions_xyzw": orientations if isinstance(orientations, list) else [],
            "joint_valid": valid if isinstance(valid, list) else [],
            "joint_names": names if isinstance(names, list) else None,
            "source": "camera_overlay_hand_log",
            "hand": sample.get("hand"),
        }

    def _command_from_sample(self, sample: dict[str, object]):
        from teleop_stack.models import GripperCommand, Pose7, SingleArmTeleopCommand
        from teleop_stack.retargeting.linker_l10_dex_retargeter import retarget_openxr_hand_to_linker_l10_right

        positions = np.asarray(sample["raw_hand_positions_xyz"], dtype=np.float32)
        orientations = np.asarray(sample.get("raw_hand_orientations_xyzw"), dtype=np.float32)
        joint_valid = np.asarray(sample["joint_valid"], dtype=np.uint8)
        wrist_pos = tuple(float(v) for v in positions[1])
        if self.use_teleop_orientation and orientations.shape == (26, 4):
            wrist_quat = _normalize_quat_xyzw(tuple(float(v) for v in orientations[1]))
        else:
            wrist_quat = (0.0, 0.0, 0.0, 1.0)
        try:
            hand_target = retarget_openxr_hand_to_linker_l10_right(
                positions,
                joint_orientations_xyzw=orientations if orientations.shape == (26, 4) else None,
                joint_valid=joint_valid,
            )
        except Exception:
            hand_target = None
        self.frame_count += 1
        return SingleArmTeleopCommand(
            arm_side=self.arm_side,
            ee_target=Pose7(position_xyz=wrist_pos, quaternion_xyzw=wrist_quat),
            gripper=GripperCommand(normalized_position=0.0),
            source_name="camera_overlay_hand_log",
            timestamp_s=float(sample.get("monotonic_time_s", time.monotonic())),
            frame_id=self.frame_count,
            hand_target=hand_target,
        )

    def step(self) -> None:
        sample = self._read_latest_sample()
        now = time.monotonic()
        if sample is None:
            if now - self._last_warn_time_s > 2.0:
                print(f"[add-scene-vr] waiting for overlay hand samples: {self.trace_path}", flush=True)
                self._last_warn_time_s = now
            _step_scene_with_attached_parts(self.robot.scene)
            return
        age_s = self._sample_age_s(sample)
        if age_s is not None and age_s > self.stale_after_s:
            if now - self._last_warn_time_s > 2.0:
                print(
                    "[add-scene-vr] overlay hand samples are stale "
                    f"age_s={age_s:.2f} path={self.trace_path}",
                    flush=True,
                )
                self._last_warn_time_s = now
            _step_scene_with_attached_parts(self.robot.scene)
            return
        command = self._command_from_sample(sample)
        self.robot.update_hand_debug(self._hand_debug_from_sample(sample), timestamp_s=command.timestamp_s)
        self.robot.send_command(command)
        if self.frame_count == 1 or self.frame_count % self.print_every_n == 0:
            print(
                f"[add-scene-vr] overlay frame={self.frame_count} "
                f"hand={sample.get('hand')} valid={sample.get('valid_joint_count')} "
                f"hand_target={'yes' if command.hand_target is not None else 'no'}",
                flush=True,
            )


class _AddSceneNeroTeleopRobot:
    def __init__(
        self,
        scene: gs.Scene,
        *,
        arm_side: str,
        translation_scale_xyz: tuple[float, float, float],
        workspace_origin_xyz: tuple[float, float, float],
        input_axis_map: tuple[str, str, str],
        openxr_coordinate_adapter: str,
        use_teleop_orientation: bool,
        orientation_source: str,
        orientation_axis_map: tuple[str, str, str],
        orientation_max_speed_rad_s: float,
        orientation_tool_offset_wxyz: tuple[float, float, float, float],
        orientation_reference_mode: str,
        openxr_yaw_recenter: bool,
        relative_control: bool,
        drive_ik: bool,
        require_engage: bool,
        print_every_n: int,
        max_solver_iters: int = 32,
        ik_damping: float = 0.02,
        pos_tol: float = 1e-3,
        max_joint_step: float = 0.045,
        min_joint_step: float = 0.001,
    ) -> None:
        self.scene = scene
        self.arm_side = "left" if arm_side == "left" else "right"
        self.translation_scale_xyz = tuple(float(v) for v in translation_scale_xyz)
        self.workspace_origin_xyz = tuple(float(v) for v in workspace_origin_xyz)
        self.input_axis_map = input_axis_map
        self.openxr_coordinate_adapter = "openxr_genesis" if openxr_coordinate_adapter == "openxr_genesis" else "none"
        self.use_teleop_orientation = bool(use_teleop_orientation)
        self.orientation_source = str(orientation_source)
        self.orientation_axis_map = tuple(str(v) for v in orientation_axis_map)
        self.orientation_reference_mode = str(orientation_reference_mode)
        self.openxr_yaw_recenter_enabled = bool(openxr_yaw_recenter)
        self.openxr_yaw_correction_rad: float | None = None
        self.openxr_yaw_recenter_debug: dict[str, object] | None = None
        self.orientation_tracker = (
            OrientationTargetTracker(
                OrientationTrackerConfig(
                    axis_map=self.orientation_axis_map,  # type: ignore[arg-type]
                    max_speed_rad_s=float(orientation_max_speed_rad_s),
                    tool_offset_wxyz=tuple(float(v) for v in orientation_tool_offset_wxyz),  # type: ignore[arg-type]
                    reference_mode=self.orientation_reference_mode,  # type: ignore[arg-type]
                )
            )
            if self.use_teleop_orientation
            else None
        )
        self.relative_control = bool(relative_control)
        self.drive_ik = bool(drive_ik)
        self.require_engage = bool(require_engage)
        self.print_every_n = max(1, int(print_every_n))
        self.solver_args = SimpleNamespace(
            max_solver_iters=int(max_solver_iters),
            ik_damping=float(ik_damping),
            pos_tol=float(pos_tol),
            max_joint_step=float(max_joint_step),
        )
        self.min_joint_step = max(float(min_joint_step), 0.0)
        self.connected = False
        self.command_count = 0
        self.arm = None
        self.eef_link = None
        self.arm_dofs: list[int] = []
        self.q_state: np.ndarray | None = None
        self.human_anchor_xyz: tuple[float, float, float] | None = None
        self.target_anchor_xyz: tuple[float, float, float] | None = None
        self.target_anchor_quaternion_wxyz: tuple[float, float, float, float] | None = None
        self.last_orientation_timestamp_s: float | None = None
        self.last_orientation_source_quaternion_xyzw: tuple[float, float, float, float] | None = None
        self.last_orientation_source_debug: dict[str, object] | None = None
        self.orientation_anchor_source_actual: str | None = None
        self.orientation_debug = None
        self.beavr_hand_frame_smoother = BeavrHandFrameSmoother(moving_average_limit=5)
        self.mode = "ready"
        self.last_event = "initialized"
        self.latest_command = None
        self.latest_hand_debug: dict[str, object] | None = None

    def connect(self) -> None:
        assembly = getattr(self.scene, "nero_assembly_info", None)
        if not isinstance(assembly, dict):
            raise RuntimeError("add_scene_glb scene does not contain a Nero assembly. Remove --no-arm-assembly.")
        self.arm = assembly[self.arm_side]
        self.eef_link = self.arm.get_link(str(assembly.get("eef_link", DEFAULT_EEF_LINK)))
        self.arm_dofs = _arm_dofs(self.arm)
        self.q_state = _tensor_to_np(self.arm.get_qpos()).reshape(-1)[self.arm_dofs].astype(np.float32)
        self.connected = True
        print(
            f"[add-scene-vr] connected side={self.arm_side} "
            f"drive_ik={'on' if self.drive_ik else 'off'} relative={'on' if self.relative_control else 'off'} "
            f"require_engage={'on' if self.require_engage else 'off'} "
            f"openxr_adapter={self.openxr_coordinate_adapter}",
            flush=True,
        )

    def _ensure_connected(self) -> None:
        if not self.connected or self.arm is None or self.eef_link is None or self.q_state is None:
            raise RuntimeError("VR teleop robot is not connected")

    def _map_delta(self, delta_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        source = (
            map_openxr_vector_to_genesis(delta_xyz)
            if self.openxr_coordinate_adapter == "openxr_genesis"
            else delta_xyz
        )
        mapped = _map_vec3_axes(source, self.input_axis_map)
        mapped = self._apply_openxr_yaw_correction_to_vector(mapped)
        return tuple(float(self.translation_scale_xyz[i]) * float(mapped[i]) for i in range(3))  # type: ignore[return-value]

    def _target_position(self, pose) -> tuple[float, float, float]:
        if not self.relative_control:
            source = tuple(float(v) for v in pose.position_xyz)
            if self.openxr_coordinate_adapter == "openxr_genesis":
                source = map_openxr_vector_to_genesis(source)
            mapped = _map_vec3_axes(source, self.input_axis_map)
            mapped = self._apply_openxr_yaw_correction_to_vector(mapped)
            return tuple(
                float(self.workspace_origin_xyz[i]) + float(self.translation_scale_xyz[i]) * float(mapped[i])
                for i in range(3)
            )  # type: ignore[return-value]
        if self.human_anchor_xyz is None or self.target_anchor_xyz is None:
            self._reset_relative_anchor(pose)
        delta = tuple(float(pose.position_xyz[i]) - float(self.human_anchor_xyz[i]) for i in range(3))
        mapped_delta = self._map_delta(delta)  # type: ignore[arg-type]
        return tuple(float(self.target_anchor_xyz[i]) + float(mapped_delta[i]) for i in range(3))  # type: ignore[return-value]

    def update_hand_debug(self, hand_debug: dict[str, object] | None, *, timestamp_s: float) -> None:
        self.latest_hand_debug = hand_debug
        return

    def _current_target_quat_wxyz(self) -> tuple[float, float, float, float]:
        self._ensure_connected()
        return tuple(float(v) for v in _tensor_to_np(self.eef_link.get_quat()).reshape(4))

    @staticmethod
    def _yaw_rotation_matrix(rad: float) -> np.ndarray:
        cos_v = math.cos(float(rad))
        sin_v = math.sin(float(rad))
        return np.asarray(
            (
                (cos_v, -sin_v, 0.0),
                (sin_v, cos_v, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float], ...]:
        return tuple(tuple(float(value) for value in row) for row in np.asarray(matrix, dtype=np.float64))

    def _set_openxr_yaw_correction_from_genesis_forward(
        self,
        forward_xyz: tuple[float, float, float],
        *,
        source: str,
    ) -> bool:
        if not self.openxr_yaw_recenter_enabled or self.openxr_coordinate_adapter != "openxr_genesis":
            self.openxr_yaw_correction_rad = None
            self.openxr_yaw_recenter_debug = {
                "enabled": False,
                "source": source,
                "reason": (
                    "disabled"
                    if not self.openxr_yaw_recenter_enabled
                    else "openxr_coordinate_adapter_not_openxr_genesis"
                ),
                "openxr_coordinate_adapter": self.openxr_coordinate_adapter,
            }
            return False

        measured = np.asarray(forward_xyz, dtype=np.float64).reshape(3)
        measured_xy = np.asarray((measured[0], measured[1]), dtype=np.float64)
        norm_xy = float(np.linalg.norm(measured_xy))
        if norm_xy <= 1e-9:
            self.openxr_yaw_correction_rad = None
            self.openxr_yaw_recenter_debug = {
                "enabled": False,
                "source": source,
                "reason": "forward_axis_horizontal_norm_too_small",
                "measured_forward_xyz": [float(v) for v in measured],
            }
            return False

        measured_xy /= norm_xy
        target_xy = np.asarray((-1.0, 0.0), dtype=np.float64)
        cross_z = float(measured_xy[0] * target_xy[1] - measured_xy[1] * target_xy[0])
        dot = float(np.dot(measured_xy, target_xy))
        yaw_rad = math.atan2(cross_z, dot)
        self.openxr_yaw_correction_rad = float(yaw_rad)
        corrected = self._apply_openxr_yaw_correction_to_vector(tuple(float(v) for v in measured))
        self.openxr_yaw_recenter_debug = {
            "enabled": True,
            "source": source,
            "yaw_correction_rad": float(yaw_rad),
            "yaw_correction_deg": float(math.degrees(yaw_rad)),
            "measured_forward_xyz": [float(v) for v in measured],
            "measured_forward_xy_normalized": [float(v) for v in measured_xy],
            "target_forward_xyz": [-1.0, 0.0, 0.0],
            "corrected_forward_xyz": [float(v) for v in corrected],
        }
        print(
            f"[add-scene-vr] openxr_yaw_recenter "
            f"source={source} yaw_deg={math.degrees(yaw_rad):+.2f} "
            f"measured_forward=({measured[0]:+.3f},{measured[1]:+.3f},{measured[2]:+.3f})",
            flush=True,
        )
        return True

    def _apply_openxr_yaw_correction_to_vector(
        self,
        vector_xyz: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        if self.openxr_yaw_correction_rad is None:
            return tuple(float(v) for v in vector_xyz)
        corrected = self._yaw_rotation_matrix(self.openxr_yaw_correction_rad) @ np.asarray(vector_xyz, dtype=np.float64)
        return (float(corrected[0]), float(corrected[1]), float(corrected[2]))

    def _apply_openxr_yaw_correction_to_quaternion(
        self,
        quaternion_xyzw: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        if self.openxr_yaw_correction_rad is None:
            return tuple(float(v) for v in quaternion_xyzw)
        yaw_matrix = self._yaw_rotation_matrix(self.openxr_yaw_correction_rad)
        rotation = np.asarray(quat_xyzw_to_matrix(quaternion_xyzw), dtype=np.float64)
        return matrix_to_quat_xyzw(self._matrix_tuple(yaw_matrix @ rotation))  # type: ignore[return-value]

    def _apply_openxr_yaw_correction_to_frame(self, frame: HandAnatomicalFrame) -> HandAnatomicalFrame:
        if self.openxr_yaw_correction_rad is None:
            return frame
        axes = FrameAxes(
            x=self._apply_openxr_yaw_correction_to_vector(tuple(float(v) for v in frame.axes.x)),
            y=self._apply_openxr_yaw_correction_to_vector(tuple(float(v) for v in frame.axes.y)),
            z=self._apply_openxr_yaw_correction_to_vector(tuple(float(v) for v in frame.axes.z)),
        )
        return HandAnatomicalFrame(
            origin_xyz=self._apply_openxr_yaw_correction_to_vector(tuple(float(v) for v in frame.origin_xyz)),
            axes=axes,
            quaternion_xyzw=matrix_to_quat_xyzw(matrix_from_axes(axes)),
            handedness_det=float(frame.handedness_det),
            thumb_alignment=float(frame.thumb_alignment),
            legacy_palm_normal_alignment=frame.legacy_palm_normal_alignment,
            construction=f"{frame.construction}_openxr_yaw_recentered",
            raw_axes=frame.raw_axes,
            axis_adapter={
                **(frame.axis_adapter or {}),
                "session_yaw_recenter": "yaw-only Genesis +Z correction; operator front -> robot front",
            },
        )

    def _hand_orientation_frame(
        self,
        requested: str,
        *,
        apply_openxr_yaw_correction: bool = True,
    ) -> HandAnatomicalFrame | None:
        if not isinstance(self.latest_hand_debug, dict):
            return None
        if requested == "hand_anatomical_frame":
            frame = hand_anatomical_frame_from_debug(self.latest_hand_debug)
        elif requested in {"hand_beavr_anatomical_frame", "hand_genesis_wrist_frame"}:
            frame = self.beavr_hand_frame_smoother.update(self.latest_hand_debug)
            if frame is None:
                frame = hand_beavr_anatomical_frame_from_debug(self.latest_hand_debug)
        else:
            return None
        if frame is None:
            return None
        if requested == "hand_genesis_wrist_frame":
            frame = adapt_openxr_hand_frame_to_genesis_wrist_frame(frame)
        elif self.openxr_coordinate_adapter == "openxr_genesis":
            frame = adapt_openxr_hand_frame_to_genesis_parent(frame)
        return self._apply_openxr_yaw_correction_to_frame(frame) if apply_openxr_yaw_correction else frame

    def _recenter_openxr_yaw_from_hand(self, pose, *, source: str) -> bool:
        frame = self._hand_orientation_frame("hand_genesis_wrist_frame", apply_openxr_yaw_correction=False)
        if frame is None:
            source_quat = tuple(float(v) for v in pose.quaternion_xyzw)
            if self.openxr_coordinate_adapter == "openxr_genesis":
                source_quat = map_openxr_quaternion_to_genesis_parent(source_quat)
            forward = quat_xyzw_to_matrix(source_quat)
            return self._set_openxr_yaw_correction_from_genesis_forward(
                (float(forward[0][2]), float(forward[1][2]), float(forward[2][2])),
                source=f"{source}:wrist_quat_fallback",
            )
        return self._set_openxr_yaw_correction_from_genesis_forward(
            tuple(float(v) for v in frame.axes.z),
            source=f"{source}:hand_genesis_wrist_frame_z",
        )

    def _reset_relative_anchor(self, pose) -> None:
        self._ensure_connected()
        if self.openxr_yaw_correction_rad is None:
            self._recenter_openxr_yaw_from_hand(pose, source="anchor")
        current_target = _tensor_to_np(self.eef_link.get_pos()).reshape(3).astype(np.float64)
        self.human_anchor_xyz = tuple(float(v) for v in pose.position_xyz)
        self.target_anchor_xyz = tuple(float(v) for v in current_target)
        self.target_anchor_quaternion_wxyz = self._current_target_quat_wxyz()
        if self.orientation_tracker is not None:
            source_quat_xyzw, source_debug = self._orientation_source_quaternion_xyzw(pose)
            self.orientation_tracker.reset_anchor(source_quat_xyzw, self.target_anchor_quaternion_wxyz)
            self.orientation_debug = None
            self.last_orientation_timestamp_s = None
            self.orientation_anchor_source_actual = str(source_debug.get("actual", "unknown"))
        print(
            f"[add-scene-vr] anchor side={self.arm_side} "
            f"human={tuple(round(v, 4) for v in self.human_anchor_xyz)} "
            f"target={tuple(round(v, 4) for v in self.target_anchor_xyz)} "
            f"orientation={'on' if self.orientation_tracker is not None else 'off'} "
            f"orientation_source={self.orientation_source} "
            f"orientation_reference_mode={self.orientation_reference_mode}",
            flush=True,
        )

    def _orientation_source_quaternion_xyzw(self, command_pose) -> tuple[tuple[float, float, float, float], dict[str, object]]:
        requested = self.orientation_source
        wrist_quat = tuple(float(v) for v in command_pose.quaternion_xyzw)
        if requested == "wrist_quat":
            adapted = (
                map_openxr_quaternion_to_genesis_parent(wrist_quat)
                if self.openxr_coordinate_adapter == "openxr_genesis"
                else wrist_quat
            )
            adapted = self._apply_openxr_yaw_correction_to_quaternion(adapted)
            adapted = _normalize_quat_xyzw(adapted)
            debug = {
                "requested": requested,
                "actual": "wrist_quat",
                "fallback": False,
                "reason": None,
                "openxr_coordinate_adapter": self.openxr_coordinate_adapter,
                "openxr_yaw_recenter": self.openxr_yaw_recenter_debug,
            }
            self.last_orientation_source_quaternion_xyzw = adapted
            self.last_orientation_source_debug = debug
            return adapted, debug

        if requested not in {"hand_anatomical_frame", "hand_beavr_anatomical_frame", "hand_genesis_wrist_frame"}:
            raise ValueError(f"unsupported VR orientation source: {requested!r}")

        frame = self._hand_orientation_frame(requested)
        if frame is not None:
            quat = _normalize_quat_xyzw(tuple(float(v) for v in frame.quaternion_xyzw))
            debug = {
                "requested": requested,
                "actual": requested,
                "fallback": False,
                "reason": None,
                "openxr_coordinate_adapter": self.openxr_coordinate_adapter,
                "openxr_yaw_recenter": self.openxr_yaw_recenter_debug,
                requested: frame.as_dict(),
            }
            self.last_orientation_source_quaternion_xyzw = quat
            self.last_orientation_source_debug = debug
            return quat, debug

        if self.last_orientation_source_quaternion_xyzw is not None:
            debug = {
                "requested": requested,
                "actual": f"last_{requested}",
                "fallback": True,
                "reason": f"{requested}_unavailable",
                "openxr_coordinate_adapter": self.openxr_coordinate_adapter,
                "openxr_yaw_recenter": self.openxr_yaw_recenter_debug,
            }
            self.last_orientation_source_debug = debug
            return self.last_orientation_source_quaternion_xyzw, debug

        adapted = (
            map_openxr_quaternion_to_genesis_parent(wrist_quat)
            if self.openxr_coordinate_adapter == "openxr_genesis"
            else wrist_quat
        )
        adapted = self._apply_openxr_yaw_correction_to_quaternion(adapted)
        adapted = _normalize_quat_xyzw(adapted)
        debug = {
            "requested": requested,
            "actual": "wrist_quat",
            "fallback": True,
            "reason": f"{requested}_unavailable",
            "openxr_coordinate_adapter": self.openxr_coordinate_adapter,
            "openxr_yaw_recenter": self.openxr_yaw_recenter_debug,
        }
        self.last_orientation_source_quaternion_xyzw = adapted
        self.last_orientation_source_debug = debug
        return adapted, debug

    def xr_status_snapshot(self) -> dict[str, object]:
        input_tracking_state = "tracked" if self.latest_command is not None else "missing"
        hand_valid_count = 0
        if isinstance(self.latest_hand_debug, dict):
            try:
                hand_valid_count = int(self.latest_hand_debug.get("joint_valid_count", 0))
            except (TypeError, ValueError):
                hand_valid_count = 0
            if hand_valid_count >= 10:
                input_tracking_state = "tracked"
        return {
            "mode": self.mode,
            "last_event": self.last_event,
            "input_tracking_state": input_tracking_state,
            "hand_pose_gate_state": "stable" if input_tracking_state == "tracked" else "reacquiring",
            "mapper_control_profile": "voice",
            "controller_available": False,
            "guard_events": (),
            "arm_side": self.arm_side,
            "hand_valid_count": hand_valid_count,
        }

    def _target_quat_wxyz(self, command) -> np.ndarray:
        if self.orientation_tracker is None:
            return _tensor_to_np(self.eef_link.get_quat()).reshape(4).astype(np.float32)

        if self.target_anchor_quaternion_wxyz is None:
            self._reset_relative_anchor(command.ee_target)

        source_quat_xyzw, source_debug = self._orientation_source_quaternion_xyzw(command.ee_target)
        source_actual = str(source_debug.get("actual", "unknown"))
        if (
            self.orientation_anchor_source_actual is not None
            and source_actual in {"hand_anatomical_frame", "hand_beavr_anatomical_frame", "hand_genesis_wrist_frame"}
            and self.orientation_anchor_source_actual != source_actual
        ):
            self._reset_relative_anchor(command.ee_target)
            source_quat_xyzw, source_debug = self._orientation_source_quaternion_xyzw(command.ee_target)
            source_actual = str(source_debug.get("actual", "unknown"))

        timestamp_s = float(getattr(command, "timestamp_s", time.monotonic()))
        if self.last_orientation_timestamp_s is None:
            dt_s = 1.0 / 60.0
        else:
            dt_s = max(0.0, timestamp_s - float(self.last_orientation_timestamp_s))
            if dt_s <= 0.0:
                dt_s = 1.0 / 60.0
        self.orientation_debug = self.orientation_tracker.update(source_quat_xyzw, dt_s=dt_s)
        self.last_orientation_timestamp_s = timestamp_s
        self.orientation_anchor_source_actual = source_actual
        return np.asarray(self.orientation_debug.cmd_target_quat_wxyz, dtype=np.float32)

    def _current_human_pose(self):
        return getattr(self.latest_command, "ee_target", None)

    def _clear_relative_anchor(self) -> None:
        self.human_anchor_xyz = None
        self.target_anchor_xyz = None
        self.target_anchor_quaternion_wxyz = None
        self.last_orientation_timestamp_s = None
        self.orientation_anchor_source_actual = None
        self.orientation_debug = None

    def engage_teleop(self) -> None:
        pose = self._current_human_pose()
        if pose is None:
            self.last_event = "engage_waiting_for_hand"
            print("[add-scene-vr] engage_waiting_for_hand", flush=True)
            return
        self._clear_relative_anchor()
        self.beavr_hand_frame_smoother.reset()
        self._recenter_openxr_yaw_from_hand(pose, source="engage")
        self._target_position(pose)
        self.mode = "engaged"
        self.last_event = "engaged"
        print(f"[add-scene-vr] engage_teleop mode={self.mode}", flush=True)

    def enter_clutch(self) -> None:
        if self.mode == "engaged":
            self.mode = "clutched"
            self.last_event = "entered_clutch"
        print(f"[add-scene-vr] enter_clutch mode={self.mode}", flush=True)

    def resume_teleop(self) -> None:
        self.engage_teleop()
        if self.mode == "engaged":
            self.last_event = "resumed_from_clutch"

    def recenter_teleop(self) -> None:
        self._ensure_connected()
        initial_q = INITIAL_LEFT_ARM_Q if self.arm_side == "left" else INITIAL_RIGHT_ARM_Q
        _set_arm_initial_pose(self.arm, initial_q)
        self.q_state = _tensor_to_np(self.arm.get_qpos()).reshape(-1)[self.arm_dofs].astype(np.float32)
        self._clear_relative_anchor()
        self.beavr_hand_frame_smoother.reset()
        pose = self._current_human_pose()
        if pose is not None:
            self._recenter_openxr_yaw_from_hand(pose, source="recenter")
        self.mode = "ready" if self.require_engage else "engaged"
        self.last_event = "recentered"
        _step_scene_with_attached_parts(self.scene)
        print(f"[add-scene-vr] recenter_teleop mode={self.mode}", flush=True)

    def disengage_teleop(self) -> None:
        self.mode = "ready"
        self.last_event = "disengaged"
        self._clear_relative_anchor()
        self.beavr_hand_frame_smoother.reset()
        print("[add-scene-vr] disengage_teleop mode=ready", flush=True)

    def estop_teleop(self) -> None:
        self.mode = "fault"
        self.last_event = "estop"
        print("[add-scene-vr] estop mode=fault", flush=True)

    def send_command(self, command) -> None:
        self._ensure_connected()
        self.latest_command = command
        if self.require_engage and self.mode != "engaged":
            _step_scene_with_attached_parts(self.scene)
            self.command_count += 1
            if self.command_count == 1 or self.command_count % self.print_every_n == 0:
                print(
                    f"[add-scene-vr] frame={command.frame_id} mode={self.mode} "
                    f"event={self.last_event} holding_until_voice_start",
                    flush=True,
                )
            return
        assembly = getattr(self.scene, "nero_assembly_info", None)
        if isinstance(assembly, dict) and getattr(command, "hand_target", None) is not None:
            _set_linker_hand_target(assembly, self.arm_side, command.hand_target)
        target_pos = np.asarray(self._target_position(command.ee_target), dtype=np.float32)
        target_quat = self._target_quat_wxyz(command)

        if self.drive_ik:
            qpos_init = _tensor_to_np(self.arm.get_qpos()).reshape(-1).astype(np.float32)
            qpos, error = self.arm.inverse_kinematics(
                link=self.eef_link,
                pos=target_pos,
                quat=target_quat,
                init_qpos=qpos_init,
                dofs_idx_local=self.arm_dofs,
                max_samples=1,
                max_solver_iters=int(self.solver_args.max_solver_iters),
                damping=float(self.solver_args.ik_damping),
                pos_tol=float(self.solver_args.pos_tol),
                max_step_size=float(self.solver_args.max_joint_step),
                return_error=True,
            )
            solved = _tensor_to_np(qpos).reshape(-1)[self.arm_dofs].astype(np.float32)
            dq = np.clip(
                solved - self.q_state,
                -float(self.solver_args.max_joint_step),
                float(self.solver_args.max_joint_step),
            )
            if self.min_joint_step > 0.0:
                dq = np.where(np.abs(dq) < self.min_joint_step, 0.0, dq)
            self.q_state = self.q_state + dq
            self.arm.set_dofs_position(self.q_state, self.arm_dofs, zero_velocity=True)
            self.arm.control_dofs_position(self.q_state, self.arm_dofs)
            error_vec = _tensor_to_np(error).reshape(-1)
        else:
            error_vec = np.zeros(3, dtype=np.float32)

        _step_scene_with_attached_parts(self.scene)
        self.command_count += 1
        if self.command_count == 1 or self.command_count % self.print_every_n == 0:
            eef_pos = _tensor_to_np(self.eef_link.get_pos()).reshape(3)
            print(
                f"[add-scene-vr] frame={command.frame_id} side={self.arm_side} "
                f"target={tuple(round(float(v), 4) for v in target_pos)} "
                f"eef={tuple(round(float(v), 4) for v in eef_pos)} "
                f"ik_error={tuple(round(float(v), 5) for v in error_vec)} "
                f"gripper={command.gripper.normalized_position:.3f} "
                f"hand_target={'yes' if getattr(command, 'hand_target', None) is not None else 'no'}",
                flush=True,
            )

    def stop(self) -> None:
        return

    def disconnect(self) -> None:
        self.connected = False


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
    linker_hand_collision: bool = False,
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
                collision=bool(linker_hand_collision),
                convexify=bool(linker_hand_collision),
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
        "linker_hand_urdf": linker_hand_urdf,
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
    collision: bool = False,
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
    linker_hand_collision: bool = False,
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
            collision=bool(collision),
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
            linker_hand_collision=linker_hand_collision,
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
    parser.add_argument(
        "--linker-l10-retargeter",
        choices=("heuristic", "dex_vector", "dex_position", "dex_dexpilot", "l10_adaptive", "holo_layered"),
        default=os.environ.get("TELEOP_LINKER_L10_RETARGETER", "holo_layered"),
        help="Linker L10 hand retargeter. Default matches the remote Nero setup: holo_layered.",
    )
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
    parser.add_argument("--enable-vr-teleop", action="store_true", help="Drive this add_scene_glb Nero assembly from Quest/OpenXR.")
    parser.add_argument("--vr-arm-side", choices=("left", "right"), default="right")
    parser.add_argument("--vr-pose-input-mode", choices=("controller_abs", "hand_abs"), default="hand_abs")
    parser.add_argument("--vr-markers-only", action="store_true", help="Receive VR targets but do not solve IK or move the arm.")
    parser.add_argument("--vr-loop-hz", type=float, default=60.0)
    parser.add_argument("--vr-print-every", type=int, default=30)
    parser.add_argument("--vr-isaac-teleop-root", default=None)
    parser.add_argument("--vr-startup-timeout-s", type=float, default=300.0)
    parser.add_argument("--vr-teleop-trace-path", default=None)
    parser.add_argument("--vr-translation-scale-xyz", type=_vec3, default=(0.15, 0.15, 0.15))
    parser.add_argument("--vr-workspace-origin-xyz", type=_vec3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--vr-input-axis-map", type=_parse_axis_map, default=_parse_axis_map("x,y,z"))
    parser.add_argument("--vr-openxr-coordinate-adapter", choices=("none", "openxr_genesis"), default="openxr_genesis")
    parser.add_argument("--vr-openxr-yaw-recenter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--vr-input-source",
        choices=("auto", "quest", "overlay-log"),
        default="auto",
        help="VR input source. auto uses camera overlay hand JSONL when it is fresh, otherwise QuestRobotSession.",
    )
    parser.add_argument("--vr-overlay-hand-trace-path", type=Path, default=DEFAULT_OVERLAY_HAND_TRACE_PATH)
    parser.add_argument("--vr-overlay-hand-side", choices=("auto", "left", "right"), default="auto")
    parser.add_argument("--vr-overlay-stale-after-s", type=float, default=1.0)
    parser.add_argument("--vr-use-teleop-orientation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--vr-orientation-source",
        choices=("wrist_quat", "hand_anatomical_frame", "hand_beavr_anatomical_frame", "hand_genesis_wrist_frame"),
        default="hand_genesis_wrist_frame",
    )
    parser.add_argument("--vr-orientation-axis-map", type=_parse_axis_map, default=DEFAULT_NERO_ORIENTATION_AXIS_MAP)
    parser.add_argument("--vr-orientation-max-speed-rad-s", type=float, default=3.0)
    parser.add_argument("--vr-orientation-tool-offset-wxyz", type=_parse_quat4, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument(
        "--vr-orientation-reference-mode",
        choices=("world_delta", "tool_local_delta", "calibrated_tool_local"),
        default="calibrated_tool_local",
    )
    parser.add_argument("--vr-palm-plane-wrist-orientation-blend-alpha", type=float, default=1.0)
    parser.add_argument("--vr-enable-voice-controls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--vr-require-engage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require voice/controller engage before following wrist targets.",
    )
    parser.add_argument(
        "--vr-voice-control-host",
        default=os.environ.get("TELEOP_QUEST_VOICE_UDP_HOST", os.environ.get("TELEOP_VOICE_UDP_HOST", "127.0.0.1")),
    )
    parser.add_argument("--vr-voice-control-port", type=int, default=_default_voice_control_port())
    parser.add_argument("--vr-xr-status-path", default=None, help="Optional teleop_xr_status.json path used by the VR overlay.")
    parser.add_argument("--vr-disable-synthetic-hands-plugin", action="store_true")
    parser.add_argument(
        "--vr-cloudxr-env-path",
        type=Path,
        default=Path.home() / ".cloudxr" / "run" / "cloudxr.env",
        help="CloudXR env file to auto-load before starting OpenXR.",
    )
    parser.add_argument("--vr-no-auto-cloudxr-env", action="store_true")
    parser.add_argument("--vr-no-cloudxr-preflight", action="store_true")
    parser.add_argument("--vr-absolute-control", action="store_true", help="Map raw Quest position directly instead of using a relative anchor.")
    parser.add_argument("--headless", action="store_true", help="Build the scene without opening the viewer.")
    parser.add_argument("--steps", type=int, default=0, help="Simulation steps to run in headless mode.")
    args = parser.parse_args()
    if args.no_collision:
        print("[scene] --no-collision ignored: scene.glb is visual-only; table collision uses --no-table-collider.", flush=True)
    if args.dynamic:
        print("[scene] --dynamic ignored: scene.glb stays fixed as the support surface.", flush=True)
    os.environ["TELEOP_LINKER_L10_RETARGETER"] = str(args.linker_l10_retargeter)
    if args.enable_vr_teleop:
        if args.no_arm_assembly:
            raise SystemExit("--enable-vr-teleop requires the Nero arm assembly. Remove --no-arm-assembly.")
        if args.headless:
            raise SystemExit("--enable-vr-teleop currently requires the Genesis viewer. Remove --headless.")
        if not args.vr_no_auto_cloudxr_env:
            loaded = _load_export_env_file(args.vr_cloudxr_env_path.expanduser())
            if loaded and "NV_CXR_RUNTIME_DIR" in os.environ:
                print(f"[add-scene-vr] loaded CloudXR env: {args.vr_cloudxr_env_path.expanduser()}", flush=True)
        if not args.vr_no_cloudxr_preflight:
            ok, message = _check_cloudxr_runtime()
            if not ok:
                raise SystemExit(
                    "[add-scene-vr] CloudXR runtime is not ready.\n"
                    f"  {message}\n"
                    "  Start it in another terminal and keep that terminal open:\n"
                    "    conda activate genesis\n"
                    "    python -m isaacteleop.cloudxr --accept-eula\n"
                    "  Then rerun add_scene_glb.py with --enable-vr-teleop."
                )
            print(f"[add-scene-vr] {message}", flush=True)

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
        not args.enable_vr_teleop and not args.no_arm_assembly and not args.no_base_pose_panel,
        FIXED_ASSEMBLY_TRANSLATION,
        FIXED_ASSEMBLY_EULER,
    )
    last_panel_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = (
        (FIXED_ASSEMBLY_TRANSLATION, FIXED_ASSEMBLY_EULER) if base_pose_panel else None
    )
    last_reset_counter = 0
    ego_view_enabled = bool(args.d455_rgb_gui and not args.no_d455)
    d405_view_enabled = bool(args.d405_camera_gui and not args.no_d405)
    vr_session = None
    vr_robot = None
    voice_policy = None
    xr_status_publisher = None
    try:
        if args.enable_vr_teleop:
            xr_status_publisher = XrTeleopStatusPublisher(args.vr_xr_status_path)
            vr_robot = _AddSceneNeroTeleopRobot(
                scene,
                arm_side=args.vr_arm_side,
                translation_scale_xyz=args.vr_translation_scale_xyz,
                workspace_origin_xyz=args.vr_workspace_origin_xyz,
                input_axis_map=args.vr_input_axis_map,
                openxr_coordinate_adapter=args.vr_openxr_coordinate_adapter,
                use_teleop_orientation=bool(args.vr_use_teleop_orientation),
                orientation_source=args.vr_orientation_source,
                orientation_axis_map=args.vr_orientation_axis_map,
                orientation_max_speed_rad_s=float(args.vr_orientation_max_speed_rad_s),
                orientation_tool_offset_wxyz=args.vr_orientation_tool_offset_wxyz,
                orientation_reference_mode=str(args.vr_orientation_reference_mode),
                openxr_yaw_recenter=bool(args.vr_openxr_yaw_recenter),
                relative_control=not args.vr_absolute_control,
                drive_ik=not args.vr_markers_only,
                require_engage=bool(args.vr_require_engage),
                print_every_n=args.vr_print_every,
            )
            vr_input_source = str(args.vr_input_source)
            if vr_input_source == "auto":
                vr_input_source = (
                    "overlay-log"
                    if _overlay_hand_trace_is_fresh(args.vr_overlay_hand_trace_path.expanduser())
                    else "quest"
                )
                print(f"[add-scene-vr] auto input source selected: {vr_input_source}", flush=True)
            if vr_input_source == "overlay-log":
                vr_session = _OverlayHandLogTeleopSession(
                    vr_robot,
                    arm_side=args.vr_arm_side,
                    trace_path=args.vr_overlay_hand_trace_path,
                    hand_side=args.vr_overlay_hand_side,
                    use_teleop_orientation=bool(args.vr_use_teleop_orientation),
                    print_every_n=args.vr_print_every,
                    stale_after_s=float(args.vr_overlay_stale_after_s),
                )
                vr_session.__enter__()
            else:
                from teleop_stack.session import QuestRobotSession, QuestRobotSessionConfig

                vr_session = QuestRobotSession(
                    QuestRobotSessionConfig(
                        arm_side=args.vr_arm_side,
                        pose_input_mode=args.vr_pose_input_mode,
                        arm_pose_command_mode=_vr_arm_pose_command_mode(
                            pose_input_mode=args.vr_pose_input_mode,
                            use_teleop_orientation=bool(args.vr_use_teleop_orientation),
                        ),
                        use_wrist_position_for_hand=args.vr_pose_input_mode == "hand_abs",
                        use_wrist_rotation_for_hand=bool(args.vr_use_teleop_orientation),
                        palm_plane_wrist_orientation_blend_alpha=float(
                            args.vr_palm_plane_wrist_orientation_blend_alpha
                        ),
                        loop_hz=float(args.vr_loop_hz),
                        print_every_n_frames=int(args.vr_print_every),
                        enable_synthetic_hands_plugin=not args.vr_disable_synthetic_hands_plugin,
                        isaac_teleop_root=args.vr_isaac_teleop_root,
                        startup_timeout_s=float(args.vr_startup_timeout_s),
                        teleop_trace_path=args.vr_teleop_trace_path,
                    ),
                    vr_robot,
                )
                vr_session.__enter__()
            if args.vr_enable_voice_controls:
                voice_policy = VoiceTeleopControlPolicy(
                    VoiceTeleopControlConfig(
                        host=args.vr_voice_control_host,
                        port=int(args.vr_voice_control_port),
                    )
                )
                voice_policy.connect()
            xr_status_publisher.publish(
                snapshot=vr_robot.xr_status_snapshot(),
                lifecycle_event="session_started",
                force=True,
            )
            print(
                f"[add-scene-vr] {vr_input_source} teleop is driving the add_scene_glb assembly. "
                "Say 开始 to engage, 暂停 to clutch, 继续 to resume, 重置 to recenter, 停止 to hold.",
                flush=True,
            )
        _render_ego_view(scene, ego_view_enabled)
        _render_d405_view(scene, d405_view_enabled)
        while scene.viewer.is_alive():
            if voice_policy is not None and vr_robot is not None:
                if _apply_voice_control_events(vr_robot, voice_policy.update()):
                    print("[add-scene-vr] voice exit requested", flush=True)
                    break
            if vr_session is not None:
                vr_session.step()
            elif base_pose_panel:
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
            if xr_status_publisher is not None and vr_robot is not None:
                xr_status_publisher.publish(snapshot=vr_robot.xr_status_snapshot())
            time.sleep(1.0 / 60.0)
    finally:
        if xr_status_publisher is not None and vr_robot is not None:
            xr_status_publisher.publish(
                snapshot=vr_robot.xr_status_snapshot(),
                lifecycle_event="session_stopped",
                force=True,
            )
        if voice_policy is not None:
            voice_policy.disconnect()
        if vr_session is not None:
            vr_session.__exit__(None, None, None)
        _shutdown_base_pose_panel(base_pose_panel)


if __name__ == "__main__":
    main()
