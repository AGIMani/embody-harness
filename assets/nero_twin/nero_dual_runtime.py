#!/usr/bin/env python3
"""Reusable Nero dual-arm Genesis IK runtime.

This module keeps the original demo script intact while exposing the useful
pieces as a small class that other projects can drive programmatically.
Targets are expressed in the Genesis world frame. Quaternions use Genesis'
``wxyz`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

import dual_nero_arm_ik_demo as ik
import genesis as gs


NeroSide = str
DEFAULT_LINKER_L10_ACTIVE_JOINTS = (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
    "index_mcp_roll",
    "ring_mcp_roll",
    "pinky_mcp_roll",
    "thumb_cmc_roll",
)
HIDDEN_MARKER_POS = np.asarray((0.0, 0.0, -10.0), dtype=np.float32)
IDENTITY_QUAT_WXYZ = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
PALM_PLANE_AXIS_COLORS = {
    "origin": (1.0, 0.92, 0.16, 1.0),
    "across": (1.0, 0.12, 0.12, 1.0),
    "forward": (0.05, 0.8, 0.24, 1.0),
    "normal": (0.12, 0.35, 1.0, 1.0),
}


def _quat_multiply_wxyz(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _quat_to_matrix_wxyz(quat: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
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


def _normalize_np3(values: object) -> np.ndarray | None:
    if values is None:
        return None
    try:
        vector = np.asarray(values, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return None
    return (vector / norm).astype(np.float32)


def _quat_wxyz_from_z_axis_to_vector(values: object) -> tuple[float, float, float, float] | None:
    axis = _normalize_np3(values)
    if axis is None:
        return None
    source = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    target = np.asarray(axis, dtype=np.float64)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    if dot < -1.0 + 1e-9:
        return (0.0, 1.0, 0.0, 0.0)
    cross = np.cross(source, target)
    quat = np.asarray((1.0 + dot, cross[0], cross[1], cross[2]), dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return tuple(float(value) for value in quat)  # type: ignore[return-value]


def _add_palm_plane_axis_arrows(
    scene: object,
    *,
    shaft_radius_m: float,
    axis_length_m: float,
) -> dict[str, object]:
    shaft_height = max(0.01, float(axis_length_m) * 0.78)
    head_height = max(0.006, float(axis_length_m) * 0.18)
    head_radius = max(float(shaft_radius_m) * 2.4, float(shaft_radius_m) + 0.002)
    markers: dict[str, object] = {
        "origin": scene.add_entity(
            gs.morphs.Box(
                pos=tuple(float(v) for v in HIDDEN_MARKER_POS),
                size=(shaft_radius_m * 3.2, shaft_radius_m * 3.2, shaft_radius_m * 3.2),
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Plastic(color=PALM_PLANE_AXIS_COLORS["origin"]),
        )
    }
    for name in ("across", "forward", "normal"):
        markers[name] = {
            "shaft": scene.add_entity(
                gs.morphs.Cylinder(
                    pos=tuple(float(v) for v in HIDDEN_MARKER_POS),
                    radius=shaft_radius_m,
                    height=shaft_height,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Plastic(color=PALM_PLANE_AXIS_COLORS[name]),
            ),
            "head": scene.add_entity(
                gs.morphs.Cylinder(
                    pos=tuple(float(v) for v in HIDDEN_MARKER_POS),
                    radius=head_radius,
                    height=head_height,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Plastic(color=PALM_PLANE_AXIS_COLORS[name]),
            ),
        }
    return markers


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

    out = ik.NERO_TWIN_TMP_ROOT / f"{source_urdf.stem}_genesis_abs_mesh.urdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print(
        f"[nero-runtime] sanitized URDF relative meshes: {source_urdf} -> {out} rewritten_meshes={rewritten}",
        flush=True,
    )
    return out


@dataclass(frozen=True)
class NeroArmTarget:
    position_xyz: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class NeroDualRuntimeConfig:
    backend: str = "cpu"
    show_viewer: bool = False
    dt: float = 0.01
    max_fps: int = 60
    command_hz: float = 30.0
    real_command_hz: float = 10.0
    speed_percent: int = 10
    max_joint_step: float = 0.045
    min_joint_step: float = 0.001
    real_joint_command_deadband: float = 0.002
    ik_joint4_limit_enabled: bool = True
    ik_joint4_limit_rad: tuple[float, float] = (0.0, 2.14)
    max_solver_iters: int = 32
    ik_damping: float = 0.02
    pos_tol: float = 1e-3
    initial_arm_q: tuple[float, ...] = tuple(float(v) for v in ik.DEFAULT_ARM_Q)
    initial_left_arm_q: tuple[float, ...] | None = None
    initial_right_arm_q: tuple[float, ...] | None = None
    connect_can: bool = False
    interface: str = "socketcan"
    firmware: str = "default"
    left_channel: str = "can1"
    right_channel: str = "can2"
    can_sides: tuple[NeroSide, ...] = ("left", "right")
    enable_feedback_push: bool = True
    send_real_commands: bool = False
    base_mesh: Path | None = None
    base_scale: float = 0.001
    base_euler: str = "90,0,0"
    base_foot_center_mm: str | None = None
    nero_urdf: Path | None = None
    package_root: Path | None = None
    right_support_hole_z_mm: float | None = None
    eef_link: str = ik.DEFAULT_EEF_LINK
    add_revo2_flange: bool = True
    arm_collision: bool = False
    base_collision: bool = False
    show_hole_markers: bool = False
    show_target_markers: bool = False
    linker_hand_urdf: Path | None = None
    linker_hand_side: NeroSide = "right"
    linker_hand_mount_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linker_hand_mount_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    linker_hand_joint_names: tuple[str, ...] = DEFAULT_LINKER_L10_ACTIVE_JOINTS
    show_palm_plane_axes: bool = False
    palm_plane_axis_length_m: float = 0.08
    palm_plane_axis_marker_radius_m: float = 0.004

    def solver_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(
            max_solver_iters=int(self.max_solver_iters),
            ik_damping=float(self.ik_damping),
            pos_tol=float(self.pos_tol),
            max_joint_step=float(self.max_joint_step),
        )

    def can_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(
            connect_can=bool(self.connect_can),
            left_channel=str(self.left_channel),
            right_channel=str(self.right_channel),
            interface=str(self.interface),
            firmware=str(self.firmware),
            no_enable_push=not bool(self.enable_feedback_push),
        )


@dataclass
class NeroDualRuntimeStatus:
    q_left: tuple[float, ...]
    q_right: tuple[float, ...]
    target_left_xyz: tuple[float, float, float]
    target_right_xyz: tuple[float, float, float]
    real_enabled: bool
    sent_real_commands: bool
    position_error_left: tuple[float, ...] = field(default_factory=tuple)
    position_error_right: tuple[float, ...] = field(default_factory=tuple)


class NeroDualArmRuntime:
    def __init__(self, config: NeroDualRuntimeConfig | None = None) -> None:
        self.config = config or NeroDualRuntimeConfig()
        self.scene = None
        self.robots: dict[NeroSide, object] = {}
        self.arm_dofs: dict[NeroSide, list[int]] = {}
        self.eef_links: dict[NeroSide, object] = {}
        self.targets: dict[NeroSide, np.ndarray] = {}
        self.target_quats: dict[NeroSide, np.ndarray] = {}
        self.q_state: dict[NeroSide, np.ndarray] = {}
        self.markers: dict[str, object] = {}
        self.can_robots: dict[NeroSide, object] = {}
        self.linker_hand: object | None = None
        self.linker_hand_side: NeroSide = str(self.config.linker_hand_side)
        self.linker_hand_dofs: list[int] = []
        self.linker_hand_joint_names: list[str] = []
        self.linker_hand_mimic_by_name: dict[str, tuple[str, float, float]] = {}
        self.linker_hand_joint_limits_by_name: dict[str, tuple[float, float]] = {}
        self._pending_linker_hand_target: object | None = None
        self.palm_plane_axis_markers: dict[str, object] = {}
        self.real_enabled = False
        self.estopped = False
        self._last_real_command_time = 0.0
        self._last_real_q_command: dict[NeroSide, np.ndarray] = {}
        self._ik_joint_limit_hit_count: dict[tuple[NeroSide, int], int] = {}
        self._solver_args = self.config.solver_namespace()

    def connect(self) -> None:
        if self.scene is not None:
            return

        assembly = ik._import_assembly()
        base_mesh = (self.config.base_mesh or assembly.DEFAULT_BASE_MESH).expanduser().resolve()
        nero_urdf = (self.config.nero_urdf or assembly.DEFAULT_NERO_URDF).expanduser().resolve()
        package_root = (self.config.package_root or assembly.WORKSPACE_ROOT).expanduser().resolve()
        if not base_mesh.exists():
            raise FileNotFoundError(f"Base mesh not found: {base_mesh}")
        if not nero_urdf.exists():
            raise FileNotFoundError(f"Nero URDF not found: {nero_urdf}")
        if self.config.add_revo2_flange:
            nero_urdf = ik._make_revo2_flange_urdf(nero_urdf)

        base_euler = assembly._parse_vec3(self.config.base_euler, name="base_euler")
        base_foot_center_mm = assembly._parse_vec3(
            self.config.base_foot_center_mm
            or ",".join(str(v) for v in assembly.DEFAULT_BASE_FOOT_CENTER_MM),
            name="base_foot_center_mm",
        )
        base_pos = assembly._pose_from_local_anchor(base_foot_center_mm, base_euler, self.config.base_scale)

        right_support_holes_mm = assembly.SUPPORT_HOLES_MM.copy()
        if self.config.right_support_hole_z_mm is not None:
            right_support_holes_mm[:, 2] = float(self.config.right_support_hole_z_mm)
        else:
            right_support_holes_mm[:, 2] = float(assembly.RIGHT_SUPPORT_HOLE_Z_MM)

        base_rotation = assembly._rotation_from_euler_deg(base_euler)
        mount_normal = base_rotation @ np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        mount_rotation = ik._rotation_about_axis(mount_normal, np.deg2rad(ik.DEFAULT_MOUNT_HOLE_YAW_DEG))
        rotations = {
            "left": mount_rotation @ base_rotation,
            "right": mount_rotation @ assembly._rotation_from_euler_deg(
                (base_euler[0] + 180.0, base_euler[1], base_euler[2])
            ),
        }
        left_pos, rotations["left"] = ik._hole_aligned_arm_pose_from_rotation(
            assembly,
            base_pos,
            base_euler,
            assembly.SUPPORT_HOLES_MM,
            rotations["left"],
            label="left",
        )
        right_pos, rotations["right"] = ik._hole_aligned_arm_pose_from_rotation(
            assembly,
            base_pos,
            base_euler,
            right_support_holes_mm,
            rotations["right"],
            label="right",
        )
        lift = np.asarray((0.0, 0.0, ik.DEFAULT_ARM_LIFT_M), dtype=np.float64)
        positions = {
            "left": tuple(float(v) for v in np.asarray(left_pos, dtype=np.float64) + lift),
            "right": tuple(float(v) for v in np.asarray(right_pos, dtype=np.float64) + lift),
        }
        quats = {side: assembly._quat_wxyz_from_rotation(rotations[side]) for side in ("left", "right")}
        genesis_nero_urdf = assembly._sanitize_urdf_for_genesis(nero_urdf, package_root)

        if self.config.connect_can:
            requested_sides = {str(side) for side in self.config.can_sides}
            self.can_robots = {}
            if "left" in requested_sides:
                self.can_robots["left"] = ik._connect_can_arm(
                    self.config.left_channel,
                    self.config.interface,
                    self.config.firmware,
                    enable_push=bool(self.config.enable_feedback_push),
                )
            if "right" in requested_sides:
                self.can_robots["right"] = ik._connect_can_arm(
                    self.config.right_channel,
                    self.config.interface,
                    self.config.firmware,
                    enable_push=bool(self.config.enable_feedback_push),
                )

        gs.init(backend=gs.gpu if self.config.backend == "gpu" else gs.cpu)
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-2.25, 0.0, 1.05),
                camera_lookat=(0.0, 0.0, 0.75),
                camera_fov=35,
                res=(1280, 720),
                max_FPS=int(self.config.max_fps),
            ),
            sim_options=gs.options.SimOptions(dt=float(self.config.dt)),
            rigid_options=gs.options.RigidOptions(
                dt=float(self.config.dt),
                enable_self_collision=False,
                enable_adjacent_collision=False,
            ),
            show_viewer=bool(self.config.show_viewer),
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.scene.add_entity(
            gs.morphs.Mesh(
                file=str(base_mesh),
                pos=base_pos,
                euler=base_euler,
                scale=float(self.config.base_scale),
                fixed=True,
                collision=bool(self.config.base_collision),
                convexify=False,
            )
        )
        if self.config.show_hole_markers:
            ik._add_hole_markers(
                self.scene,
                assembly,
                base_pos=base_pos,
                base_euler=base_euler,
                support_holes_mm=assembly.SUPPORT_HOLES_MM,
                arm_pos=positions["left"],
                arm_rotation=rotations["left"],
            )
            ik._add_hole_markers(
                self.scene,
                assembly,
                base_pos=base_pos,
                base_euler=base_euler,
                support_holes_mm=right_support_holes_mm,
                arm_pos=positions["right"],
                arm_rotation=rotations["right"],
            )

        self.robots = {
            side: self.scene.add_entity(
                gs.morphs.URDF(
                    file=str(genesis_nero_urdf),
                    pos=positions[side],
                    quat=tuple(float(v) for v in quats[side]),
                    fixed=True,
                    collision=bool(self.config.arm_collision),
                    convexify=False,
                    merge_fixed_links=False,
                    prioritize_urdf_material=True,
                    links_to_keep=(self.config.eef_link,),
                    requires_jac_and_IK=True,
                )
            )
            for side in ("left", "right")
        }
        if self.config.linker_hand_urdf is not None:
            hand_urdf = self.config.linker_hand_urdf.expanduser().resolve()
            if not hand_urdf.exists():
                raise FileNotFoundError(f"Linker Hand URDF not found: {hand_urdf}")
            self.linker_hand_side = (
                str(self.config.linker_hand_side)
                if str(self.config.linker_hand_side) in ("left", "right")
                else "right"
            )
            self.linker_hand = self.scene.add_entity(
                gs.morphs.URDF(
                    file=str(_sanitize_relative_mesh_urdf(hand_urdf)),
                    fixed=True,
                    collision=False,
                    convexify=False,
                    merge_fixed_links=False,
                    prioritize_urdf_material=True,
                )
            )
        if self.config.show_target_markers:
            self.markers = {
                "left": ik._add_target_marker(self.scene, ik.LEFT_COLOR, 0.025),
                "right": ik._add_target_marker(self.scene, ik.RIGHT_COLOR, 0.025),
                "selected": ik._add_target_marker(self.scene, ik.SELECTED_COLOR, 0.029),
                "roll_axis": ik._add_target_marker(self.scene, ik.ROLL_AXIS_COLOR, ik.AXIS_MARKER_RADIUS_M),
                "pitch_axis": ik._add_target_marker(self.scene, ik.PITCH_AXIS_COLOR, ik.AXIS_MARKER_RADIUS_M),
                "yaw_axis": ik._add_target_marker(self.scene, ik.YAW_AXIS_COLOR, ik.AXIS_MARKER_RADIUS_M),
            }
        if self.config.show_palm_plane_axes:
            self.palm_plane_axis_markers = _add_palm_plane_axis_arrows(
                self.scene,
                shaft_radius_m=max(0.001, float(self.config.palm_plane_axis_marker_radius_m)),
                axis_length_m=max(0.01, float(self.config.palm_plane_axis_length_m)),
            )

        print("[nero-runtime] building dual-arm IK scene...", flush=True)
        self.scene.build()
        self.arm_dofs = {side: ik._arm_dofs(robot) for side, robot in self.robots.items()}
        for side, robot in self.robots.items():
            ik._set_gains(robot, self.arm_dofs[side])
        self._initialize_linker_hand()

        initial_left_q = np.asarray(
            self.config.initial_left_arm_q or self.config.initial_arm_q,
            dtype=np.float32,
        ).reshape(7)
        initial_right_q = np.asarray(
            self.config.initial_right_arm_q or self.config.initial_arm_q,
            dtype=np.float32,
        ).reshape(7)
        self.q_state = {"left": initial_left_q.copy(), "right": initial_right_q.copy()}
        for side, robot in self.robots.items():
            robot.set_dofs_position(self.q_state[side], self.arm_dofs[side], zero_velocity=True)
            robot.control_dofs_position(self.q_state[side], self.arm_dofs[side])
        self.scene.step()

        self.eef_links = {side: robot.get_link(self.config.eef_link) for side, robot in self.robots.items()}
        self.targets = {
            side: ik._tensor_to_np(self.eef_links[side].get_pos()).reshape(3).astype(np.float32)
            for side in ("left", "right")
        }
        self.target_quats = {
            side: ik._tensor_to_np(self.eef_links[side].get_quat()).reshape(4).astype(np.float32)
            for side in ("left", "right")
        }
        self.refresh_target_markers("right")
        self._mount_linker_hand()
        print(
            "[nero-runtime] ready "
            f"can={'connected' if self.can_robots else 'disabled'} "
            f"left={np.round(self.targets['left'], 4).tolist()} "
            f"right={np.round(self.targets['right'], 4).tolist()} "
            f"linker_hand={'on' if self.linker_hand is not None else 'off'}",
            flush=True,
        )

    def enable_real(self) -> None:
        if not self.can_robots:
            raise RuntimeError("Cannot enable real Nero arms without CAN connections.")
        if self.estopped:
            ik._safe_can_call(self.can_robots, "reset")
            self.estopped = False
        ik._safe_can_call(self.can_robots, "set_normal_mode")
        ik._safe_can_call(self.can_robots, "set_auto_set_motion_mode_enabled", False)
        ik._safe_can_call(self.can_robots, "set_speed_percent", int(self.config.speed_percent))
        self.real_enabled = ik._safe_can_call(self.can_robots, "enable")

    def reset_to_initial_joint_pose(
        self,
        *,
        sides: tuple[NeroSide, ...] = ("left", "right"),
        enable_real: bool = False,
        send_real: bool = False,
    ) -> bool:
        self._require_connected()
        requested_sides = tuple(side for side in sides if side in ("left", "right"))
        if not requested_sides:
            raise ValueError(f"No valid Nero sides requested: {sides}")

        initial_by_side = {
            "left": np.asarray(
                self.config.initial_left_arm_q or self.config.initial_arm_q,
                dtype=np.float32,
            ).reshape(7),
            "right": np.asarray(
                self.config.initial_right_arm_q or self.config.initial_arm_q,
                dtype=np.float32,
            ).reshape(7),
        }
        for side in requested_sides:
            self.q_state[side] = initial_by_side[side].copy()
            self.robots[side].set_dofs_position(self.q_state[side], self.arm_dofs[side], zero_velocity=True)
            self.robots[side].control_dofs_position(self.q_state[side], self.arm_dofs[side])
        self.scene.step()
        for side in requested_sides:
            self.targets[side] = ik._tensor_to_np(self.eef_links[side].get_pos()).reshape(3).astype(np.float32)
            self.target_quats[side] = ik._tensor_to_np(self.eef_links[side].get_quat()).reshape(4).astype(np.float32)
        self.refresh_target_markers(requested_sides[-1])
        self._apply_linker_hand_target()

        if enable_real:
            self.enable_real()
        if send_real:
            return self.send_real_joint_targets_once(sides=requested_sides)
        return False

    def disable_real(self) -> None:
        self.real_enabled = False
        if self.can_robots:
            ik._safe_can_call(self.can_robots, "disable")

    def estop(self) -> None:
        self.real_enabled = False
        self.estopped = True
        if self.can_robots:
            ik._safe_can_call(self.can_robots, "electronic_emergency_stop")

    def set_target(self, side: NeroSide, target: NeroArmTarget) -> None:
        self._require_connected()
        if side not in ("left", "right"):
            raise ValueError(f"Unsupported Nero side: {side}")
        self.targets[side] = np.asarray(target.position_xyz, dtype=np.float32).reshape(3)
        if target.quaternion_wxyz is not None:
            self.target_quats[side] = np.asarray(target.quaternion_wxyz, dtype=np.float32).reshape(4)

    def refresh_target_markers(self, selected: NeroSide = "right") -> None:
        if not self.markers:
            return
        selected = selected if selected in ("left", "right") else "right"
        ik._refresh_markers(self.markers, self.targets, self.target_quats, selected)

    def update_palm_plane_axis_markers(self, side: NeroSide, axes: dict[str, object] | None) -> None:
        if not self.palm_plane_axis_markers:
            return
        if axes is None or side not in ("left", "right") or side not in self.eef_links:
            self.hide_palm_plane_axis_markers()
            return
        if self.linker_hand is not None and side == self.linker_hand_side:
            origin = ik._tensor_to_np(self.linker_hand.get_pos()).reshape(3).astype(np.float32)
        else:
            origin = ik._tensor_to_np(self.eef_links[side].get_pos()).reshape(3).astype(np.float32)
        self.palm_plane_axis_markers["origin"].set_pos(origin, zero_velocity=True)
        self.palm_plane_axis_markers["origin"].set_quat(IDENTITY_QUAT_WXYZ, zero_velocity=True)

        axis_length = max(0.01, float(self.config.palm_plane_axis_length_m))
        shaft_height = np.float32(axis_length * 0.78)
        head_height = np.float32(axis_length * 0.18)
        for name in ("across", "forward", "normal"):
            axis = _normalize_np3(axes.get(name))
            quat = _quat_wxyz_from_z_axis_to_vector(axis)
            if axis is None or quat is None:
                self.hide_palm_plane_axis_markers()
                return
            shaft_center = origin + axis * (shaft_height * np.float32(0.5))
            head_center = origin + axis * (shaft_height + head_height * np.float32(0.5))
            marker = self.palm_plane_axis_markers[name]
            marker["shaft"].set_pos(shaft_center.astype(np.float32), zero_velocity=True)
            marker["shaft"].set_quat(np.asarray(quat, dtype=np.float32), zero_velocity=True)
            marker["head"].set_pos(head_center.astype(np.float32), zero_velocity=True)
            marker["head"].set_quat(np.asarray(quat, dtype=np.float32), zero_velocity=True)

    def hide_palm_plane_axis_markers(self) -> None:
        for marker in self.palm_plane_axis_markers.values():
            if isinstance(marker, dict):
                for part in marker.values():
                    part.set_pos(HIDDEN_MARKER_POS, zero_velocity=True)
                    part.set_quat(IDENTITY_QUAT_WXYZ, zero_velocity=True)
            else:
                marker.set_pos(HIDDEN_MARKER_POS, zero_velocity=True)
                marker.set_quat(IDENTITY_QUAT_WXYZ, zero_velocity=True)

    def _initialize_linker_hand(self) -> None:
        if self.linker_hand is None:
            self.linker_hand_dofs = []
            self.linker_hand_joint_names = []
            self.linker_hand_mimic_by_name = {}
            self.linker_hand_joint_limits_by_name = {}
            return

        self.linker_hand_mimic_by_name = self._load_linker_hand_mimic_specs()
        self.linker_hand_joint_limits_by_name = self._load_linker_hand_joint_limits()
        names: list[str] = []
        dofs: list[int] = []
        joint_names = list(self.config.linker_hand_joint_names)
        for mimic_joint_name in self.linker_hand_mimic_by_name:
            if mimic_joint_name not in joint_names:
                joint_names.append(mimic_joint_name)
        for joint_name in joint_names:
            try:
                joint = self.linker_hand.get_joint(joint_name)
            except Exception:
                continue
            joint_dofs = list(getattr(joint, "dofs_idx_local", ()))
            if not joint_dofs:
                continue
            names.append(str(joint_name))
            dofs.append(int(joint_dofs[0]))
        self.linker_hand_joint_names = names
        self.linker_hand_dofs = dofs
        if self.linker_hand_dofs:
            open_pose = np.zeros(len(self.linker_hand_dofs), dtype=np.float32)
            self.linker_hand.set_dofs_position(open_pose, self.linker_hand_dofs, zero_velocity=True)
            self.linker_hand.control_dofs_position(open_pose, self.linker_hand_dofs)
        print(
            "[nero-runtime] linker hand ready "
            f"side={self.linker_hand_side} "
            f"links={getattr(self.linker_hand, 'n_links', 'unknown')} "
            f"joints={getattr(self.linker_hand, 'n_joints', 'unknown')} "
            f"dofs={getattr(self.linker_hand, 'n_dofs', 'unknown')} "
            f"active_dofs={len(self.linker_hand_dofs)}",
            flush=True,
        )

    def _load_linker_hand_mimic_specs(self) -> dict[str, tuple[str, float, float]]:
        if self.config.linker_hand_urdf is None:
            return {}
        try:
            root = ET.parse(self.config.linker_hand_urdf.expanduser().resolve()).getroot()
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

    def _load_linker_hand_joint_limits(self) -> dict[str, tuple[float, float]]:
        if self.config.linker_hand_urdf is None:
            return {}
        try:
            root = ET.parse(self.config.linker_hand_urdf.expanduser().resolve()).getroot()
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

    def _mount_linker_hand(self) -> None:
        if self.linker_hand is None or self.linker_hand_side not in self.eef_links:
            return
        eef_link = self.eef_links[self.linker_hand_side]
        eef_pos = ik._tensor_to_np(eef_link.get_pos()).reshape(3).astype(np.float64)
        eef_quat = tuple(float(v) for v in ik._tensor_to_np(eef_link.get_quat()).reshape(4))
        rotation = _quat_to_matrix_wxyz(eef_quat)
        offset = np.asarray(self.config.linker_hand_mount_offset_xyz, dtype=np.float64)
        mounted_pos = eef_pos + rotation @ offset
        mounted_quat = _quat_multiply_wxyz(
            eef_quat,
            tuple(float(v) for v in self.config.linker_hand_mount_quat_wxyz),
        )
        self.linker_hand.set_pos(mounted_pos.astype(np.float32), zero_velocity=True)
        self.linker_hand.set_quat(np.asarray(mounted_quat, dtype=np.float32), zero_velocity=True)

    def set_linker_hand_target(self, side: NeroSide, joint_values: object | None) -> None:
        if side != self.linker_hand_side:
            return
        self._pending_linker_hand_target = joint_values

    def _apply_linker_hand_target(self) -> None:
        if self.linker_hand is None:
            return
        self._mount_linker_hand()
        if self._pending_linker_hand_target is None or not self.linker_hand_dofs:
            return
        target_names = tuple(getattr(self._pending_linker_hand_target, "joint_names", ()))
        target_positions = tuple(getattr(self._pending_linker_hand_target, "joint_positions", ()))
        values_by_name = dict(zip(target_names, target_positions, strict=False))
        for mimic_name, (source_name, multiplier, offset) in self.linker_hand_mimic_by_name.items():
            if mimic_name not in values_by_name and source_name in values_by_name:
                values_by_name[mimic_name] = multiplier * float(values_by_name[source_name]) + offset
        values = np.asarray(
            [
                float(
                    np.clip(
                        values_by_name.get(name, 0.0),
                        *self.linker_hand_joint_limits_by_name.get(name, (-np.inf, np.inf)),
                    )
                )
                for name in self.linker_hand_joint_names
            ],
            dtype=np.float32,
        )
        self.linker_hand.set_dofs_position(values, self.linker_hand_dofs, zero_velocity=True)
        self.linker_hand.control_dofs_position(values, self.linker_hand_dofs)

    def freeze_targets_at_current_eef(
        self,
        *,
        sides: tuple[NeroSide, ...] = ("left", "right"),
        selected: NeroSide = "right",
    ) -> None:
        self._require_connected()
        for side in sides:
            if side not in ("left", "right"):
                continue
            self.targets[side] = ik._tensor_to_np(self.eef_links[side].get_pos()).reshape(3).astype(np.float32)
            self.target_quats[side] = ik._tensor_to_np(self.eef_links[side].get_quat()).reshape(4).astype(np.float32)
        self.refresh_target_markers(selected)

    def step_targets_only(
        self,
        targets: dict[NeroSide, NeroArmTarget] | None = None,
        *,
        selected: NeroSide = "right",
    ) -> NeroDualRuntimeStatus:
        """Advance the Genesis scene while only moving visual target markers.

        This intentionally skips IK and joint control. It is useful for checking
        XR-to-Genesis target mapping before allowing targets to drive the arms.
        """

        self._require_connected()
        if targets:
            for side, target in targets.items():
                self.set_target(side, target)
        self.refresh_target_markers(selected)
        self._apply_linker_hand_target()
        self.scene.step()
        return NeroDualRuntimeStatus(
            q_left=tuple(float(v) for v in self.q_state["left"]),
            q_right=tuple(float(v) for v in self.q_state["right"]),
            target_left_xyz=tuple(float(v) for v in self.targets["left"]),
            target_right_xyz=tuple(float(v) for v in self.targets["right"]),
            real_enabled=bool(self.real_enabled),
            sent_real_commands=False,
        )

    def step(
        self,
        targets: dict[NeroSide, NeroArmTarget] | None = None,
        *,
        selected: NeroSide = "right",
    ) -> NeroDualRuntimeStatus:
        self._require_connected()
        if targets:
            for side, target in targets.items():
                self.set_target(side, target)
            if len(targets) == 1:
                selected = next(iter(targets.keys()))

        errors: dict[NeroSide, tuple[float, ...]] = {"left": (), "right": ()}
        for side, robot in self.robots.items():
            qpos_init = ik._tensor_to_np(robot.get_qpos()).reshape(-1)
            qpos_init = self._apply_ik_joint4_search_limit(side, qpos_init)
            qpos, error = ik._solve_ik(
                robot,
                self.eef_links[side],
                self.targets[side],
                self.target_quats[side],
                qpos_init,
                self.arm_dofs[side],
                self._solver_args,
            )
            qpos = self._apply_ik_joint4_search_limit(side, qpos)
            solved = qpos[self.arm_dofs[side]].astype(np.float32)
            dq = np.clip(
                solved - self.q_state[side],
                -float(self.config.max_joint_step),
                float(self.config.max_joint_step),
            )
            min_joint_step = max(float(self.config.min_joint_step), 0.0)
            if min_joint_step > 0.0:
                dq = np.where(np.abs(dq) < min_joint_step, 0.0, dq)
            self.q_state[side] = self.q_state[side] + dq
            robot.set_dofs_position(self.q_state[side], self.arm_dofs[side], zero_velocity=True)
            robot.control_dofs_position(self.q_state[side], self.arm_dofs[side])
            errors[side] = tuple(float(v) for v in error)

        self.refresh_target_markers(selected)
        self._apply_linker_hand_target()
        self.scene.step()
        sent = self._send_real_if_due()
        return NeroDualRuntimeStatus(
            q_left=tuple(float(v) for v in self.q_state["left"]),
            q_right=tuple(float(v) for v in self.q_state["right"]),
            target_left_xyz=tuple(float(v) for v in self.targets["left"]),
            target_right_xyz=tuple(float(v) for v in self.targets["right"]),
            real_enabled=bool(self.real_enabled),
            sent_real_commands=bool(sent),
            position_error_left=errors["left"],
            position_error_right=errors["right"],
        )

    def stop(self) -> None:
        if self.real_enabled:
            self.disable_real()

    def disconnect(self) -> None:
        self.stop()
        if self.can_robots:
            ik._disconnect_can_arms(self.can_robots)
        self.can_robots = {}
        self.scene = None

    def _send_real_if_due(self) -> bool:
        if not self.can_robots or not self.real_enabled or self.estopped or not self.config.send_real_commands:
            return False
        now = time.monotonic()
        real_dt = 1.0 / max(float(self.config.real_command_hz), 1e-6)
        if now - self._last_real_command_time < real_dt:
            return False
        self._last_real_command_time = now
        sent_any = False
        for side, can_robot in self.can_robots.items():
            q_command = self.q_state[side].astype(np.float32)
            previous_q_command = self._last_real_q_command.get(side)
            if previous_q_command is not None:
                q_delta = float(np.max(np.abs(q_command - previous_q_command)))
                if q_delta < max(float(self.config.real_joint_command_deadband), 0.0):
                    continue
            try:
                can_robot.move_j([float(v) for v in q_command])
                can_robot.set_motion_mode("j")
                self._last_real_q_command[side] = q_command.copy()
                sent_any = True
            except Exception as exc:
                self.real_enabled = False
                print(f"[nero-runtime:{side}] move_j failed; disabling real commands: {exc}", flush=True)
                return False
        return sent_any

    def _apply_ik_joint4_search_limit(self, side: NeroSide, qpos: np.ndarray) -> np.ndarray:
        limited = np.asarray(qpos, dtype=np.float32).copy()
        if not bool(self.config.ik_joint4_limit_enabled):
            return limited
        if side not in self.arm_dofs or len(self.arm_dofs[side]) < 4:
            return limited
        dof_index = int(self.arm_dofs[side][3])
        lower, upper = (float(v) for v in self.config.ik_joint4_limit_rad)
        if lower > upper:
            lower, upper = upper, lower
        before = float(limited[dof_index])
        after = float(np.clip(before, lower, upper))
        if after != before:
            limited[dof_index] = after
            key = (side, dof_index)
            hit_count = self._ik_joint_limit_hit_count.get(key, 0) + 1
            self._ik_joint_limit_hit_count[key] = hit_count
            if hit_count == 1 or hit_count % 60 == 0:
                print(
                    f"[nero-runtime:{side}] ik_joint4_limit "
                    f"before={before:+.3f} after={after:+.3f} range=({lower:+.3f},{upper:+.3f})",
                    flush=True,
                )
        return limited

    def send_real_joint_targets_once(
        self,
        *,
        sides: tuple[NeroSide, ...] = ("left", "right"),
        verbose: bool = True,
    ) -> bool:
        if not self.can_robots or not self.real_enabled or self.estopped:
            return False
        if not self.config.send_real_commands:
            print("[nero-runtime] reset real move skipped because send_real_commands is disabled", flush=True)
            return False
        ok = True
        for side in sides:
            can_robot = self.can_robots.get(side)
            if can_robot is None:
                continue
            q_command = self.q_state[side].astype(np.float32)
            real_q_before = self._read_real_q(side, can_robot)
            delta_text = "real_before=unavailable"
            if real_q_before is not None:
                max_delta = float(np.max(np.abs(q_command - real_q_before)))
                delta_text = (
                    f"real_before={np.round(real_q_before, 3).tolist()} "
                    f"max_abs_delta={max_delta:.4f}"
                )
            try:
                can_robot.set_motion_mode("j")
                can_robot.move_j([float(v) for v in q_command])
                can_robot.set_motion_mode("j")
                self._last_real_q_command[side] = q_command.copy()
                if verbose:
                    print(
                        f"[nero-runtime:{side}] reset_move_j "
                        f"q={np.round(q_command, 3).tolist()} {delta_text}",
                        flush=True,
                    )
            except Exception as exc:
                ok = False
                self.real_enabled = False
                print(f"[nero-runtime:{side}] reset move_j failed; disabling real commands: {exc}", flush=True)
        self._last_real_command_time = time.monotonic()
        return ok

    def real_joint_target_error(self, side: NeroSide) -> float | None:
        can_robot = self.can_robots.get(side)
        if can_robot is None:
            return None
        real_q = self._read_real_q(side, can_robot)
        if real_q is None:
            return None
        return float(np.max(np.abs(self.q_state[side].astype(np.float32) - real_q)))

    def _read_real_q(self, side: NeroSide, can_robot: object) -> np.ndarray | None:
        try:
            msg = can_robot.get_joint_angles()
        except Exception as exc:
            print(f"[nero-runtime:{side}] get_joint_angles warning: {exc}", flush=True)
            return None
        values = getattr(msg, "msg", None)
        if values is None or len(values) < 7:
            return None
        return np.asarray(values[:7], dtype=np.float32)

    def _require_connected(self) -> None:
        if self.scene is None:
            raise RuntimeError("NeroDualArmRuntime is not connected.")
