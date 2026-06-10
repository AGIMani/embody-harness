#!/usr/bin/env python3
"""Dual Nero arm IK demo on the current Genesis assembly.

Controls:
    Arrow keys/buttons: move the selected wrist target in world X/Y
    PageUp/PageDown or E/Q: move the selected wrist target in world Z
    Tab or Space: switch the selected target ball
    Active mode: Genesis target balls drive IK and optionally the real arms
    Passive mode: Genesis robots and target balls follow CAN feedback
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import can_config


NERO_TWIN_TMP_ROOT = Path(os.environ.get("NERO_TWIN_TMPDIR", f"/tmp/teleop_nero_{os.environ.get('USER', 'user')}"))
NERO_TWIN_TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NERO_TWIN_TMP_ROOT / "genesis_numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(NERO_TWIN_TMP_ROOT / "genesis_mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(NERO_TWIN_TMP_ROOT / "genesis_xdg_cache"))
os.environ.setdefault("GS_CACHE_FILE_PATH", str(NERO_TWIN_TMP_ROOT / "genesis_cache"))
os.environ.setdefault("QD_OFFLINE_CACHE_FILE_PATH", str(NERO_TWIN_TMP_ROOT / "genesis_qd_cache"))

import genesis as gs


THIS_FILE = Path(__file__).resolve()
ASSEMBLY_SCRIPT = THIS_FILE.with_name("load_dual_arm_assembly.py")
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
DEFAULT_ARM_Q = np.zeros(7, dtype=np.float32)
DEFAULT_EEF_LINK = "revo2_flange"
KEYS = ("left", "right", "up", "down", "pageup", "pagedown", "e", "q")
MODE_PASSIVE = 0
MODE_ACTIVE = 1
MODE_NAMES = {MODE_PASSIVE: "passive", MODE_ACTIVE: "active"}
LEFT_COLOR = (0.95, 0.22, 0.18, 1.0)
RIGHT_COLOR = (0.15, 0.45, 1.0, 1.0)
SELECTED_COLOR = (0.1, 0.95, 0.25, 1.0)
HIDDEN_POS = np.array((0.0, 0.0, -10.0), dtype=np.float32)
AXIS_MARKER_OFFSET_M = 0.075
AXIS_MARKER_RADIUS_M = 0.011
ROLL_AXIS_COLOR = (0.1, 0.25, 1.0, 1.0)
PITCH_AXIS_COLOR = (0.05, 0.95, 0.1, 1.0)
YAW_AXIS_COLOR = (1.0, 0.05, 0.05, 1.0)
DEFAULT_MOUNT_HOLE_YAW_DEG = 90.0
DEFAULT_ARM_LIFT_M = 0.005
REVO2_FLANGE_VISUAL_MESH = "package://agx_arm_description/agx_arm_urdf/nero/meshes/dae/revo2_flange.dae"
REVO2_FLANGE_COLLISION_MESH = "package://agx_arm_description/agx_arm_urdf/nero/meshes/revo2_flange.stl"
REVO2_FLANGE_JOINT_XYZ = "0.032 0 -0.0235"
REVO2_FLANGE_JOINT_RPY = "-1.5708 0 -1.5708"


def _import_assembly():
    spec = importlib.util.spec_from_file_location("dual_arm_assembly", ASSEMBLY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import assembly script: {ASSEMBLY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_vec(text: str, length: int, *, name: str) -> np.ndarray:
    values = np.asarray([float(v.strip()) for v in text.split(",") if v.strip()], dtype=np.float32)
    if values.shape != (length,):
        raise ValueError(f"{name} must contain {length} comma-separated floats, got {values.shape[0]}")
    return values


def _tensor_to_np(value) -> np.ndarray:
    return value.detach().cpu().numpy()


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

    out = NERO_TWIN_TMP_ROOT / f"{source_urdf.stem}_with_revo2_flange.urdf"
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print(
        "[ik] added fixed revo2_flange to URDF: "
        f"parent=link7 xyz={REVO2_FLANGE_JOINT_XYZ} rpy={REVO2_FLANGE_JOINT_RPY} -> {out}",
        flush=True,
    )
    return out


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


def _hole_aligned_arm_pose_from_rotation(
    assembly,
    base_pos: tuple[float, float, float],
    base_euler: tuple[float, float, float],
    support_holes_mm: np.ndarray,
    arm_rotation: np.ndarray,
    *,
    label: str,
) -> tuple[tuple[float, float, float], np.ndarray]:
    support_holes_m = np.asarray(support_holes_mm, dtype=np.float64) * 0.001
    arm_holes_m = np.asarray(assembly.ARM_HOLES_MM, dtype=np.float64) * 0.001
    support_center_m = support_holes_m.mean(axis=0)
    arm_center_m = arm_holes_m.mean(axis=0)
    support_world = assembly._transform_point(support_center_m, base_pos, base_euler)
    arm_world_pos = support_world - np.asarray(arm_rotation, dtype=np.float64) @ arm_center_m
    print(
        f"[align:{label}] support hole center in base mesh frame mm="
        f"{np.round(support_center_m * 1000.0, 4).tolist()} "
        f"arm hole center in URDF root/base_link frame mm={np.round(arm_center_m * 1000.0, 4).tolist()} "
        f"arm_world_pos={np.round(arm_world_pos, 5).tolist()}",
        flush=True,
    )
    return tuple(float(v) for v in arm_world_pos), np.asarray(arm_rotation, dtype=np.float64)


def _add_hole_markers(
    scene,
    assembly,
    *,
    base_pos: tuple[float, float, float],
    base_euler: tuple[float, float, float],
    support_holes_mm: np.ndarray,
    arm_pos: tuple[float, float, float],
    arm_rotation: np.ndarray,
) -> None:
    support_holes_m = np.asarray(support_holes_mm, dtype=np.float64) * 0.001
    arm_holes_m = np.asarray(assembly.ARM_HOLES_MM, dtype=np.float64) * 0.001
    support_world = assembly._transform_points(support_holes_m, base_pos, base_euler)
    arm_world = np.asarray(arm_pos, dtype=np.float64) + arm_holes_m @ np.asarray(arm_rotation, dtype=np.float64).T
    for pos in support_world:
        scene.add_entity(
            gs.morphs.Sphere(pos=tuple(float(v) for v in pos), radius=0.007, fixed=True, collision=False),
            surface=gs.surfaces.Plastic(color=(1.0, 0.85, 0.05, 1.0)),
        )
    for pos in arm_world:
        scene.add_entity(
            gs.morphs.Sphere(pos=tuple(float(v) for v in pos), radius=0.0045, fixed=True, collision=False),
            surface=gs.surfaces.Plastic(color=(0.0, 1.0, 1.0, 1.0)),
        )


def _get_joint_dofs(robot, joint_name: str) -> list[int]:
    try:
        joint = robot.get_joint(joint_name)
    except Exception:
        return []
    return list(joint.dofs_idx_local)


def _arm_dofs(robot) -> list[int]:
    dofs = [idx for name in ARM_JOINT_NAMES for idx in _get_joint_dofs(robot, name)]
    if len(dofs) != 7:
        raise RuntimeError(f"Expected 7 Nero arm DOFs, got {dofs}")
    return dofs


def _set_gains(robot, dofs: list[int]) -> None:
    robot.set_dofs_kp(np.full(len(dofs), 3600.0, dtype=np.float32), dofs)
    robot.set_dofs_kv(np.full(len(dofs), 180.0, dtype=np.float32), dofs)
    robot.set_dofs_force_range(
        np.full(len(dofs), -100.0, dtype=np.float32),
        np.full(len(dofs), 100.0, dtype=np.float32),
        dofs,
    )


def _add_target_marker(scene, color: tuple[float, float, float, float], radius: float):
    return scene.add_entity(
        gs.morphs.Sphere(pos=(0.0, 0.0, 0.0), radius=radius, fixed=True, collision=False),
        surface=gs.surfaces.Plastic(color=color),
    )


def _set_marker_pos(marker, pos: np.ndarray) -> None:
    marker.set_pos(np.asarray(pos, dtype=np.float32), zero_velocity=True)


def _quat_wxyz_to_rotation(quat: np.ndarray) -> np.ndarray:
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


def _refresh_markers(
    markers: dict[str, object],
    targets: dict[str, np.ndarray],
    target_quats: dict[str, np.ndarray],
    selected: str,
) -> None:
    _set_marker_pos(markers["selected"], targets[selected])
    _set_marker_pos(markers["left"], HIDDEN_POS if selected == "left" else targets["left"])
    _set_marker_pos(markers["right"], HIDDEN_POS if selected == "right" else targets["right"])
    rotation = _quat_wxyz_to_rotation(target_quats[selected])
    axis_specs = (
        ("yaw_axis", rotation[:, 2]),
        ("pitch_axis", rotation[:, 1]),
        ("roll_axis", rotation[:, 0]),
    )
    for marker_name, axis in axis_specs:
        _set_marker_pos(markers[marker_name], targets[selected] + np.asarray(axis, dtype=np.float32) * AXIS_MARKER_OFFSET_M)


def _firmware(name: str):
    from pyAgxArm import NeroFW

    return {
        "default": NeroFW.DEFAULT,
        "v111": NeroFW.V111,
        "v112": NeroFW.V112,
    }[name]


def _connect_can_arm(channel: str, interface: str, firmware: str, *, enable_push: bool):
    from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config

    cfg = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=_firmware(firmware),
        interface=interface,
        channel=channel,
    )
    robot = AgxArmFactory.create_arm(cfg)
    print(f"[can] connecting channel={channel} interface={interface} firmware={firmware}", flush=True)
    robot.connect()
    robot.set_auto_set_motion_mode_enabled(False)
    if enable_push:
        print(f"[can:{channel}] set_normal_mode() for feedback push", flush=True)
        robot.set_normal_mode()
    return robot


def _connect_can_arms(args: argparse.Namespace) -> dict[str, object]:
    if not args.connect_can:
        return {}
    return {
        "left": _connect_can_arm(
            args.left_channel,
            args.interface,
            args.firmware,
            enable_push=not args.no_enable_push,
        ),
        "right": _connect_can_arm(
            args.right_channel,
            args.interface,
            args.firmware,
            enable_push=not args.no_enable_push,
        ),
    }


def _disconnect_can_arms(can_robots: dict[str, object]) -> None:
    for robot in can_robots.values():
        try:
            robot.disconnect()
        except Exception as exc:
            print(f"[can] disconnect warning: {exc}", flush=True)


def _safe_can_call(can_robots: dict[str, object], method: str, *args) -> bool:
    ok = True
    for side, robot in can_robots.items():
        try:
            getattr(robot, method)(*args)
        except Exception as exc:
            ok = False
            print(f"[can:{side}] {method} failed: {exc}", flush=True)
    return ok


def _read_can_joints(can_robots: dict[str, object]) -> dict[str, np.ndarray | None]:
    joints: dict[str, np.ndarray | None] = {"left": None, "right": None}
    for side, robot in can_robots.items():
        try:
            msg = robot.get_joint_angles()
        except Exception as exc:
            print(f"[can:{side}] get_joint_angles failed: {exc}", flush=True)
            continue
        if msg is not None:
            joints[side] = np.asarray(msg.msg, dtype=np.float32)
    return joints


def _read_can_joint(can_robots: dict[str, object], side: str) -> np.ndarray | None:
    robot = can_robots.get(side)
    if robot is None:
        return None
    try:
        msg = robot.get_joint_angles()
    except Exception as exc:
        print(f"[can:{side}] get_joint_angles failed: {exc}", flush=True)
        return None
    if msg is None:
        return None
    return np.asarray(msg.msg, dtype=np.float32)


def _control_window_main(keys, selected_idx, mode_value, enable_count, disable_count, estop_count, stop_flag) -> None:
    import tkinter as tk
    from tkinter import ttk

    key_to_idx = {key: i for i, key in enumerate(KEYS)}

    def set_key(key: str, pressed: bool) -> None:
        keys[key_to_idx[key]] = bool(pressed)

    def toggle() -> None:
        selected_idx.value = 1 - int(selected_idx.value)

    def set_mode(mode: int) -> None:
        mode_value.value = int(mode)

    def request_enable() -> None:
        enable_count.value = int(enable_count.value) + 1

    def request_disable() -> None:
        disable_count.value = int(disable_count.value) + 1

    def request_estop() -> None:
        estop_count.value = int(estop_count.value) + 1

    def close() -> None:
        stop_flag.value = True
        root.destroy()
        root.quit()

    root = tk.Tk()
    root.title("Dual Nero Wrist IK")
    root.geometry("460x390")
    root.minsize(460, 390)

    title = ttk.Label(root, text="Arrow keys move the selected wrist target", font=("Arial", 12, "bold"))
    title.pack(padx=12, pady=(12, 4), anchor=tk.W)
    selected_label = ttk.Label(root, text="")
    selected_label.pack(padx=12, pady=(0, 8), anchor=tk.W)
    mode_label = ttk.Label(root, text="")
    mode_label.pack(padx=12, pady=(0, 8), anchor=tk.W)

    mode_frame = ttk.LabelFrame(root, text="Mode")
    mode_frame.pack(padx=12, pady=6, fill=tk.X)
    ttk.Button(mode_frame, text="Passive / 从动", command=lambda: set_mode(MODE_PASSIVE)).pack(
        side=tk.LEFT, padx=6, pady=6, expand=True, fill=tk.X
    )
    ttk.Button(mode_frame, text="Active / 主动", command=lambda: set_mode(MODE_ACTIVE)).pack(
        side=tk.LEFT, padx=6, pady=6, expand=True, fill=tk.X
    )

    grid = ttk.Frame(root)
    grid.pack(padx=12, pady=6, fill=tk.BOTH, expand=True)
    button_specs = (
        ("Up", "up", 0, 1),
        ("Left", "left", 1, 0),
        ("Right", "right", 1, 2),
        ("Down", "down", 2, 1),
        ("Z+", "pageup", 0, 3),
        ("Z-", "pagedown", 2, 3),
    )
    for label, key, row, col in button_specs:
        button = ttk.Button(grid, text=label)
        button.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        button.bind("<ButtonPress-1>", lambda _event, k=key: set_key(k, True))
        button.bind("<ButtonRelease-1>", lambda _event, k=key: set_key(k, False))
    for col in range(4):
        grid.columnconfigure(col, weight=1)

    safety_frame = ttk.LabelFrame(root, text="Real Arm")
    safety_frame.pack(padx=12, pady=6, fill=tk.X)
    ttk.Button(safety_frame, text="Enable / 使能", command=request_enable).pack(
        side=tk.LEFT, padx=6, pady=6, expand=True, fill=tk.X
    )
    ttk.Button(safety_frame, text="Disable / 失能", command=request_disable).pack(
        side=tk.LEFT, padx=6, pady=6, expand=True, fill=tk.X
    )
    ttk.Button(safety_frame, text="E-STOP / 急停", command=request_estop).pack(
        side=tk.LEFT, padx=6, pady=6, expand=True, fill=tk.X
    )

    bottom = ttk.Frame(root)
    bottom.pack(padx=12, pady=(4, 12), fill=tk.X)
    ttk.Button(bottom, text="Switch Target", command=toggle).pack(side=tk.LEFT)
    ttk.Button(bottom, text="Close", command=close).pack(side=tk.RIGHT)

    def on_key_press(event) -> None:
        key = str(event.keysym).lower()
        if key in ("tab", "space"):
            toggle()
            return "break"
        if key in key_to_idx:
            set_key(key, True)
        if key == "prior":
            set_key("pageup", True)
        elif key == "next":
            set_key("pagedown", True)

    def on_key_release(event) -> None:
        key = str(event.keysym).lower()
        if key in key_to_idx:
            set_key(key, False)
        if key == "prior":
            set_key("pageup", False)
        elif key == "next":
            set_key("pagedown", False)

    def poll() -> None:
        selected_label.config(text=f"Selected target: {'left' if selected_idx.value == 0 else 'right'}")
        mode_label.config(text=f"Mode: {MODE_NAMES.get(int(mode_value.value), 'unknown')}")
        if stop_flag.value:
            close()
        else:
            root.after(80, poll)

    root.bind("<KeyPress>", on_key_press)
    root.bind("<KeyRelease>", on_key_release)
    root.protocol("WM_DELETE_WINDOW", close)
    root.after(80, poll)
    root.mainloop()


def _create_control_window(enabled: bool, initial_mode: int) -> dict[str, object] | None:
    if not enabled:
        return None
    keys = multiprocessing.RawArray("b", [False] * len(KEYS))
    selected_idx = multiprocessing.RawValue("i", 0)
    mode_value = multiprocessing.RawValue("i", int(initial_mode))
    enable_count = multiprocessing.RawValue("i", 0)
    disable_count = multiprocessing.RawValue("i", 0)
    estop_count = multiprocessing.RawValue("i", 0)
    stop_flag = multiprocessing.RawValue("b", False)
    process = multiprocessing.Process(
        target=_control_window_main,
        args=(keys, selected_idx, mode_value, enable_count, disable_count, estop_count, stop_flag),
        daemon=True,
    )
    process.start()
    print(
        "[ik] control window: arrows move target, mode switches active/passive, buttons request enable/disable/e-stop.",
        flush=True,
    )
    return {
        "keys": keys,
        "selected_idx": selected_idx,
        "mode_value": mode_value,
        "enable_count": enable_count,
        "disable_count": disable_count,
        "estop_count": estop_count,
        "stop_flag": stop_flag,
        "process": process,
    }


def _shutdown_control_window(control_window: dict[str, object] | None) -> None:
    if not control_window:
        return
    control_window["stop_flag"].value = True
    process = control_window["process"]
    if process.is_alive():
        process.join(timeout=1.0)


def _pressed(control_window: dict[str, object] | None) -> dict[str, bool]:
    if not control_window:
        return {key: False for key in KEYS}
    values = control_window["keys"]
    return {key: bool(values[i]) for i, key in enumerate(KEYS)}


def _target_delta(keys: dict[str, bool], speed: float, dt: float) -> np.ndarray:
    delta = np.zeros(3, dtype=np.float32)
    delta[0] += float(keys["up"]) - float(keys["down"])
    delta[1] += float(keys["left"]) - float(keys["right"])
    delta[2] += float(keys["pageup"] or keys["e"]) - float(keys["pagedown"] or keys["q"])
    norm = float(np.linalg.norm(delta))
    if norm > 1.0:
        delta /= norm
    return delta * float(speed) * float(dt)


def _solve_ik(robot, eef_link, target_pos, target_quat, qpos_init, arm_dofs, args) -> tuple[np.ndarray, np.ndarray]:
    qpos, error = robot.inverse_kinematics(
        link=eef_link,
        pos=np.asarray(target_pos, dtype=np.float32),
        quat=np.asarray(target_quat, dtype=np.float32),
        init_qpos=np.asarray(qpos_init, dtype=np.float32),
        dofs_idx_local=arm_dofs,
        max_samples=1,
        max_solver_iters=int(args.max_solver_iters),
        damping=float(args.ik_damping),
        pos_tol=float(args.pos_tol),
        max_step_size=float(args.max_joint_step),
        return_error=True,
    )
    return _tensor_to_np(qpos).reshape(-1), _tensor_to_np(error).reshape(-1)


def _parse_args() -> argparse.Namespace:
    assembly = _import_assembly()
    parser = argparse.ArgumentParser(description=__doc__)
    can_config.add_config_arg(parser)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--max-fps", type=int, default=60)
    parser.add_argument("--command-hz", type=float, default=30.0)
    parser.add_argument("--real-command-hz", type=float, default=10.0)
    parser.add_argument("--speed-percent", type=int, default=10)
    parser.add_argument("--target-speed", type=float, default=0.08)
    parser.add_argument("--max-joint-step", type=float, default=0.045)
    parser.add_argument("--max-solver-iters", type=int, default=32)
    parser.add_argument("--ik-damping", type=float, default=0.02)
    parser.add_argument("--pos-tol", type=float, default=1e-3)
    parser.add_argument("--base-mesh", type=Path, default=assembly.DEFAULT_BASE_MESH)
    parser.add_argument("--base-scale", type=float, default=0.001)
    parser.add_argument("--base-euler", type=str, default="90,0,0")
    parser.add_argument("--base-foot-center-mm", type=str, default=",".join(str(v) for v in assembly.DEFAULT_BASE_FOOT_CENTER_MM))
    parser.add_argument("--nero-urdf", type=Path, default=assembly.DEFAULT_NERO_URDF)
    parser.add_argument("--package-root", type=Path, default=assembly.WORKSPACE_ROOT)
    parser.add_argument("--right-support-hole-z-mm", type=float, default=assembly.RIGHT_SUPPORT_HOLE_Z_MM)
    parser.add_argument("--eef-link", type=str, default=DEFAULT_EEF_LINK)
    parser.add_argument("--left-target-xyz", type=str, default=None)
    parser.add_argument("--right-target-xyz", type=str, default=None)
    parser.add_argument("--initial-arm-q", type=str, default=",".join(str(float(v)) for v in DEFAULT_ARM_Q))
    parser.add_argument("--no-control-window", action="store_true")
    parser.add_argument("--arm-collision", action="store_true")
    parser.add_argument("--base-collision", action="store_true")
    parser.add_argument("--show-hole-markers", action="store_true")
    parser.add_argument("--no-revo2-flange", action="store_true", help="Use the bare Nero URDF without the Revo2 adapter flange.")
    parser.add_argument("--connect-can", action="store_true", help="Connect can1/can2 and allow real-arm control after Enable.")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--firmware", choices=("default", "v111", "v112"), default="default")
    parser.add_argument("--left-channel", default="can1")
    parser.add_argument("--right-channel", default="can2")
    parser.add_argument("--no-enable-push", action="store_true", help="Do not call set_normal_mode() when connecting CAN.")
    parser.add_argument(
        "--initial-mode",
        choices=("active", "passive", "auto"),
        default="auto",
        help="auto starts passive when --connect-can is used, otherwise active.",
    )
    return can_config.parse_args(parser, sections=("ik_demo",))


def main() -> None:
    args = _parse_args()
    assembly = _import_assembly()
    if args.connect_can and args.no_control_window:
        print("[ik] warning: --connect-can without control window can read feedback, but enable/e-stop buttons are unavailable.", flush=True)
    if args.speed_percent < 0 or args.speed_percent > 100:
        raise ValueError("--speed-percent must be in [0, 100]")

    base_mesh = args.base_mesh.expanduser().resolve()
    nero_urdf = args.nero_urdf.expanduser().resolve()
    package_root = args.package_root.expanduser().resolve()
    if not base_mesh.exists():
        raise FileNotFoundError(f"Base mesh not found: {base_mesh}")
    if not nero_urdf.exists():
        raise FileNotFoundError(f"Nero URDF not found: {nero_urdf}")
    if not args.no_revo2_flange:
        nero_urdf = _make_revo2_flange_urdf(nero_urdf)

    can_robots = _connect_can_arms(args)

    base_euler = assembly._parse_vec3(args.base_euler, name="--base-euler")
    base_foot_center_mm = assembly._parse_vec3(args.base_foot_center_mm, name="--base-foot-center-mm")
    base_pos = assembly._pose_from_local_anchor(base_foot_center_mm, base_euler, args.base_scale)

    right_support_holes_mm = assembly.SUPPORT_HOLES_MM.copy()
    right_support_holes_mm[:, 2] = float(args.right_support_hole_z_mm)
    base_rotation = assembly._rotation_from_euler_deg(base_euler)
    mount_normal = base_rotation @ np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    mount_rotation = _rotation_about_axis(mount_normal, np.deg2rad(DEFAULT_MOUNT_HOLE_YAW_DEG))
    left_rotation = mount_rotation @ base_rotation
    right_rotation = mount_rotation @ assembly._rotation_from_euler_deg(
        (base_euler[0] + 180.0, base_euler[1], base_euler[2])
    )
    left_pos, left_rotation = _hole_aligned_arm_pose_from_rotation(
        assembly,
        base_pos,
        base_euler,
        assembly.SUPPORT_HOLES_MM,
        left_rotation,
        label="left",
    )
    right_pos, right_rotation = _hole_aligned_arm_pose_from_rotation(
        assembly,
        base_pos,
        base_euler,
        right_support_holes_mm,
        right_rotation,
        label="right",
    )
    lift = np.asarray((0.0, 0.0, DEFAULT_ARM_LIFT_M), dtype=np.float64)
    left_pos = tuple(float(v) for v in np.asarray(left_pos, dtype=np.float64) + lift)
    right_pos = tuple(float(v) for v in np.asarray(right_pos, dtype=np.float64) + lift)
    left_quat = assembly._quat_wxyz_from_rotation(left_rotation)
    right_quat = assembly._quat_wxyz_from_rotation(right_rotation)
    genesis_nero_urdf = assembly._sanitize_urdf_for_genesis(nero_urdf, package_root)

    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-2.25, 0.0, 1.05),
            camera_lookat=(0.0, 0.0, 0.75),
            camera_fov=35,
            res=(1280, 720),
            max_FPS=args.max_fps,
        ),
        sim_options=gs.options.SimOptions(dt=args.dt),
        rigid_options=gs.options.RigidOptions(dt=args.dt, enable_self_collision=False, enable_adjacent_collision=False),
        show_viewer=not args.no_viewer,
    )

    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Mesh(
            file=str(base_mesh),
            pos=base_pos,
            euler=base_euler,
            scale=args.base_scale,
            fixed=True,
            collision=args.base_collision,
            convexify=False,
        )
    )
    if args.show_hole_markers:
        _add_hole_markers(
            scene,
            assembly,
            base_pos=base_pos,
            base_euler=base_euler,
            support_holes_mm=assembly.SUPPORT_HOLES_MM,
            arm_pos=left_pos,
            arm_rotation=left_rotation,
        )
        _add_hole_markers(
            scene,
            assembly,
            base_pos=base_pos,
            base_euler=base_euler,
            support_holes_mm=right_support_holes_mm,
            arm_pos=right_pos,
            arm_rotation=right_rotation,
        )
    robots = {
        "left": scene.add_entity(
            gs.morphs.URDF(
                file=str(genesis_nero_urdf),
                pos=left_pos,
                quat=tuple(float(v) for v in left_quat),
                fixed=True,
                collision=args.arm_collision,
                convexify=False,
                merge_fixed_links=False,
                prioritize_urdf_material=True,
                links_to_keep=(args.eef_link,),
                requires_jac_and_IK=True,
            )
        ),
        "right": scene.add_entity(
            gs.morphs.URDF(
                file=str(genesis_nero_urdf),
                pos=right_pos,
                quat=tuple(float(v) for v in right_quat),
                fixed=True,
                collision=args.arm_collision,
                convexify=False,
                merge_fixed_links=False,
                prioritize_urdf_material=True,
                links_to_keep=(args.eef_link,),
                requires_jac_and_IK=True,
            )
        ),
    }
    markers = {
        "left": _add_target_marker(scene, LEFT_COLOR, 0.025),
        "right": _add_target_marker(scene, RIGHT_COLOR, 0.025),
        "selected": _add_target_marker(scene, SELECTED_COLOR, 0.029),
        "roll_axis": _add_target_marker(scene, ROLL_AXIS_COLOR, AXIS_MARKER_RADIUS_M),
        "pitch_axis": _add_target_marker(scene, PITCH_AXIS_COLOR, AXIS_MARKER_RADIUS_M),
        "yaw_axis": _add_target_marker(scene, YAW_AXIS_COLOR, AXIS_MARKER_RADIUS_M),
    }

    print("[ik] building dual-arm IK scene...", flush=True)
    scene.build()

    arm_dofs = {name: _arm_dofs(robot) for name, robot in robots.items()}
    for name, robot in robots.items():
        _set_gains(robot, arm_dofs[name])

    arm_q = _parse_vec(args.initial_arm_q, 7, name="--initial-arm-q")
    q_state = {"left": arm_q.copy(), "right": arm_q.copy()}
    for name, robot in robots.items():
        robot.set_dofs_position(q_state[name], arm_dofs[name], zero_velocity=True)
        robot.control_dofs_position(q_state[name], arm_dofs[name])
    scene.step()

    eef_links = {name: robot.get_link(args.eef_link) for name, robot in robots.items()}
    targets = {
        "left": _tensor_to_np(eef_links["left"].get_pos()).reshape(3).astype(np.float32),
        "right": _tensor_to_np(eef_links["right"].get_pos()).reshape(3).astype(np.float32),
    }
    target_quats = {
        "left": _tensor_to_np(eef_links["left"].get_quat()).reshape(4).astype(np.float32),
        "right": _tensor_to_np(eef_links["right"].get_quat()).reshape(4).astype(np.float32),
    }
    if args.left_target_xyz:
        targets["left"] = _parse_vec(args.left_target_xyz, 3, name="--left-target-xyz")
    if args.right_target_xyz:
        targets["right"] = _parse_vec(args.right_target_xyz, 3, name="--right-target-xyz")

    if args.initial_mode == "auto":
        mode = MODE_PASSIVE if can_robots else MODE_ACTIVE
    else:
        mode = MODE_ACTIVE if args.initial_mode == "active" else MODE_PASSIVE

    control_window = _create_control_window(enabled=not args.no_control_window, initial_mode=mode)
    selected = "left"
    _refresh_markers(markers, targets, target_quats, selected)
    print(
        "[ik] ready. Selected target is green. Axis balls show roll/pitch/yaw as red/green/blue. "
        f"mode={MODE_NAMES[mode]} can={'connected' if can_robots else 'not connected'} "
        f"left={np.round(targets['left'], 4).tolist()} right={np.round(targets['right'], 4).tolist()}",
        flush=True,
    )

    step = 0
    last_command_time = 0.0
    last_real_command_time = 0.0
    last_selected_idx = 0
    last_mode = mode
    last_enable_count = 0
    last_disable_count = 0
    last_estop_count = 0
    real_enabled = False
    estopped = False
    last_print = 0.0
    command_dt = 1.0 / max(float(args.command_hz), 1e-6)
    real_command_dt = 1.0 / max(float(args.real_command_hz), 1e-6)
    try:
        while True:
            now = time.monotonic()
            if control_window:
                selected_idx = int(control_window["selected_idx"].value)
                selected = "left" if selected_idx == 0 else "right"
                if selected_idx != last_selected_idx:
                    last_selected_idx = selected_idx
                    _refresh_markers(markers, targets, target_quats, selected)
                mode = int(control_window["mode_value"].value)
                enable_count = int(control_window["enable_count"].value)
                disable_count = int(control_window["disable_count"].value)
                estop_count = int(control_window["estop_count"].value)
                if mode != last_mode:
                    last_mode = mode
                    print(f"[ik] mode -> {MODE_NAMES.get(mode, 'unknown')}", flush=True)
                    if mode == MODE_ACTIVE:
                        for side in ("left", "right"):
                            targets[side] = _tensor_to_np(eef_links[side].get_pos()).reshape(3).astype(np.float32)
                            target_quats[side] = _tensor_to_np(eef_links[side].get_quat()).reshape(4).astype(np.float32)
                        _refresh_markers(markers, targets, target_quats, selected)
                if estop_count != last_estop_count:
                    last_estop_count = estop_count
                    estopped = True
                    real_enabled = False
                    if can_robots:
                        print("[can] E-STOP requested", flush=True)
                        _safe_can_call(can_robots, "electronic_emergency_stop")
                    else:
                        print("[ik] E-STOP requested, but CAN is not connected.", flush=True)
                if disable_count != last_disable_count:
                    last_disable_count = disable_count
                    real_enabled = False
                    if can_robots:
                        print("[can] disable requested", flush=True)
                        _safe_can_call(can_robots, "disable")
                    else:
                        print("[ik] disable requested, but CAN is not connected.", flush=True)
                if enable_count != last_enable_count:
                    last_enable_count = enable_count
                    if can_robots:
                        if estopped:
                            print("[can] reset after E-STOP before enabling", flush=True)
                            _safe_can_call(can_robots, "reset")
                            estopped = False
                        print(f"[can] enable requested, speed_percent={args.speed_percent}", flush=True)
                        _safe_can_call(can_robots, "set_normal_mode")
                        _safe_can_call(can_robots, "set_auto_set_motion_mode_enabled", False)
                        _safe_can_call(can_robots, "set_speed_percent", int(args.speed_percent))
                        real_enabled = _safe_can_call(can_robots, "enable")
                    else:
                        real_enabled = False
                        print("[ik] enable requested, but CAN is not connected.", flush=True)

            if now - last_command_time >= command_dt:
                elapsed = max(now - last_command_time, command_dt) if last_command_time > 0.0 else command_dt
                last_command_time = now
                if mode == MODE_PASSIVE and can_robots:
                    feedback = _read_can_joints(can_robots)
                    for name, q_feedback in feedback.items():
                        if q_feedback is None:
                            continue
                        q_state[name] = q_feedback
                        robots[name].set_dofs_position(q_state[name], arm_dofs[name], zero_velocity=True)
                        robots[name].control_dofs_position(q_state[name], arm_dofs[name])
                    scene.step()
                    for name in ("left", "right"):
                        targets[name] = _tensor_to_np(eef_links[name].get_pos()).reshape(3).astype(np.float32)
                        target_quats[name] = _tensor_to_np(eef_links[name].get_quat()).reshape(4).astype(np.float32)
                    if now - last_print > 1.0:
                        last_print = now
                        print(
                            f"[passive] q_left={np.round(q_state['left'], 3).tolist()} "
                            f"q_right={np.round(q_state['right'], 3).tolist()}",
                            flush=True,
                        )
                else:
                    keys = _pressed(control_window)
                    delta = _target_delta(keys, args.target_speed, elapsed)
                    if np.linalg.norm(delta) > 0.0:
                        targets[selected] = targets[selected] + delta

                    for name, robot in robots.items():
                        qpos_init = _tensor_to_np(robot.get_qpos()).reshape(-1)
                        qpos, error = _solve_ik(
                            robot,
                            eef_links[name],
                            targets[name],
                            target_quats[name],
                            qpos_init,
                            arm_dofs[name],
                            args,
                        )
                        solved = qpos[arm_dofs[name]].astype(np.float32)
                        dq = np.clip(solved - q_state[name], -float(args.max_joint_step), float(args.max_joint_step))
                        q_state[name] = q_state[name] + dq
                        robot.control_dofs_position(q_state[name], arm_dofs[name])
                        if name == selected and now - last_print > 1.0:
                            last_print = now
                            real_q = _read_can_joint(can_robots, name)
                            real_text = ""
                            if real_q is not None:
                                cmd_minus_real = q_state[name] - real_q
                                real_text = (
                                    f" real_q={np.round(real_q, 3).tolist()}"
                                    f" cmd_minus_real={np.round(cmd_minus_real, 3).tolist()}"
                                )
                            print(
                                f"[active:{name}] target={np.round(targets[name], 4).tolist()} "
                                f"q={np.round(q_state[name], 3).tolist()} "
                                f"pos_err={float(np.linalg.norm(error[:3])):.5f} "
                                f"rot_err={float(np.linalg.norm(error[3:])):.5f} "
                                f"real_enabled={real_enabled}{real_text}",
                                flush=True,
                            )
                    if (
                        can_robots
                        and real_enabled
                        and not estopped
                        and mode == MODE_ACTIVE
                        and now - last_real_command_time >= real_command_dt
                    ):
                        last_real_command_time = now
                        for name, can_robot in can_robots.items():
                            try:
                                can_robot.move_j([float(v) for v in q_state[name]])
                                can_robot.set_motion_mode("j")
                            except Exception as exc:
                                real_enabled = False
                                print(f"[can:{name}] move_j failed; disabling real commands: {exc}", flush=True)
                _refresh_markers(markers, targets, target_quats, selected)

            scene.step()
            step += 1
            if control_window and control_window["stop_flag"].value:
                break
            if args.steps >= 0 and step >= args.steps:
                break
            if args.steps < 0 and args.no_viewer and not control_window:
                break
            if not args.no_viewer and not scene.viewer.is_alive():
                break
    finally:
        _shutdown_control_window(control_window)
        if real_enabled and can_robots:
            print("[can] disabling arms on exit", flush=True)
            _safe_can_call(can_robots, "disable")
        _disconnect_can_arms(can_robots)


if __name__ == "__main__":
    main()
