#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import genesis as gs
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import add_scene_glb as harness  # noqa: E402


DEFAULT_GLB = ROOT_DIR / "scene" / "bottle.glb"
DEFAULT_OUTPUT_JSON = ROOT_DIR / "assets" / "generated" / "bottle_cylinder_collision_proxy.json"
DEFAULT_OUTPUT_URDF = ROOT_DIR / "assets" / "generated" / "bottle_cylinder_collision_proxy.urdf"
DEFAULT_PROXY_VISUAL_MESH = ROOT_DIR / "assets" / "generated" / "unit_cylinder_proxy.obj"

DEFAULT_PROXY_POS = (0.0, 0.0, 0.0)
DEFAULT_PROXY_EULER = (0.0, 0.0, 0.0)
DEFAULT_PROXY_DIAMETER = 0.070
DEFAULT_PROXY_HEIGHT = 0.190

POS_RANGE_M = 0.30
ROT_RANGE_DEG = 180.0
DIAMETER_RANGE_M = 0.30
HEIGHT_RANGE_M = 0.50
POS_STEP_M = 0.001
ROT_STEP_DEG = 0.1
SIZE_STEP_M = 0.001
HIDDEN_POS = (0.0, 0.0, -10.0)


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


def _payload_from_values(values: tuple[float, ...], *, glb: Path, output_urdf: Path) -> dict[str, object]:
    pos = tuple(float(v) for v in values[:3])
    euler = tuple(float(v) for v in values[3:6])
    diameter = float(values[6])
    height = float(values[7])
    return {
        "description": "GLB visual plus simple cylinder collision proxy. Cylinder frame is relative to the GLB object root.",
        "glb": str(glb.expanduser().resolve()),
        "output_urdf": str(output_urdf.expanduser().resolve()),
        "collision": {
            "type": "cylinder",
            "pos_m": list(pos),
            "euler_xyz_deg": list(euler),
            "diameter_m": diameter,
            "radius_m": diameter * 0.5,
            "height_m": height,
        },
    }


def _print_payload(payload: dict[str, object]) -> None:
    collision = payload["collision"]
    pos = tuple(float(v) for v in collision["pos_m"])
    euler = tuple(float(v) for v in collision["euler_xyz_deg"])
    diameter = float(collision["diameter_m"])
    height = float(collision["height_m"])
    print("[glb-cylinder-proxy]", flush=True)
    print(f"  glb={payload['glb']}", flush=True)
    print(
        f"  cylinder pos={_format_tuple(pos)} euler_deg={_format_tuple(euler, 3)} "
        f"diameter={diameter:.6f} height={height:.6f}",
        flush=True,
    )
    print(
        "  debug_args:\n"
        f"    --proxy-pos {pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f} "
        f"--proxy-euler {euler[0]:.3f},{euler[1]:.3f},{euler[2]:.3f} "
        f"--proxy-diameter {diameter:.6f} --proxy-height {height:.6f}",
        flush=True,
    )
    print(f"  output_urdf={payload['output_urdf']}", flush=True)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[glb-cylinder-proxy] saved json {path}", flush=True)


def _origin(parent: ET.Element, xyz: tuple[float, float, float], rpy_deg: tuple[float, float, float]) -> None:
    rpy = tuple(float(np.deg2rad(value)) for value in rpy_deg)
    ET.SubElement(
        parent,
        "origin",
        {
            "xyz": " ".join(f"{float(value):.9g}" for value in xyz),
            "rpy": " ".join(f"{float(value):.9g}" for value in rpy),
        },
    )


def _write_urdf(path: Path, *, glb: Path, values: tuple[float, ...], mass_kg: float) -> None:
    pos = tuple(float(v) for v in values[:3])
    euler = tuple(float(v) for v in values[3:6])
    diameter = float(values[6])
    height = float(values[7])
    radius = diameter * 0.5
    mass = max(float(mass_kg), 1.0e-6)

    root = ET.Element("robot", {"name": f"{glb.stem}_cylinder_collision_proxy"})
    link = ET.SubElement(root, "link", {"name": "object"})

    inertial = ET.SubElement(link, "inertial")
    _origin(inertial, pos, euler)
    ET.SubElement(inertial, "mass", {"value": f"{mass:.9g}"})
    ixx = (1.0 / 12.0) * mass * (3.0 * radius * radius + height * height)
    izz = 0.5 * mass * radius * radius
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{ixx:.9g}",
            "ixy": "0",
            "ixz": "0",
            "iyy": f"{ixx:.9g}",
            "iyz": "0",
            "izz": f"{izz:.9g}",
        },
    )

    visual = ET.SubElement(link, "visual")
    _origin(visual, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    visual_geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(visual_geometry, "mesh", {"filename": str(glb.expanduser().resolve())})

    collision = ET.SubElement(link, "collision", {"name": "cylinder_proxy"})
    _origin(collision, pos, euler)
    collision_geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(collision_geometry, "cylinder", {"radius": f"{radius:.9g}", "length": f"{height:.9g}"})

    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    print(f"[glb-cylinder-proxy] wrote urdf {path}", flush=True)


def _ensure_unit_cylinder_mesh(path: Path, *, segments: int = 64) -> Path:
    """Write a unit cylinder OBJ: radius=0.5, height=1.0, axis=+Z."""
    path = path.expanduser().resolve()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices: list[tuple[float, float, float]] = []
    for z in (-0.5, 0.5):
        for idx in range(int(segments)):
            theta = 2.0 * np.pi * float(idx) / float(segments)
            vertices.append((0.5 * float(np.cos(theta)), 0.5 * float(np.sin(theta)), float(z)))
    bottom_center_index = len(vertices) + 1
    vertices.append((0.0, 0.0, -0.5))
    top_center_index = len(vertices) + 1
    vertices.append((0.0, 0.0, 0.5))

    faces: list[tuple[int, int, int]] = []
    for idx in range(int(segments)):
        nxt = (idx + 1) % int(segments)
        bottom_a = idx + 1
        bottom_b = nxt + 1
        top_a = int(segments) + idx + 1
        top_b = int(segments) + nxt + 1
        faces.append((bottom_a, bottom_b, top_b))
        faces.append((bottom_a, top_b, top_a))
        faces.append((bottom_center_index, bottom_b, bottom_a))
        faces.append((top_center_index, top_a, top_b))

    with path.open("w", encoding="ascii") as file:
        file.write("# unit cylinder radius=0.5 height=1.0 axis=+Z\n")
        for vertex in vertices:
            file.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        for face in faces:
            file.write(f"f {face[0]} {face[1]} {face[2]}\n")
    return path


def _panel_main(initial_values, values, running, print_counter, save_counter, urdf_counter, reset_counter, stop_flag):
    import tkinter as tk
    from tkinter import ttk

    specs = (
        ("pos x", -POS_RANGE_M, POS_RANGE_M, POS_STEP_M, "m"),
        ("pos y", -POS_RANGE_M, POS_RANGE_M, POS_STEP_M, "m"),
        ("pos z", -POS_RANGE_M, POS_RANGE_M, POS_STEP_M, "m"),
        ("roll", -ROT_RANGE_DEG, ROT_RANGE_DEG, ROT_STEP_DEG, "deg"),
        ("pitch", -ROT_RANGE_DEG, ROT_RANGE_DEG, ROT_STEP_DEG, "deg"),
        ("yaw", -ROT_RANGE_DEG, ROT_RANGE_DEG, ROT_STEP_DEG, "deg"),
        ("diameter", 0.001, DIAMETER_RANGE_M, SIZE_STEP_M, "m"),
        ("height", 0.001, HEIGHT_RANGE_M, SIZE_STEP_M, "m"),
    )

    def set_value(idx: int, value: float | str, *, update_slider: bool = False) -> None:
        _, lower, upper, _, _ = specs[idx]
        clamped = max(float(lower), min(float(upper), float(value)))
        values[idx] = clamped
        labels[idx].config(text=f"{clamped: .5f}")
        if update_slider:
            sliders[idx].set(clamped)

    def nudge(idx: int, direction: int) -> None:
        set_value(idx, float(values[idx]) + float(direction) * float(specs[idx][3]), update_slider=True)

    def set_running(enabled: bool) -> None:
        running.value = bool(enabled)
        start_button.config(text="Pause Step" if running.value else "Start Step")
        status_label.config(text="Stepping" if running.value else "Paused")

    def reset() -> None:
        for idx in range(len(specs)):
            set_value(idx, float(initial_values[idx]), update_slider=True)
        reset_counter.value += 1

    def close() -> None:
        stop_flag.value = True
        root.destroy()
        root.quit()

    root = tk.Tk()
    root.title("GLB Cylinder Collision Proxy")
    root.geometry("900x610")
    root.minsize(760, 520)

    ttk.Label(root, text="Cylinder collision proxy relative to GLB", font=("Arial", 12, "bold")).pack(
        fill=tk.X, padx=12, pady=(12, 4)
    )
    ttk.Label(root, text="Adjust the translucent cylinder until it overlaps the visual object, then Save URDF.").pack(
        fill=tk.X, padx=12, pady=(0, 8)
    )

    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)
    sliders = []
    labels = []
    for idx, (label, lower, upper, _, unit) in enumerate(specs):
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text=f"{label} ({unit})", width=13).pack(side=tk.LEFT)
        ttk.Button(row, text="-", width=3, command=lambda i=idx: nudge(i, -1)).pack(side=tk.LEFT)
        slider = ttk.Scale(
            row,
            from_=float(lower),
            to=float(upper),
            orient=tk.HORIZONTAL,
            command=lambda value, i=idx: set_value(i, value),
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(row, text="+", width=3, command=lambda i=idx: nudge(i, 1)).pack(side=tk.LEFT)
        value_label = ttk.Label(row, text="", width=12)
        value_label.pack(side=tk.RIGHT, padx=(8, 0))
        sliders.append(slider)
        labels.append(value_label)
        set_value(idx, float(initial_values[idx]), update_slider=True)

    buttons = ttk.Frame(root)
    buttons.pack(fill=tk.X, padx=12, pady=(4, 12))
    start_button = ttk.Button(buttons, text="Start Step", command=lambda: set_running(not bool(running.value)))
    start_button.pack(side=tk.LEFT)
    ttk.Button(buttons, text="Reset", command=reset).pack(side=tk.LEFT, padx=8)
    ttk.Button(buttons, text="Print", command=lambda: setattr(print_counter, "value", print_counter.value + 1)).pack(
        side=tk.LEFT
    )
    ttk.Button(buttons, text="Save JSON", command=lambda: setattr(save_counter, "value", save_counter.value + 1)).pack(
        side=tk.LEFT, padx=8
    )
    ttk.Button(buttons, text="Save URDF", command=lambda: setattr(urdf_counter, "value", urdf_counter.value + 1)).pack(
        side=tk.LEFT
    )
    ttk.Button(buttons, text="Close", command=close).pack(side=tk.RIGHT)
    status_label = ttk.Label(buttons, text="Paused")
    status_label.pack(side=tk.RIGHT, padx=12)

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()


def _create_panel(initial_values: tuple[float, ...]) -> dict[str, object]:
    values = multiprocessing.Array("d", initial_values)
    running = multiprocessing.Value("b", False)
    print_counter = multiprocessing.Value("i", 0)
    save_counter = multiprocessing.Value("i", 0)
    urdf_counter = multiprocessing.Value("i", 0)
    reset_counter = multiprocessing.Value("i", 0)
    stop_flag = multiprocessing.Value("b", False)
    process = multiprocessing.Process(
        target=_panel_main,
        args=(initial_values, values, running, print_counter, save_counter, urdf_counter, reset_counter, stop_flag),
        daemon=True,
    )
    process.start()
    return {
        "values": values,
        "running": running,
        "print_counter": print_counter,
        "save_counter": save_counter,
        "urdf_counter": urdf_counter,
        "reset_counter": reset_counter,
        "stop_flag": stop_flag,
        "process": process,
    }


def _read_panel(panel: dict[str, object]) -> tuple[tuple[float, ...], bool, int, int, int, int, bool]:
    values = tuple(float(panel["values"][idx]) for idx in range(8))
    return (
        values,
        bool(panel["running"].value),
        int(panel["print_counter"].value),
        int(panel["save_counter"].value),
        int(panel["urdf_counter"].value),
        int(panel["reset_counter"].value),
        bool(panel["stop_flag"].value),
    )


def _shutdown_panel(panel: dict[str, object] | None) -> None:
    if not panel:
        return
    panel["stop_flag"].value = True
    process = panel.get("process")
    if process is not None:
        process.join(timeout=1.0)


def _set_entity_pose(entity: object, pos: tuple[float, float, float], euler_deg: tuple[float, float, float]) -> None:
    harness._set_entity_pose(  # noqa: SLF001
        entity,
        np.asarray(pos, dtype=np.float64),
        harness._rotation_from_euler_deg(euler_deg),  # noqa: SLF001
    )


def _make_proxy_visual(scene: gs.Scene, values: tuple[float, ...]) -> object:
    diameter = max(float(values[6]), 1.0e-4)
    height = max(float(values[7]), 1.0e-4)
    mesh = _ensure_unit_cylinder_mesh(DEFAULT_PROXY_VISUAL_MESH)
    return scene.add_entity(
        gs.morphs.Mesh(
            file=str(mesh),
            pos=HIDDEN_POS,
            scale=(diameter, diameter, height),
            fixed=True,
            collision=False,
            convexify=False,
        ),
        surface=gs.surfaces.Plastic(color=(0.0, 0.85, 1.0, 0.35), roughness=0.4),
        name="cylinder_collision_proxy_visual",
    )


def _apply_proxy_pose(proxy: object, values: tuple[float, ...]) -> None:
    pos = tuple(float(v) for v in values[:3])
    euler = tuple(float(v) for v in values[3:6])
    _set_entity_pose(proxy, pos, euler)


def _build_debug_scene(args: argparse.Namespace, glb: Path, values: tuple[float, ...]) -> tuple[gs.Scene, object]:
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.45, -0.65, 0.35),
            camera_lookat=(0.0, 0.0, 0.08),
            camera_fov=35,
            res=(1280, 720),
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0)),
        show_viewer=True,
    )
    scene.add_entity(
        gs.morphs.Mesh(
            file=str(glb),
            pos=(0.0, 0.0, 0.0),
            euler=(0.0, 0.0, 0.0),
            fixed=True,
            collision=False,
            convexify=False,
        ),
        surface=gs.surfaces.Default(vis_mode="visual"),
        name="glb_visual",
    )
    proxy = _make_proxy_visual(scene, values)
    scene.build()
    _apply_proxy_pose(proxy, values)
    print(
        "[glb-cylinder-proxy] preview rebuilt "
        f"diameter={float(values[6]):.6f} height={float(values[7]):.6f}",
        flush=True,
    )
    return scene, proxy


def _destroy_scene(scene: gs.Scene | None) -> None:
    if scene is None:
        return
    try:
        scene.destroy()
    except Exception:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a cylinder collision proxy for a visual GLB object.")
    parser.add_argument("--glb", type=Path, default=DEFAULT_GLB)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--proxy-pos", type=_vec3, default=DEFAULT_PROXY_POS)
    parser.add_argument("--proxy-euler", type=_vec3, default=DEFAULT_PROXY_EULER)
    parser.add_argument("--proxy-diameter", type=float, default=DEFAULT_PROXY_DIAMETER)
    parser.add_argument("--proxy-height", type=float, default=DEFAULT_PROXY_HEIGHT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-urdf", type=Path, default=DEFAULT_OUTPUT_URDF)
    parser.add_argument("--mass", type=float, default=0.12)
    parser.add_argument("--print-every", type=int, default=120)
    parser.add_argument(
        "--size-rebuild-delay-s",
        type=float,
        default=0.25,
        help="Debounce delay before rebuilding the preview after diameter/height changes.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    glb = args.glb.expanduser().resolve()
    if not glb.exists():
        raise FileNotFoundError(f"GLB not found: {glb}")

    initial_values = (
        *tuple(float(v) for v in args.proxy_pos),
        *tuple(float(v) for v in args.proxy_euler),
        max(float(args.proxy_diameter), 1.0e-4),
        max(float(args.proxy_height), 1.0e-4),
    )

    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu)
    scene, proxy = _build_debug_scene(args, glb, initial_values)

    panel = _create_panel(initial_values)
    last_values = initial_values
    last_size = initial_values[6:8]
    pending_rebuild_values: tuple[float, ...] | None = None
    pending_rebuild_after_s: float | None = None
    last_print_counter = 0
    last_save_counter = 0
    last_urdf_counter = 0
    last_reset_counter = 0
    frame = 0
    try:
        while True:
            values, running, print_counter, save_counter, urdf_counter, reset_counter, stop_requested = _read_panel(panel)
            if stop_requested:
                break
            if values != last_values or reset_counter != last_reset_counter:
                _apply_proxy_pose(proxy, values)
                if values[6:8] != last_size:
                    pending_rebuild_values = values
                    pending_rebuild_after_s = time.monotonic() + max(0.0, float(args.size_rebuild_delay_s))
                last_values = values
                last_reset_counter = reset_counter

            if pending_rebuild_values is not None and pending_rebuild_after_s is not None:
                if time.monotonic() >= pending_rebuild_after_s:
                    _destroy_scene(scene)
                    scene, proxy = _build_debug_scene(args, glb, pending_rebuild_values)
                    last_size = pending_rebuild_values[6:8]
                    last_values = pending_rebuild_values
                    pending_rebuild_values = None
                    pending_rebuild_after_s = None

            payload = _payload_from_values(values, glb=glb, output_urdf=args.output_urdf)
            if print_counter != last_print_counter:
                _print_payload(payload)
                last_print_counter = print_counter
            if save_counter != last_save_counter:
                _write_json(args.output_json.expanduser().resolve(), payload)
                last_save_counter = save_counter
            if urdf_counter != last_urdf_counter:
                output_urdf = args.output_urdf.expanduser().resolve()
                _write_urdf(output_urdf, glb=glb, values=values, mass_kg=float(args.mass))
                payload = _payload_from_values(values, glb=glb, output_urdf=output_urdf)
                _write_json(args.output_json.expanduser().resolve(), payload)
                _print_payload(payload)
                last_urdf_counter = urdf_counter

            if running:
                scene.step()
            else:
                scene.visualizer.update(force=True)
                time.sleep(1.0 / 60.0)
            frame += 1
            if frame % max(1, int(args.print_every)) == 0:
                _print_payload(payload)
    finally:
        _destroy_scene(scene)
        _shutdown_panel(panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
