#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import add_scene_glb as harness  # noqa: E402


DEFAULT_OUTPUT = ROOT_DIR / "assets" / "scene_support_collider_debug.json"
DEFAULT_BOTTLE_POS = (-0.016, 0.32889, 0.82667)
DEFAULT_BOTTLE_EULER = (0.0, 0.0, 0.0)
DEFAULT_COLLIDER_POS = (-0.016, 0.32889, 0.805)
DEFAULT_COLLIDER_SIZE = (0.70, 0.70, 0.04)

POS_RANGE_M = 1.5
Z_RANGE_M = 1.5
SIZE_RANGE_M = 2.0
SIZE_Z_RANGE_M = 0.30
FINE_POS_STEP_M = 0.001
FINE_SIZE_STEP_M = 0.001


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


def _payload_from_values(values: tuple[float, ...]) -> dict[str, object]:
    pos = tuple(float(v) for v in values[:3])
    size = tuple(float(v) for v in values[3:6])
    lower = tuple(float(pos[i] - size[i] * 0.5) for i in range(3))
    upper = tuple(float(pos[i] + size[i] * 0.5) for i in range(3))
    return {
        "description": "Scene table/support collision proxy box. The visual scene.glb remains visual-only.",
        "pos_m": list(pos),
        "size_m": list(size),
        "lower_m": list(lower),
        "upper_m": list(upper),
    }


def _print_payload(payload: dict[str, object]) -> None:
    pos = tuple(float(v) for v in payload["pos_m"])
    size = tuple(float(v) for v in payload["size_m"])
    lower = tuple(float(v) for v in payload["lower_m"])
    upper = tuple(float(v) for v in payload["upper_m"])
    print("[table-collider-debug] scene_support_proxy", flush=True)
    print(f"  pos={_format_tuple(pos)} size={_format_tuple(size)}", flush=True)
    print(f"  lower={_format_tuple(lower)} upper={_format_tuple(upper)}", flush=True)
    print(
        "  add_scene_glb_args:\n"
        f"    --table-collider-pos {pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f} "
        f"--table-collider-size {size[0]:.6f},{size[1]:.6f},{size[2]:.6f} "
        "--show-table-collider",
        flush=True,
    )
    print(
        "  python_constants:\n"
        f"    DEFAULT_SCENE_SUPPORT_COLLIDER_POS = {_format_tuple(pos)}\n"
        f"    DEFAULT_SCENE_SUPPORT_COLLIDER_SIZE = {_format_tuple(size)}",
        flush=True,
    )


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[table-collider-debug] saved {path}", flush=True)


def _panel_main(initial_values, values, running, print_counter, save_counter, reset_counter, bottle_reset_counter, stop_flag):
    import tkinter as tk
    from tkinter import ttk

    initializing = True
    specs = (
        ("pos x", -POS_RANGE_M, POS_RANGE_M, FINE_POS_STEP_M, "m"),
        ("pos y", -POS_RANGE_M, POS_RANGE_M, FINE_POS_STEP_M, "m"),
        ("pos z", -1.0, Z_RANGE_M, FINE_POS_STEP_M, "m"),
        ("size x", 0.001, SIZE_RANGE_M, FINE_SIZE_STEP_M, "m"),
        ("size y", 0.001, SIZE_RANGE_M, FINE_SIZE_STEP_M, "m"),
        ("size z", 0.001, SIZE_Z_RANGE_M, FINE_SIZE_STEP_M, "m"),
    )

    def set_value(idx: int, value: float | str) -> None:
        _, lower, upper, _, _ = specs[idx]
        clamped = max(float(lower), min(float(upper), float(value)))
        values[idx] = clamped
        value_labels[idx].config(text=f"{clamped: .5f}")
        if idx >= 3 and not initializing:
            size_note.config(text="Size changed: restart script with printed args to rebuild collision geometry.")

    def step_value(idx: int, direction: int) -> None:
        current = float(sliders[idx].get())
        step = float(specs[idx][3])
        next_value = current + float(direction) * step
        sliders[idx].set(next_value)
        set_value(idx, next_value)

    def reset_collider() -> None:
        for idx, slider in enumerate(sliders):
            slider.set(float(initial_values[idx]))
            set_value(idx, float(initial_values[idx]))
        size_note.config(text="Collider reset to initial values.")
        reset_counter.value += 1

    def set_running(enabled: bool) -> None:
        running.value = bool(enabled)
        start_button.config(text="Pause Step" if running.value else "Start Step")
        status_label.config(text="Stepping" if running.value else "Paused")

    def close() -> None:
        stop_flag.value = True
        root.destroy()
        root.quit()

    root = tk.Tk()
    root.title("Table Collider Calibration")
    root.geometry("880x500")
    root.minsize(760, 430)

    ttk.Label(root, text="Scene table collision proxy", font=("Arial", 12, "bold")).pack(
        fill=tk.X, padx=12, pady=(12, 4)
    )
    ttk.Label(
        root,
        text="Position updates live. Size is stored/printed and takes effect after rebuilding the scene.",
    ).pack(fill=tk.X, padx=12, pady=(0, 6))

    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

    sliders = []
    value_labels = []
    for idx, (label, lower, upper, step, unit) in enumerate(specs):
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text=f"{label} ({unit})", width=12).pack(side=tk.LEFT)
        slider = ttk.Scale(
            row,
            from_=float(lower),
            to=float(upper),
            orient=tk.HORIZONTAL,
            command=lambda value, i=idx: set_value(i, value),
        )
        slider.set(float(initial_values[idx]))
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(row, text="-", width=3, command=lambda i=idx: step_value(i, -1)).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(row, text="+", width=3, command=lambda i=idx: step_value(i, 1)).pack(side=tk.LEFT)
        value_label = ttk.Label(row, text=f"{float(initial_values[idx]): .5f}", width=12)
        value_label.pack(side=tk.RIGHT, padx=(8, 0))
        sliders.append(slider)
        value_labels.append(value_label)

    size_note = ttk.Label(root, text="Collision size is fixed after Genesis scene.build().")
    size_note.pack(fill=tk.X, padx=12, pady=(0, 6))
    initializing = False

    buttons = ttk.Frame(root)
    buttons.pack(fill=tk.X, padx=12, pady=(4, 12))
    start_button = ttk.Button(buttons, text="Start Step", command=lambda: set_running(not bool(running.value)))
    start_button.pack(side=tk.LEFT)
    ttk.Button(buttons, text="Reset Bottle", command=lambda: setattr(bottle_reset_counter, "value", bottle_reset_counter.value + 1)).pack(
        side=tk.LEFT, padx=8
    )
    ttk.Button(buttons, text="Reset Collider", command=reset_collider).pack(side=tk.LEFT)
    ttk.Button(buttons, text="Print", command=lambda: setattr(print_counter, "value", print_counter.value + 1)).pack(
        side=tk.LEFT, padx=8
    )
    ttk.Button(buttons, text="Save JSON", command=lambda: setattr(save_counter, "value", save_counter.value + 1)).pack(
        side=tk.LEFT
    )
    ttk.Button(buttons, text="Close", command=close).pack(side=tk.RIGHT)
    status_label = ttk.Label(buttons, text="Paused")
    status_label.pack(side=tk.RIGHT, padx=12)

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()


def _create_panel(initial_values: tuple[float, ...]) -> dict[str, object]:
    values = multiprocessing.RawArray("d", initial_values)
    running = multiprocessing.RawValue("b", False)
    print_counter = multiprocessing.RawValue("i", 0)
    save_counter = multiprocessing.RawValue("i", 0)
    reset_counter = multiprocessing.RawValue("i", 0)
    bottle_reset_counter = multiprocessing.RawValue("i", 0)
    stop_flag = multiprocessing.RawValue("b", False)
    process = multiprocessing.Process(
        target=_panel_main,
        args=(initial_values, values, running, print_counter, save_counter, reset_counter, bottle_reset_counter, stop_flag),
        daemon=True,
    )
    process.start()
    return {
        "values": values,
        "running": running,
        "print_counter": print_counter,
        "save_counter": save_counter,
        "reset_counter": reset_counter,
        "bottle_reset_counter": bottle_reset_counter,
        "stop_flag": stop_flag,
        "process": process,
    }


def _shutdown_panel(panel: dict[str, object] | None) -> None:
    if panel is None:
        return
    panel["stop_flag"].value = True
    process = panel["process"]
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)


def _read_values(panel: dict[str, object]) -> tuple[float, ...]:
    return tuple(float(panel["values"][idx]) for idx in range(6))


def _apply_table_pose(table_entity: object | None, pos: tuple[float, float, float]) -> None:
    if table_entity is None:
        return
    harness._set_entity_pose(table_entity, np.asarray(pos, dtype=np.float64), np.eye(3, dtype=np.float64))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug the visual scene's table/support Box collision proxy.")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--glb", type=Path, default=harness.DEFAULT_GLB)
    parser.add_argument("--bottle-glb", type=Path, default=harness.DEFAULT_BOTTLE_GLB)
    parser.add_argument("--bottle-pos", type=_vec3, default=DEFAULT_BOTTLE_POS)
    parser.add_argument("--bottle-euler", type=_vec3, default=DEFAULT_BOTTLE_EULER)
    parser.add_argument("--initial-pos", type=_vec3, default=DEFAULT_COLLIDER_POS)
    parser.add_argument("--initial-size", type=_vec3, default=DEFAULT_COLLIDER_SIZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-arm-assembly", action="store_true", help="Skip the Nero assembly for faster collider-only tuning.")
    parser.add_argument("--linker-hand-collision", action="store_true", help="Enable Linker Hand collision while debugging.")
    parser.add_argument("--no-d455-gui", action="store_true", help="Disable the D455 camera GUI.")
    parser.add_argument("--no-d405-gui", action="store_true", help="Disable the D405 camera GUI.")
    parser.add_argument("--no-viewer", action="store_true", help="Build once and print the initial collider values.")
    parser.add_argument("--start-step", action="store_true", help="Start stepping immediately.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    initial_values = tuple(float(v) for v in (*args.initial_pos, *args.initial_size))

    scene, _ = harness.create_scene(
        args.glb,
        show_viewer=not args.no_viewer,
        backend=args.backend,
        collision=False,
        add_bottle=True,
        bottle_path=args.bottle_glb,
        bottle_pos=args.bottle_pos,
        bottle_euler=args.bottle_euler,
        bottle_collision=True,
        add_table_collider=True,
        table_collider_pos=args.initial_pos,
        table_collider_size=args.initial_size,
        show_table_collider=True,
        add_arm_assembly=not args.no_arm_assembly,
        linker_hand_collision=bool(args.linker_hand_collision),
        d455_rgb_gui=not args.no_d455_gui,
        d405_camera_gui=not args.no_d405_gui,
    )

    payload = _payload_from_values(initial_values)
    _print_payload(payload)
    if args.no_viewer:
        return

    panel = _create_panel(initial_values)
    panel["running"].value = bool(args.start_step)
    last_values = initial_values
    last_print_counter = 0
    last_save_counter = 0
    last_bottle_reset_counter = 0
    size_warning_printed = False

    try:
        while scene.viewer.is_alive():
            if panel["stop_flag"].value:
                break

            values = _read_values(panel)
            if values[:3] != last_values[:3]:
                _apply_table_pose(scene.table_collider_entity, tuple(values[:3]))
            if values[3:] != last_values[3:] and not size_warning_printed:
                print(
                    "[table-collider-debug] size sliders changed. "
                    "Genesis collision geometry is fixed after build; print/save and restart with --initial-size to verify.",
                    flush=True,
                )
                size_warning_printed = True
            last_values = values

            if int(panel["print_counter"].value) != last_print_counter:
                last_print_counter = int(panel["print_counter"].value)
                _print_payload(_payload_from_values(values))
            if int(panel["save_counter"].value) != last_save_counter:
                last_save_counter = int(panel["save_counter"].value)
                _write_payload(args.output.expanduser().resolve(), _payload_from_values(values))
            if int(panel["bottle_reset_counter"].value) != last_bottle_reset_counter:
                last_bottle_reset_counter = int(panel["bottle_reset_counter"].value)
                harness._apply_bottle_pose(scene.bottle_entity, args.bottle_pos, args.bottle_euler)

            if bool(panel["running"].value):
                harness._step_scene_with_attached_parts(scene)
            else:
                scene.visualizer.update(force=True)
                time.sleep(1.0 / 60.0)
    finally:
        _shutdown_panel(panel)


if __name__ == "__main__":
    main()
