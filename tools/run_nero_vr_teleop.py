#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop_stack.robots import NeroRuntimeRobotConfig, NeroRuntimeRobotInterface, NeroTeleopMappingConfig  # noqa: E402
from teleop_stack.session import QuestRobotSession, QuestRobotSessionConfig  # noqa: E402


def _parse_vec3(text: str, *, name: str) -> tuple[float, float, float]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"{name} must contain 3 comma-separated floats")
    return values


def _parse_axis_map(text: str) -> tuple[str, str, str]:
    values = tuple(part.strip().lower() for part in text.split(",") if part.strip())
    if len(values) != 3:
        raise argparse.ArgumentTypeError("axis map must contain 3 comma-separated tokens")
    valid = {"x", "y", "z", "+x", "+y", "+z", "-x", "-y", "-z"}
    invalid = [value for value in values if value not in valid]
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid axis map token(s): {', '.join(invalid)}")
    return values  # type: ignore[return-value]


def _arm_pose_command_mode(*, pose_input_mode: str, use_teleop_orientation: bool) -> str:
    if pose_input_mode != "hand_abs":
        return "legacy_retargeted_ee"
    return "raw_wrist_position_full_orientation" if use_teleop_orientation else "raw_wrist_position_fixed_orientation"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Quest/OpenXR VR teleop into the local Genesis Nero runtime.")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run. 0 means run until interrupted.")
    parser.add_argument("--arm-side", choices=["left", "right"], default="right")
    parser.add_argument("--pose-input-mode", choices=["controller_abs", "hand_abs"], default="hand_abs")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--headless", action="store_true", help="Do not open the Genesis viewer.")
    parser.add_argument("--markers-only", action="store_true", help="Move only target markers; do not solve IK or move arms.")
    parser.add_argument("--loop-hz", type=float, default=60.0)
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--isaac-teleop-root", default=None, help="Override IsaacTeleop root. Also supports ISAAC_TELEOP_ROOT.")
    parser.add_argument("--startup-timeout-s", type=float, default=300.0)
    parser.add_argument("--teleop-trace-path", default=None, help="Optional JSONL trace path.")
    parser.add_argument(
        "--cloudxr-env-path",
        type=Path,
        default=Path.home() / ".cloudxr" / "run" / "cloudxr.env",
        help="CloudXR env file to auto-load before starting OpenXR.",
    )
    parser.add_argument("--no-auto-cloudxr-env", action="store_true", help="Do not auto-load ~/.cloudxr/run/cloudxr.env.")
    parser.add_argument("--no-cloudxr-preflight", action="store_true", help="Skip the CloudXR IPC socket check before building Genesis.")
    parser.add_argument("--translation-scale-xyz", default="0.15,0.15,0.15")
    parser.add_argument("--workspace-origin-xyz", default="0,0,0")
    parser.add_argument("--input-axis-map", type=_parse_axis_map, default="z,x,y")
    parser.add_argument("--use-teleop-orientation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--palm-plane-wrist-orientation-blend-alpha", type=float, default=1.0)
    parser.add_argument("--disable-synthetic-hands-plugin", action="store_true")
    parser.add_argument("--no-palm-plane-axes", action="store_true", help="Hide palm-plane debug axes in Genesis.")
    parser.add_argument("--absolute-control", action="store_true", help="Map raw Quest position directly instead of relative anchor control.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.no_auto_cloudxr_env:
        loaded = _load_export_env_file(args.cloudxr_env_path.expanduser())
        if loaded and "NV_CXR_RUNTIME_DIR" in os.environ:
            print(f"[nero-vr] loaded CloudXR env: {args.cloudxr_env_path.expanduser()}")
    if not args.no_cloudxr_preflight:
        ok, message = _check_cloudxr_runtime()
        if not ok:
            raise SystemExit(
                "[nero-vr] CloudXR runtime is not ready.\n"
                f"  {message}\n"
                "  Start it in another terminal and keep that terminal open:\n"
                "    conda activate genesis\n"
                "    python -m isaacteleop.cloudxr --accept-eula\n"
                "  Then rerun this command."
            )
        print(f"[nero-vr] {message}")
    mapping = NeroTeleopMappingConfig(
        translation_scale_xyz=_parse_vec3(args.translation_scale_xyz, name="--translation-scale-xyz"),
        workspace_origin_xyz=_parse_vec3(args.workspace_origin_xyz, name="--workspace-origin-xyz"),
        input_axis_map=args.input_axis_map,
        use_teleop_orientation=bool(args.use_teleop_orientation),
    )
    robot = NeroRuntimeRobotInterface(
        NeroRuntimeRobotConfig(
            arm_side=args.arm_side,
            backend=args.backend,
            show_viewer=not args.headless,
            linker_hand_side=args.arm_side,
            show_palm_plane_axes=not args.no_palm_plane_axes,
            drive_ik=not args.markers_only,
            relative_control=not args.absolute_control,
            mapping=mapping,
        ),
        print_every_n=args.print_every,
    )
    session_config = QuestRobotSessionConfig(
        arm_side=args.arm_side,
        pose_input_mode=args.pose_input_mode,
        arm_pose_command_mode=_arm_pose_command_mode(
            pose_input_mode=args.pose_input_mode,
            use_teleop_orientation=bool(args.use_teleop_orientation),
        ),
        use_wrist_position_for_hand=args.pose_input_mode == "hand_abs",
        use_wrist_rotation_for_hand=bool(args.use_teleop_orientation),
        palm_plane_wrist_orientation_blend_alpha=float(args.palm_plane_wrist_orientation_blend_alpha),
        loop_hz=float(args.loop_hz),
        print_every_n_frames=int(args.print_every),
        enable_synthetic_hands_plugin=not args.disable_synthetic_hands_plugin,
        isaac_teleop_root=args.isaac_teleop_root,
        startup_timeout_s=float(args.startup_timeout_s),
        teleop_trace_path=args.teleop_trace_path,
    )
    with QuestRobotSession(session_config, robot) as session:
        session.run(duration_s=float(args.duration) if float(args.duration) > 0 else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
