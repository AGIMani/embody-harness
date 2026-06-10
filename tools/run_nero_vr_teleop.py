#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    parser.add_argument("--startup-timeout-s", type=float, default=30.0)
    parser.add_argument("--teleop-trace-path", default=None, help="Optional JSONL trace path.")
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
