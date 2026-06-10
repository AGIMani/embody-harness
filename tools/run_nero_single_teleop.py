#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop_palm_plane import (  # noqa: E402
    apply_palm_plane_wrist_orientation_correction,
    palm_plane_orientation_from_hand_debug,
)
from teleop_palm_plane.palm_plane import wxyz_to_xyzw  # noqa: E402


def _parse_quat4(text: str, *, name: str) -> tuple[float, float, float, float]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(values) != 4:
        raise argparse.ArgumentTypeError(f"{name} must contain 4 comma-separated floats")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extracted AGIMani Nero single-teleop palm-plane helper. "
            "Reads a Quest/OpenXR hand debug JSON payload and prints palm-plane orientation."
        )
    )
    parser.add_argument(
        "--hand-debug-json",
        type=Path,
        required=True,
        help="JSON file containing joint_positions_xyz and joint_valid arrays.",
    )
    parser.add_argument(
        "--raw-wrist-quat-xyzw",
        default=None,
        help="Optional raw wrist quaternion in OpenXR xyzw order.",
    )
    parser.add_argument(
        "--raw-wrist-quat-wxyz",
        default=None,
        help="Optional raw wrist quaternion in Genesis wxyz order.",
    )
    parser.add_argument(
        "--palm-plane-wrist-orientation-blend-alpha",
        type=float,
        default=1.0,
        help="Blend raw wrist orientation toward palm-plane orientation; 1 fully uses the palm plane.",
    )
    parser.add_argument(
        "--pretty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.hand_debug_json.expanduser().open("r", encoding="utf-8") as handle:
        hand_debug = json.load(handle)

    palm_plane = palm_plane_orientation_from_hand_debug(hand_debug)
    if palm_plane is None:
        raise ValueError("hand debug payload does not contain a valid palm plane")

    payload: dict[str, object] = {
        "palm_plane": palm_plane.as_dict(),
    }
    raw_xyzw = None
    if args.raw_wrist_quat_xyzw is not None and args.raw_wrist_quat_wxyz is not None:
        raise ValueError("Use only one of --raw-wrist-quat-xyzw or --raw-wrist-quat-wxyz")
    if args.raw_wrist_quat_xyzw is not None:
        raw_xyzw = _parse_quat4(args.raw_wrist_quat_xyzw, name="--raw-wrist-quat-xyzw")
    elif args.raw_wrist_quat_wxyz is not None:
        raw_xyzw = wxyz_to_xyzw(_parse_quat4(args.raw_wrist_quat_wxyz, name="--raw-wrist-quat-wxyz"))

    if raw_xyzw is not None:
        correction = apply_palm_plane_wrist_orientation_correction(
            raw_xyzw,
            hand_debug,
            blend_alpha=float(args.palm_plane_wrist_orientation_blend_alpha),
        )
        if correction is not None:
            correction_payload = correction.as_dict()
            correction_payload["raw_to_palm_error_deg"] = math.degrees(correction.raw_to_palm_error_rad)
            payload["wrist_orientation_correction"] = correction_payload

    indent = 2 if bool(args.pretty) else None
    print(json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
