from __future__ import annotations

import math

import pytest

from teleop_palm_plane.palm_plane import (
    apply_palm_plane_wrist_orientation_correction,
    palm_plane_orientation_from_hand_debug,
    quat_angle_between_xyzw,
)


def _z_rotation_xyzw(angle_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(angle_rad)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _open_palm_debug() -> dict[str, object]:
    positions = [[0.0, 0.0, 0.0] for _ in range(26)]
    valid = [False for _ in range(26)]
    positions[1] = [0.0, 0.0, 0.0]
    positions[7] = [0.08, 0.08, 0.0]
    positions[12] = [0.0, 0.12, 0.0]
    positions[22] = [-0.08, 0.08, 0.0]
    for index in (1, 7, 12, 22):
        valid[index] = True
    return {
        "joint_positions_xyz": positions,
        "joint_valid": valid,
    }


def test_palm_plane_orientation_uses_palm_axes() -> None:
    orientation = palm_plane_orientation_from_hand_debug(_open_palm_debug())

    assert orientation is not None
    assert orientation.palm_origin_xyz == pytest.approx((0.0, 0.07, 0.0))
    assert orientation.palm_across_xyz == pytest.approx((1.0, 0.0, 0.0))
    assert orientation.palm_forward_xyz == pytest.approx((0.0, 1.0, 0.0))
    assert orientation.palm_normal_xyz == pytest.approx((0.0, 0.0, 1.0))
    assert quat_angle_between_xyzw(orientation.quaternion_xyzw, (0.0, 0.0, 0.0, 1.0)) == pytest.approx(0.0)


def test_palm_plane_correction_blends_wrist_toward_palm_plane() -> None:
    correction = apply_palm_plane_wrist_orientation_correction(
        _z_rotation_xyzw(math.radians(90.0)),
        _open_palm_debug(),
        blend_alpha=0.5,
    )

    assert correction is not None
    assert correction.raw_to_palm_error_rad == pytest.approx(math.radians(90.0))
    assert quat_angle_between_xyzw(
        correction.corrected_quaternion_xyzw,
        _z_rotation_xyzw(math.radians(90.0)),
    ) == pytest.approx(math.radians(45.0))
