from __future__ import annotations

from teleop_stack.teleop.openxr_genesis_adapter import map_openxr_vector_to_genesis


def test_openxr_right_maps_to_canonical_genesis_right() -> None:
    assert map_openxr_vector_to_genesis((1.0, 0.0, 0.0)) == (0.0, 1.0, 0.0)


def test_openxr_up_maps_to_canonical_genesis_up() -> None:
    assert map_openxr_vector_to_genesis((0.0, 1.0, 0.0)) == (0.0, 0.0, 1.0)


def test_openxr_front_maps_to_canonical_genesis_front() -> None:
    assert map_openxr_vector_to_genesis((0.0, 0.0, -1.0)) == (-1.0, 0.0, 0.0)
