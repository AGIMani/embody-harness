#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import add_scene_glb as harness  # noqa: E402


def _fmt(values: tuple[float, ...] | list[float] | np.ndarray) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _rpy_rad_from_rotation(rotation: np.ndarray) -> tuple[float, float, float]:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sy = math.sqrt(float(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0]))
    singular = sy < 1.0e-9
    if not singular:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        pitch = math.atan2(float(-rotation[2, 0]), sy)
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = math.atan2(float(-rotation[1, 2]), float(rotation[1, 1]))
        pitch = math.atan2(float(-rotation[2, 0]), sy)
        yaw = 0.0
    return roll, pitch, yaw


def _rpy_rad_from_euler_deg(euler_deg: tuple[float, float, float]) -> tuple[float, float, float]:
    return _rpy_rad_from_rotation(harness._rotation_from_euler_deg(euler_deg))  # noqa: SLF001


def _rpy_rad_from_quat_wxyz(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float]:
    return _rpy_rad_from_rotation(harness._rotation_from_quat_wxyz(np.asarray(quat_wxyz, dtype=np.float64)))  # noqa: SLF001


def _origin(parent: ET.Element, xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> ET.Element:
    return ET.SubElement(parent, "origin", {"xyz": _fmt(xyz), "rpy": _fmt(rpy)})


def _add_fixed_joint(
    root: ET.Element,
    *,
    name: str,
    parent: str,
    child: str,
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> None:
    joint = ET.SubElement(root, "joint", {"name": name, "type": "fixed"})
    _origin(joint, xyz, rpy)
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})


def _add_mesh_link(
    root: ET.Element,
    *,
    name: str,
    mesh: Path,
    scale: float,
    collision: bool = True,
) -> None:
    link = ET.SubElement(root, "link", {"name": name})
    inertial = ET.SubElement(link, "inertial")
    _origin(inertial, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    ET.SubElement(inertial, "mass", {"value": "1.0"})
    ET.SubElement(
        inertial,
        "inertia",
        {"ixx": "1e-3", "ixy": "0", "ixz": "0", "iyy": "1e-3", "iyz": "0", "izz": "1e-3"},
    )
    for tag in ("visual", "collision") if collision else ("visual",):
        item = ET.SubElement(link, tag)
        _origin(item, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        geometry = ET.SubElement(item, "geometry")
        ET.SubElement(
            geometry,
            "mesh",
            {
                "filename": str(mesh.expanduser().resolve()),
                "scale": _fmt((scale, scale, scale)),
            },
        )


def _prefix_tree(
    source: Path,
    *,
    prefix: str,
    drop_transmissions: bool = True,
    strip_collisions: bool = False,
) -> tuple[list[ET.Element], set[str], set[str]]:
    tree = ET.parse(source)
    source_root = tree.getroot()
    link_names = {link.attrib["name"] for link in source_root.findall("link") if "name" in link.attrib}
    joint_names = {joint.attrib["name"] for joint in source_root.findall("joint") if "name" in joint.attrib}
    prefixed_link_names = {prefix + name for name in link_names}
    prefixed_joint_names = {prefix + name for name in joint_names}
    elements: list[ET.Element] = []

    for child in list(source_root):
        if drop_transmissions and child.tag == "transmission":
            continue
        item = copy.deepcopy(child)
        if strip_collisions:
            for collision in list(item.findall(".//collision")):
                for parent in item.iter():
                    if collision in list(parent):
                        parent.remove(collision)
                        break
        if item.tag == "link" and "name" in item.attrib:
            item.attrib["name"] = prefix + item.attrib["name"]
        elif item.tag == "joint" and "name" in item.attrib:
            item.attrib["name"] = prefix + item.attrib["name"]
            parent = item.find("parent")
            if parent is not None and "link" in parent.attrib:
                parent.attrib["link"] = prefix + parent.attrib["link"]
            child_link = item.find("child")
            if child_link is not None and "link" in child_link.attrib:
                child_link.attrib["link"] = prefix + child_link.attrib["link"]
            mimic = item.find("mimic")
            if mimic is not None and "joint" in mimic.attrib:
                mimic.attrib["joint"] = prefix + mimic.attrib["joint"]
        elements.append(item)
    return elements, prefixed_link_names, prefixed_joint_names


def _append_prefixed_urdf(root: ET.Element, source: Path, *, prefix: str, strip_collisions: bool = False) -> set[str]:
    elements, links, _ = _prefix_tree(source, prefix=prefix, strip_collisions=strip_collisions)
    for item in elements:
        root.append(item)
    return links


def build_combined_urdf(output: Path, *, hand_side: str) -> Path:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    base_mesh = Path(harness.DEFAULT_BASE_MESH).expanduser().resolve()
    connector_mesh = Path(harness.DEFAULT_CONNECTOR_MESH).expanduser().resolve()
    package_root = Path(harness.DEFAULT_PACKAGE_ROOT).expanduser().resolve()
    nero_source = Path(harness.DEFAULT_NERO_URDF).expanduser().resolve()
    nero_with_flange = harness._make_revo2_flange_urdf(nero_source)  # noqa: SLF001
    nero_urdf = harness._sanitize_urdf_for_genesis(nero_with_flange, package_root)  # noqa: SLF001
    hand_urdf = Path(harness.NERO_LINKER_CONFIG.linker_hand_urdf).expanduser().resolve()
    if hand_side == "left":
        hand_urdf = hand_urdf.parents[1] / "left" / "linkerhand_l10_left.urdf"
    hand_urdf = harness._sanitize_relative_mesh_urdf(hand_urdf)  # noqa: SLF001

    root = ET.Element("robot", {"name": "dual_nero_linker_l10_combined"})
    root.append(ET.Comment("Generated by tools/build_add_scene_combined_urdf.py from add_scene_glb.py assembly constants."))

    _add_mesh_link(
        root,
        name="base_stl",
        mesh=base_mesh,
        scale=float(harness.DEFAULT_BASE_SCALE),
        collision=False,
    )

    left_links = _append_prefixed_urdf(root, nero_urdf, prefix="left_", strip_collisions=True)
    right_links = _append_prefixed_urdf(root, nero_urdf, prefix="right_", strip_collisions=True)

    _add_fixed_joint(
        root,
        name="base_stl_to_left_nero",
        parent="base_stl",
        child="left_world" if "left_world" in left_links else "left_base_link",
        xyz=harness.LEFT_ARM_REL_POS_M,
        rpy=_rpy_rad_from_euler_deg(harness.LEFT_ARM_REL_EULER_DEG),
    )
    _add_fixed_joint(
        root,
        name="base_stl_to_right_nero",
        parent="base_stl",
        child="right_world" if "right_world" in right_links else "right_base_link",
        xyz=harness.RIGHT_ARM_REL_POS_M,
        rpy=_rpy_rad_from_euler_deg(harness.RIGHT_ARM_REL_EULER_DEG),
    )

    _add_mesh_link(
        root,
        name="left_connector",
        mesh=connector_mesh,
        scale=float(harness.DEFAULT_CONNECTOR_SCALE),
        collision=False,
    )
    _add_mesh_link(
        root,
        name="right_connector",
        mesh=connector_mesh,
        scale=float(harness.DEFAULT_CONNECTOR_SCALE),
        collision=False,
    )
    _add_fixed_joint(
        root,
        name="left_revo2_flange_to_connector",
        parent=f"left_{harness.DEFAULT_EEF_LINK}",
        child="left_connector",
        xyz=harness.LEFT_CONNECTOR_MOUNT_OFFSET_XYZ,
        rpy=_rpy_rad_from_euler_deg(harness.LEFT_CONNECTOR_MOUNT_EULER_DEG),
    )
    _add_fixed_joint(
        root,
        name="right_revo2_flange_to_connector",
        parent=f"right_{harness.DEFAULT_EEF_LINK}",
        child="right_connector",
        xyz=harness.RIGHT_CONNECTOR_MOUNT_OFFSET_XYZ,
        rpy=_rpy_rad_from_euler_deg(harness.RIGHT_CONNECTOR_MOUNT_EULER_DEG),
    )

    hand_prefix = f"{hand_side}_l10_"
    hand_links = _append_prefixed_urdf(root, hand_urdf, prefix=hand_prefix)
    hand_root_link = hand_prefix + "hand_base_link"
    if hand_root_link not in hand_links:
        raise RuntimeError(f"Expected hand root link not found: {hand_root_link}")
    _add_fixed_joint(
        root,
        name=f"{hand_side}_revo2_flange_to_l10_hand",
        parent=f"{hand_side}_{harness.DEFAULT_EEF_LINK}",
        child=hand_root_link,
        xyz=tuple(float(v) for v in harness.NERO_LINKER_CONFIG.linker_hand_mount_offset_xyz),
        rpy=_rpy_rad_from_quat_wxyz(tuple(float(v) for v in harness.NERO_LINKER_CONFIG.linker_hand_mount_quat_wxyz)),
    )

    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a single URDF for the add_scene_glb Nero + connector + L10 assembly.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "assets" / "generated" / "dual_nero_linker_l10_combined.urdf",
    )
    parser.add_argument("--hand-side", choices=("left", "right"), default=harness.NERO_LINKER_CONFIG.linker_hand_side)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = build_combined_urdf(args.output, hand_side=args.hand_side)
    print(f"[combined-urdf] wrote {output}", flush=True)
    print(
        "[combined-urdf] load pose matching add_scene_glb default base: "
        f"pos={harness.DEFAULT_INITIAL_BASE_WORLD_POS} euler_deg={harness.DEFAULT_INITIAL_BASE_WORLD_EULER}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
