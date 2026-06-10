from __future__ import annotations

from pathlib import Path
import sys


BUNDLE_ROOT = Path(__file__).resolve().parent
NERO_ROOT = BUNDLE_ROOT / "nero_twin"
LINKERHAND_ROOT = BUNDLE_ROOT / "linkerhand-urdf"

if str(NERO_ROOT) not in sys.path:
    sys.path.insert(0, str(NERO_ROOT))

from nero_dual_runtime import NeroDualArmRuntime, NeroDualRuntimeConfig  # noqa: E402


ACTIVE_LINKER_L10_JOINTS = (
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

INITIAL_LEFT_ARM_Q = (
    -0.3010692959690218,
    1.4731277018532938,
    -1.1596840214876325,
    1.3072865163287928,
    0.005689773361501515,
    -0.06812020070533868,
    0.21753783796857323,
)

INITIAL_RIGHT_ARM_Q = (
    0.2530727415391778,
    1.5579507035002182,
    1.2218002895661106,
    1.3225232406987033,
    -0.0004886921905584122,
    -0.11129964639967839,
    0.11606439525762292,
)


def make_runtime_config(
    *,
    backend: str = "gpu",
    show_viewer: bool = True,
    linker_hand_side: str = "right",
) -> NeroDualRuntimeConfig:
    hand_side = "left" if linker_hand_side == "left" else "right"
    return NeroDualRuntimeConfig(
        backend=backend,
        show_viewer=show_viewer,
        show_target_markers=True,
        dt=0.01,
        max_fps=60,
        max_solver_iters=32,
        ik_damping=0.02,
        pos_tol=1e-3,
        max_joint_step=0.045,
        min_joint_step=0.001,
        ik_joint4_limit_enabled=True,
        ik_joint4_limit_rad=(0.0, 2.14),
        base_mesh=NERO_ROOT / "assets" / "mesh" / "base.STL",
        base_scale=0.001,
        base_euler="90,0,0",
        base_foot_center_mm="-51.439,-842.036,-50.0",
        nero_urdf=NERO_ROOT / "assets" / "agx_arm_urdf" / "nero" / "urdf" / "nero_description.urdf",
        package_root=NERO_ROOT / "assets",
        right_support_hole_z_mm=-109.0,
        eef_link="revo2_flange",
        add_revo2_flange=True,
        arm_collision=False,
        base_collision=False,
        linker_hand_urdf=LINKERHAND_ROOT / "l10" / hand_side / f"linkerhand_l10_{hand_side}.urdf",
        linker_hand_side=hand_side,
        linker_hand_mount_offset_xyz=(0.0, 0.0, 0.0),
        linker_hand_mount_quat_wxyz=(0.70710678, 0.0, 0.0, 0.70710678),
        linker_hand_joint_names=ACTIVE_LINKER_L10_JOINTS,
        initial_left_arm_q=INITIAL_LEFT_ARM_Q,
        initial_right_arm_q=INITIAL_RIGHT_ARM_Q,
        connect_can=False,
        send_real_commands=False,
    )


def make_runtime(
    *,
    backend: str = "gpu",
    show_viewer: bool = True,
    linker_hand_side: str = "right",
) -> NeroDualArmRuntime:
    return NeroDualArmRuntime(
        make_runtime_config(
            backend=backend,
            show_viewer=show_viewer,
            linker_hand_side=linker_hand_side,
        )
    )

