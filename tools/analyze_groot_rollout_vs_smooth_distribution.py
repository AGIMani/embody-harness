#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMOOTH_DIR = ROOT.parent / "Isaac-GR00T" / "outputs" / "IsaacLab" / "nero" / "mission2" / "smooth"
DEFAULT_TRACE_DIR = ROOT / "logs" / "groot_finetune_policy_trace_current_180x320_no_rtc_160"
DEFAULT_COMPARE_TRACE_DIR = ROOT / "logs" / "groot_finetune_policy_trace_smooth_hand_ref_160"
DEFAULT_OUTPUT_DIR = ROOT / "logs" / "groot_rollout_vs_smooth_distribution"
DEFAULT_RESET_ARM_Q = (
    0.272428,
    1.601217,
    1.453545,
    1.264351,
    0.299394,
    -0.053442,
    0.182823,
)

ARM_NAMES = tuple(f"j{i}" for i in range(7))
EEF_NAMES = (
    "x",
    "y",
    "z",
    "r6_0",
    "r6_1",
    "r6_2",
    "r6_3",
    "r6_4",
    "r6_5",
)
HAND_NAMES = (
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
HAND_PITCH_NAMES = (
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)
HAND_PITCH_INDICES = tuple(HAND_NAMES.index(name) for name in HAND_PITCH_NAMES)
NON_INDEX_HAND_PITCH_NAMES = (
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)
NON_INDEX_HAND_PITCH_INDICES = tuple(HAND_NAMES.index(name) for name in NON_INDEX_HAND_PITCH_NAMES)


def _vec7(text: str) -> tuple[float, ...]:
    parts = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if len(parts) != 7:
        raise argparse.ArgumentTypeError("expected seven comma-separated values")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected numeric comma-separated values") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_npz_path(trace_dir: Path, text: str) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    return trace_dir / path


def _load_smooth(smooth_dir: Path) -> dict[str, np.ndarray]:
    paths = sorted((smooth_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No smooth parquet episodes found under {smooth_dir / 'data'}")
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    frame_ids: list[np.ndarray] = []
    for path in paths:
        table = pq.read_table(path, columns=["observation.state", "action"])
        state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
        action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        if state.ndim != 2 or state.shape[1] < 26:
            raise ValueError(f"Unexpected observation.state shape in {path}: {state.shape}")
        if action.ndim != 2 or action.shape[1] < 19:
            raise ValueError(f"Unexpected action shape in {path}: {action.shape}")
        episode_index = int(path.stem.split("_")[-1])
        states.append(state)
        actions.append(action)
        episode_ids.append(np.full((state.shape[0],), episode_index, dtype=np.int32))
        frame_ids.append(np.arange(state.shape[0], dtype=np.int32))
    state_all = np.concatenate(states, axis=0)
    action_all = np.concatenate(actions, axis=0)
    return {
        "state": state_all,
        "action": action_all,
        "arm_state": state_all[:, 0:7],
        "eef_state": state_all[:, 7:16],
        "hand_state": state_all[:, 16:26],
        "eef_action": action_all[:, 0:9],
        "hand_action": action_all[:, 9:19],
        "episode": np.concatenate(episode_ids, axis=0),
        "frame": np.concatenate(frame_ids, axis=0),
    }


def _latest_session(rows: list[dict[str, Any]]) -> str | None:
    sessions = [str(row.get("session_id")) for row in rows if row.get("record_type") == "replan" and row.get("session_id")]
    return sessions[-1] if sessions else None


def _load_trace(trace_dir: Path, *, session_id: str | None = None) -> dict[str, Any]:
    trace_jsonl = trace_dir / "trace.jsonl"
    rows = _read_jsonl(trace_jsonl)
    selected_session = session_id or _latest_session(rows)
    if selected_session is None:
        raise RuntimeError(f"No replan session found in {trace_jsonl}")
    replan_rows = [
        row
        for row in rows
        if row.get("record_type") == "replan"
        and str(row.get("session_id")) == str(selected_session)
        and row.get("npz_path")
    ]
    if not replan_rows:
        raise RuntimeError(f"No replan NPZ rows found in {trace_jsonl} for session {selected_session}")

    steps: list[int] = []
    arm_obs: list[np.ndarray] = []
    eef_obs: list[np.ndarray] = []
    hand_obs: list[np.ndarray] = []
    hand_ref: list[np.ndarray] = []
    hand_exec_first: list[np.ndarray] = []
    hand_exec_chunks: list[np.ndarray] = []
    eef_ref: list[np.ndarray] = []
    eef_exec_first: list[np.ndarray] = []
    arm_target_first: list[np.ndarray] = []
    sources: list[str] = []
    ik_debug: list[dict[str, Any]] = []

    for row in replan_rows:
        npz_path = _resolve_npz_path(trace_dir, str(row["npz_path"]))
        with np.load(npz_path) as z:
            steps.append(int(row.get("step_count", len(steps))))
            arm_obs.append(np.asarray(z["observation.state.arm_joint_pos"], dtype=np.float64).reshape(-1)[:7])
            eef_obs.append(np.asarray(z["observation.state.eef_9d"], dtype=np.float64).reshape(-1)[:9])
            hand_obs.append(np.asarray(z["observation.state.hand_joint_pos"], dtype=np.float64).reshape(-1)[:10])
            hand_ref.append(np.asarray(z["reference_action.hand_joint_target"], dtype=np.float64).reshape(-1)[:10])
            eef_ref.append(np.asarray(z["reference_action.eef_9d"], dtype=np.float64).reshape(-1)[:9])
            hand_exec = np.asarray(z["execution_action.hand_joint_target"], dtype=np.float64).reshape(-1, 10)
            hand_exec_first.append(hand_exec[0, :10])
            hand_exec_chunks.append(hand_exec[:, :10])
            eef_exec = np.asarray(z["execution_action.eef_9d"], dtype=np.float64).reshape(-1, 9)
            eef_exec_first.append(eef_exec[0, :9])
            arm_target = np.asarray(z["execution_action.arm_joint_target"], dtype=np.float64).reshape(-1, 7)
            arm_target_first.append(arm_target[0, :7])
        hand_debug = row.get("hand") or {}
        sources.append(str(hand_debug.get("reference_hand_source") or row.get("reference_action_source") or ""))
        ik_debug.append((row.get("executor") or {}).get("last_differential_ik_debug") or {})

    return {
        "trace_dir": str(trace_dir),
        "session_id": selected_session,
        "rows": replan_rows,
        "steps": np.asarray(steps, dtype=np.int32),
        "arm_obs": np.stack(arm_obs, axis=0),
        "eef_obs": np.stack(eef_obs, axis=0),
        "hand_obs": np.stack(hand_obs, axis=0),
        "hand_ref": np.stack(hand_ref, axis=0),
        "hand_exec_first": np.stack(hand_exec_first, axis=0),
        "hand_exec_chunks": np.concatenate(hand_exec_chunks, axis=0),
        "eef_ref": np.stack(eef_ref, axis=0),
        "eef_exec_first": np.stack(eef_exec_first, axis=0),
        "arm_target_first": np.stack(arm_target_first, axis=0),
        "sources": sources,
        "ik_debug": ik_debug,
    }


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": np.nanmin(arr, axis=0).astype(float).tolist(),
        "q001": np.nanquantile(arr, 0.001, axis=0).astype(float).tolist(),
        "q01": np.nanquantile(arr, 0.01, axis=0).astype(float).tolist(),
        "mean": np.nanmean(arr, axis=0).astype(float).tolist(),
        "q99": np.nanquantile(arr, 0.99, axis=0).astype(float).tolist(),
        "q999": np.nanquantile(arr, 0.999, axis=0).astype(float).tolist(),
        "max": np.nanmax(arr, axis=0).astype(float).tolist(),
    }


def _outside_fraction(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    return np.nanmean(arr < low[None, :], axis=0), np.nanmean(arr > high[None, :], axis=0)


def _percentile_ranks(values: np.ndarray, train: np.ndarray) -> np.ndarray:
    train_arr = np.asarray(train, dtype=np.float64)
    value_arr = np.asarray(values, dtype=np.float64)
    if value_arr.ndim == 1:
        value_arr = value_arr[None, :]
    out = np.zeros_like(value_arr, dtype=np.float64)
    for dim in range(value_arr.shape[1]):
        col = train_arr[:, dim]
        for row in range(value_arr.shape[0]):
            out[row, dim] = float(np.mean(col <= value_arr[row, dim]) * 100.0)
    return out


def _oob_report(values: np.ndarray, train: np.ndarray, names: tuple[str, ...]) -> list[dict[str, float | str]]:
    train_stats = _stats(train)
    low = np.asarray(train_stats["q01"], dtype=np.float64)
    high = np.asarray(train_stats["q99"], dtype=np.float64)
    below, above = _outside_fraction(values, low, high)
    value_stats = _stats(values)
    rows: list[dict[str, float | str]] = []
    for idx, name in enumerate(names):
        rows.append(
            {
                "name": name,
                "trace_min": float(value_stats["min"][idx]),
                "trace_mean": float(value_stats["mean"][idx]),
                "trace_max": float(value_stats["max"][idx]),
                "smooth_q01": float(low[idx]),
                "smooth_q99": float(high[idx]),
                "below_q01_fraction": float(below[idx]),
                "above_q99_fraction": float(above[idx]),
            }
        )
    return rows


def _reset_arm_report(reset_q: np.ndarray, smooth_arm: np.ndarray) -> dict[str, Any]:
    reset = np.asarray(reset_q, dtype=np.float64).reshape(7)
    stats = _stats(smooth_arm)
    percentile = _percentile_ranks(reset, smooth_arm).reshape(7)
    rows: list[dict[str, float | str]] = []
    for idx, name in enumerate(ARM_NAMES):
        rows.append(
            {
                "name": name,
                "reset": float(reset[idx]),
                "smooth_min": float(stats["min"][idx]),
                "smooth_q01": float(stats["q01"][idx]),
                "smooth_median": float(np.nanquantile(smooth_arm[:, idx], 0.5)),
                "smooth_q99": float(stats["q99"][idx]),
                "smooth_max": float(stats["max"][idx]),
                "reset_percentile": float(percentile[idx]),
            }
        )
    return {
        "right_arm_q": [float(v) for v in reset],
        "by_joint": rows,
        "j4_1_based": {
            "maps_to_zero_based": "j3",
            "reset": float(reset[3]),
            "smooth_q01": float(stats["q01"][3]),
            "smooth_median": float(np.nanquantile(smooth_arm[:, 3], 0.5)),
            "smooth_q99": float(stats["q99"][3]),
            "reset_percentile": float(percentile[3]),
            "note": "If j4 means the fourth joint in 1-based human naming, ~1.0 rad is in the dense smooth training range.",
        },
        "j4_0_based": {
            "maps_to_zero_based": "j4",
            "reset": float(reset[4]),
            "smooth_q01": float(stats["q01"][4]),
            "smooth_median": float(np.nanquantile(smooth_arm[:, 4], 0.5)),
            "smooth_q99": float(stats["q99"][4]),
            "reset_percentile": float(percentile[4]),
            "note": "If j4 means zero-based index 4, the observed reset is near the smooth upper tail but is not ~1.0 rad.",
        },
    }


def _non_index_pitch_floor_report(values: np.ndarray, smooth_hand_state: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    train_stats = _stats(smooth_hand_state)
    floors = np.asarray(train_stats["q01"], dtype=np.float64)
    rows: list[dict[str, float | str]] = []
    for idx, name in zip(NON_INDEX_HAND_PITCH_INDICES, NON_INDEX_HAND_PITCH_NAMES, strict=True):
        floor = float(floors[idx])
        column = arr[:, idx]
        rows.append(
            {
                "name": name,
                "smooth_state_q01_floor": floor,
                "trace_min": float(np.nanmin(column)),
                "trace_mean": float(np.nanmean(column)),
                "trace_max": float(np.nanmax(column)),
                "near_floor_fraction": float(np.nanmean(column <= floor + 1.0e-4)),
            }
        )
    return {
        "definition": "near_floor means observation <= smooth hand_state q01 + 1e-4 for middle/ring/pinky pitch.",
        "by_joint": rows,
        "mean_near_floor_fraction": float(np.mean([row["near_floor_fraction"] for row in rows])),
    }


def _print_distribution_block(name: str, names: tuple[str, ...], values: np.ndarray, train: np.ndarray) -> None:
    train_stats = _stats(train)
    low = np.asarray(train_stats["q01"], dtype=np.float64)
    high = np.asarray(train_stats["q99"], dtype=np.float64)
    below, above = _outside_fraction(values, low, high)
    value_stats = _stats(values)
    print(f"\n[{name}]")
    for i, dim_name in enumerate(names):
        print(
            f"  {dim_name:18s} trace=[{value_stats['min'][i]:+.4f},{value_stats['mean'][i]:+.4f},{value_stats['max'][i]:+.4f}] "
            f"smooth_q01/q99=[{low[i]:+.4f},{high[i]:+.4f}] "
            f"below={below[i] * 100.0:5.1f}% above={above[i] * 100.0:5.1f}%"
        )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot_reports(
    *,
    output_dir: Path,
    smooth: dict[str, np.ndarray],
    traces: list[tuple[str, dict[str, Any]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    arm_stats = _stats(smooth["arm_state"])
    arm_q01 = np.asarray(arm_stats["q01"])
    arm_q99 = np.asarray(arm_stats["q99"])
    arm_min = np.asarray(arm_stats["min"])
    arm_max = np.asarray(arm_stats["max"])

    fig, axes = plt.subplots(7, 1, figsize=(11, 12), sharex=True)
    colors = ("tab:blue", "tab:orange", "tab:green")
    for j, ax in enumerate(axes):
        for label, trace in traces:
            steps = trace["steps"]
            color = colors[traces.index((label, trace)) % len(colors)]
            ax.plot(steps, trace["arm_obs"][:, j], "o-", ms=3, lw=1.4, label=label, color=color)
        ax.axhspan(arm_q01[j], arm_q99[j], color="0.85", label="smooth q01-q99" if j == 0 else None)
        ax.axhline(arm_min[j], color="0.75", ls=":", lw=1)
        ax.axhline(arm_max[j], color="0.75", ls=":", lw=1)
        ax.set_ylabel(ARM_NAMES[j])
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Trace arm_joint_pos observations vs smooth training distribution")
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("policy step")
    fig.tight_layout()
    fig.savefig(output_dir / "arm_joint_pos_trace_vs_smooth.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(HAND_PITCH_NAMES), 1, figsize=(11, 8), sharex=True)
    smooth_hand = smooth["hand_action"]
    for ax, name, idx in zip(axes, HAND_PITCH_NAMES, HAND_PITCH_INDICES, strict=True):
        ax.plot(np.arange(smooth_hand.shape[0]), smooth_hand[:, idx], color="0.82", lw=1.2, label="smooth action all episodes concat")
        for label, trace in traces:
            steps = trace["steps"]
            color = colors[traces.index((label, trace)) % len(colors)]
            ax.plot(steps, trace["hand_ref"][:, idx], "o-", ms=3, lw=1.2, color=color, label=f"{label} ref")
            ax.plot(steps, trace["hand_obs"][:, idx], "x--", ms=3, lw=1.0, color=color, alpha=0.8, label=f"{label} obs")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Hand pitch reference/observation vs smooth action distribution")
    axes[0].legend(ncol=3, fontsize=7)
    axes[-1].set_xlabel("smooth frame / policy step")
    fig.tight_layout()
    fig.savefig(output_dir / "hand_pitch_trace_vs_smooth.png", dpi=160)
    plt.close(fig)

    eef_stats = _stats(smooth["eef_state"])
    eef_q01 = np.asarray(eef_stats["q01"])
    eef_q99 = np.asarray(eef_stats["q99"])
    fig, axes = plt.subplots(9, 1, figsize=(11, 13), sharex=True)
    for j, ax in enumerate(axes):
        ax.axhspan(eef_q01[j], eef_q99[j], color="0.87", label="smooth q01-q99" if j == 0 else None)
        for label, trace in traces:
            steps = trace["steps"]
            color = colors[traces.index((label, trace)) % len(colors)]
            ax.plot(steps, trace["eef_obs"][:, j], "o-", ms=3, lw=1.2, label=label, color=color)
        ax.set_ylabel(EEF_NAMES[j])
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Trace observation.state.eef_9d vs smooth training distribution")
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("policy step")
    fig.tight_layout()
    fig.savefig(output_dir / "eef_9d_trace_vs_smooth.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare GR00T Genesis rollout traces against Isaac-GR00T smooth training distributions."
    )
    parser.add_argument("--smooth-dir", type=Path, default=DEFAULT_SMOOTH_DIR)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--trace-label", default="online")
    parser.add_argument("--compare-trace-dir", type=Path, default=DEFAULT_COMPARE_TRACE_DIR)
    parser.add_argument("--compare-trace-label", default="smooth-hand-ref")
    parser.add_argument("--no-compare-trace", action="store_true")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--compare-session-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reset-arm-q", type=_vec7, default=DEFAULT_RESET_ARM_Q)
    args = parser.parse_args()

    smooth_dir = args.smooth_dir.expanduser().resolve()
    trace_dir = args.trace_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    smooth = _load_smooth(smooth_dir)
    traces: list[tuple[str, dict[str, Any]]] = [(str(args.trace_label), _load_trace(trace_dir, session_id=args.session_id))]
    if not args.no_compare_trace and args.compare_trace_dir is not None and args.compare_trace_dir.exists():
        traces.append(
            (
                str(args.compare_trace_label),
                _load_trace(args.compare_trace_dir.expanduser().resolve(), session_id=args.compare_session_id),
            )
        )

    print(f"[smooth] dir={smooth_dir} frames={smooth['state'].shape[0]}")
    print(f"[smooth] episodes={int(np.unique(smooth['episode']).size)}")
    for label, trace in traces:
        print(f"[trace] {label}: dir={trace['trace_dir']} session={trace['session_id']} replans={trace['arm_obs'].shape[0]}")
        print(f"[trace] {label}: reference_sources={sorted(set(trace['sources']))}")

    for label, trace in traces:
        _print_distribution_block(f"{label} arm_joint_pos observation vs smooth", ARM_NAMES, trace["arm_obs"], smooth["arm_state"])
        _print_distribution_block(f"{label} eef_9d observation vs smooth", EEF_NAMES, trace["eef_obs"], smooth["eef_state"])
        _print_distribution_block(
            f"{label} hand_joint_pos observation vs smooth state",
            HAND_NAMES,
            trace["hand_obs"],
            smooth["hand_state"],
        )
        _print_distribution_block(
            f"{label} hand reference vs smooth action",
            HAND_NAMES,
            trace["hand_ref"],
            smooth["hand_action"],
        )

    summary = {
        "smooth_dir": str(smooth_dir),
        "smooth_frames": int(smooth["state"].shape[0]),
        "smooth_episodes": int(np.unique(smooth["episode"]).size),
        "smooth": {
            "arm_state": _stats(smooth["arm_state"]),
            "eef_state": _stats(smooth["eef_state"]),
            "hand_state": _stats(smooth["hand_state"]),
            "hand_action": _stats(smooth["hand_action"]),
        },
        "reset": {
            "arm": _reset_arm_report(np.asarray(args.reset_arm_q, dtype=np.float64), smooth["arm_state"]),
        },
        "traces": {},
        "interpretation": {
            "j4_note": (
                "Disambiguate the joint name first. If j4 means the fourth joint in human 1-based naming, "
                "it is zero-based j3 and ~1.0 rad is normal in smooth. If j4 means zero-based index 4, "
                "the smooth q01/q99 range is around -0.53..+0.31 rad and the current reset is near q99, not 1.0 rad."
            ),
            "eef_frame_note": (
                "observation.state.eef_9d is expected to be Nero CAN/SDK feedback-base FK; "
                "do not interpret its z as Genesis/world height for a right-side mount."
            ),
            "hand_note": (
                "Current ablations indicate wrist_view dominates the non-index finger behavior: replacing only the "
                "sim wrist image with smooth wrist recovers middle/ring/pinky, while replacing only ego does not. "
                "Use the D405 smooth wrist calibration tool and then rerun this report."
            ),
        },
    }
    for label, trace in traces:
        summary["traces"][label] = {
            "trace_dir": trace["trace_dir"],
            "session_id": trace["session_id"],
            "replans": int(trace["arm_obs"].shape[0]),
            "reference_sources": sorted(set(trace["sources"])),
            "arm_obs": _stats(trace["arm_obs"]),
            "eef_obs": _stats(trace["eef_obs"]),
            "hand_obs": _stats(trace["hand_obs"]),
            "hand_ref": _stats(trace["hand_ref"]),
            "hand_exec_first": _stats(trace["hand_exec_first"]),
            "arm_obs_oob_vs_smooth_q01_q99": _oob_report(trace["arm_obs"], smooth["arm_state"], ARM_NAMES),
            "eef_obs_oob_vs_smooth_q01_q99": _oob_report(trace["eef_obs"], smooth["eef_state"], EEF_NAMES),
            "hand_obs_non_index_pitch_floor": _non_index_pitch_floor_report(trace["hand_obs"], smooth["hand_state"]),
        }
    _write_json(output_dir / "summary.json", summary)
    _plot_reports(output_dir=output_dir, smooth=smooth, traces=traces)
    print(f"[output] {output_dir / 'summary.json'}")
    print(f"[output] {output_dir / 'arm_joint_pos_trace_vs_smooth.png'}")
    print(f"[output] {output_dir / 'eef_9d_trace_vs_smooth.png'}")
    print(f"[output] {output_dir / 'hand_pitch_trace_vs_smooth.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
