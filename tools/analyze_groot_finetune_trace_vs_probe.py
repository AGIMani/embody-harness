#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Diagnose online GR00T finetune hand rollout against Isaac-GR00T probe logs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_JSONL = ROOT / "logs" / "groot_finetune_policy_trace" / "trace.jsonl"
DEFAULT_STATISTICS = ROOT / "checkpoints" / "finetune" / "checkpoint-59000" / "statistics.json"
DEFAULT_PROBE_CHUNKS = (
    Path("/home/whf/Project/Isaac-GR00T")
    / "outputs"
    / "IsaacLab"
    / "nero"
    / "mission2"
    / "rtc_probe"
    / "nero_mission2_checkpoint-59000_ep0_full_rtc_r8_o24_f4_r20_seedanchor_fixed_actionstored_20260612"
    / "policy_raw_chunks.jsonl"
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _hand_array(value: Any) -> np.ndarray:
    if value is None:
        return np.full((len(HAND_NAMES),), np.nan, dtype=np.float64)
    if isinstance(value, dict):
        if "first" in value:
            return _hand_array(value["first"])
        return np.asarray([float(value.get(name, np.nan)) for name in HAND_NAMES], dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    out = np.full((len(HAND_NAMES),), np.nan, dtype=np.float64)
    out[: min(out.size, arr.size)] = arr[: min(out.size, arr.size)]
    return out


def _latest_session(rows: list[dict[str, Any]]) -> str:
    sessions = [str(r.get("session_id")) for r in rows if r.get("record_type") == "replan" and r.get("session_id")]
    if not sessions:
        raise RuntimeError("No replan records found in trace.")
    return sessions[-1]


def _trace_session_rows(rows: list[dict[str, Any]], session_id: str) -> list[dict[str, Any]]:
    out = [
        r
        for r in rows
        if r.get("record_type") == "replan" and str(r.get("session_id")) == str(session_id)
    ]
    if not out:
        raise RuntimeError(f"No replan records found for session {session_id!r}.")
    return out


def _collect_trace_hand(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    ref = []
    raw = []
    guarded = []
    sim = []
    obs = []
    actual = []
    sources = []
    rtc_reasons = []
    rtc_options_present = []
    for row in rows:
        hand = row.get("hand") or {}
        ref.append(_hand_array(hand.get("reference_hand_first")))
        raw.append(_hand_array((hand.get("policy_raw_unclipped") or {}).get("first")))
        guarded.append(_hand_array((hand.get("guarded_command_range") or {}).get("first")))
        sim.append(_hand_array((hand.get("sim_hand_projected_range") or {}).get("first")))
        obs.append(_hand_array(hand.get("observation_hand_state")))
        actual.append(_hand_array(hand.get("actual_hand_state")))
        sources.append(str(hand.get("reference_hand_source") or row.get("reference_action_source") or ""))
        rtc = row.get("rtc") or {}
        rtc_reasons.append(str((rtc.get("metadata") or {}).get("reason") or ""))
        rtc_options_present.append(rtc.get("options") is not None)
    return {
        "reference": np.stack(ref, axis=0),
        "raw": np.stack(raw, axis=0),
        "guarded": np.stack(guarded, axis=0),
        "sim": np.stack(sim, axis=0),
        "observation": np.stack(obs, axis=0),
        "actual": np.stack(actual, axis=0),
        "sources": np.asarray(sources, dtype=object),
        "rtc_reasons": np.asarray(rtc_reasons, dtype=object),
        "rtc_options_present": np.asarray(rtc_options_present, dtype=bool),
    }


def _hand_chunk_from_npz(npz: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    full_key = f"{key}.hand_joint_target"
    if full_key not in npz:
        return None
    arr = np.asarray(npz[full_key], dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    return arr[:, : len(HAND_NAMES)].astype(np.float64, copy=False)


def _trace_seed_storage_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_diffs = []
    exec_diffs = []
    checked = 0
    for row in rows:
        npz_path_text = row.get("npz_path")
        if not npz_path_text:
            continue
        npz_path = Path(str(npz_path_text))
        if not npz_path.exists():
            continue
        try:
            with np.load(npz_path) as npz:
                rtc_store = _hand_chunk_from_npz(npz, "rtc_store_action")
                raw = _hand_chunk_from_npz(npz, "raw_stored_action")
                execution = _hand_chunk_from_npz(npz, "execution_action")
        except Exception:
            continue
        if rtc_store is None or raw is None or execution is None:
            continue
        horizon = min(rtc_store.shape[0], raw.shape[0], execution.shape[0])
        rtc_store = rtc_store[:horizon]
        raw = raw[:horizon]
        execution = execution[:horizon]
        checked += 1
        raw_diffs.append(float(np.nanmax(np.abs(rtc_store - raw))))
        exec_diffs.append(float(np.nanmax(np.abs(rtc_store - execution))))
    if checked == 0:
        return {"checked": 0}
    raw_arr = np.asarray(raw_diffs, dtype=np.float64)
    exec_arr = np.asarray(exec_diffs, dtype=np.float64)
    return {
        "checked": int(checked),
        "rtc_store_vs_raw_max_abs": float(np.nanmax(raw_arr)),
        "rtc_store_vs_execution_max_abs": float(np.nanmax(exec_arr)),
        "rtc_store_matches_raw_count": int(np.sum(raw_arr <= 1.0e-5)),
        "rtc_store_matches_execution_count": int(np.sum(exec_arr <= 1.0e-5)),
    }


def _stats_array(stats: dict[str, Any], section: str, group: str, key: str) -> np.ndarray:
    return np.asarray(stats["new_embodiment"][section][group][key], dtype=np.float64)


def _print_distribution(
    title: str,
    values: np.ndarray,
    *,
    q01: np.ndarray,
    q99: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    threshold_pct: float = 10.0,
) -> None:
    print(f"\n[{title}]")
    below = np.nanmean(values < q01[None, :], axis=0) * 100.0
    above = np.nanmean(values > q99[None, :], axis=0) * 100.0
    zmax = np.nanmax(np.abs((values - mean[None, :]) / (std[None, :] + 1.0e-9)), axis=0)
    any_printed = False
    for idx, name in enumerate(HAND_NAMES):
        if below[idx] >= threshold_pct or above[idx] >= threshold_pct or zmax[idx] >= 3.0:
            any_printed = True
            print(
                f"  {name:18s} range=[{np.nanmin(values[:, idx]):+.4f},{np.nanmax(values[:, idx]):+.4f}] "
                f"q01/q99=[{q01[idx]:+.4f},{q99[idx]:+.4f}] "
                f"below={below[idx]:5.1f}% above={above[idx]:5.1f}% max|z|={zmax[idx]:.2f}"
            )
    if not any_printed:
        print("  ok: no hand dimension exceeded reporting thresholds")


def _distribution_issue_names(
    values: np.ndarray,
    *,
    q01: np.ndarray,
    q99: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    outside_pct_threshold: float = 50.0,
    z_threshold: float = 6.0,
) -> list[str]:
    below = np.nanmean(values < q01[None, :], axis=0) * 100.0
    above = np.nanmean(values > q99[None, :], axis=0) * 100.0
    zmax = np.nanmax(np.abs((values - mean[None, :]) / (std[None, :] + 1.0e-9)), axis=0)
    names: list[str] = []
    for idx, name in enumerate(HAND_NAMES):
        if below[idx] >= outside_pct_threshold or above[idx] >= outside_pct_threshold or zmax[idx] >= z_threshold:
            names.append(name)
    return names


def _print_delta_summary(title: str, raw: np.ndarray, ref: np.ndarray) -> None:
    delta = raw - ref
    print(f"\n[{title}]")
    for idx, name in enumerate(HAND_NAMES):
        neg = np.nanmean(delta[:, idx] < 0.0) * 100.0
        pos = np.nanmean(delta[:, idx] > 0.0) * 100.0
        print(
            f"  {name:18s} delta_mean={np.nanmean(delta[:, idx]):+.5f} "
            f"delta_range=[{np.nanmin(delta[:, idx]):+.5f},{np.nanmax(delta[:, idx]):+.5f}] "
            f"neg={neg:5.1f}% pos={pos:5.1f}%"
        )


def _load_probe_hand(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    rows = _read_jsonl(path)
    hands = []
    sources = []
    rtc_on = []
    for row in rows:
        action = row.get("raw_action") or {}
        hand = action.get("hand_joint_target")
        if hand is None:
            continue
        arr = np.asarray(hand, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        hands.append(arr[0, : len(HAND_NAMES)])
        metadata = row.get("metadata") or {}
        sources.append(str(metadata.get("reference_action_source") or ""))
        rtc_on.append(row.get("rtc_options") is not None)
    if not hands:
        return None
    return {
        "first_hand": np.stack(hands, axis=0),
        "sources": np.asarray(sources, dtype=object),
        "rtc_options_present": np.asarray(rtc_on, dtype=bool),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-jsonl", type=Path, default=DEFAULT_TRACE_JSONL)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--probe-chunks", type=Path, default=DEFAULT_PROBE_CHUNKS)
    parser.add_argument("--fail-on-issues", action="store_true")
    args = parser.parse_args()

    trace_rows = _read_jsonl(args.trace_jsonl)
    session_id = str(args.session_id or _latest_session(trace_rows))
    rows = _trace_session_rows(trace_rows, session_id)
    hand = _collect_trace_hand(rows)
    stats = json.loads(args.statistics.read_text(encoding="utf-8"))

    action_q01 = _stats_array(stats, "action", "hand_joint_target", "q01")
    action_q99 = _stats_array(stats, "action", "hand_joint_target", "q99")
    action_mean = _stats_array(stats, "action", "hand_joint_target", "mean")
    action_std = _stats_array(stats, "action", "hand_joint_target", "std")
    state_q01 = _stats_array(stats, "state", "hand_joint_pos", "q01")
    state_q99 = _stats_array(stats, "state", "hand_joint_pos", "q99")
    state_mean = _stats_array(stats, "state", "hand_joint_pos", "mean")
    state_std = _stats_array(stats, "state", "hand_joint_pos", "std")

    print(f"[trace] path={args.trace_jsonl}")
    print(f"[trace] session={session_id} replans={len(rows)}")
    print(f"[trace] reference_sources={dict(Counter(hand['sources']))}")
    print(f"[trace] rtc_reasons={dict(Counter(hand['rtc_reasons']))}")
    print(f"[trace] rtc_options_present={int(hand['rtc_options_present'].sum())}/{len(rows)}")
    seed_storage = _trace_seed_storage_diagnostics(rows)
    if seed_storage.get("checked", 0):
        print(f"[trace] seed_storage={seed_storage}")
    issues: list[str] = []

    _print_delta_summary("trace raw_action_first - reference_hand_first", hand["raw"], hand["reference"])
    _print_distribution(
        "trace reference_action.hand_joint_target vs checkpoint action q01/q99",
        hand["reference"],
        q01=action_q01,
        q99=action_q99,
        mean=action_mean,
        std=action_std,
    )
    _print_distribution(
        "trace raw first hand_joint_target vs checkpoint action q01/q99",
        hand["raw"],
        q01=action_q01,
        q99=action_q99,
        mean=action_mean,
        std=action_std,
    )
    _print_distribution(
        "trace observation.state.hand_joint_pos vs checkpoint state q01/q99",
        hand["observation"],
        q01=state_q01,
        q99=state_q99,
        mean=state_mean,
        std=state_std,
    )

    reference_bad = _distribution_issue_names(
        hand["reference"],
        q01=action_q01,
        q99=action_q99,
        mean=action_mean,
        std=action_std,
    )
    observation_bad = _distribution_issue_names(
        hand["observation"],
        q01=state_q01,
        q99=state_q99,
        mean=state_mean,
        std=state_std,
    )
    ref_obs_gap = np.nanmax(np.abs(hand["reference"] - hand["observation"]), axis=0)
    ref_obs_bad = [name for idx, name in enumerate(HAND_NAMES) if ref_obs_gap[idx] >= 0.5]
    if reference_bad:
        issues.append("reference_hand_out_of_training_distribution")
        print(f"\n[likely issue] reference hand is out of checkpoint action distribution: {reference_bad}")
    if observation_bad:
        issues.append("observation_hand_out_of_training_distribution")
        print(f"[likely issue] observation hand is out of checkpoint state distribution: {observation_bad}")
    if ref_obs_bad:
        issues.append("reference_observation_hand_mismatch")
        print("[likely issue] reference hand and observation hand disagree by >=0.5 rad:")
        for idx, name in enumerate(HAND_NAMES):
            if name in ref_obs_bad:
                print(f"  {name:18s} max_abs_gap={ref_obs_gap[idx]:.4f}")

    sim_minus_raw = np.nanmax(np.abs(hand["sim"] - np.clip(hand["raw"], -0.6, 1.6)), axis=0)
    if np.nanmax(sim_minus_raw) > 1.0e-5:
        print("\n[warning] execution/projection differs from probe [-0.6,1.6] clipped raw action:")
        for idx, name in enumerate(HAND_NAMES):
            if sim_minus_raw[idx] > 1.0e-5:
                print(f"  {name:18s} max_abs_diff={sim_minus_raw[idx]:.6f}")
    else:
        print("\n[projection] ok: sim hand first-step matches probe-range clipped raw action")

    if len(rows) > 1 and not np.any(hand["sources"][1:] == "rtc_seed"):
        issues.append("missing_rtc_seed_reference")
        print(
            "\n[likely issue] No later replan uses reference_hand_source=rtc_seed. "
            "This is not comparable to the full RTC probe."
        )
    if len(rows) > 1 and not np.any(hand["rtc_options_present"][1:]):
        issues.append("missing_rtc_options")
        print(
            "[likely issue] RTC options are absent after the first replan. "
            "The online rollout is running no-RTC/open-loop chunks while the probe uses RTC."
        )
    if seed_storage.get("checked", 0):
        checked = int(seed_storage["checked"])
        raw_matches = int(seed_storage["rtc_store_matches_raw_count"])
        exec_matches = int(seed_storage["rtc_store_matches_execution_count"])
        if raw_matches == checked and exec_matches < checked and ref_obs_bad:
            issues.append("rtc_seed_storage_uses_unexecutable_raw_hand")
            print(
                "[likely issue] RTC seed storage matches raw while reference/observation hand disagree. "
                "For online execution, next-step hand reference should come from executable command-space seed, "
                "not unbounded raw decoded hand values."
            )
        elif exec_matches >= max(1, checked // 2):
            print(
                "[seed] ok: RTC seed storage is close to execution command space for most checked chunks."
            )

    probe = _load_probe_hand(args.probe_chunks)
    if probe is None:
        print(f"\n[probe] not available: {args.probe_chunks}")
    else:
        print(f"\n[probe] chunks={args.probe_chunks}")
        print(f"[probe] rows={probe['first_hand'].shape[0]} reference_sources={dict(Counter(probe['sources']))}")
        print(f"[probe] rtc_options_present={int(probe['rtc_options_present'].sum())}/{probe['first_hand'].shape[0]}")
        _print_delta_summary(
            "probe adjacent first-step hand deltas",
            probe["first_hand"][1:],
            probe["first_hand"][:-1],
        )
        _print_distribution(
            "probe first hand_joint_target vs checkpoint action q01/q99",
            probe["first_hand"],
            q01=action_q01,
            q99=action_q99,
            mean=action_mean,
            std=action_std,
        )

    print("\n[interpretation]")
    print("  If same-input probe/local outputs match, processor normalization/decode/key-order are not the root cause.")
    print("  Differences then come from online reference_action/RTC semantics, execution projection, or observation distribution.")
    print("  Online RTC hand seed should stay in executable command space; raw decoded chunks are kept for diagnostics.")
    if issues:
        print(f"[issues] {issues}")
    elif args.fail_on_issues:
        print("[issues] none")
    return 2 if issues and args.fail_on_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
