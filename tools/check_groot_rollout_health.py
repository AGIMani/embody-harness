#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY = Path("logs/groot_rollout_vs_smooth_distribution_goal_report/summary.json")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _trace_labels(summary: dict[str, Any]) -> list[str]:
    traces = summary.get("traces")
    if not isinstance(traces, dict):
        return []
    return list(traces.keys())


def _fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"[FAIL] {message}")


def _pass(message: str) -> None:
    print(f"[PASS] {message}")


def _check_arm(
    *,
    trace_label: str,
    trace: dict[str, Any],
    max_arm_oob_fraction: float,
    failures: list[str],
) -> None:
    rows = trace.get("arm_obs_oob_vs_smooth_q01_q99")
    if not isinstance(rows, list):
        _fail(f"{trace_label}: missing arm_obs_oob_vs_smooth_q01_q99", failures)
        return
    worst = 0.0
    worst_name = ""
    for row in rows:
        if not isinstance(row, dict):
            continue
        frac = max(float(row.get("below_q01_fraction", 0.0)), float(row.get("above_q99_fraction", 0.0)))
        if frac > worst:
            worst = frac
            worst_name = str(row.get("name", ""))
    if worst > float(max_arm_oob_fraction):
        _fail(
            f"{trace_label}: arm OOD fraction too high: {worst_name}={worst:.3f} > {max_arm_oob_fraction:.3f}",
            failures,
        )
    else:
        _pass(f"{trace_label}: arm OOD fraction ok: worst {worst_name or '<none>'}={worst:.3f}")


def _check_hand_floor(
    *,
    trace_label: str,
    trace: dict[str, Any],
    max_non_index_floor_fraction: float,
    failures: list[str],
) -> None:
    report = trace.get("hand_obs_non_index_pitch_floor")
    if not isinstance(report, dict):
        _fail(f"{trace_label}: missing hand_obs_non_index_pitch_floor", failures)
        return
    mean_fraction = float(report.get("mean_near_floor_fraction", 1.0))
    rows = report.get("by_joint") if isinstance(report.get("by_joint"), list) else []
    details = ", ".join(
        f"{row.get('name')}={float(row.get('near_floor_fraction', 1.0)):.3f}"
        for row in rows
        if isinstance(row, dict)
    )
    if mean_fraction > float(max_non_index_floor_fraction):
        _fail(
            f"{trace_label}: non-index hand pitch stuck near floor: mean={mean_fraction:.3f} "
            f"> {max_non_index_floor_fraction:.3f} ({details})",
            failures,
        )
    else:
        _pass(f"{trace_label}: non-index hand pitch floor ok: mean={mean_fraction:.3f} ({details})")


def _print_j4(summary: dict[str, Any]) -> None:
    arm = (summary.get("reset") or {}).get("arm") if isinstance(summary.get("reset"), dict) else {}
    if not isinstance(arm, dict):
        return
    j4_one = arm.get("j4_1_based")
    j4_zero = arm.get("j4_0_based")
    if isinstance(j4_one, dict):
        print(
            "[INFO] j4 1-based -> zero-based j3: "
            f"reset={float(j4_one.get('reset', 0.0)):.4f}, "
            f"smooth_q01={float(j4_one.get('smooth_q01', 0.0)):.4f}, "
            f"median={float(j4_one.get('smooth_median', 0.0)):.4f}, "
            f"q99={float(j4_one.get('smooth_q99', 0.0)):.4f}, "
            f"percentile={float(j4_one.get('reset_percentile', 0.0)):.2f}%"
        )
    if isinstance(j4_zero, dict):
        print(
            "[INFO] j4 zero-based: "
            f"reset={float(j4_zero.get('reset', 0.0)):.4f}, "
            f"smooth_q01={float(j4_zero.get('smooth_q01', 0.0)):.4f}, "
            f"median={float(j4_zero.get('smooth_median', 0.0)):.4f}, "
            f"q99={float(j4_zero.get('smooth_q99', 0.0)):.4f}, "
            f"percentile={float(j4_zero.get('reset_percentile', 0.0)):.2f}%"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate a GR00T rollout-vs-smooth summary for arm and hand health.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--trace-label", default=None, help="Trace label in summary.json. Defaults to all traces.")
    parser.add_argument("--max-arm-oob-fraction", type=float, default=0.25)
    parser.add_argument("--max-non-index-floor-fraction", type=float, default=0.80)
    parser.add_argument("--fail-on-issues", action="store_true")
    args = parser.parse_args()

    summary = _load_json(args.summary)
    traces = summary.get("traces")
    if not isinstance(traces, dict) or not traces:
        raise SystemExit(f"No traces found in {args.summary}")
    labels = [str(args.trace_label)] if args.trace_label else _trace_labels(summary)
    failures: list[str] = []

    print(f"[summary] {args.summary.expanduser().resolve()}")
    _print_j4(summary)
    for label in labels:
        trace = traces.get(label)
        if not isinstance(trace, dict):
            _fail(f"trace label not found: {label}", failures)
            continue
        _check_arm(
            trace_label=label,
            trace=trace,
            max_arm_oob_fraction=float(args.max_arm_oob_fraction),
            failures=failures,
        )
        _check_hand_floor(
            trace_label=label,
            trace=trace,
            max_non_index_floor_fraction=float(args.max_non_index_floor_fraction),
            failures=failures,
        )

    if failures:
        print(f"[health] issues={len(failures)}")
        if bool(args.fail_on_issues):
            return 1
    else:
        print("[health] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
