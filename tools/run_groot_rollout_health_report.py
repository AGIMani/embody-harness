#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMOOTH_DIR = ROOT.parent / "Isaac-GR00T" / "outputs" / "IsaacLab" / "nero" / "mission2" / "smooth"
DEFAULT_OUTPUT_DIR = ROOT / "logs" / "groot_rollout_vs_smooth_distribution_report"


def _run(cmd: list[str]) -> int:
    print("[run] " + " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a GR00T rollout-vs-smooth report and run the health gate in one command."
    )
    parser.add_argument("--smooth-dir", type=Path, default=DEFAULT_SMOOTH_DIR)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--trace-label", default="online")
    parser.add_argument("--compare-trace-dir", type=Path, default=None)
    parser.add_argument("--compare-trace-label", default="compare")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--compare-session-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--health-trace-label", default=None, help="Trace label to gate. Defaults to --trace-label.")
    parser.add_argument("--max-arm-oob-fraction", type=float, default=0.25)
    parser.add_argument("--max-non-index-floor-fraction", type=float, default=0.80)
    parser.add_argument("--fail-on-issues", action="store_true")
    args = parser.parse_args()

    analyzer = ROOT / "tools" / "analyze_groot_rollout_vs_smooth_distribution.py"
    gate = ROOT / "tools" / "check_groot_rollout_health.py"
    analyze_cmd = [
        sys.executable,
        str(analyzer),
        "--smooth-dir",
        str(args.smooth_dir.expanduser()),
        "--trace-dir",
        str(args.trace_dir.expanduser()),
        "--trace-label",
        str(args.trace_label),
        "--output-dir",
        str(args.output_dir.expanduser()),
    ]
    if args.session_id:
        analyze_cmd.extend(["--session-id", str(args.session_id)])
    if args.compare_trace_dir is not None:
        analyze_cmd.extend(
            [
                "--compare-trace-dir",
                str(args.compare_trace_dir.expanduser()),
                "--compare-trace-label",
                str(args.compare_trace_label),
            ]
        )
        if args.compare_session_id:
            analyze_cmd.extend(["--compare-session-id", str(args.compare_session_id)])
    else:
        analyze_cmd.append("--no-compare-trace")

    rc = _run(analyze_cmd)
    if rc != 0:
        return rc

    health_label = str(args.health_trace_label or args.trace_label)
    gate_cmd = [
        sys.executable,
        str(gate),
        "--summary",
        str(args.output_dir.expanduser() / "summary.json"),
        "--trace-label",
        health_label,
        "--max-arm-oob-fraction",
        str(float(args.max_arm_oob_fraction)),
        "--max-non-index-floor-fraction",
        str(float(args.max_non_index_floor_fraction)),
    ]
    if bool(args.fail_on_issues):
        gate_cmd.append("--fail-on-issues")
    rc = _run(gate_cmd)
    if rc == 0:
        print(f"[report] output_dir={args.output_dir.expanduser().resolve()}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
