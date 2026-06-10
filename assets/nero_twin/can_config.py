"""Shared JSON configuration helpers for Nero CAN demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path(__file__).resolve().with_name("nero_can_config.json")
_SECTION_KEYS = {"common", "can", "digital_twin", "dual_digital_twin", "ik_demo", "read_feedback"}


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"JSON config file for CAN/runtime defaults. Default: {DEFAULT_CONFIG}",
    )


def parse_args(parser: argparse.ArgumentParser, *, sections: tuple[str, ...]) -> argparse.Namespace:
    probe, _ = parser.parse_known_args()
    config_path = probe.config.expanduser().resolve()
    if config_path.exists():
        values = _load_config_defaults(config_path, parser, sections)
        if values:
            parser.set_defaults(**values)
    args = parser.parse_args()
    args.config = args.config.expanduser().resolve()
    return args


def _load_config_defaults(
    config_path: Path,
    parser: argparse.ArgumentParser,
    sections: tuple[str, ...],
) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a JSON object: {config_path}")

    merged: dict[str, Any] = {}
    _merge_scalars(merged, raw)
    for section in ("common", "can", *sections):
        value = raw.get(section)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"Config section '{section}' must be a JSON object: {config_path}")
        _merge_scalars(merged, value)

    channels = raw.get("channels")
    if isinstance(channels, dict):
        _apply_channels(merged, channels)
    can = raw.get("can")
    if isinstance(can, dict) and isinstance(can.get("channels"), dict):
        _apply_channels(merged, can["channels"])

    if "enable_push" in merged:
        merged["no_enable_push"] = not bool(merged.pop("enable_push"))

    known = {action.dest: action for action in parser._actions if action.dest != argparse.SUPPRESS}
    return {
        key: _coerce_value(value, known[key])
        for key, value in merged.items()
        if key in known and key != "config"
    }


def _merge_scalars(out: dict[str, Any], values: dict[str, Any]) -> None:
    for key, value in values.items():
        if key in _SECTION_KEYS or key == "channels":
            continue
        if isinstance(value, dict):
            continue
        out[key.replace("-", "_")] = value


def _apply_channels(out: dict[str, Any], channels: dict[str, Any]) -> None:
    if "left" in channels:
        out["left_channel"] = channels["left"]
    if "right" in channels:
        out["right_channel"] = channels["right"]
    if "single" in channels:
        out["channel"] = channels["single"]
    if "default" in channels and "channel" not in out:
        out["channel"] = channels["default"]


def _coerce_value(value: Any, action: argparse.Action) -> Any:
    if value is None:
        return value
    if action.type is Path and isinstance(value, str):
        return Path(value)
    return value
