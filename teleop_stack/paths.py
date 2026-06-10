from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_isaac_teleop_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_value = os.environ.get("ISAAC_TELEOP_ROOT")
    if env_value:
        candidates.append(Path(env_value))
    root = repo_root()
    candidates.extend((root / "external" / "IsaacTeleop", root.parent / "IsaacTeleop"))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.expanduser().resolve()
    raise FileNotFoundError(
        "Could not locate IsaacTeleop. Set ISAAC_TELEOP_ROOT or clone it into external/IsaacTeleop or ../IsaacTeleop."
    )


def resolve_plugin_root_dir(isaac_teleop_root: str | os.PathLike[str] | None = None) -> Path | None:
    root = resolve_isaac_teleop_root(isaac_teleop_root)
    plugin_dir = root / "plugins"
    return plugin_dir if plugin_dir.is_dir() else None
