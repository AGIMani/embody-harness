# Nero Genesis + Linker Hand L10 Bundle

This folder is self-contained for Genesis loading of the Nero dual-arm scene with a Linker Hand L10 mounted on the selected flange.

## Contents

- `nero_arm_linker_l10_genesis_config.py`: importable config and runtime factory.
- `smoke_test.py`: minimal Genesis load test.
- `nero_twin/`: Nero Genesis runtime scripts and arm/base assets.
- `linkerhand-urdf/`: Linker Hand L10 URDF and meshes.

## Requirements

Install a Python environment with `genesis-world` available. This bundle was verified with `genesis-world==0.4.7`.

## Use

From this directory:

```bash
python smoke_test.py --backend cpu
```

In another project:

```python
from pathlib import Path
import sys

bundle = Path("/path/to/nero_genesis_linker_l10_bundle")
sys.path.insert(0, str(bundle))

from nero_arm_linker_l10_genesis_config import make_runtime

runtime = make_runtime(backend="gpu", show_viewer=True)
runtime.connect()
runtime.step(selected="right")
```

All asset paths are resolved relative to this bundle directory.

