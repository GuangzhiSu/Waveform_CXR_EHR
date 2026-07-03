# Runtime Artifacts

Generated files for the current contrastive learning pipeline live here.

| Path | Contents |
|------|----------|
| `cache/` | Pair JSON files and frozen CXR/ECG embedding arrays |
| `checkpoints/` | Downloaded frozen Bio-ViL-T and ECG-CoCa weights |
| `outputs/` | Training, staged experiment, sweep, and label-probe outputs |
| `logs/` | Slurm stdout/stderr and small build logs |
| `pylibs/` | Workspace-local Python packages installed by setup scripts |

The large contents are ignored by git; only this layout marker is tracked.
