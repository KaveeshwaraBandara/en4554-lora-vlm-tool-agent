"""Environment helpers: Colab detection, output/checkpoint directories, GPU checks.

Output location (first match wins):
  1. $EN4554_OUT if set
  2. On Colab: /content/drive/MyDrive/en4554-a1  (Drive is mounted on first use,
     so a disconnect does not lose checkpoints)
  3. Locally: <repo>/outputs
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVE_ROOT = Path("/content/drive")
DRIVE_OUT = DRIVE_ROOT / "MyDrive" / "en4554-a1"


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def _mount_drive() -> None:
    if (DRIVE_ROOT / "MyDrive").exists():
        return
    from google.colab import drive

    drive.mount(str(DRIVE_ROOT))


def get_out_dir(*parts: str, mount: bool = True) -> Path:
    """Return (and create) an output dir, e.g. get_out_dir("part1", "pissa_lm_r16_s42")."""
    if os.environ.get("EN4554_OUT"):
        root = Path(os.environ["EN4554_OUT"])
    elif in_colab():
        if mount:
            _mount_drive()
        root = DRIVE_OUT
    else:
        root = REPO_ROOT / "outputs"
    out = root.joinpath(*parts)
    out.mkdir(parents=True, exist_ok=True)
    return out


def gpu_summary() -> str:
    try:
        import torch
    except ImportError:
        return "torch not installed"
    if not torch.cuda.is_available():
        return "No CUDA GPU (CPU only)"
    p = torch.cuda.get_device_properties(0)
    cc = f"{p.major}.{p.minor}"
    bf16 = torch.cuda.is_bf16_supported()
    return (
        f"{p.name} | {p.total_memory / 2**30:.1f} GiB | compute capability {cc} | "
        f"torch {torch.__version__} | CUDA {torch.version.cuda} | bf16 supported: {bf16}"
    )


def assert_fp16_only(dtype_name: str) -> None:
    """Fail fast if a run is configured with bf16.

    The brief requires fp16 because the Colab T4 has no hardware bf16. We use fp16
    everywhere (including on faster local GPUs) so results match the Colab setup.
    """
    if "bf16" in dtype_name.lower() or "bfloat16" in dtype_name.lower():
        raise ValueError("Use fp16, not bf16: the Colab T4 does not support bf16 in hardware.")
