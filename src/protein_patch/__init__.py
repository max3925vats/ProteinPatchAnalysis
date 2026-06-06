"""Pure-Python protein surface-patch voxelization and 3D VAE."""
from .spec import PatchSpec
from .config import TrainConfig
from .clean import load_clean_structure
from .patches import extract_patches

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "PatchSpec",
    "TrainConfig",
    "load_clean_structure",
    "extract_patches",
]
