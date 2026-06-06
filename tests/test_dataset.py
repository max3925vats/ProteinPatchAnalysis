import pickle
import numpy as np
import torch
from protein_patch.model.dataset import PatchDataset


def test_dataset_loads_channel_first(tmp_path, rng):
    shape = (4, 8, 8, 8)  # dataset is shape-agnostic; small for speed
    for i in range(3):
        arr = rng.random(shape).astype(np.float32)
        with open(tmp_path / f"patch{i}.pickle", "wb") as f:
            pickle.dump(arr, f)
    ds = PatchDataset(tmp_path)
    assert len(ds) == 3
    x = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == shape           # no axis swap needed: already channel-first
    assert x.dtype == torch.float32


def test_dataset_raises_on_empty_dir(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        PatchDataset(tmp_path)
