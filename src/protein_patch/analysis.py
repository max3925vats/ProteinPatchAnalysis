import numpy as np
import torch


def embed_patches(model: "torch.nn.Module", dataset, device: str = "cpu") -> np.ndarray:
    """Encode every patch in *dataset* to its latent mean vector (mu).

    Args:
        model: A ``ConvVAE3D`` instance with an ``encode(x) -> (mu, logvar)``
               method.
        dataset: Any object supporting ``__len__`` and ``__getitem__`` that
                 returns a ``(C, L, L, L)`` float tensor.
        device: Torch device string; defaults to ``"cpu"``.

    Returns:
        NumPy array of shape ``(n, latent_dim)``.
    """
    model.eval().to(device)
    rows = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x = dataset[i].unsqueeze(0).to(device)   # (1, C, L, L, L)
            mu, _ = model.encode(x)
            rows.append(mu.cpu().numpy()[0])          # (latent_dim,)
    return np.asarray(rows)


def pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """Project an ``(n, d)`` embedding matrix to its first two PCs.

    Uses numpy SVD on the zero-mean data; no external dependency. The
    returned columns are ordered by descending explained variance, so
    ``out[:, 0].var() >= out[:, 1].var()`` is guaranteed.

    Args:
        embeddings: Array of shape ``(n, d)`` with ``d >= 2``.

    Returns:
        Array of shape ``(n, 2)``.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    x = x - x.mean(axis=0)          # center columns
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T              # project onto first two right-singular vectors
