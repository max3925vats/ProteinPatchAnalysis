import numpy as np


def knn_accuracy(train_emb, train_labels, val_emb, val_labels, k: int = 5) -> float:
    """Probe-free kNN accuracy of frozen latents.

    A weak, parameter-free read of how much a property is linearly/locally
    accessible in the latent: for each val embedding, majority-vote the labels
    of its k nearest train embeddings (Euclidean). No training, so it can't
    overfit the probe — the latent does the work.
    """
    tr = np.asarray(train_emb, dtype=np.float64)
    va = np.asarray(val_emb, dtype=np.float64)
    tr_lab = np.asarray(train_labels)
    va_lab = np.asarray(val_labels)
    k = min(k, tr.shape[0])

    # (n_val, n_train) squared Euclidean distances
    d2 = ((va[:, None, :] - tr[None, :, :]) ** 2).sum(axis=2)
    nn_idx = np.argsort(d2, axis=1)[:, :k]            # k nearest train indices
    preds = np.empty(va.shape[0], dtype=tr_lab.dtype)
    for i, idx in enumerate(nn_idx):
        vals, counts = np.unique(tr_lab[idx], return_counts=True)
        preds[i] = vals[np.argmax(counts)]            # majority vote (ties: lowest label)
    return float((preds == va_lab).mean())
