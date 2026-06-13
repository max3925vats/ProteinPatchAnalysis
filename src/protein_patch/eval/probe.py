import numpy as np
import torch


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


def linear_probe_accuracy(train_emb, train_labels, val_emb, val_labels,
                          epochs: int = 100, seed: int = 0) -> float:
    """Trained linear probe: a seeded multinomial logistic regression on the
    FROZEN embeddings. A single Linear layer keeps it weak, so high accuracy
    means the property is linearly accessible in the latent — not that the probe
    is powerful. No sklearn dependency.
    """
    torch.manual_seed(seed)
    Xtr = torch.as_tensor(np.asarray(train_emb), dtype=torch.float32)
    Xva = torch.as_tensor(np.asarray(val_emb), dtype=torch.float32)
    ytr = torch.as_tensor(np.asarray(train_labels), dtype=torch.long)
    yva = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    n_classes = int(max(ytr.max().item(), yva.max().item())) + 1

    clf = torch.nn.Linear(Xtr.shape[1], n_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(clf(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        preds = clf(Xva).argmax(dim=1)
    return float((preds == yva).float().mean())


def run_probes(train_emb, train_labels, val_emb, val_labels) -> dict[str, float]:
    """Both probes for one (embedding, label) pairing."""
    return {
        "knn": knn_accuracy(train_emb, train_labels, val_emb, val_labels),
        "linear": linear_probe_accuracy(train_emb, train_labels, val_emb, val_labels),
    }
