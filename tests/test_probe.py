import numpy as np

from protein_patch.eval.probe import knn_accuracy


def _two_clusters(rng, n, dim=4, sep=10.0):
    a = rng.normal(0.0, 0.5, (n, dim))
    b = rng.normal(sep, 0.5, (n, dim))
    emb = np.vstack([a, b])
    labels = np.array([0] * n + [1] * n)
    return emb, labels


def test_knn_recovers_separable_labels():
    rng = np.random.default_rng(0)
    tr_emb, tr_lab = _two_clusters(rng, 30)
    va_emb, va_lab = _two_clusters(rng, 15)
    acc = knn_accuracy(tr_emb, tr_lab, va_emb, va_lab, k=5)
    assert acc > 0.95


def test_knn_chance_on_shuffled_labels():
    # non-vacuity: destroying the label-embedding relationship -> ~chance
    rng = np.random.default_rng(1)
    tr_emb, tr_lab = _two_clusters(rng, 200)
    va_emb, va_lab = _two_clusters(rng, 100)
    shuffled = tr_lab.copy()
    rng.shuffle(shuffled)
    acc = knn_accuracy(tr_emb, shuffled, va_emb, va_lab, k=5)
    assert 0.3 < acc < 0.7
