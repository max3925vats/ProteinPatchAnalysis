"""Headline comparison: the representation × probe matrix.

For each representation (voxel, point) and probe task, train a frozen contrastive
encoder, embed, and score the latents (kNN + linear), aggregated over seeds. The
masked-identity task uses a SEPARATELY trained center-excluded encoder (latent never
saw the central residue) — so its column is a different latent, flagged as such.
"""
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from .config import ContrastiveConfig
from .contrastive_train import encode_patches, train_contrastive
from .eval.probe import run_probes
from .spec import PatchSpec
from .tasks.burial import assign_classes, fit_quantile_edges, read_rel_sasa
from .tasks.identity import identity_labels
from .tasks.io import read_patch_meta
from .tasks.secondary_structure import ss_labels_from_dssp

_KINDS = ("voxel", "point")

# A label loader maps (train_dir, val_dir) -> (train_labels, val_labels), aligned
# to sorted-path order. Pairing train+val lets tasks fit on train, apply to val.
LabelFn = Callable[[str, str], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class ProbeTask:
    name: str
    labels: LabelFn
    use_excluded_encoder: bool          # True -> environment-only encoder (identity)


def burial_task(n_classes: int = 3) -> ProbeTask:
    def labels(train_dir: str, val_dir: str):
        _, tr = read_rel_sasa(train_dir)
        _, va = read_rel_sasa(val_dir)
        edges = fit_quantile_edges(tr, n_classes)        # fit on train only
        return assign_classes(tr, edges), assign_classes(va, edges)
    return ProbeTask("burial", labels, use_excluded_encoder=False)


def identity_task() -> ProbeTask:
    def labels(train_dir: str, val_dir: str):
        _, tr = identity_labels(train_dir)
        _, va = identity_labels(val_dir)
        return tr, va
    return ProbeTask("identity", labels, use_excluded_encoder=True)


def ss_task(dssp_by_dir: Callable[[str], dict]) -> ProbeTask:
    """Secondary-structure probe. `dssp_by_dir(root) -> {(chain,resseq): 8-state}`
    (real use: build it per protein via tasks.secondary_structure.compute_dssp;
    tests pass a synthetic map). Labels align to sorted patches; -1 where DSSP has
    no entry (filtered before probing)."""
    def labels(train_dir: str, val_dir: str):
        def lab(root: str) -> np.ndarray:
            _, prov = read_patch_meta(root)
            keys = [(p[1], p[2]) for p in prov]          # (chain, resseq)
            return ss_labels_from_dssp(dssp_by_dir(root), keys)
        return lab(train_dir), lab(val_dir)
    return ProbeTask("ss", labels, use_excluded_encoder=False)


def _labeled(emb: np.ndarray, lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows with an unknown label (-1) so they never reach the probes."""
    keep = np.asarray(lab) >= 0
    return np.asarray(emb)[keep], np.asarray(lab)[keep]


def probe_cell(kind: str, task: ProbeTask, train_dir: str, val_dir: str,
               spec: PatchSpec, ccfg: ContrastiveConfig) -> dict[str, float]:
    """Train the right encoder for one (kind, task), embed both splits, probe."""
    _, encoder = train_contrastive(kind, train_dir, spec, ccfg,
                                   exclude_center_atoms=task.use_excluded_encoder)
    tr_emb = encode_patches(encoder, train_dir, spec, kind, task.use_excluded_encoder)
    va_emb = encode_patches(encoder, val_dir, spec, kind, task.use_excluded_encoder)
    tr_lab, va_lab = task.labels(train_dir, val_dir)
    tr_emb, tr_lab = _labeled(tr_emb, tr_lab)        # drop unknown-label rows (-1)
    va_emb, va_lab = _labeled(va_emb, va_lab)
    return run_probes(tr_emb, tr_lab, va_emb, va_lab)


def build_matrix(train_dir: str, val_dir: str, spec: PatchSpec,
                 ccfg: ContrastiveConfig, tasks: list[ProbeTask],
                 seeds: tuple[int, ...] = (0,)) -> dict:
    """{(kind, task) -> {knn/linear mean+std, use_excluded_encoder}} over seeds."""
    out: dict = {}
    for kind in _KINDS:
        for task in tasks:
            knn, lin = [], []
            for seed in seeds:
                cell = probe_cell(kind, task, train_dir, val_dir, spec,
                                  replace(ccfg, seed=seed))
                knn.append(cell["knn"]); lin.append(cell["linear"])
            out[(kind, task.name)] = {
                "knn_mean": float(np.mean(knn)), "knn_std": float(np.std(knn)),
                "linear_mean": float(np.mean(lin)), "linear_std": float(np.std(lin)),
                "n_seeds": len(seeds),
                "use_excluded_encoder": task.use_excluded_encoder,
            }
    return out


def capacity_sensitivity(train_dir: str, val_dir: str, spec: PatchSpec,
                         task: ProbeTask, configs: dict[str, dict[str, ContrastiveConfig]]
                         ) -> dict:
    """Re-run voxel-vs-point under each capacity-matching convention.

    `configs`: {convention_label -> {"voxel": cfg, "point": cfg}}. Returns, per
    label, the per-kind linear accuracy and the winner — so a ranking *flip*
    across conventions (params vs FLOPs vs width) is visible (that flip is itself
    the finding).
    """
    report: dict = {}
    for label, per_kind in configs.items():
        acc = {kind: probe_cell(kind, task, train_dir, val_dir, spec, per_kind[kind])["linear"]
               for kind in _KINDS}
        report[label] = {"acc": acc, "winner": max(acc, key=acc.get)}
    return report
