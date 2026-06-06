# ProteinPatchAnalysis

Turn PDB protein structures into voxelized "surface patches" and learn a compact
representation of them with a 3D convolutional variational autoencoder (VAE).

A patch is a cubic grid centred on a solvent-exposed residue, with one channel per
atom type (C, N, O, S) holding a Gaussian atomic-density field. Patches are stored
**channel-first** as `(n_channels, L, L, L)`.

## Modernized pipeline (2026)

Pure Python — no GROMACS, AmberTools, or Keras. Everything lives in
`src/protein_patch/`:

**Prep** (`Biopython` + `NumPy`)
- `clean.load_clean_structure(name, handle)` — parse a PDB, strip waters/heteroatoms
- `sasa.relative_sasa(structure)` — per-residue relative solvent accessibility via
  Biopython's Shrake–Rupley (replaces GROMACS `g_sas`)
- `voxelize.voxelize_atoms(coords, elements, grid_min, spec)` — Gaussian density grid
- `patches.extract_patches(structure, spec)` — one patch per exposed residue →
  `(n, 4, L, L, L)`

**Model** (`PyTorch`)
- `model.ConvVAE3D(spec, latent_dim)` — 3D conv VAE with a correct
  `exp(0.5·logvar)` reparameterization and an MSE + softplus reconstruction head
  suited to continuous density fields
- `model.PatchDataset(dir)` — streams patch pickles
- `model.train.train(train_dir, val_dir, spec, cfg)` — seeded training loop

`PatchSpec` (`spec.py`) is the single source of geometry truth — default
**64³ voxels × 0.375 Å = a 24 Å cube**. Both prep and model import it, so the
patch shape can never silently drift between them.

## Quickstart

```bash
uv sync --extra dev
uv run pytest                      # full suite
uv run pytest -m "not integration" # skip the network-gated 1UBQ test
```

## Legacy

The original 2019 standalone-Keras / GROMACS / HPC prototype is preserved under
`legacy/` (the MNIST autoencoder experiments, the old `SamplePrep` shell+Python
pipeline, and the old `ModelOptimization_3D` Keras VAE). It is kept for reference
and is not maintained. The sample data (`SamplePrep/1raw_pdb` etc.) remains in
place at the repo root.
