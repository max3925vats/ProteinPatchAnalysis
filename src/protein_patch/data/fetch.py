"""Download PDB files from RCSB with local caching and a deterministic random sampler."""
import json
import logging
import random
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
_HOLDINGS_URL = "https://data.rcsb.org/rest/v1/holdings/current/entry_ids"


def read_id_file(path: str | Path) -> list[str]:
    """Read PDB IDs from a text file (one per line; '#' comments, blanks ignored)."""
    out = []
    for line in Path(path).read_text().splitlines():
        # Strip everything after a comment marker, then whitespace
        s = line.split("#", 1)[0].strip()
        if s:
            out.append(s)
    return out


def _fetch_current_entry_ids() -> list[str]:
    """Fetch the full set of current PDB entry IDs from RCSB."""
    with urllib.request.urlopen(_HOLDINGS_URL, timeout=60) as r:
        return json.loads(r.read().decode())


def sample_random_pdb_ids(
    n: int,
    seed: int,
    holdings: list[str] | None = None,
) -> list[str]:
    """Return n distinct PDB IDs sampled deterministically from RCSB holdings.

    Args:
        n: Number of IDs to sample.
        seed: Random seed for reproducibility.
        holdings: Optional pre-fetched list of IDs; if None, fetches from RCSB.

    Returns:
        List of n distinct PDB ID strings.
    """
    ids = holdings if holdings is not None else _fetch_current_entry_ids()
    return random.Random(seed).sample(ids, n)


def _download_one(pdb_id: str, dest: Path) -> None:
    """Fetch a single PDB file from RCSB and write it to dest."""
    url = _DOWNLOAD_URL.format(pdb_id=pdb_id)
    with urllib.request.urlopen(url, timeout=30) as r:
        dest.write_bytes(r.read())


def fetch_pdbs(
    ids: list[str],
    cache_dir: str | Path,
    delay: float = 0.5,
) -> list[Path]:
    """Download each ID to cache_dir, skipping any already present.

    Sleeps `delay` seconds after each NETWORK download (not after cache hits).
    Per-ID failures are logged and skipped — never raised to the caller.

    Args:
        ids: PDB IDs to fetch.
        cache_dir: Directory where <ID>.pdb files are stored.
        delay: Polite inter-request sleep in seconds.

    Returns:
        Paths to all successfully cached files (hits + downloads).
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for pdb_id in ids:
        dest = cache / f"{pdb_id.upper()}.pdb"
        if dest.exists():
            paths.append(dest)
            continue
        try:
            _download_one(pdb_id, dest)
            paths.append(dest)
            if delay:
                time.sleep(delay)
        except Exception as e:
            dest.unlink(missing_ok=True)  # drop any partial write so a retry re-fetches
            logger.warning("failed to fetch %s: %s", pdb_id, e)
    return paths
