"""CLI: sample N random distinct PDBs from RCSB and download them to a cache."""
import argparse

from protein_patch.data.fetch import sample_random_pdb_ids, fetch_pdbs


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the download pipeline.

    Args:
        argv: Argument list (defaults to sys.argv when None).
    """
    ap = argparse.ArgumentParser(description="Download N random PDBs from RCSB.")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="cache directory")
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args(argv)
    ids = sample_random_pdb_ids(args.n, args.seed)
    paths = fetch_pdbs(ids, args.out, delay=args.delay)
    print(f"fetched/cached {len(paths)} PDBs into {args.out}")


if __name__ == "__main__":
    main()
