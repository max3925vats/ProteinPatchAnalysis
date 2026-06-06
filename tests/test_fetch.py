from protein_patch.data.fetch import read_id_file, sample_random_pdb_ids, fetch_pdbs
from protein_patch.data import fetch as fetch_mod


def test_read_id_file_skips_blanks_and_comments(tmp_path):
    f = tmp_path / "ids.txt"
    f.write_text("1UBQ\n# a comment\n\n4HHB  \n")
    assert read_id_file(f) == ["1UBQ", "4HHB"]


def test_sample_random_pdb_ids_is_deterministic_and_distinct():
    holdings = [f"{i:04X}" for i in range(1000)]
    a = sample_random_pdb_ids(50, seed=7, holdings=holdings)
    b = sample_random_pdb_ids(50, seed=7, holdings=holdings)
    assert a == b
    assert len(set(a)) == 50
    assert sample_random_pdb_ids(50, seed=8, holdings=holdings) != a


def test_fetch_pdbs_skips_existing(tmp_path, monkeypatch):
    calls = []

    def fake_download(pdb_id, dest):
        calls.append(pdb_id)
        dest.write_text("ATOM\n")

    monkeypatch.setattr("protein_patch.data.fetch._download_one", fake_download)
    (tmp_path / "1UBQ.pdb").write_text("already here")
    paths = fetch_pdbs(["1UBQ", "4HHB"], tmp_path, delay=0.0)
    assert calls == ["4HHB"]
    assert {p.name for p in paths} == {"1UBQ.pdb", "4HHB.pdb"}


def test_cli_main_samples_and_fetches(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_mod, "_fetch_current_entry_ids",
                        lambda: [f"{i:04X}" for i in range(500)])
    monkeypatch.setattr(fetch_mod, "_download_one",
                        lambda pid, dest: dest.write_text("ATOM\n"))
    from scripts.fetch_random_pdbs import main
    main(["--n", "5", "--seed", "1", "--out", str(tmp_path), "--delay", "0"])
    assert len(list(tmp_path.glob("*.pdb"))) == 5
