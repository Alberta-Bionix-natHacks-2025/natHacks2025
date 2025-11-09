# tools/find_and_copy_bnci_cache.py
import os, sys, shutil
from pathlib import Path

REPO_RAW = Path("data/eeg/raw").resolve()
REPO_RAW.mkdir(parents=True, exist_ok=True)

def uniq(seq):
    seen = set()
    out = []
    for s in seq:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

def _flatten_paths(p):
    """Return a list of file paths from various structures MOABB may return."""
    if p is None:
        return []
    if isinstance(p, (str, os.PathLike)):
        return [str(p)]
    if isinstance(p, dict):
        out = []
        for v in p.values():
            out.extend(_flatten_paths(v))
        return out
    try:
        # list/tuple/iterable
        out = []
        for x in p:
            out.extend(_flatten_paths(x))
        return out
    except TypeError:
        return []

def from_moabb() -> list[str]:
    """Ask MOABB directly where it put BNCI2014_001."""
    try:
        from moabb.datasets import BNCI2014_001
        ds = BNCI2014_001()
        try:
            paths = ds.data_path(subject=1)  # older MOABB API
        except TypeError:
            # newer MOABB API sometimes uses 'subjects' or positional
            try:
                paths = ds.data_path(subjects=[1])
            except TypeError:
                paths = ds.data_path([1])
        files = _flatten_paths(paths)
        parents = [str(Path(f).parent) for f in files]
        return uniq(parents)
    except Exception as e:
        print(f"[info] MOABB probe failed: {e}")
        return []

def scan_common_places() -> list[str]:
    """Heuristic scan of typical Windows cache locations & names."""
    home = Path.home()
    roots = [
        home / "moabb_data",
        home / ".moabb",
        home / "mne_data",
        home / "AppData/Local/mne",
        home / "AppData/Local/mne_data",
        home / "AppData/Local/moabb",
        home / "AppData/Local/Temp",
        home / ".cache",                 # sometimes used by pooch
        home / "AppData/Local/pypoetry", # rare
    ]
    # BNCI2014_001 == BCI Competition IV 2a; files look like A01T.gdf, A01E.gdf, etc.
    patterns = [
        "**/*BNCI*2014*001*",
        "**/*BCI*Competition*IV*2a*",
        "**/*BCICIV_2a*",
        "**/A0*T.gdf",
        "**/A0*E.gdf",
    ]
    hits = []
    for root in roots:
        if not root.is_dir():
            continue
        for pat in patterns:
            for p in root.glob(pat):
                hits.append(p if p.is_dir() else p.parent)
    return uniq([str(Path(h)) for h in hits])

def copy_tree(src: Path, dst: Path):
    if dst.exists():
        print(f"Destination already exists: {dst}")
        return
    print(f"\nCopying\n  from: {src}\n    to: {dst}")
    shutil.copytree(src, dst)
    print("Done.")

def main():
    print("Looking for BNCI2014-001 cache...")
    candidates = from_moabb()
    if not candidates:
        candidates = scan_common_places()

    print("Candidates:")
    for c in candidates:
        print(" -", c)

    if not candidates:
        print("\nNo cache found. Run your loader once (it will download), then re-run this script.")
        sys.exit(1)

    src = Path(candidates[0]).resolve()
    # keep original folder name so we don’t mix datasets
    dst = (REPO_RAW / src.name).resolve()

    if str(dst).lower().startswith(str(src).lower()):
        print("\nCache is already inside the repo. Nothing to copy.")
        return

    copy_tree(src, dst)
    print("\nRepo raw dir now contains:")
    for p in REPO_RAW.iterdir():
        print(" -", p)

if __name__ == "__main__":
    main()
