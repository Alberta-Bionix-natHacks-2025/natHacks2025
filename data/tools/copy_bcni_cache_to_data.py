# tools/copy_bnci_cache_into_repo.py
import os, glob, shutil, sys

REPO_RAW = os.path.abspath("data/eeg/raw")
os.makedirs(REPO_RAW, exist_ok=True)

candidates = [
    os.path.expanduser("~/moabb_data"),
    os.path.expanduser("~/.moabb"),
    os.path.expanduser("~/mne_data"),
    os.path.expanduser("~/mne-data"),
    os.path.expanduser("~/mne"),  # just in case
]

patterns = [
    "BNCI2014-001",
    "*BNCI*2014*001*",
    "MNE-BNCI*2014*001*",
]

found = []
for root in candidates:
    if not os.path.isdir(root):
        continue
    for pat in patterns:
        matches = glob.glob(os.path.join(root, "**", pat), recursive=True)
        for m in matches:
            if os.path.isdir(m):
                found.append(os.path.abspath(m))

print("Candidate cache roots:", candidates)
print("Found BNCI dirs:")
for p in found:
    print(" -", p)

if not found:
    print("\nNo BNCI2014-001 found in typical caches. "
          "Try loading the dataset once (it will download), "
          "then re-run this script.")
    sys.exit(1)

src = found[0]  # pick the first hit
dst = os.path.join(REPO_RAW, os.path.basename(src))

if os.path.abspath(src) == os.path.abspath(dst):
    print("\nAlready inside the repo:", dst)
    sys.exit(0)

if os.path.exists(dst):
    print("\nDestination already exists:", dst)
else:
    print("\nCopying\n  from:", src, "\n  to:  ", dst)
    shutil.copytree(src, dst)

print("\nDone. Repo raw dir now contains:")
for p in glob.glob(os.path.join(REPO_RAW, "*")):
    print(" -", p)
