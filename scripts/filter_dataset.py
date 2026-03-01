import os, glob, shutil
import numpy as np

IN_DIR = "data/processed_npz"
OUT_DIR = "data/processed_npz_clean"
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(f"{IN_DIR}/*.npz"))

kept = 0
dropped = 0

for f in files:
    d = np.load(f, allow_pickle=True)
    room_count = int(d["room_count"])
    uniq = np.unique(d["sem"])
    uniq_count = len(uniq)

    # ✅ cleaning rules (adjust if you want)
    if room_count >= 3 and uniq_count >= 3:
        shutil.copy2(f, os.path.join(OUT_DIR, os.path.basename(f)))
        kept += 1
    else:
        dropped += 1

print("Total:", len(files))
print("Kept:", kept)
print("Dropped:", dropped)
print("Clean dataset folder:", OUT_DIR)