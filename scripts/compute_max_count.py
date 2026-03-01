import glob
import os
import numpy as np

DATA_DIR = "data/processed_npz_clean"

files = glob.glob(os.path.join(DATA_DIR, "*.npz"))

print("Files found:", len(files))
print(files[:5])

counts = []

for f in files:
    d = np.load(f, allow_pickle=True)
    counts.append(int(d["room_count"]))

if len(counts) == 0:
    print("No samples found — check path.")
else:
    print("Num samples:", len(counts))
    print("Min room_count:", min(counts))
    print("Max room_count:", max(counts))
    print("Mean room_count:", sum(counts) / len(counts))