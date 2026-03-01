import glob
import numpy as np

NPZ_DIR = "data/processed_npz"

files = sorted(glob.glob(f"{NPZ_DIR}/*.npz"))
print("Found", len(files), "npz files")
print("First file:", files[0])

d = np.load(files[0], allow_pickle=True)

print("\nKeys inside npz:", d.files)
print("sample_id:", d["sample_id"])
print("room_count:", int(d["room_count"]))

print("\nShapes / dtypes")
print("sem:", d["sem"].shape, d["sem"].dtype)           # (256,256) class ids
print("outline:", d["outline"].shape, d["outline"].dtype)  # (256,256) 0/1

# Show what label ids exist in this sample
unique = np.unique(d["sem"])
print("\nUnique class ids in sem:", unique)