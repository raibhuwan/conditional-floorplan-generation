import os, glob, random
import numpy as np
import cv2

NPZ_DIR = "data/processed_npz_clean"
OUT_DIR = "outputs/vis"
os.makedirs(OUT_DIR, exist_ok=True)

files = glob.glob(f"{NPZ_DIR}/*.npz")
random.shuffle(files)

def colorize_sem(sem: np.ndarray) -> np.ndarray:
    # turn class ids into colors for viewing
    m = ((sem.astype(np.int32) * 29) % 255).astype(np.uint8)
    return cv2.applyColorMap(m, cv2.COLORMAP_TURBO)

for f in files[:20]:
    d = np.load(f, allow_pickle=True)
    sem = d["sem"]
    outline = d["outline"]
    room_count = int(d["room_count"])

    img = colorize_sem(sem)

    # draw outline edges in white
    edges = cv2.Canny((outline * 255).astype(np.uint8), 50, 150)
    img[edges > 0] = (255, 255, 255)

    base = os.path.splitext(os.path.basename(f))[0]
    cv2.putText(img, f"rooms={room_count}", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    out_path = os.path.join(OUT_DIR, f"{base}.png")
    cv2.imwrite(out_path, img)

print("Saved 20 images to:", OUT_DIR)