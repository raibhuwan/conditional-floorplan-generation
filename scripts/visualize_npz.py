import os, glob, random
import numpy as np
import cv2

NPZ_DIR = "data/processed_npz_clean"
OUT_DIR = "outputs/vis"

# Create the output directory used for sample visualisations.
os.makedirs(OUT_DIR, exist_ok=True)

# Locate all processed NPZ samples and randomise their order before
# selecting a small subset for visual inspection.
files = glob.glob(f"{NPZ_DIR}/*.npz")
random.shuffle(files)


def colorize_sem(sem: np.ndarray) -> np.ndarray:
    # Convert semantic class IDs into display values and apply
    # an OpenCV colour map for quick qualitative inspection.
    m = ((sem.astype(np.int32) * 29) % 255).astype(np.uint8)
    return cv2.applyColorMap(m, cv2.COLORMAP_TURBO)


# Visualise up to 20 randomly selected processed floor samples.
for f in files[:20]:
    d = np.load(f, allow_pickle=True)

    # Load the semantic target, filled binary support mask and historical
    # room_count field containing the encoded connected-region count.
    sem = d["sem"]
    outline = d["outline"]
    room_count = int(d["room_count"])

    # Convert the semantic class-ID mask into a colour image.
    img = colorize_sem(sem)

    # draw outline edges in white
    # Extract the support-mask boundary and overlay it on the semantic image.
    edges = cv2.Canny(
        (outline * 255).astype(np.uint8),
        50,
        150,
    )
    img[edges > 0] = (255, 255, 255)

    # Use the source NPZ filename as the output image identifier.
    base = os.path.splitext(
        os.path.basename(f)
    )[0]

    # Display the stored encoded connected-region count on the preview.
    # The existing "rooms=" label is retained for compatibility.
    cv2.putText(
        img,
        f"rooms={room_count}",
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    # Save the completed semantic/support preview as a PNG image.
    out_path = os.path.join(
        OUT_DIR,
        f"{base}.png",
    )
    cv2.imwrite(out_path, img)

print("Saved 20 images to:", OUT_DIR)