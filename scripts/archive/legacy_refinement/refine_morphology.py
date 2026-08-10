import os, csv, random
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader

from src.data.dataset import FloorplanNPZDataset
from src.models.unet import UNet

DATA_DIR = "data/processed_npz_clean"
CKPT_PATH = "outputs/checkpoints/unet_base16_best.pt"
MAX_COUNT = 17

NUM_CLASSES = 9
BG = 0
WALL = 8

OUT_CSV = "outputs/metrics_refined_morph.csv"
OUT_DIR = "outputs/refine_morph_samples"
os.makedirs("outputs", exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# candidates
KERNELS = [3, 5]     # try 3x3 and 5x5
OPS = ["close", "open"]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compactness(bin_mask):
    area = float(bin_mask.sum())
    if area <= 0:
        return 0.0
    contours, _ = cv2.findContours(bin_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    perim = sum(cv2.arcLength(c, True) for c in contours)
    if perim <= 1e-6:
        return 0.0
    return float((4.0 * np.pi * area) / (perim * perim))


def mean_iou(a, b, ignore=(BG,)):
    ious = []
    for c in range(NUM_CLASSES):
        if c in ignore:
            continue
        A = (a == c)
        B = (b == c)
        inter = np.logical_and(A, B).sum()
        union = np.logical_or(A, B).sum()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def objective(refined, outline, base_pred):
    # self-consistency (don’t drift)
    base_iou = mean_iou(refined, base_pred, ignore=(BG,))

    # compactness term across classes
    comps = []
    for c in range(1, NUM_CLASSES):
        if c == WALL:
            continue
        bin_c = (refined == c).astype(np.uint8)
        if bin_c.sum() == 0:
            continue
        comps.append(compactness(bin_c))
    comp_mean = float(np.mean(comps)) if comps else 0.0

    # outside outline penalty
    outside_pred = float(((outline == 0) & (refined != BG)).sum())
    outside_tot = float((outline == 0).sum())
    outside_pen = outside_pred / max(1.0, outside_tot)

    # score: prefer compact, consistent, inside outline
    return (1.0 * comp_mean) + (1.0 * base_iou) - (1.0 * outside_pen)


def apply_op(mask, class_id, op, k):
    """
    Apply morphology to a single class binary mask and reinsert into multi-class mask
    without overwriting other classes (only modify pixels currently class_id).
    """
    bin_c = (mask == class_id).astype(np.uint8)
    kernel = np.ones((k, k), np.uint8)

    if op == "close":
        new_bin = cv2.morphologyEx(bin_c, cv2.MORPH_CLOSE, kernel)
    elif op == "open":
        new_bin = cv2.morphologyEx(bin_c, cv2.MORPH_OPEN, kernel)
    else:
        return mask

    out = mask.copy()

    # Only allow changes where pixels are currently class_id OR background
    # (this lets close fill tiny holes, but doesn't overwrite other rooms)
    can_write = (out == class_id) | (out == BG)

    # write class_id where new_bin is 1 and allowed
    out[np.logical_and(new_bin == 1, can_write)] = class_id

    # if opening removed pixels, set removed pixels to BG (only if they were class_id)
    removed = (bin_c == 1) & (new_bin == 0)
    out[removed] = BG

    return out


def refine_mask(pred_mask, outline):
    cur = pred_mask.copy()
    best_score = objective(cur, outline, base_pred=pred_mask)

    # iterate over room classes
    for c in range(1, NUM_CLASSES):
        if c == WALL:
            continue

        # try each candidate op+kernel and accept if better
        for op in OPS:
            for k in KERNELS:
                cand = apply_op(cur, c, op, k)
                cand[outline == 0] = BG  # enforce constraint

                sc = objective(cand, outline, base_pred=pred_mask)
                if sc > best_score:
                    cur = cand
                    best_score = sc

    return cur


def colorize(mask):
    m = ((mask.astype(np.int32) * 29) % 255).astype(np.uint8)
    return cv2.applyColorMap(m, cv2.COLORMAP_TURBO)


def main():
    device = get_device()
    print("Device:", device)

    ds = FloorplanNPZDataset(DATA_DIR, max_count=MAX_COUNT)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = UNet(in_channels=2, out_channels=NUM_CLASSES, base=16).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rows = []
    saved = 0

    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        gt = y[0].numpy().astype(np.uint8)
        outline = (x[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

        with torch.no_grad():
            logits = model(x)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        # before/after metrics
        miou_b = mean_iou(pred, gt, ignore=(BG,))
        miou_base = mean_iou(pred, pred, ignore=(BG,))  # =1 sanity

        refined = refine_mask(pred, outline)

        miou_a = mean_iou(refined, gt, ignore=(BG,))
        base_consistency = mean_iou(refined, pred, ignore=(BG,))

        # compactness (pred vs refined)
        def comp_avg(m):
            vals = []
            for c in range(1, NUM_CLASSES):
                if c == WALL: 
                    continue
                bin_c = (m == c).astype(np.uint8)
                if bin_c.sum() == 0:
                    continue
                vals.append(compactness(bin_c))
            return float(np.mean(vals)) if vals else 0.0

        comp_b = comp_avg(pred)
        comp_a = comp_avg(refined)

        rows.append({
            "idx": idx,
            "miou_before": miou_b,
            "miou_after": miou_a,
            "compact_before": comp_b,
            "compact_after": comp_a,
            "consistency_iou": base_consistency
        })

        # save some examples
        if saved < 12 and idx % 60 == 0:
            o = cv2.cvtColor((outline * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            gt_rgb = colorize(gt)
            pr_rgb = colorize(pred)
            rf_rgb = colorize(refined)

            combo = np.concatenate([
                np.concatenate([o, gt_rgb], axis=1),
                np.concatenate([pr_rgb, rf_rgb], axis=1)
            ], axis=0)

            cv2.imwrite(os.path.join(OUT_DIR, f"morph_{idx:04d}.png"), combo)
            saved += 1

        if (idx + 1) % 100 == 0:
            print(f"[{idx+1}/{len(ds)}] mIoU {miou_b:.3f}->{miou_a:.3f} comp {comp_b:.3f}->{comp_a:.3f} cons {base_consistency:.3f}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def avg(k): return float(np.mean([r[k] for r in rows]))
    print("\nSaved:", OUT_CSV)
    print("Summary over", len(rows), "samples:")
    print(f" mean mIoU:      {avg('miou_before'):.3f} -> {avg('miou_after'):.3f}")
    print(f" mean compact:   {avg('compact_before'):.3f} -> {avg('compact_after'):.3f}")
    print(f" mean consistency IoU (to pred): {avg('consistency_iou'):.3f}")


if __name__ == "__main__":
    main()