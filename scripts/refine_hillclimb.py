import os, csv, random
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader

from src.data.dataset import FloorplanNPZDataset
from src.models.unet import UNet

from scripts.refine_utils import extract_rect_instances, paint_rects_safe

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data/processed_npz_clean"
CKPT_PATH = "outputs/checkpoints/unet_base16_best.pt"  # adjust if needed
MAX_COUNT = 17

NUM_CLASSES = 9
BG = 0
WALL = 8
HALLWAY = 5  # option 1: hallway is circulation anchor

OUT_CSV = "outputs/metrics_refined.csv"
OUT_DIR = "outputs/refine_samples"
os.makedirs("outputs", exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# hill-climb parameters (lightweight)
ITERS = 80               # per sample
STEP_CHOICES = [-2, -1, 1, 2]   # pixel nudges
MIN_W, MIN_H = 6, 6       # keep rectangles from collapsing
MAX_SHIFT = 6             # clamp x/y within image after moves
MIN_AREA = 50             # ignore tiny components


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------
# Metrics (same spirit as earlier)
# -----------------------------
def mean_iou(pred, gt, num_classes=NUM_CLASSES, ignore=(BG,)):
    ious = []
    for c in range(num_classes):
        if c in ignore:
            continue
        p = (pred == c)
        g = (gt == c)
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if len(ious) else 0.0


def compactness_of_instance(inst_mask):
    area = float(inst_mask.sum())
    if area <= 0:
        return 0.0
    contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    perim = 0.0
    for cnt in contours:
        perim += cv2.arcLength(cnt, True)
    if perim <= 1e-6:
        return 0.0
    return float((4.0 * np.pi * area) / (perim * perim))


def extract_instances(mask, ignore_ids=(BG, WALL), min_area=30):
    instances = []
    for c in range(NUM_CLASSES):
        if c in ignore_ids:
            continue
        bin_c = (mask == c).astype(np.uint8)
        if bin_c.sum() < min_area:
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_c, connectivity=8)
        for comp_id in range(1, n):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            inst_mask = (labels == comp_id).astype(np.uint8)
            instances.append({"class_id": c, "mask": inst_mask})
    return instances


def adjacency_edges(instances):
    edges = set()
    kernel = np.ones((3, 3), np.uint8)

    dilated = []
    for inst in instances:
        d = cv2.dilate(inst["mask"].astype(np.uint8), kernel, iterations=1)
        dilated.append(d)

    for i in range(len(instances)):
        ci = instances[i]["class_id"]
        for j in range(i + 1, len(instances)):
            cj = instances[j]["class_id"]
            if ci == cj:
                continue
            touch = np.logical_and(dilated[i] > 0, dilated[j] > 0).any()
            if touch:
                a, b = (ci, cj) if ci < cj else (cj, ci)
                edges.add((a, b))
    return edges


def f1_edges(pred_edges, gt_edges):
    if len(pred_edges) == 0 and len(gt_edges) == 0:
        return 1.0
    if len(pred_edges) == 0 or len(gt_edges) == 0:
        return 0.0
    tp = len(pred_edges & gt_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if (prec + rec) == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def layout_metrics(pred_mask, gt_mask):
    miou = mean_iou(pred_mask, gt_mask, ignore=(BG,))
    pred_inst = extract_instances(pred_mask)
    gt_inst = extract_instances(gt_mask)

    pred_edges = adjacency_edges(pred_inst)
    gt_edges = adjacency_edges(gt_inst)
    adjf1 = f1_edges(pred_edges, gt_edges)

    pred_comp = [compactness_of_instance(i["mask"]) for i in pred_inst]
    comp_mean = float(np.mean(pred_comp)) if pred_comp else 0.0

    return miou, adjf1, comp_mean, len(pred_inst)

def iou_to_base(mask, base_pred, ignore=(BG,)):
    # mean IoU between refined mask and original prediction
    ious = []
    for c in range(NUM_CLASSES):
        if c in ignore:
            continue
        p = (mask == c)
        b = (base_pred == c)
        inter = np.logical_and(p, b).sum()
        union = np.logical_or(p, b).sum()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if len(ious) else 0.0

# -----------------------------
# Hill-climb objective (NO GT)
# -----------------------------
def objective(mask, outline, base_pred):
    """
    Optimise only on predicted layout quality + constraints (no ground truth).
    Score higher is better.
    """
    inst = extract_instances(mask, ignore_ids=(BG, WALL), min_area=MIN_AREA)

    # compactness
    comps = [compactness_of_instance(i["mask"]) for i in inst]
    comp_mean = float(np.mean(comps)) if comps else 0.0

    # adjacency richness (proxy): more reasonable adjacencies tends to help connectivity
    edges = adjacency_edges(inst)
    adj_term = min(len(edges) / 20.0, 1.0)  # cap

    # penalties
    outside = float((outline == 0).sum())
    outside_pred = float(((outline == 0) & (mask != BG)).sum())
    outside_pen = outside_pred / max(1.0, outside)

    # overlap penalty approximation:
    # rectangles painting already overwrites; we approximate “messiness” by number of instances exploding
    inst_count = len(inst)
    inst_pen = max(0.0, (inst_count - 20) / 20.0)  # penalize too many fragments

    base_iou = iou_to_base(mask, base_pred, ignore=(BG,))
    chg = change_ratio(mask, base_pred)

    score = (1.0 * comp_mean) + (0.2 * base_iou) - (0.2 * outside_pen) - (0.2 * chg)
    return float(score)


def propose_move(rect, H, W):
    r = rect.copy()
    # random choice: move or resize a bit
    if random.random() < 0.7:
        dx = random.choice(STEP_CHOICES)
        dy = random.choice(STEP_CHOICES)
        r["x"] += dx
        r["y"] += dy
    else:
        dw = random.choice(STEP_CHOICES)
        dh = random.choice(STEP_CHOICES)
        r["w"] = max(MIN_W, r["w"] + dw)
        r["h"] = max(MIN_H, r["h"] + dh)

    # clamp to image bounds
    r["x"] = int(np.clip(r["x"], 0, W - 1))
    r["y"] = int(np.clip(r["y"], 0, H - 1))
    r["w"] = int(np.clip(r["w"], MIN_W, W - r["x"]))
    r["h"] = int(np.clip(r["h"], MIN_H, H - r["y"]))
    return r

def change_ratio(mask, base_pred):
    changed = (mask != base_pred)
    return float(changed.sum()) / float(mask.size)

def refine_by_hillclimb(pred_mask, outline):
    H, W = pred_mask.shape
    rects = extract_rect_instances(pred_mask, ignore_ids=(BG, WALL), min_area=MIN_AREA)

    # Only refine a subset to avoid destroying IoU:
    # refine small or messy rectangles only
    filtered = []
    for r in rects:
        # build a mask for this rectangle region within pred
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        crop = pred_mask[y:y+h, x:x+w]
        # measure how "pure" it is (how much of crop belongs to this class)
        purity = (crop == r["class_id"]).mean() if crop.size else 0.0

        # Heuristic: if purity is high, the shape is already close to rectangular → skip
        # If purity is low, it's fragmented/jagged → refine
        if purity < 0.75 or r["area"] < 300:
            filtered.append(r)

    rects = filtered

    # If nothing to refine, return as-is
    if len(rects) == 0:
        return pred_mask

    cur_mask = paint_rects_safe(pred_mask, rects, outline=outline)

    chg0 = ((cur_mask != pred_mask).sum() / cur_mask.size)
    print("initial change ratio:", chg0)

    cur_score = objective(cur_mask, outline, base_pred=pred_mask)

    for _ in range(ITERS):
        k = random.randrange(len(rects))
        old = rects[k]
        new = propose_move(old, H, W)

        rects[k] = new
        cand_mask = paint_rects_safe(pred_mask, rects, outline=outline)
        cand_score = objective(cand_mask, outline, base_pred=pred_mask)

        if cand_score >= cur_score:
            cur_score = cand_score
            cur_mask = cand_mask
        else:
            rects[k] = old  # revert

    chg1 = ((cur_mask != pred_mask).sum() / cur_mask.size)
    print("final change ratio:", chg1, "rects:", len(rects), "score:", cur_score)

    return cur_mask


def colorize(mask):
    m = ((mask.astype(np.int32) * 29) % 255).astype(np.uint8)
    return cv2.applyColorMap(m, cv2.COLORMAP_TURBO)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = get_device()
    print("Device:", device)

    # dataset
    ds = FloorplanNPZDataset(DATA_DIR, max_count=MAX_COUNT)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # model
    model = UNet(in_channels=2, out_channels=NUM_CLASSES, base=16).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rows = []
    saved = 0

    for idx, (x, y) in enumerate(loader):
        if idx == 1:
            break

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            logits = model(x)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        gt = y[0].cpu().numpy().astype(np.uint8)
        outline = (x[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

        # metrics before
        miou_b, adj_b, comp_b, inst_b = layout_metrics(pred, gt)

        # refine (no GT used inside)
        refined = refine_by_hillclimb(pred, outline)

        # metrics after
        miou_a, adj_a, comp_a, inst_a = layout_metrics(refined, gt)

        rows.append({
            "idx": idx,
            "miou_before": miou_b,
            "miou_after": miou_a,
            "adjF1_before": adj_b,
            "adjF1_after": adj_a,
            "compact_before": comp_b,
            "compact_after": comp_a,
            "num_inst_before": inst_b,
            "num_inst_after": inst_a,
        })

        # Save a few qualitative examples
        if saved < 12 and (idx % 40 == 0):
            o = (outline * 255).astype(np.uint8)
            o_rgb = cv2.cvtColor(o, cv2.COLOR_GRAY2BGR)

            gt_rgb = colorize(gt)
            pr_rgb = colorize(pred)
            rf_rgb = colorize(refined)

            top = np.concatenate([o_rgb, gt_rgb], axis=1)
            bottom = np.concatenate([pr_rgb, rf_rgb], axis=1)
            combo = np.concatenate([top, bottom], axis=0)

            cv2.putText(combo, f"idx={idx}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(combo, f"mIoU {miou_b:.3f}->{miou_a:.3f}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(combo, f"adjF1 {adj_b:.3f}->{adj_a:.3f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(combo, f"comp {comp_b:.3f}->{comp_a:.3f}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            out_path = os.path.join(OUT_DIR, f"refine_{idx:04d}.png")
            cv2.imwrite(out_path, combo)
            saved += 1

        if (idx + 1) % 50 == 0:
            print(f"[{idx+1}/{len(ds)}] adj {adj_b:.3f}->{adj_a:.3f} comp {comp_b:.3f}->{comp_a:.3f}")

    # save csv
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # print summary
    def avg(key): return float(np.mean([r[key] for r in rows]))
    print("\nSaved:", OUT_CSV)
    print("Summary over", len(rows), "samples:")
    print(f" mean mIoU:      {avg('miou_before'):.3f} -> {avg('miou_after'):.3f}")
    print(f" mean adjF1:     {avg('adjF1_before'):.3f} -> {avg('adjF1_after'):.3f}")
    print(f" mean compact:   {avg('compact_before'):.3f} -> {avg('compact_after'):.3f}")
    print(f" mean #inst:     {avg('num_inst_before'):.2f} -> {avg('num_inst_after'):.2f}")


if __name__ == "__main__":
    main()