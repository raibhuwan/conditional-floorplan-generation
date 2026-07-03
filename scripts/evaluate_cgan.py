import os, csv, argparse
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader, Subset

from src.data.dataset import FloorplanNPZDataset
from src.data.splits import load_split
from src.models.unet import UNet

DATA_DIR = "data/processed_npz_clean_full"
SPLIT_PATH = "outputs/splits/split_seed42_full.json"
CKPT_PATH = "outputs/checkpoints/cgan_unet_patchgan_best.pt"

MAX_COUNT = 32
NUM_CLASSES = 9

# merged ids (from your merge_map)
BG = 0
WALL = 8


OUT_CSV = "outputs/metrics_cgan.csv"
os.makedirs("outputs", exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate cGAN generator on the held-out test set.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--split_path", type=str, default=SPLIT_PATH)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--max_count", type=int, default=MAX_COUNT)
    parser.add_argument("--out_csv", type=str, default=OUT_CSV)
    return parser.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def extract_instances(mask, ignore_ids=(BG, WALL), min_area=30):
    """
    Turn a semantic mask into room instances (connected components).
    Returns:
      instances: list of dict {class_id, comp_id, area, bbox, binary_mask}
    """
    instances = []
    h, w = mask.shape

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
            x = int(stats[comp_id, cv2.CC_STAT_LEFT])
            y = int(stats[comp_id, cv2.CC_STAT_TOP])
            bw = int(stats[comp_id, cv2.CC_STAT_WIDTH])
            bh = int(stats[comp_id, cv2.CC_STAT_HEIGHT])

            inst_mask = (labels == comp_id).astype(np.uint8)

            instances.append({
                "class_id": c,
                "area": area,
                "bbox": (x, y, bw, bh),
                "mask": inst_mask
            })
    return instances


def compactness_of_instance(inst_mask):
    """
    Compute 4*pi*A / P^2 for a binary instance mask.
    """
    area = float(inst_mask.sum())
    if area <= 0:
        return 0.0

    # perimeter from contours
    contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    perim = 0.0
    for cnt in contours:
        perim += cv2.arcLength(cnt, True)

    if perim <= 1e-6:
        return 0.0

    return float((4.0 * np.pi * area) / (perim * perim))


def adjacency_edges(instances):
    """
    Build adjacency edges between room instances based on boundary touch.
    Output edges are between class IDs (not instance IDs) to keep it simple.
    Returns set of tuples like (min_class, max_class).
    """
    edges = set()
    # Precompute dilated masks for touch detection
    kernel = np.ones((3, 3), np.uint8)

    dilated = []
    for inst in instances:
        m = inst["mask"].astype(np.uint8)
        d = cv2.dilate(m, kernel, iterations=1)
        dilated.append(d)

    for i in range(len(instances)):
        ci = instances[i]["class_id"]
        for j in range(i + 1, len(instances)):
            cj = instances[j]["class_id"]
            if ci == cj:
                continue  # adjacency between same class not useful here

            # If dilated boundaries overlap, consider adjacent
            touch = np.logical_and(dilated[i] > 0, dilated[j] > 0).any()
            if touch:
                a, b = (ci, cj) if ci < cj else (cj, ci)
                edges.add((a, b))
    return edges


def f1_edges(pred_edges, gt_edges):
    """
    F1 score on sets of edges.
    """
    if len(pred_edges) == 0 and len(gt_edges) == 0:
        return 1.0
    if len(pred_edges) == 0 or len(gt_edges) == 0:
        return 0.0

    tp = len(pred_edges.intersection(gt_edges))
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if (prec + rec) == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


# --- Additional metrics ---
def boundary_violation_rate(pred_mask, outline_mask):
    """
    Measure the proportion of predicted non-background pixels that fall outside
    the building outline. Lower values indicate better boundary consistency.
    """
    pred_non_bg = pred_mask != BG
    total_pred = int(pred_non_bg.sum())

    if total_pred == 0:
        return 0.0

    outside = np.logical_and(pred_non_bg, outline_mask == 0).sum()
    return float(outside / total_pred)


def get_expected_room_count_from_input(x, max_count):
    """
    Recover the expected room count from the second input channel.
    Channel 0 contains the outline and channel 1 contains the normalised room count.
    """
    count_channel = x[0, 1].detach().cpu().numpy()
    normalised_count = float(count_channel.max())
    return int(round(normalised_count * max_count))


def room_count_error(expected_count, predicted_instances):
    """
    Calculate absolute room-count error for one prediction. Lower values are better.
    """
    predicted_count = len(predicted_instances)
    return abs(expected_count - predicted_count), predicted_count


def load_model(device, ckpt_path):
    model = UNet(in_channels=2, out_channels=NUM_CLASSES, base=16).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    if "generator_state" in ckpt:
        model.load_state_dict(ckpt["generator_state"])
    else:
        model.load_state_dict(ckpt["model_state"])

    model.eval()
    print("Loaded checkpoint epoch:", ckpt.get("epoch", "unknown"))
    print("Checkpoint val IoU:", ckpt.get("val_iou", "unknown"))
    return model


def main():
    args = parse_args()

    device = get_device()
    print("Device:", device)

    ds = FloorplanNPZDataset(args.data_dir, max_count=args.max_count)
    split = load_split(args.split_path)
    test_ds = Subset(ds, split["test"])
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    print(f"Data folder: {args.data_dir}")
    print(f"Split file: {args.split_path}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"MAX_COUNT: {args.max_count}")
    print(f"Output CSV: {args.out_csv}")
    print(f"Evaluating cGAN on held-out test samples: {len(test_ds)}")

    model = load_model(device, args.ckpt_path)

    rows = []
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            logits = model(x)
            pred = torch.argmax(logits, dim=1)

        pred_np = pred[0].cpu().numpy().astype(np.uint8)
        gt_np = y[0].cpu().numpy().astype(np.uint8)
        outline_np = (x[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        expected_room_count = get_expected_room_count_from_input(x, max_count=args.max_count)

        # Metrics
        miou = mean_iou(pred_np, gt_np, ignore=(BG,))  # ignore background by default

        gt_inst = extract_instances(gt_np)
        pr_inst = extract_instances(pred_np)
        bvr = boundary_violation_rate(pred_np, outline_np)
        rc_error, predicted_room_count = room_count_error(expected_room_count, pr_inst)

        # adjacency similarity between class-level edges
        gt_edges = adjacency_edges(gt_inst)
        pr_edges = adjacency_edges(pr_inst)
        adj_f1 = f1_edges(pr_edges, gt_edges)

        # compactness: average over instances
        gt_comp = [compactness_of_instance(i["mask"]) for i in gt_inst]
        pr_comp = [compactness_of_instance(i["mask"]) for i in pr_inst]
        gt_comp_mean = float(np.mean(gt_comp)) if gt_comp else 0.0
        pr_comp_mean = float(np.mean(pr_comp)) if pr_comp else 0.0

        rows.append({
            "idx": idx,
            "miou_no_bg": miou,
            "adj_f1": adj_f1,
            "compact_gt": gt_comp_mean,
            "compact_pred": pr_comp_mean,
            "num_inst_gt": len(gt_inst),
            "num_inst_pred": len(pr_inst),
            "boundary_violation_rate": bvr,
            "expected_room_count": expected_room_count,
            "predicted_room_count": predicted_room_count,
            "room_count_error": rc_error,
        })

        if (idx + 1) % 50 == 0:
            print(f"[{idx+1}/{len(test_ds)}] mIoU={miou:.3f} adjF1={adj_f1:.3f} comp_pred={pr_comp_mean:.3f} BVR={bvr:.3f} RCerr={rc_error}")

    # Save CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    miou_mean = float(np.mean([r["miou_no_bg"] for r in rows]))
    adj_mean = float(np.mean([r["adj_f1"] for r in rows]))
    comp_mean = float(np.mean([r["compact_pred"] for r in rows]))
    bvr_mean = float(np.mean([r["boundary_violation_rate"] for r in rows]))
    room_count_mae = float(np.mean([r["room_count_error"] for r in rows]))

    print("\nSaved:", args.out_csv)
    print(f"Summary over {len(rows)} samples:")
    print(f" mean mIoU (no bg): {miou_mean:.3f}")
    print(f" mean adjacency F1: {adj_mean:.3f}")
    print(f" mean compactness (pred): {comp_mean:.3f}")
    print(f" mean boundary violation rate: {bvr_mean:.3f}")
    print(f" mean room-count MAE: {room_count_mae:.3f}")


if __name__ == "__main__":
    main()