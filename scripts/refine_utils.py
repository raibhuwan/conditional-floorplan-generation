import numpy as np
import cv2

BG = 0
WALL = 8

def extract_rect_instances(mask, ignore_ids=(BG, WALL), min_area=50):
    """
    Extract connected components per class and represent each instance as a rectangle.
    Returns list of dicts:
      {class_id, x,y,w,h, area}
    """
    rects = []
    for c in range(mask.max() + 1):
        if c in ignore_ids:
            continue
        bin_c = (mask == c).astype(np.uint8)
        if bin_c.sum() < min_area:
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_c, connectivity=8)
        for comp in range(1, n):
            area = int(stats[comp, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[comp, cv2.CC_STAT_LEFT])
            y = int(stats[comp, cv2.CC_STAT_TOP])
            w = int(stats[comp, cv2.CC_STAT_WIDTH])
            h = int(stats[comp, cv2.CC_STAT_HEIGHT])
            rects.append({"class_id": int(c), "x": x, "y": y, "w": w, "h": h, "area": area})
    return rects

def paint_rects_safe(base_mask, rects, outline=None):
    BG = 0
    WALL = 8
    out = base_mask.copy()
    h, w = out.shape

    for r in rects:
        x, y, rw, rh = r["x"], r["y"], r["w"], r["h"]
        c = r["class_id"]
        x2 = min(w, x + rw)
        y2 = min(h, y + rh)
        x1 = max(0, x)
        y1 = max(0, y)
        if x2 <= x1 or y2 <= y1:
            continue

        region = out[y1:y2, x1:x2]

        # don't touch walls
        not_wall = (region != WALL)

        # only fill background pixels
        fill_bg = (region == BG)

        # if outline exists, only fill inside outline
        if outline is not None:
            oreg = outline[y1:y2, x1:x2] > 0
            mask_ok = fill_bg & not_wall & oreg
        else:
            mask_ok = fill_bg & not_wall

        region[mask_ok] = c
        out[y1:y2, x1:x2] = region

    if outline is not None:
        out[outline == 0] = BG
    return out