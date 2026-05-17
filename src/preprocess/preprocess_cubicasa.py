import argparse
import os
import glob
import numpy as np
import cv2
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

from merge_map import TOKEN_TO_MERGED, BG  # from the file you created

TARGET_SIZE = 256
SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


# ---------------------------------------------------
# 1) Find floor groups inside the SVG
#    Human meaning: a house may have Floor-1, Floor-2...
# ---------------------------------------------------
def list_floors(root):
    floors = []
    for g in root.findall(".//svg:g", SVG_NS):
        gid = g.attrib.get("id", "")
        if gid.startswith("Floor-"):
            floors.append(gid)
    return sorted(set(floors))


# ---------------------------------------------------
# 2) Get viewBox scaling info from SVG
#    Human meaning: SVG coords are big; we scale them into 256x256 pixels.
# ---------------------------------------------------
def get_viewbox(root):
    vb = root.attrib.get("viewBox", None)
    if vb is None:
        raise ValueError("SVG has no viewBox attribute")
    minx, miny, w, h = map(float, vb.split())
    return minx, miny, w, h


# ---------------------------------------------------
# 3) Get label from polygon's parent group
#    Human meaning: polygons don't have class name; parent <g> does.
# ---------------------------------------------------
def get_parent_class(poly, parent_map, max_hops=8):
    cur = poly
    for _ in range(max_hops):
        if "class" in cur.attrib and cur.attrib["class"]:
            return cur.attrib["class"]
        cur = parent_map.get(cur)
        if cur is None:
            break
    return ""


# ---------------------------------------------------
# 4) Convert SVG polygon points -> list of (x,y) pixel coords
# ---------------------------------------------------
def polygon_points_to_pixels(points_str, minx, miny, w, h):
    pts = points_str.replace(",", " ").split()
    if len(pts) < 6 or len(pts) % 2 != 0:
        return None

    coords = []
    for i in range(0, len(pts), 2):
        x = (float(pts[i]) - minx) / w * TARGET_SIZE
        y = (float(pts[i + 1]) - miny) / h * TARGET_SIZE
        coords.append((x, y))
    return coords


# ---------------------------------------------------
# 5) Rasterise ONE floor into a merged semantic mask
#    Human meaning: this outputs a 256x256 image where pixels = room type id.
# ---------------------------------------------------
def rasterize_floor(svg_path, floor_id):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    minx, miny, w, h = get_viewbox(root)

    # build child->parent map so we can climb up to <g class="Space ...">
    parent_map = {c: p for p in root.iter() for c in p}

    # find the chosen floor group (e.g., <g id="Floor-1">)
    floor_group = None
    for g in root.findall(".//svg:g", SVG_NS):
        if g.attrib.get("id") == floor_id:
            floor_group = g
            break
    if floor_group is None:
        # fallback: render everything if no floor group
        floor_group = root

    # create blank mask
    mask_img = Image.new("L", (TARGET_SIZE, TARGET_SIZE), BG)
    draw = ImageDraw.Draw(mask_img)

    unknown_tokens = set()

    # draw all polygons within this floor
    for poly in floor_group.findall(".//svg:polygon", SVG_NS):

        label = get_parent_class(poly, parent_map)
        if not label:
            continue

        # Example label: "Space Kitchen" or "Wall"
        tokens = label.split()
        if tokens[0].lower() == "space" and len(tokens) > 1:
            token = tokens[1]  # Kitchen, Bedroom, Bath, Storage...
        else:
            token = tokens[0]  # Wall, etc.

        token = token.strip()

        # merge to our 9 classes
        if token not in TOKEN_TO_MERGED:
            unknown_tokens.add(token)
            continue

        class_id = int(TOKEN_TO_MERGED[token])

        pts_str = poly.attrib.get("points", "")
        coords = polygon_points_to_pixels(pts_str, minx, miny, w, h)
        if coords is None:
            continue

        draw.polygon(coords, fill=class_id)

    sem = np.array(mask_img, dtype=np.uint8)
    return sem, unknown_tokens


# ---------------------------------------------------
# 6) Outline mask
#    Human meaning: where the building exists (anything not background).
# ---------------------------------------------------
def make_outline(sem):
    return (sem != BG).astype(np.uint8)


# ---------------------------------------------------
# 7) Room count
#    Human meaning: count separate connected blobs, ignoring walls.
# ---------------------------------------------------
def count_rooms(sem, wall_id=8):
    room_mask = (sem != BG) & (sem != wall_id)
    room_mask = room_mask.astype(np.uint8)
    n, _, _, _ = cv2.connectedComponentsWithStats(room_mask, connectivity=8)
    return n - 1


# ---------------------------------------------------
# 8) Main preprocessing loop
# ---------------------------------------------------
def preprocess(data_root, out_dir, category=None, max_samples=50):
    os.makedirs(out_dir, exist_ok=True)

    # find all model.svg
    if category:
        svg_files = sorted(glob.glob(os.path.join(data_root, category, "*", "model.svg")))
    else:
        svg_files = sorted(glob.glob(os.path.join(data_root, "*", "*", "model.svg")))

    if max_samples:
        svg_files = svg_files[:max_samples]

    all_unknown = set()

    for i, svg_path in enumerate(svg_files):
        rel = os.path.relpath(svg_path, data_root)  # category/id/model.svg
        cat, sid, _ = rel.split(os.sep)

        # parse once to find floors
        tree = ET.parse(svg_path)
        root = tree.getroot()
        floors = list_floors(root)
        if not floors:
            floors = ["ALL"]

        for floor_id in floors:
            sem, unknown = rasterize_floor(svg_path, floor_id)
            outline = make_outline(sem)
            room_count = count_rooms(sem, wall_id=8)

            all_unknown |= unknown

            out_name = f"{cat}_{sid}_{floor_id}.npz"
            out_path = os.path.join(out_dir, out_name)

            np.savez_compressed(
                out_path,
                sem=sem,
                outline=outline,
                room_count=room_count,
                sample_id=f"{cat}/{sid}/{floor_id}",
            )

            uniq = np.unique(sem)
            print(f"[{i}] {cat}/{sid}/{floor_id} rooms={room_count} uniq={uniq}")

    print("\nUnknown tokens found (add these to merge_map.py if needed):")
    print(sorted(all_unknown))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess CubiCasa5K SVG floor plans into NPZ semantic masks.")
    parser.add_argument("--data_root", type=str, default="data/cubicasa5k")
    parser.add_argument("--out_dir", type=str, default="data/processed_npz")
    parser.add_argument("--category", type=str, default="high_quality_architectural")
    parser.add_argument("--max_samples", type=int, default=2000)

    args = parser.parse_args()

    preprocess(
        data_root=args.data_root,
        out_dir=args.out_dir,
        category=args.category,
        max_samples=args.max_samples,
    )