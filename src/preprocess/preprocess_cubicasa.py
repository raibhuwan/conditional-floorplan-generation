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

# Identify the individual floor groups contained in a CubiCasa5K SVG file.
def list_floors(root):
    floors = []

    # Search all SVG group elements for identifiers such as Floor-1 or Floor-2.
    for g in root.findall(".//svg:g", SVG_NS):
        gid = g.attrib.get("id", "")
        if gid.startswith("Floor-"):
            floors.append(gid)

    # Remove duplicate identifiers and return them in a consistent order.
    return sorted(set(floors))


# ---------------------------------------------------
# 2) Get viewBox scaling info from SVG
#    Human meaning: SVG coords are big; we scale them into 256x256 pixels.
# ---------------------------------------------------

# Read the SVG viewBox values required to scale vector coordinates to raster pixels.
def get_viewbox(root):
    vb = root.attrib.get("viewBox", None)

    # A viewBox is required because its dimensions define the SVG coordinate system.
    if vb is None:
        raise ValueError("SVG has no viewBox attribute")

    minx, miny, w, h = map(float, vb.split())
    return minx, miny, w, h


# ---------------------------------------------------
# 3) Get label from polygon's parent group
#    Human meaning: polygons don't have class name; parent <g> does.
# ---------------------------------------------------

# Traverse parent SVG groups to find the semantic class associated with a polygon.
def get_parent_class(poly, parent_map, max_hops=8):
    cur = poly

    # Move upwards through the XML hierarchy until a class attribute is found.
    for _ in range(max_hops):
        if "class" in cur.attrib and cur.attrib["class"]:
            return cur.attrib["class"]

        cur = parent_map.get(cur)

        # Stop if the root of the available parent hierarchy has been reached.
        if cur is None:
            break

    # Return an empty label when no semantic class can be identified.
    return ""


# ---------------------------------------------------
# 4) Convert SVG polygon points -> list of (x,y) pixel coords
# ---------------------------------------------------

# Convert polygon coordinates from the SVG coordinate system to 256x256 pixels.
def polygon_points_to_pixels(points_str, minx, miny, w, h):
    pts = points_str.replace(",", " ").split()

    # A valid polygon requires at least three coordinate pairs.
    if len(pts) < 6 or len(pts) % 2 != 0:
        return None

    coords = []

    # Scale each SVG coordinate pair relative to the original viewBox.
    for i in range(0, len(pts), 2):
        x = (float(pts[i]) - minx) / w * TARGET_SIZE
        y = (float(pts[i + 1]) - miny) / h * TARGET_SIZE
        coords.append((x, y))

    return coords


# ---------------------------------------------------
# 5) Rasterise ONE floor into a merged semantic mask
#    Human meaning: this outputs a 256x256 image where pixels = room type id.
# ---------------------------------------------------

# Rasterise one selected SVG floor into the merged semantic class representation.
def rasterize_floor(svg_path, floor_id):
    # Parse the source SVG and obtain its root XML element.
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Read the coordinate system used by the original SVG.
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
    # Start with a background-only raster that will receive semantic polygons.
    mask_img = Image.new("L", (TARGET_SIZE, TARGET_SIZE), BG)
    draw = ImageDraw.Draw(mask_img)

    # Keep track of source labels that are not represented in the merge map.
    unknown_tokens = set()

    # draw all polygons within this floor
    for poly in floor_group.findall(".//svg:polygon", SVG_NS):

        # Recover the semantic label from the polygon or one of its parent groups.
        label = get_parent_class(poly, parent_map)
        if not label:
            continue

        # Example label: "Space Kitchen" or "Wall"
        tokens = label.split()

        # For Space labels, use the room-type token; otherwise use
        # the first token directly for structural elements such as Wall.
        if tokens[0].lower() == "space" and len(tokens) > 1:
            token = tokens[1]  # Kitchen, Bedroom, Bath, Storage...
        else:
            token = tokens[0]  # Wall, etc.

        token = token.strip()

        # merge to our 9 classes
        # Ignore unsupported source labels while recording them for inspection.
        if token not in TOKEN_TO_MERGED:
            unknown_tokens.add(token)
            continue

        # Convert the source annotation token to its merged semantic class ID.
        class_id = int(TOKEN_TO_MERGED[token])

        # Convert the polygon geometry from SVG coordinates to raster coordinates.
        pts_str = poly.attrib.get("points", "")
        coords = polygon_points_to_pixels(pts_str, minx, miny, w, h)

        # Skip malformed polygons that could not be converted safely.
        if coords is None:
            continue

        # Fill the polygon using its merged semantic class ID.
        draw.polygon(coords, fill=class_id)

    # Convert the completed raster image into the semantic NumPy mask.
    sem = np.array(mask_img, dtype=np.uint8)

    return sem, unknown_tokens


# ---------------------------------------------------
# 6) Outline mask
#    Human meaning: where the building exists (anything not background).
# ---------------------------------------------------

# Create the filled binary support mask from all non-background semantic pixels.
def make_outline(sem):
    return (sem != BG).astype(np.uint8)


# ---------------------------------------------------
# 7) Room count
#    Human meaning: count separate connected blobs, ignoring walls.
# ---------------------------------------------------

# Count eight-connected non-background and non-wall semantic regions.
# This is an encoded connected-region count rather than a verified room-instance count.
def count_rooms(sem, wall_id=8):
    # Combine all supported room-type classes into one binary foreground mask.
    room_mask = (sem != BG) & (sem != wall_id)
    room_mask = room_mask.astype(np.uint8)

    # Eight-connectivity treats horizontally, vertically and diagonally
    # touching foreground pixels as belonging to the same connected component.
    n, _, _, _ = cv2.connectedComponentsWithStats(room_mask, connectivity=8)

    # OpenCV includes the background as one component, so subtract it.
    return n - 1


# ---------------------------------------------------
# 8) Main preprocessing loop
# ---------------------------------------------------

# Process the selected CubiCasa5K samples and save one NPZ file per floor.
def preprocess(data_root, out_dir, category=None, max_samples=50):
    # Create the destination directory if it does not already exist.
    os.makedirs(out_dir, exist_ok=True)

    # find all model.svg
    # Restrict the search to the selected dataset category when one is supplied.
    if category:
        svg_files = sorted(glob.glob(os.path.join(data_root, category, "*", "model.svg")))
    else:
        svg_files = sorted(glob.glob(os.path.join(data_root, "*", "*", "model.svg")))

    # Optionally limit the number of source buildings processed.
    if max_samples:
        svg_files = svg_files[:max_samples]

    # Accumulate unsupported source tokens across all processed samples.
    all_unknown = set()

    for i, svg_path in enumerate(svg_files):
        # Recover the source category and building identifier from the file path.
        rel = os.path.relpath(svg_path, data_root)  # category/id/model.svg
        cat, sid, _ = rel.split(os.sep)

        # parse once to find floors
        tree = ET.parse(svg_path)
        root = tree.getroot()
        floors = list_floors(root)

        # Use the complete SVG as one sample when explicit floor groups are absent.
        if not floors:
            floors = ["ALL"]

        # Process each floor as a separate floor-level sample.
        for floor_id in floors:
            # Rasterise the semantic target and derive both conditional values.
            sem, unknown = rasterize_floor(svg_path, floor_id)
            outline = make_outline(sem)
            room_count = count_rooms(sem, wall_id=8)

            # Add any unsupported labels from this floor to the global audit set.
            all_unknown |= unknown

            # Construct a traceable output filename using source identifiers.
            out_name = f"{cat}_{sid}_{floor_id}.npz"
            out_path = os.path.join(out_dir, out_name)

            # Save the semantic target, filled support mask, encoded connected-region
            # count and source identifier together as one compressed sample.
            np.savez_compressed(
                out_path,
                sem=sem,
                outline=outline,
                room_count=room_count,
                sample_id=f"{cat}/{sid}/{floor_id}",
            )

            # Display the encoded count and represented semantic classes for inspection.
            uniq = np.unique(sem)
            print(f"[{i}] {cat}/{sid}/{floor_id} rooms={room_count} uniq={uniq}")

    # Report source annotation tokens that were not included in the merge map.
    print("\nUnknown tokens found (add these to merge_map.py if needed):")
    print(sorted(all_unknown))


if __name__ == "__main__":

    # Parse command-line settings for preprocessing the CubiCasa5K source files.
    parser = argparse.ArgumentParser(description="Preprocess CubiCasa5K SVG floor plans into NPZ semantic masks.")
    parser.add_argument("--data_root", type=str, default="data/cubicasa5k")
    parser.add_argument("--out_dir", type=str, default="data/processed_npz")
    parser.add_argument("--category", type=str, default="high_quality_architectural")
    parser.add_argument("--max_samples", type=int, default=2000)

    args = parser.parse_args()

    # Start preprocessing using the selected source directory and settings.
    preprocess(
        data_root=args.data_root,
        out_dir=args.out_dir,
        category=args.category,
        max_samples=args.max_samples,
    )