import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from decoder import build_mask, tiling, extract_boxes, save_yolo_labels

RAW_IMAGES_DIR = Path("raw_data/train_images")
CSV_PATH = Path("raw_data/train.csv")
OUT_IMAGES_DIR = Path("processed_data/images")
OUT_LABELS_DIR = Path("processed_data/labels")
NUM_CLASSES = 4


def load_annotations(csv_path: Path) -> dict[str, list[str | None]]:
    df = pd.read_csv(csv_path)
    annotations: dict[str, list] = {}

    for _, row in df.iterrows():
        img_id = row["ImageId"]
        class_id = int(row["ClassId"]) - 1
        rle = row["EncodedPixels"]

        if img_id not in annotations:
            annotations[img_id] = [None] * NUM_CLASSES

        annotations[img_id][class_id] = rle if isinstance(rle, str) else None

    return annotations


def process_image(img_path: Path, rle_list: list[str | None], out_images_dir: Path, out_labels_dir: Path) -> dict:
    image = cv2.imread(str(img_path))
    if image is None:
        return {"tiles": 0, "skipped": True}

    mask  = build_mask(rle_list)
    tiles = tiling(image, mask)

    stem  = img_path.stem
    stats = {"tiles": 0, "defect_tiles": 0, "empty_tiles": 0}

    for idx, (img_tile, mask_tile) in enumerate(tiles):
        tile_name = f"{stem}_tile{idx}"

        cv2.imwrite(str(out_images_dir / f"{tile_name}.jpg"), img_tile)

        bboxes = extract_boxes(mask_tile)
        label_path = out_labels_dir / f"{tile_name}.txt"
        save_yolo_labels(bboxes, label_path)

        stats["tiles"] += 1
        if bboxes:
            stats["defect_tiles"] += 1
        else:
            stats["empty_tiles"] += 1

    return stats

def main():
    OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Annotations...")
    annotations = load_annotations(CSV_PATH)

    all_images = sorted(RAW_IMAGES_DIR.glob("*.jpg"))
    print(f"Found images: {len(all_images)}")

    total_tiles = 0
    defect_tiles = 0
    empty_tiles = 0
    skipped = 0

    for img_path in tqdm(all_images, desc="Processing", unit="img"):
        img_id = img_path.name
        rle_list = annotations.get(img_id, [None] * NUM_CLASSES)

        result = process_image(img_path, rle_list, OUT_IMAGES_DIR, OUT_LABELS_DIR)

        if result.get("skipped"):
            skipped += 1
            continue

        total_tiles += result["tiles"]
        defect_tiles += result["defect_tiles"]
        empty_tiles += result["empty_tiles"]

    print("\n--- Completed ---")
    print(f"Images skipped:        {skipped}")
    print(f"Tiles created:         {total_tiles}")
    print(f"  — with defects:      {defect_tiles}")
    print(f"  — defect-free:       {empty_tiles}")

if __name__ == "__main__":
    main()