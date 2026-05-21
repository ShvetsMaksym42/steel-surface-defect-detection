import numpy as np
import pandas as pd
import cv2
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from decoder import rle_decode

CSV_PATH = Path("raw_data/train.csv")
NUM_CLASSES = 4
THRESHOLD_SETTINGS = [0.20, 0.15, 0.30, 0.25]

def get_blob_areas(rle: str, shape: tuple = (256, 1600)) -> list[int]:
    mask = rle_decode(rle, shape).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]

def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["EncodedPixels"])

    areas_per_class: dict[int, list[int]] = {i: [] for i in range(NUM_CLASSES)}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing masks", unit="mask"):
        class_id = int(row["ClassId"]) - 1
        rle = row["EncodedPixels"]
        areas = get_blob_areas(rle)
        areas_per_class[class_id].extend(areas)

    header = f"{'Class':<7} {'Blobs':<8} {'Min':<8} {'Median':<10} {'Mean':<10} {'Max':<10} {'Suggested threshold'}"
    separator = "-" * len(header)

    print(f"\n{separator}")
    print(header)
    print(separator)

    for class_id, areas in areas_per_class.items():
        if not areas:
            print(f"{class_id + 1:<7} {'—':<8} {'—':<8} {'—':<10} {'—':<10} {'—':<10} —")
            continue

        arr = np.array(areas)
        median = np.median(arr)
        mean = arr.mean()
        mn = arr.min()
        mx = arr.max()

        pct = THRESHOLD_SETTINGS[class_id]
        threshold = int(median * pct)

        print(
            f"{class_id + 1:<7} {len(arr):<8} {int(mn):<8} {int(median):<10} "
            f"{int(mean):<10} {int(mx):<10} {threshold}  ({int(pct*100)}% of median)"
        )

    print(separator)

if __name__ == "__main__":
    main()