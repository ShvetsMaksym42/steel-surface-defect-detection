import shutil
import random
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

CSV_PATH       = Path("raw_data/train.csv")
PROCESSED_DIR  = Path("processed_data")
IMAGES_DIR     = PROCESSED_DIR / "images"
LABELS_DIR     = PROCESSED_DIR / "labels"

TRAIN_RATIO = 0.75
VAL_RATIO   = 0.15
TEST_RATIO  = 0.10

SEED = 42

CLASS_PRIORITY = [1, 0, 3, 2]

CLASS_NAMES = ["class1", "class2", "class3", "class4"]

def load_image_classes(csv_path: Path) -> dict[str, set[int]]:
    df = pd.read_csv(csv_path).dropna(subset=["EncodedPixels"])
    image_classes: dict[str, set[int]] = defaultdict(set)

    for _, row in df.iterrows():
        image_classes[row["ImageId"]].add(int(row["ClassId"]) - 1)

    return dict(image_classes)

def assign_stratum(classes: set[int]) -> int:
    for class_id in CLASS_PRIORITY:
        if class_id in classes:
            return class_id
    return -1

def split_images(all_stems: list[str], image_classes: dict[str, set[int]]) -> tuple[list[str], list[str], list[str]]:
    strata: dict[int, list[str]] = defaultdict(list)
    for stem in all_stems:
        img_id  = f"{stem}.jpg"
        classes = image_classes.get(img_id, set())
        strata[assign_stratum(classes)].append(stem)

    train, val, test = [], [], []

    for stratum_id, stems in strata.items():
        random.shuffle(stems)
        n       = len(stems)
        n_val   = max(1, round(n * VAL_RATIO))
        n_test  = max(1, round(n * TEST_RATIO))
        n_train = n - n_val - n_test

        train += stems[:n_train]
        val   += stems[n_train:n_train + n_val]
        test  += stems[n_train + n_val:]

    return train, val, test

def move_tiles(stems: list[str], split: str) -> None:
    out_images = IMAGES_DIR / split
    out_labels = LABELS_DIR / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for stem in tqdm(stems, desc=f"Moving {split}", unit="img"):
        for tile_idx in range(3):
            tile_name = f"{stem}_tile{tile_idx}"

            src_img = IMAGES_DIR / f"{tile_name}.jpg"
            src_lbl = LABELS_DIR / f"{tile_name}.txt"

            if src_img.exists():
                shutil.move(str(src_img), str(out_images / f"{tile_name}.jpg"))
            if src_lbl.exists():
                shutil.move(str(src_lbl), str(out_labels / f"{tile_name}.txt"))

def save_yaml(out_path: Path) -> None:
    config = {
        "path"  : str(PROCESSED_DIR.resolve()),
        "train" : "images/train",
        "val"   : "images/val",
        "test"  : "images/test",
        "nc"    : len(CLASS_NAMES),
        "names" : CLASS_NAMES,
    }
    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def print_summary(train: list, val: list, test: list) -> None:
    total = len(train) + len(val) + len(test)
    print("\n=== Split summary (original images) ===")
    print(f"  Train: {len(train):>5}  ({len(train)/total*100:.1f}%)")
    print(f"  Val:   {len(val):>5}  ({len(val)/total*100:.1f}%)")
    print(f"  Test:  {len(test):>5}  ({len(test)/total*100:.1f}%)")
    print(f"  Total: {total:>5}")
    print(f"\n  Tiles per split (approx): train={len(train)*3}  val={len(val)*3}  test={len(test)*3}")

def main():
    random.seed(SEED)

    print("Loading annotations...")
    image_classes = load_image_classes(CSV_PATH)

    all_stems = sorted({p.stem.rsplit("_tile", 1)[0] for p in IMAGES_DIR.glob("*.jpg")})
    print(f"Found unique images: {len(all_stems)}")

    print("Splitting...")
    train, val, test = split_images(all_stems, image_classes)
    print_summary(train, val, test)

    move_tiles(train, "train")
    move_tiles(val,   "val")
    move_tiles(test,  "test")

    yaml_path = Path("data.yaml")
    save_yaml(yaml_path)
    print(f"\ndata.yaml saved to: {yaml_path.resolve()}")
    print("Done.")

if __name__ == "__main__":
    main()