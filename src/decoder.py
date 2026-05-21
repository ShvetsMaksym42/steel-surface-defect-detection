import numpy as np
import cv2
from pathlib import Path

TILE_WIDTH = 608
TILE_POSITIONS = [0, 495, 991]
BBOX_PAD = 3

# Values derived from median blob area analysis in eda/analyze_blobs.py
MIN_AREA = [175, 340, 1250, 1900]

def rle_decode(rle_string: str | None, shape: tuple[int, int] =(256, 1600)) -> np.ndarray:
    mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    if not rle_string:
        return mask.reshape(shape)
    
    s = list(map(int, rle_string.split()))
    starts, lengths = s[0::2], s[1::2]

    for start, length in zip(starts, lengths):
        start -= 1
        mask[start:start+length] = 1

    return mask.reshape(shape, order='F')

def build_mask(rle_list: list[str | None], shape: tuple[int, int] = (256, 1600)) -> np.ndarray:
    if not rle_list:
        return np.zeros((*shape, 0), dtype=np.uint8)
 
    return np.stack([rle_decode(rle) for rle in rle_list], axis=-1)

def tiling(image: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    tiles = []

    for x_start in TILE_POSITIONS:
        x_end = x_start + TILE_WIDTH
        tiles.append((
            image[:, x_start:x_end],
            mask[:, x_start:x_end]
        ))
    
    return tiles

def extract_boxes(mask_tile: np.ndarray) -> list[tuple[int, float, float, float, float]]:
    H, W, C = mask_tile.shape
    bboxes = []

    for class_id in range(C):
        binary = mask_tile[:, :, class_id].astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < MIN_AREA[class_id]:
                continue
            x1 = max(0, x - BBOX_PAD)
            y1 = max(0, y - BBOX_PAD)
            x2 = min(W, x + w + BBOX_PAD)
            y2 = min(H, y + h + BBOX_PAD)
            w_pad = x2 - x1
            h_pad = y2 - y1
            if w_pad<=0 or h_pad<=0:
                continue

            bboxes.append((
                class_id,
                (x1 + x2) / 2 / W,
                (y1 + y2) / 2 / H,
                w_pad / W,
                h_pad / H,
            ))

    return bboxes

def save_yolo_labels(bboxes: list[tuple[int, float, float, float, float]], file_path: str | Path) -> None:
    with open(file_path, 'w') as f:
        for cls, x, y, w, h in bboxes:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")