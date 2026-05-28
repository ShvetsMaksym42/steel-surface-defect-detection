import time
import platform
import numpy as np
from pathlib import Path
from ultralytics import YOLO

model = YOLO(Path('weights/yolo/v1/best_first_attempt_85th_epoch.pt'))

TILE_H, TILE_W = 256, 608
NUM_WARMUP     = 10
NUM_RUNS       = 100

dummy_tile = np.random.randint(0, 255, (TILE_H, TILE_W, 3), dtype=np.uint8)

for _ in range(NUM_WARMUP):
    model.predict(dummy_tile, device='cpu', verbose=False)

times = []
for _ in range(NUM_RUNS):
    start = time.perf_counter()
    for _ in range(3):
        model.predict(dummy_tile, device='cpu', verbose=False)
    end = time.perf_counter()
    times.append((end - start) * 1000)  # ms

avg_ms  = sum(times) / len(times)
per_tile = avg_ms / 3

print(f"\n=== Inference Speed ===")
print(f"CPU:            {platform.processor()}")
print(f"Per tile:       {per_tile:.1f} ms")
print(f"Per image (3 tiles): {avg_ms:.1f} ms")
print(f"Runs:           {NUM_RUNS}")

metrics = model.val(
    data    = 'data.yaml',
    device  = 'cpu',
    split   = 'test',
    workers = 4,
)