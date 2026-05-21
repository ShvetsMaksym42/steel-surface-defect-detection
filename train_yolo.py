import torch
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path("data.yaml")
MODEL     = "yolov8m.pt"
EPOCHS    = 100
BATCH     = 16
IMGSZ     = 608
WORKERS   = 4
PROJECT   = "runs"
RUN_NAME  = "severstal_yolov8m"


def get_device() -> str:
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        return "cuda:0"
    print("Device: CPU")
    return "cpu"


def main():
    device = get_device()
    model  = YOLO(MODEL)

    model.train(
        data     = str(DATA_YAML),
        epochs   = EPOCHS,
        batch    = BATCH,
        imgsz    = IMGSZ,
        device   = device,
        workers  = WORKERS,
        project  = PROJECT,
        name     = RUN_NAME,
        patience = 20,
        hsv_h    = 0.01,
        hsv_s    = 0.3,
        hsv_v    = 0.2,
        fliplr   = 0.5,
        flipud   = 0.0,
        mosaic   = 0.5,
        mixup    = 0.0,
    )
    print("Done.")


if __name__ == "__main__":
    main()