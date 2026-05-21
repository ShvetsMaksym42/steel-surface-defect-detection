from ultralytics import YOLO
from pathlib import Path

model = YOLO(Path('weights/yolo/v2/best.pt'))
#'weights/yolo/v1/best_first_attempt_85th_epoch.pt'

metrics = model.val(
    data   = 'data.yaml',
    device = 'cpu',
    split  = 'test',
    workers = 4
)