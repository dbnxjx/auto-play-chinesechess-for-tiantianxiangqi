from __future__ import annotations

import numpy as np
from ultralytics import YOLO


class BoardLocator:
    def __init__(self, model_path, confidence: float) -> None:
        self.model = YOLO(str(model_path))
        self.confidence = confidence

    def locate(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        results = self.model(frame, device=0, verbose=False, conf=self.confidence)
        boxes = [box for result in results for box in result.boxes]
        if not boxes:
            return None
        box = max(boxes, key=lambda item: float(item.conf[0]))
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        width, height = x2 - x1, y2 - y1
        x1 += width // 45
        y1 += height // 100
        x2 -= width // 90
        y2 -= height // 80
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)
