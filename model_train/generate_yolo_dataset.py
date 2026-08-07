from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
BOARD_DIR = ROOT / "yolo" / "source" / "boards"
BACKGROUND_DIR = ROOT / "yolo" / "source" / "backgrounds"
OUTPUT_DIR = ROOT / "yolo" / "dataset"
IMAGE_SIZE = (2560, 1440)


def image_files(directory: Path) -> list[Path]:
    return [path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]


def main() -> None:
    boards = image_files(BOARD_DIR)
    backgrounds = image_files(BACKGROUND_DIR)
    if not boards:
        raise FileNotFoundError(f"No board images in {BOARD_DIR}")
    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    for index in range(200):
        board = cv2.imread(str(random.choice(boards)))
        if backgrounds and random.random() > 0.3:
            canvas = cv2.imread(str(random.choice(backgrounds)))
            canvas = cv2.resize(canvas, IMAGE_SIZE)
        else:
            canvas = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), random.randint(20, 230), dtype=np.uint8)
        ratio = random.uniform(0.15, 0.60)
        width = int(IMAGE_SIZE[0] * ratio)
        height = int(board.shape[0] * width / board.shape[1])
        if height >= IMAGE_SIZE[1]:
            continue
        board = cv2.resize(board, (width, height))
        x = random.randint(0, IMAGE_SIZE[0] - width)
        y = random.randint(0, IMAGE_SIZE[1] - height)
        canvas[y:y + height, x:x + width] = board
        split = "val" if random.random() < 0.1 else "train"
        name = f"{index:06d}"
        cv2.imwrite(str(OUTPUT_DIR / "images" / split / f"{name}.jpg"), canvas)
        label = f"0 {(x + width / 2) / IMAGE_SIZE[0]:.6f} {(y + height / 2) / IMAGE_SIZE[1]:.6f} {width / IMAGE_SIZE[0]:.6f} {height / IMAGE_SIZE[1]:.6f}"
        (OUTPUT_DIR / "labels" / split / f"{name}.txt").write_text(label, encoding="ascii")


if __name__ == "__main__":
    main()
