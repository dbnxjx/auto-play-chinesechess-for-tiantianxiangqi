from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from shared.cnn import DeepChessCNN


PIECE_CLASS_TO_VALUE = {
    0: -1, 1: -2, 2: -3, 3: -4, 4: -5, 5: -6, 6: -7, 7: 0,
    8: 1, 9: 2, 10: 3, 11: 4, 12: 5, 13: 6, 14: 7,
}


class CellClassifier:
    def __init__(self, model_path, classes: int) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DeepChessCNN(classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def predict(self, cell: np.ndarray) -> int:
        image = Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            return int(self.model(tensor).argmax(1).item())


class ChessRecognizer:
    def __init__(self, cell_model_path, piece_model_path, marker_model_path) -> None:
        self.cell = CellClassifier(cell_model_path, 3)
        self.piece = CellClassifier(piece_model_path, 15)
        self.marker = CellClassifier(marker_model_path, 2)

    def recognize_board(self, board_image: np.ndarray) -> np.ndarray:
        result = np.zeros((10, 9), dtype=int)
        for row, column, cell in self._cells(board_image):
            result[row, column] = PIECE_CLASS_TO_VALUE[self.piece.predict(cell)]
        return result

    def recognize_markers(self, board_image: np.ndarray) -> list[tuple[int, int]]:
        # Existing marker data uses class 0 for a highlighted cell.
        return [(row, column) for row, column, cell in self._cells(board_image) if self.marker.predict(cell) == 0]

    def _cells(self, board_image: np.ndarray):
        height, width = board_image.shape[:2]
        cell_height, cell_width = height // 10, width // 9
        for row in range(10):
            for column in range(9):
                yield row, column, board_image[row * cell_height:(row + 1) * cell_height, column * cell_width:(column + 1) * cell_width]
