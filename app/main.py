from __future__ import annotations

import time

import cv2
import numpy as np

from app.board import MiniXiangqiBoard
from app.device import AdbController
from app.engine import FairyStockfishEngine
from app.settings import load_settings
from app.vision.board_locator import BoardLocator
from app.vision.recognizer import ChessRecognizer


class ChessRobot:
    def __init__(self) -> None:
        self.settings = load_settings()
        if not self.settings.use_adb:
            raise RuntimeError("Desktop capture mode was removed; configure ADB in config.ini")
        self.device = AdbController(self.settings.adb_path)
        self.locator = BoardLocator(self.settings.yolo_model_path, self.settings.yolo_confidence)
        self.recognizer = ChessRecognizer(
            self.settings.cell_model_path,
            self.settings.piece_model_path,
            self.settings.marker_model_path,
            self.settings.board_rows,
            self.settings.board_columns,
        )
        self.engine = FairyStockfishEngine(self.settings.engine_path, self.settings.nnue_path, self.settings.variant)
        self.board = MiniXiangqiBoard()

    def run(self) -> None:
        self.engine.start()
        try:
            while True:
                time.sleep(self.settings.interval)
                board_image, board_rect = self._stable_board()
                if board_image is not None:
                    state = self.recognizer.recognize_board(board_image)
                    self.board.detect_orientation_and_set_state(state)
                    our_side = "red" if not self.board.flip else "black"
                    markers = self.recognizer.recognize_markers(board_image)
                    if len(markers) > 1:
                        print("Multiple highlighted cells detected; waiting for a stable board")
                        continue
                    if markers:
                        row, column = markers[0]
                        # A highlighted piece belongs to the side that has just moved.
                        # The engine must therefore move for the opposite side.
                        side_to_move = "red" if state[row][column] < 0 else "black"
                        if side_to_move != our_side:
                            continue
                        self.board.set_current_side(side_to_move)
                    else:
                        self.board.set_current_side(our_side)

                    move = self.engine.best_move(self.board.fen(), self.settings.think_time)
                    if move:
                        self._click_move(move, board_rect)
                        self.board.apply_ucci_move(move)
                        print(f"AI move: {move}")

                # 有可能结束了，点击结束和新一局
                time.sleep(0.2)
                # 参数是手机的参数
                self.device.click(334, 2329) # 点击确定
                time.sleep(0.1)
                self.device.click(1120, 727) # 点击×
                time.sleep(0.1)
                self.device.click(894, 2363)
        finally:
            self.engine.close()

    def _stable_board(self):
        first = self._capture_board()
        time.sleep(self.settings.stability_delay)
        second = self._capture_board()
        if first is None or second is None:
            return None, None
        first_image, _ = first
        second_image, rect = second
        if first_image.shape != second_image.shape:
            return None, None
        if np.mean(cv2.absdiff(first_image, second_image)) > self.settings.image_diff_threshold:
            print("两次图片有较大区别，等待图片稳定")
            return None, None
        return second_image, rect

    def _capture_board(self):
        image = self.device.screenshot()
        if image is None:
            return None
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
        return self.locator.locate(frame)

    def _click_move(self, move: str, board_rect: tuple[int, int, int, int]) -> None:
        x, y, width, height = board_rect
        for coordinate in (move[:2], move[2:]):
            row, column = self.board.ucci_to_screen(coordinate)
            self.device.click(
                x + (column * width) // self.settings.board_columns + width // (2 * self.settings.board_columns),
                y + (row * height) // self.settings.board_rows + height // (2 * self.settings.board_rows),
            )
            time.sleep(self.settings.click_delay)


def main() -> None:
    ChessRobot().run()


if __name__ == "__main__":
    main()
