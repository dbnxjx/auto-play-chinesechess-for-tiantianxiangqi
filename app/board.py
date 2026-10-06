from __future__ import annotations

import numpy as np


class MiniXiangqiBoard:
    """State, FEN, and screen-coordinate conversion for Fairy-Stockfish minixiangqi."""

    ROWS = 7
    COLUMNS = 7
    FILES = "abcdefg"
    PIECES = {
        1: "R", 2: "N", 5: "K", 6: "C", 7: "P",
        -1: "r", -2: "n", -5: "k", -6: "c", -7: "p",
    }

    def __init__(self) -> None:
        self.flip = False
        self.current_side = "red"
        self.state = self._standard_state()

    @staticmethod
    def _standard_state() -> list[list[int]]:
        return [
            [-1, -6, -2, -5, -2, -6, -1],
            [-7, 0, -7, -7, -7, 0, -7],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [7, 0, 7, 7, 7, 0, 7],
            [1, 6, 2, 5, 2, 6, 1],
        ]

    def detect_orientation_and_set_state(self, recognized_state) -> None:
        state = [list(row) for row in recognized_state]
        if len(state) != self.ROWS or any(len(row) != self.COLUMNS for row in state):
            raise ValueError("Recognized mini xiangqi board must be 7 rows by 7 columns")
        red_rows = [row for row, values in enumerate(state) for piece in values if piece > 0]
        black_rows = [row for row, values in enumerate(state) for piece in values if piece < 0]
        if red_rows and black_rows:
            self.flip = sum(red_rows) / len(red_rows) < sum(black_rows) / len(black_rows)
        self.state = state

    def set_current_side(self, side: str) -> None:
        if side not in {"red", "black"}:
            raise ValueError("Side must be red or black")
        self.current_side = side

    def ucci_to_screen(self, coordinate: str) -> tuple[int, int]:
        if len(coordinate) != 2 or coordinate[0] not in self.FILES or coordinate[1] not in "1234567":
            raise ValueError(f"Invalid minixiangqi coordinate: {coordinate}")
        column = self.FILES.index(coordinate[0])
        rank = int(coordinate[1])
        if self.flip:
            return rank - 1, self.COLUMNS - column - 1
        return self.ROWS - rank, column

    def apply_ucci_move(self, move: str) -> None:
        if len(move) != 4:
            raise ValueError(f"Invalid minixiangqi move: {move}")
        from_row, from_column = self.ucci_to_screen(move[:2])
        to_row, to_column = self.ucci_to_screen(move[2:])
        self.state[to_row][to_column] = self.state[from_row][from_column]
        self.state[from_row][from_column] = 0
        self.current_side = "black" if self.current_side == "red" else "red"

    def fen(self) -> str:
        rows = self.state if not self.flip else [row[::-1] for row in self.state[::-1]]
        fen_rows = []
        for row in rows:
            empty = 0
            result = ""
            for piece in row:
                if piece == 0:
                    empty += 1
                    continue
                if empty:
                    result += str(empty)
                    empty = 0
                try:
                    result += self.PIECES[piece]
                except KeyError as error:
                    raise ValueError(f"Piece {piece} is not valid in minixiangqi") from error
            fen_rows.append(result + (str(empty) if empty else ""))
        side = "w" if self.current_side == "red" else "b"
        return f"{'/'.join(fen_rows)} {side} - - 0 1"

    def print_board(self) -> None:
        print(np.array(self.state))
