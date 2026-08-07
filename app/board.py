from __future__ import annotations

import numpy as np


class Board:
    def __init__(self) -> None:
        self.flip = False
        self.current_side = "red"
        self.state: list[list[int]] = self._standard_state()

    @staticmethod
    def _standard_state() -> list[list[int]]:
        return [
            [-1, -2, -3, -4, -5, -4, -3, -2, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -6, 0, 0, 0, 0, 0, -6, 0],
            [-7, 0, -7, 0, -7, 0, -7, 0, -7],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [7, 0, 7, 0, 7, 0, 7, 0, 7],
            [0, 6, 0, 0, 0, 0, 0, 6, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 3, 4, 5, 4, 3, 2, 1],
        ]

    def detect_orientation_and_set_state(self, recognized_state) -> None:
        state = [list(row) for row in recognized_state]
        if len(state) != 10 or any(len(row) != 9 for row in state):
            raise ValueError("Recognized board must be 10 rows by 9 columns")
        black_king_top = any(state[row][column] == -5 for row in range(3) for column in range(3, 6))
        red_king_bottom = any(state[row][column] == 5 for row in range(7, 10) for column in range(3, 6))
        self.flip = not (black_king_top and red_king_bottom)
        self.state = state

    def set_current_side(self, side: str) -> None:
        if side not in {"red", "black"}:
            raise ValueError("Side must be red or black")
        self.current_side = side

    def ucci_to_screen(self, coordinate: str) -> tuple[int, int]:
        if len(coordinate) != 2 or coordinate[0] not in "abcdefghi" or coordinate[1] not in "0123456789":
            raise ValueError(f"Invalid UCCI coordinate: {coordinate}")
        column = "abcdefghi".index(coordinate[0])
        rank = int(coordinate[1])
        return (rank if self.flip else 9 - rank, column)

    def apply_ucci_move(self, move: str) -> None:
        if len(move) != 4:
            raise ValueError(f"Invalid UCCI move: {move}")
        from_row, from_column = self.ucci_to_screen(move[:2])
        to_row, to_column = self.ucci_to_screen(move[2:])
        self.state[to_row][to_column] = self.state[from_row][from_column]
        self.state[from_row][from_column] = 0
        self.current_side = "black" if self.current_side == "red" else "red"

    def fen(self) -> str:
        pieces = {1: "R", 2: "N", 3: "B", 4: "A", 5: "K", 6: "C", 7: "P", -1: "r", -2: "n", -3: "b", -4: "a", -5: "k", -6: "c", -7: "p"}
        rows = self.state if not self.flip else self.state[::-1]
        fen_rows = []
        for row in rows:
            empty = 0
            result = ""
            for piece in row:
                if piece == 0:
                    empty += 1
                else:
                    if empty:
                        result += str(empty)
                        empty = 0
                    result += pieces[piece]
            fen_rows.append(result + (str(empty) if empty else ""))
        side = "w" if self.current_side == "red" else "b"
        return f"{'/'.join(fen_rows)} {side} - - 0 1"

    def print_board(self) -> None:
        print(np.array(self.state))
