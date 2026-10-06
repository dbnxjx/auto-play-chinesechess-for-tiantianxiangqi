from __future__ import annotations

import subprocess
import time
from pathlib import Path


class FairyStockfishEngine:
    """UCI client configured for the Fairy-Stockfish minixiangqi variant."""

    def __init__(self, engine_path: Path, nnue_path: Path, variant: str) -> None:
        self.engine_path = engine_path
        self.nnue_path = nnue_path
        self.variant = variant
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"Fairy-Stockfish executable not found: {self.engine_path}")
        if not self.nnue_path.is_file():
            raise FileNotFoundError(f"NNUE weight not found: {self.nnue_path}")
        self.process = subprocess.Popen(
            [str(self.engine_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self._send("uci")
        self._wait_for("uciok", 10)
        self._send(f"setoption name UCI_Variant value {self.variant}")
        self._send("setoption name EvalFile value " + str(self.nnue_path))
        self._send("setoption name Threads value 8")
        self._send("setoption name Hash value 1536")
        self._send("isready")
        self._wait_for("readyok", 10)
        self._send("ucinewgame")

    def best_move(self, fen: str, think_time: float) -> str | None:
        self._send("position fen " + fen)
        self._send(f"go movetime {max(1, round(think_time * 1000))}")
        deadline = time.monotonic() + think_time + 3
        while time.monotonic() < deadline:
            line = self._read()
            if line.startswith("bestmove"):
                move = line.split()[1]
                return None if move == "(none)" else move
        return None

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            self._send("quit")
            self.process.terminate()
        self.process = None

    def _send(self, command: str) -> None:
        if not self.process or not self.process.stdin:
            raise RuntimeError("Fairy-Stockfish has not started")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _read(self) -> str:
        if not self.process or not self.process.stdout:
            return ""
        return self.process.stdout.readline().strip()

    def _wait_for(self, expected: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._read() == expected:
                return
        raise TimeoutError(f"Fairy-Stockfish did not return {expected}")
