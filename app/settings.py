from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    use_adb: bool
    interval: float
    stability_delay: float
    image_diff_threshold: float
    yolo_model_path: Path
    yolo_confidence: float
    cell_model_path: Path
    piece_model_path: Path
    marker_model_path: Path
    think_time: float
    click_delay: float
    adb_path: str
    variant: str
    board_rows: int
    board_columns: int
    engine_path: Path
    nnue_path: Path


def load_settings(config_path: Path | None = None) -> Settings:
    parser = configparser.ConfigParser()
    config_path = config_path or PROJECT_ROOT / "config.ini"
    if not parser.read(config_path, encoding="utf-8"):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    return Settings(
        use_adb=parser.getboolean("General", "use_adb", fallback=True),
        interval=parser.getfloat("General", "interval", fallback=1.0),
        stability_delay=parser.getfloat("General", "stability_delay", fallback=0.2),
        image_diff_threshold=parser.getfloat("General", "image_diff_threshold", fallback=2.0),
        yolo_model_path=_resolve_path(parser.get("Models", "yolo_model_path")),
        yolo_confidence=parser.getfloat("Models", "yolo_confidence", fallback=0.5),
        cell_model_path=_resolve_path(parser.get("Models", "cell_model_path")),
        piece_model_path=_resolve_path(parser.get("Models", "piece_model_path")),
        marker_model_path=_resolve_path(parser.get("Models", "marker_model_path")),
        think_time=parser.getfloat("AI", "think_time", fallback=1.0),
        click_delay=parser.getfloat("Click", "click_delay", fallback=0.2),
        adb_path=parser.get("ADB", "adb_path", fallback="adb"),
        variant=parser.get("Game", "variant", fallback="minixiangqi"),
        board_rows=parser.getint("Game", "board_rows", fallback=7),
        board_columns=parser.getint("Game", "board_columns", fallback=7),
        engine_path=_resolve_path(parser.get("Engine", "engine_path")),
        nnue_path=_resolve_path(parser.get("Engine", "nnue_path")),
    )
