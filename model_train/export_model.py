from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNTIME_MODELS = ROOT.parent / "app" / "assets" / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a validated trained model to runtime assets")
    parser.add_argument("source", type=Path)
    parser.add_argument("--name", required=True, choices=["board_detector.pt", "cell_classifier_3class.pth", "piece_classifier_15class.pth", "marker_classifier_2class.pth"])
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    RUNTIME_MODELS.mkdir(parents=True, exist_ok=True)
    destination = RUNTIME_MODELS / args.name
    shutil.copy2(source, destination)
    print(f"Exported {source} to {destination}")


if __name__ == "__main__":
    main()
