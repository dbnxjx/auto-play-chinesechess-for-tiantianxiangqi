from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main() -> None:
    dataset = ROOT / "yolo" / "dataset.yaml"
    model = YOLO("yolo26n.pt")
    model.train(data=str(dataset), project=str(ROOT / "outputs" / "yolo"), name="board_detector")


if __name__ == "__main__":
    main()
