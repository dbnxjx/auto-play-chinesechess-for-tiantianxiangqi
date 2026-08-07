# Model Training

`data/` contains the original labeled cell images. `yolo/source/` contains original board and background images.
Generated datasets and trained weights belong in `yolo/dataset/` and `outputs/`; neither is runtime input.

Run from the project root with the project virtual environment activated:

```powershell
python -m model_train.generate_yolo_dataset
python -m model_train.train_yolo
python -m model_train.train_classifier cell
python -m model_train.train_classifier piece
python -m model_train.train_classifier marker
```

After evaluating a candidate model, promote it explicitly:

```powershell
python -m model_train.export_model model_train/outputs/piece_classifier_15class.pth --name piece_classifier_15class.pth
```
