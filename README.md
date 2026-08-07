# 中国象棋自动下棋

该项目通过 ADB 获取 Android 设备截图，使用 YOLO 定位棋盘、CNN 识别棋子与落子标记，再调用 Pikafish 计算并执行走子。

## 运行前准备

1. 在 Android 设备中开启 USB 调试，并确认设备已通过 ADB 连接。
2. 在 `config.ini` 中设置正确的 `adb_path`。
3. 确认项目虚拟环境存在；如需重新安装依赖，执行：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 启动

双击 `start.bat`，或在项目根目录执行：

```powershell
.venv\Scripts\python.exe -m app.main
```

程序启动后会依次检查 ADB 设备、加载模型、启动 Pikafish，并开始识别棋盘和自动走子。

## 目录说明

- `app/`：运行期代码。
- `app/main.py`：主程序入口，负责截图、识别、引擎调用与点击。
- `app/board.py`：棋盘状态、FEN 生成和 UCCI 坐标转换。
- `app/device.py`：ADB 截图和点击控制。
- `app/engine.py`：Pikafish 进程及 UCI 通信。
- `app/vision/`：YOLO 棋盘定位与 CNN 棋子识别。
- `app/assets/models/`：当前运行使用的已确认模型。
- `model_train/`：模型训练、原始训练图片、数据集生成和模型导出。
- `shared/cnn.py`：训练与运行共用的 CNN 网络定义。
- `vendor/pikafish/`：Pikafish 可执行文件与 NNUE 权重。
- `config.ini`：运行参数、模型路径、ADB 路径和引擎路径。

运行期只依赖 `app/`、`shared/`、`vendor/pikafish/`、`app/assets/models/` 和 `config.ini`。

## 模型训练

训练相关内容都在 `model_train/` 中，运行代码不会直接读取训练输出。训练完成并确认效果后，需要显式导出模型到 `app/assets/models/` 才会在运行时生效。

在项目根目录执行以下命令：

```powershell
# 生成 YOLO 棋盘检测数据集
.venv\Scripts\python.exe -m model_train.generate_yolo_dataset

# 训练 YOLO 棋盘检测模型
.venv\Scripts\python.exe -m model_train.train_yolo

# 训练三类格子识别模型：黑棋、空位、红棋
.venv\Scripts\python.exe -m model_train.train_classifier cell

# 训练十五类棋子识别模型
.venv\Scripts\python.exe -m model_train.train_classifier piece

# 训练两类落子标记识别模型
.venv\Scripts\python.exe -m model_train.train_classifier marker
```

训练权重输出到 `model_train/outputs/`。例如，将十五类棋子模型导出为运行模型：

```powershell
.venv\Scripts\python.exe -m model_train.export_model model_train/outputs/piece_classifier_15class.pth --name piece_classifier_15class.pth
```

可导出的运行模型名称为：

- `board_detector.pt`
- `cell_classifier_3class.pth`
- `piece_classifier_15class.pth`
- `marker_classifier_2class.pth`
