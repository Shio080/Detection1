# RT-DETR Real-Time Webcam Object Detection

中文 | [English](#english)

## 中文

这是一个基于 Python、OpenCV 和 Ultralytics RT-DETR 的实时摄像头目标检测项目。项目默认使用 COCO 预训练权重 `rtdetr-l.pt`，可以对摄像头画面中的人、车辆、手机、键盘、杯子等常见目标进行检测、分类并画出边界框。

RT-DETR 是一种基于 Transformer 的实时目标检测模型。相比传统检测模型，它保留了 DETR 系列端到端检测的特点，同时更适合实时推理场景。

详细中文使用说明见 [USAGE.md](USAGE.md)。

### 功能特点

- 实时调用电脑摄像头进行目标检测
- 支持 GPU 和 FP16 半精度推理
- 支持 COCO 预训练模型快速启动
- 支持自定义数据集训练
- 支持导出 ONNX / TensorRT 部署格式
- 支持 ONNX 部署版脚本输出 JSONL 检测结果

### 项目结构

```text
realtime_detect.py        开发/调试用实时摄像头检测脚本
deploy_camera_onnx.py     ONNX 部署版摄像头检测脚本
train_custom.py           自定义数据集训练脚本
export_model.py           模型导出脚本
configs/data.yaml         自定义数据集配置模板
requirements.txt          开发和训练依赖
requirements-deploy.txt   ONNX 部署依赖
USAGE.md                  详细中文使用说明
```

### 环境安装

建议使用 Python 3.10 或 3.11。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果使用 NVIDIA GPU，请先安装与你的 CUDA 环境匹配的 PyTorch GPU 版本，再安装项目依赖。CPU 也可以运行，但实时帧率会明显低很多。

### 直接运行摄像头检测

```powershell
python realtime_detect.py
```

使用 GPU 和 FP16：

```powershell
python realtime_detect.py --device 0 --half
```

常用实时参数：

```powershell
python realtime_detect.py --model rtdetr-l.pt --source 0 --conf 0.4 --imgsz 416 --width 640 --height 480 --camera-fps 30 --target-fps 30
```

只检测指定类别，例如只检测人：

```powershell
python realtime_detect.py --classes 0
```

按 `q` 或 `ESC` 退出窗口。

### 训练自定义数据集

将数据集整理成 YOLO 检测格式：

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

修改 `configs/data.yaml` 中的类别名称，然后运行：

```powershell
python train_custom.py --data configs/data.yaml --model rtdetr-l.pt --epochs 50 --imgsz 640 --batch 8 --device 0
```

训练完成后的权重通常位于：

```text
runs/train/rtdetr_custom/weights/best.pt
```

### 导出 ONNX / TensorRT

导出 ONNX：

```powershell
python export_model.py --model rtdetr-l.pt --format onnx --imgsz 416
```

导出 TensorRT engine：

```powershell
python export_model.py --model rtdetr-l.pt --format engine --device 0 --half --imgsz 416
```

### ONNX 部署运行

安装部署依赖：

```powershell
pip install -r requirements-deploy.txt
```

运行 ONNX 摄像头部署：

```powershell
python deploy_camera_onnx.py --model rtdetr-l.onnx --device 0
```

保存检测结果为 JSONL：

```powershell
python deploy_camera_onnx.py --model rtdetr-l.onnx --device 0 --jsonl runs/deploy/detections.jsonl
```

### 模型文件说明

`.pt`、`.onnx`、`.engine` 等模型文件通常较大，默认不会上传到 GitHub。你可以在本地第一次运行时自动下载 `rtdetr-l.pt`，或通过 `export_model.py` 自行导出 ONNX / TensorRT 文件。

---

## English

This project implements real-time webcam object detection with Python, OpenCV, and Ultralytics RT-DETR. It uses the COCO-pretrained `rtdetr-l.pt` model by default and can detect, classify, and draw bounding boxes for common objects such as people, vehicles, phones, keyboards, cups, and more.

RT-DETR is a Transformer-based real-time object detector. It keeps the end-to-end detection style of the DETR family while being more practical for real-time inference.

For the detailed Chinese guide, see [USAGE.md](USAGE.md).

### Features

- Real-time webcam object detection
- GPU and FP16 inference support
- Quick start with COCO-pretrained RT-DETR weights
- Custom dataset training
- ONNX and TensorRT export
- ONNX deployment script with optional JSONL detection output

### Project Structure

```text
realtime_detect.py        Real-time webcam detection script for development/testing
deploy_camera_onnx.py     ONNX deployment script for webcam inference
train_custom.py           Custom dataset training script
export_model.py           Model export script
configs/data.yaml         Dataset configuration template
requirements.txt          Development and training dependencies
requirements-deploy.txt   ONNX deployment dependencies
USAGE.md                  Detailed Chinese usage guide
```

### Installation

Python 3.10 or 3.11 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the PyTorch GPU build that matches your CUDA environment before installing the project dependencies. CPU inference also works, but real-time performance will be much lower.

### Run Webcam Detection

```powershell
python realtime_detect.py
```

Run with GPU and FP16:

```powershell
python realtime_detect.py --device 0 --half
```

Common real-time settings:

```powershell
python realtime_detect.py --model rtdetr-l.pt --source 0 --conf 0.4 --imgsz 416 --width 640 --height 480 --camera-fps 30 --target-fps 30
```

Detect only selected classes, for example COCO class `person=0`:

```powershell
python realtime_detect.py --classes 0
```

Press `q` or `ESC` to quit the video window.

### Train on a Custom Dataset

Prepare the dataset in YOLO detection format:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Edit class names in `configs/data.yaml`, then run:

```powershell
python train_custom.py --data configs/data.yaml --model rtdetr-l.pt --epochs 50 --imgsz 640 --batch 8 --device 0
```

The best checkpoint is usually saved at:

```text
runs/train/rtdetr_custom/weights/best.pt
```

### Export to ONNX / TensorRT

Export to ONNX:

```powershell
python export_model.py --model rtdetr-l.pt --format onnx --imgsz 416
```

Export to TensorRT engine:

```powershell
python export_model.py --model rtdetr-l.pt --format engine --device 0 --half --imgsz 416
```

### Run ONNX Deployment

Install deployment dependencies:

```powershell
pip install -r requirements-deploy.txt
```

Run ONNX webcam inference:

```powershell
python deploy_camera_onnx.py --model rtdetr-l.onnx --device 0
```

Save detection results as JSONL:

```powershell
python deploy_camera_onnx.py --model rtdetr-l.onnx --device 0 --jsonl runs/deploy/detections.jsonl
```

### Model Files

Large model files such as `.pt`, `.onnx`, and `.engine` are ignored by Git by default. They are not uploaded to GitHub. You can download `rtdetr-l.pt` automatically on first run, or export ONNX / TensorRT files locally with `export_model.py`.
