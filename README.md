# RT-DETR 实时摄像头目标检测

这个项目用 Python、OpenCV 和 Ultralytics RT-DETR 实现电脑摄像头实时目标检测与分类。默认使用 COCO 预训练权重 `rtdetr-l.pt`，第一次运行会自动下载模型。

详细操作步骤见 `USAGE.md`。

## 1. 创建环境

建议使用 Python 3.10 或 3.11。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果你有 NVIDIA GPU，请按你的 CUDA 版本安装对应的 PyTorch，再安装本项目依赖。CPU 也能跑，但实时帧率会低很多。

## 2. 直接运行摄像头检测

```powershell
python realtime_detect.py
```

常用参数：

```powershell
python realtime_detect.py --model rtdetr-l.pt --source 0 --conf 0.4 --imgsz 416 --width 640 --height 480 --camera-fps 30 --target-fps 30
```

使用 GPU：

```powershell
python realtime_detect.py --device 0 --half
```

默认配置已经按 30 FPS 优先做了优化：摄像头请求 `640x480 @ 30 FPS`，推理尺寸为 `416`。如果电脑有 NVIDIA CUDA，脚本会自动尝试使用 GPU 和 FP16。

保存检测后的视频：

```powershell
python realtime_detect.py --save
```

只检测部分类别，例如 COCO 里的 `person=0`：

```powershell
python realtime_detect.py --classes 0
```

按 `q` 或 `ESC` 退出窗口。

## 3. 自定义数据训练

把数据整理成 YOLO 检测格式：

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

修改 `configs/data.yaml` 里的类别名称，然后运行：

```powershell
python train_custom.py --data configs/data.yaml --model rtdetr-l.pt --epochs 50 --imgsz 640 --batch 8
```

训练完成后，权重通常在：

```text
runs/train/rtdetr_custom/weights/best.pt
```

用自定义权重实时检测：

```powershell
python realtime_detect.py --model runs/train/rtdetr_custom/weights/best.pt
```

## 4. 导出部署模型

导出 ONNX：

```powershell
python export_model.py --model runs/train/rtdetr_custom/weights/best.pt --format onnx
```

NVIDIA GPU 上导出 TensorRT：

```powershell
python export_model.py --model runs/train/rtdetr_custom/weights/best.pt --format engine --device 0 --half
```

## 5. 调速建议

- 帧率不够时，先降低 `--imgsz`，例如 `416` 或 `320`。
- 摄像头分辨率可以降到 `--width 640 --height 480`。
- 有 NVIDIA GPU 时使用 `--device 0 --half`。
- 最终部署优先导出 TensorRT engine。
- 如果仍达不到 30 FPS，说明主要瓶颈在模型推理速度，需要 GPU、TensorRT 或更小的输入尺寸。

## 6. 正式 Python 部署

部署版入口是 `deploy_camera_onnx.py`，它默认加载当前目录下的 `rtdetr-l.onnx`。

安装部署依赖：

```powershell
pip install -r requirements-deploy.txt
```

运行 ONNX 摄像头部署：

```powershell
python deploy_camera_onnx.py --model rtdetr-l.onnx --device 0
```

如果画面里目标很多但显示太少，可以降低置信度阈值：

```powershell
python deploy_camera_onnx.py --model rtdetr-l.onnx --device 0 --conf 0.25 --max-det 300
```

部署脚本默认会逐个遍历检测框并分别画出类别和置信度。RT-DETR 导出的 ONNX 默认使用 `RTDETR` 加载；如果你的环境里加载失败，可以临时切回通用加载器：

```powershell
python deploy_camera_onnx.py --model rtdetr-l.onnx --device 0 --loader yolo
```

保存每帧检测结果为 JSONL：

```powershell
python deploy_camera_onnx.py --model rtdetr-l.onnx --device 0 --jsonl runs/deploy/detections.jsonl
```

同时保存标注后的视频：

```powershell
python deploy_camera_onnx.py --model rtdetr-l.onnx --device 0 --save-video
```

这个入口适合长期运行或接入其他程序；训练、导出和实验参数调整则继续放在 `train_custom.py`、`export_model.py` 和 `realtime_detect.py` 里。
