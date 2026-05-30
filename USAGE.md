# RT-DETR 摄像头目标检测项目使用说明

本项目使用 Python、OpenCV 和 Ultralytics RT-DETR 实现实时摄像头目标检测。默认模型是 `rtdetr-l.pt`，它是基于 COCO 数据集训练的 Transformer 目标检测模型，可以识别人、车、手机、键盘、杯子等 80 类常见目标。

## 1. 项目文件说明

```text
realtime_detect.py        开发/调试用实时摄像头检测脚本
deploy_camera_onnx.py     ONNX 部署版摄像头检测脚本
train_custom.py           自定义数据集训练脚本
export_model.py           模型导出脚本，支持 ONNX / TensorRT 等格式
configs/data.yaml         自定义数据集配置模板
requirements.txt          开发和训练依赖
requirements-deploy.txt   ONNX 部署依赖
README.md                 项目简介
USAGE.md                  详细使用说明
```

## 2. 创建 Conda 环境

```powershell
conda create -n Detect python=3.11 -y
conda activate Detect
cd /d C:\Users\Shaw\Documents\目标跟踪项目
python -m pip install --upgrade pip
```

如果使用 NVIDIA 显卡，建议安装 CUDA 版 PyTorch：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

安装项目依赖：

```powershell
pip install -r requirements.txt
```

验证显卡是否可用：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

如果输出 `True` 和你的显卡名称，说明 PyTorch 已经可以使用 GPU。

## 3. 直接运行实时检测

最简单运行方式：

```powershell
python realtime_detect.py
```

使用 GPU 和 FP16 半精度：

```powershell
python realtime_detect.py --device 0 --half
```

只检测指定类别，例如只检测人：

```powershell
python realtime_detect.py --classes 0
```

检测人、车、手机：

```powershell
python realtime_detect.py --classes 0,2,67
```

如果帧率较低，可以降低推理尺寸：

```powershell
python realtime_detect.py --device 0 --half --imgsz 320 --width 640 --height 480
```

按 `q` 或 `ESC` 退出摄像头窗口。

## 4. 导出 ONNX 模型

导出默认 RT-DETR 模型：

```powershell
python export_model.py --model rtdetr-l.pt --format onnx --imgsz 416
```

导出后会生成类似文件：

```text
rtdetr-l.onnx
```

如果你训练了自己的模型，例如：

```text
runs/train/rtdetr_custom/weights/best.pt
```

可以这样导出：

```powershell
python export_model.py --model runs/train/rtdetr_custom/weights/best.pt --format onnx --imgsz 416
```

## 5. ONNX 部署版运行

安装部署依赖：

```powershell
pip install -r requirements-deploy.txt
```

运行 ONNX 摄像头检测：

```powershell
python deploy_camera_onnx.py --model rtdetr-l.onnx --device 0
```

如果显示目标太少，可以降低置信度阈值：

```powershell
python deploy_camera_onnx.py --model rtdetr-l.onnx --device 0 --conf 0.25 --max-det 300
```

保存检测结果为 JSONL：

```powershell
python deploy_camera_onnx.py --model rtdetr-l.onnx --device 0 --jsonl runs/deploy/detections.jsonl
```

保存标注后的视频：

```powershell
python deploy_camera_onnx.py --model rtdetr-l.onnx --device 0 --save-video
```

部署版和直接检测版的区别：

```text
realtime_detect.py       适合开发、调试、快速验证
deploy_camera_onnx.py    适合固定模型、稳定运行、输出 JSON 给其他程序使用
```

## 6. 训练自定义目标

数据集建议整理成 YOLO 检测格式：

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

标签文件格式为：

```text
class_id x_center y_center width height
```

坐标均为 0 到 1 之间的归一化值。

修改 `configs/data.yaml`：

```yaml
path: ./dataset
train: images/train
val: images/val

names:
  0: object_a
  1: object_b
```

开始训练：

```powershell
python train_custom.py --data configs/data.yaml --model rtdetr-l.pt --epochs 50 --imgsz 640 --batch 8 --device 0
```

训练完成后，权重通常在：

```text
runs/train/rtdetr_custom/weights/best.pt
```

使用自定义模型检测：

```powershell
python realtime_detect.py --model runs/train/rtdetr_custom/weights/best.pt --device 0 --half
```

## 7. GitHub 上传建议

本项目的 `.gitignore` 已经排除了模型权重和运行结果：

```text
*.pt
*.onnx
*.engine
runs/
dataset/
```

这些文件通常很大，不建议直接上传到 GitHub 仓库。推荐做法：

```text
代码、配置、说明文档：上传到 GitHub
模型权重、ONNX、TensorRT engine：放到 GitHub Releases、网盘，或使用 Git LFS
```

如果你已经安装 Git，并且在 GitHub 上创建了空仓库，可以用下面命令上传：

```powershell
git init
git add .
git commit -m "Initial RT-DETR camera detection project"
git branch -M main
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

如果要上传大模型文件，请先了解 Git LFS：

```powershell
git lfs install
git lfs track "*.pt"
git lfs track "*.onnx"
git lfs track "*.engine"
git add .gitattributes
```

## 8. 常见问题

### 找不到 cv2

```powershell
pip install opencv-python
```

### CUDA 不可用

先检查显卡驱动：

```powershell
nvidia-smi
```

再检查 PyTorch：

```powershell
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
```

### ONNX Runtime 报缺少 CUDA DLL

通常是 `onnxruntime-gpu`、CUDA、cuDNN 或 PyTorch CUDA 版本不匹配。可以尝试：

```powershell
pip uninstall -y onnxruntime onnxruntime-gpu torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install "onnxruntime-gpu[cuda,cudnn]"
```

然后重新运行部署脚本。
