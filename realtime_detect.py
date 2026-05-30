import argparse
import time
from pathlib import Path
from typing import Optional, Union

import cv2
from ultralytics import RTDETR

try:
    import torch
except ImportError:
    torch = None


def parse_source(value: str) -> Union[int, str]:
    """Use integer camera ids when possible, otherwise treat the value as a path/URL."""
    try:
        return int(value)
    except ValueError:
        return value


def open_capture(
    source: Union[int, str],
    width: Optional[int],
    height: Optional[int],
    fps: Optional[int],
) -> cv2.VideoCapture:
    if isinstance(source, int):
        # CAP_DSHOW usually opens USB/web cameras faster and more reliably on Windows.
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(source)

    if width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        capture.set(cv2.CAP_PROP_FPS, fps)

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    return capture


def draw_fps(frame, fps: float) -> None:
    text = f"FPS: {fps:.1f}"
    cv2.rectangle(frame, (10, 10), (125, 45), (20, 20, 20), -1)
    cv2.putText(
        frame,
        text,
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 120),
        2,
        cv2.LINE_AA,
    )


def build_predict_kwargs(args: argparse.Namespace) -> dict:
    predict_kwargs = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "max_det": args.max_det,
        "verbose": False,
    }
    if args.device:
        device = args.device
    elif args.auto_device and torch is not None and torch.cuda.is_available():
        device = "0"
    else:
        device = ""

    use_half = args.half
    if args.auto_half and device and device != "cpu" and torch is not None and torch.cuda.is_available():
        use_half = True

    if device:
        predict_kwargs["device"] = device
    if use_half:
        predict_kwargs["half"] = True
    if args.classes:
        predict_kwargs["classes"] = [int(item) for item in args.classes.split(",")]
    return predict_kwargs


def run(args: argparse.Namespace) -> None:
    cv2.setUseOptimized(True)

    model = RTDETR(args.model)
    source = parse_source(args.source)
    capture = open_capture(source, args.width, args.height, args.camera_fps)

    writer = None
    output_path = None
    if args.save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"rtdetr_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

    predict_kwargs = build_predict_kwargs(args)
    last_time = time.perf_counter()
    smoothed_fps = 0.0

    print("Press q or ESC in the video window to quit.")
    print(f"Model: {args.model}")
    print(f"Source: {source}")
    print(f"Requested camera: {args.width}x{args.height} @ {args.camera_fps} FPS")
    print(f"Inference image size: {args.imgsz}")
    if "device" in predict_kwargs:
        print(f"Device: {predict_kwargs['device']}")
    if predict_kwargs.get("half"):
        print("FP16: enabled")

    while True:
        ok, frame = capture.read()
        if not ok:
            print("No frame received. Stopping.")
            break

        results = model.predict(frame, **predict_kwargs)
        annotated = results[0].plot()

        now = time.perf_counter()
        instant_fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now
        smoothed_fps = instant_fps if smoothed_fps == 0 else (0.9 * smoothed_fps + 0.1 * instant_fps)
        draw_fps(annotated, smoothed_fps)

        if writer is None and args.save:
            height, width = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, args.output_fps, (width, height))

        if writer is not None:
            writer.write(annotated)

        if args.show:
            cv2.imshow(args.window_name, annotated)
            wait_ms = max(1, int(1000 / args.target_fps)) if args.target_fps > 0 else 1
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (27, ord("q")):
                break

    capture.release()
    if writer is not None:
        writer.release()
        print(f"Saved video: {output_path}")
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time object detection from camera/video using RT-DETR."
    )
    parser.add_argument("--model", default="rtdetr-l.pt", help="Model weight path or Ultralytics model name.")
    parser.add_argument("--source", default="0", help="Camera id, video path, image path, or stream URL.")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size.")
    parser.add_argument("--device", default="", help="Device, for example 0, cuda:0, cpu. Empty means auto.")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference when supported.")
    parser.add_argument("--auto-device", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA automatically when available.")
    parser.add_argument("--auto-half", action=argparse.BooleanOptionalAction, default=True, help="Use FP16 automatically on CUDA.")
    parser.add_argument("--classes", default="", help="Optional comma-separated class ids, for example 0,2,67.")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per frame.")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height.")
    parser.add_argument("--camera-fps", type=int, default=30, help="Requested camera FPS.")
    parser.add_argument("--target-fps", type=float, default=30.0, help="Target display FPS cap.")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True, help="Show video window.")
    parser.add_argument("--save", action="store_true", help="Save annotated video.")
    parser.add_argument("--output-dir", default="runs/camera", help="Directory for saved videos.")
    parser.add_argument("--output-fps", type=float, default=30.0, help="Saved video FPS.")
    parser.add_argument("--window-name", default="RT-DETR Realtime Detection", help="OpenCV window title.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
