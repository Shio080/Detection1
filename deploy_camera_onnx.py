import argparse
import json
import time
from pathlib import Path
from typing import Optional, Union

import cv2

try:
    import torch  # noqa: F401
except ImportError:
    pass

try:
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
except Exception as exc:
    print(f"ONNX Runtime DLL preload warning: {exc}")

from ultralytics import RTDETR, YOLO


BOX_COLORS = [
    (0, 220, 255),
    (0, 180, 80),
    (255, 120, 40),
    (220, 80, 255),
    (80, 160, 255),
    (120, 220, 120),
    (255, 210, 80),
    (255, 90, 120),
]


def parse_source(value: str) -> Union[int, str]:
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


def draw_detections(frame, result, line_width: int = 2):
    annotated = frame.copy()
    names = result.names or {}

    if result.boxes is None or len(result.boxes) == 0:
        cv2.putText(
            annotated,
            "Detections: 0",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
        color = BOX_COLORS[class_id % len(BOX_COLORS)]
        class_name = names.get(class_id, str(class_id))
        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width)

        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1, label_size[1] + 8)
        cv2.rectangle(
            annotated,
            (x1, label_y - label_size[1] - baseline - 6),
            (x1 + label_size[0] + 8, label_y + baseline - 2),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        f"Detections: {len(result.boxes)}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def result_to_record(result, frame_id: int, fps: float) -> dict:
    names = result.names or {}
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": names.get(class_id, str(class_id)),
                    "confidence": round(float(box.conf[0]), 6),
                    "box_xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                }
            )

    return {
        "frame_id": frame_id,
        "timestamp": time.time(),
        "fps": round(fps, 2),
        "detections": detections,
    }


def build_predict_kwargs(args: argparse.Namespace) -> dict:
    predict_kwargs = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "max_det": args.max_det,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device
    if args.classes:
        predict_kwargs["classes"] = [int(item) for item in args.classes.split(",")]
    return predict_kwargs


def run(args: argparse.Namespace) -> None:
    cv2.setUseOptimized(True)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Cannot find ONNX model: {model_path}. Export one first, for example: "
            "python export_model.py --model rtdetr-l.pt --format onnx --imgsz 416"
        )

    if args.loader == "rtdetr":
        model = RTDETR(str(model_path))
    else:
        model = YOLO(str(model_path), task="detect")
    source = parse_source(args.source)
    capture = open_capture(source, args.width, args.height, args.camera_fps)
    predict_kwargs = build_predict_kwargs(args)

    jsonl_file = None
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = jsonl_path.open("a", encoding="utf-8")

    writer = None
    output_path = None
    if args.save_video:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"deploy_camera_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

    last_time = time.perf_counter()
    smoothed_fps = 0.0
    frame_id = 0

    print("Deployment camera inference started. Press q or ESC to quit.")
    print(f"ONNX model: {model_path}")
    print(f"Model loader: {args.loader}")
    print(f"Source: {source}")
    print(f"Requested camera: {args.width}x{args.height} @ {args.camera_fps} FPS")
    print(f"Inference image size: {args.imgsz}")
    if args.device:
        print(f"Device: {args.device}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("No frame received. Stopping.")
                break

            results = model.predict(frame, **predict_kwargs)
            result = results[0]
            annotated = draw_detections(frame, result, args.line_width)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            smoothed_fps = instant_fps if smoothed_fps == 0 else (0.9 * smoothed_fps + 0.1 * instant_fps)
            draw_fps(annotated, smoothed_fps)

            if jsonl_file is not None:
                record = result_to_record(result, frame_id, smoothed_fps)
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                if frame_id % args.flush_every == 0:
                    jsonl_file.flush()

            if writer is None and args.save_video:
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

            frame_id += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
            print(f"Saved video: {output_path}")
        if jsonl_file is not None:
            jsonl_file.close()
            print(f"Saved detection records: {args.jsonl}")
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy RT-DETR ONNX model for camera inference.")
    parser.add_argument("--model", default="rtdetr-l.onnx", help="Path to exported ONNX model.")
    parser.add_argument("--loader", choices=["rtdetr", "yolo"], default="rtdetr", help="Ultralytics loader for the exported model.")
    parser.add_argument("--source", default="0", help="Camera id, video path, image path, or stream URL.")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size.")
    parser.add_argument("--device", default="0", help="Device, for example 0 or cpu.")
    parser.add_argument("--classes", default="", help="Optional comma-separated class ids, for example 0,2,67.")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per frame.")
    parser.add_argument("--line-width", type=int, default=2, help="Detection box line width.")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height.")
    parser.add_argument("--camera-fps", type=int, default=30, help="Requested camera FPS.")
    parser.add_argument("--target-fps", type=float, default=30.0, help="Target display FPS cap.")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True, help="Show video window.")
    parser.add_argument("--jsonl", default="", help="Optional path to save detection records as JSONL.")
    parser.add_argument("--flush-every", type=int, default=10, help="Flush JSONL records every N frames.")
    parser.add_argument("--save-video", action="store_true", help="Save annotated video.")
    parser.add_argument("--output-dir", default="runs/deploy", help="Directory for saved deployment outputs.")
    parser.add_argument("--output-fps", type=float, default=30.0, help="Saved video FPS.")
    parser.add_argument("--window-name", default="RT-DETR ONNX Deployment", help="OpenCV window title.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
