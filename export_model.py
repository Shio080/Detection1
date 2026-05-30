import argparse

from ultralytics import RTDETR


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RT-DETR weights for deployment.")
    parser.add_argument("--model", required=True, help="Path to trained .pt weights.")
    parser.add_argument(
        "--format",
        default="onnx",
        choices=["onnx", "engine", "torchscript", "openvino"],
        help="Export format. Use engine for TensorRT on NVIDIA GPUs.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size.")
    parser.add_argument("--device", default="", help="Device, for example 0, cuda:0, cpu. Empty means auto.")
    parser.add_argument("--half", action="store_true", help="Use FP16 export when supported.")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shapes where supported.")
    args = parser.parse_args()

    model = RTDETR(args.model)
    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "half": args.half,
        "dynamic": args.dynamic,
    }
    if args.device:
        export_kwargs["device"] = args.device

    model.export(**export_kwargs)


if __name__ == "__main__":
    main()
