import argparse

from ultralytics import RTDETR


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune RT-DETR on a custom object detection dataset.")
    parser.add_argument("--data", default="configs/data.yaml", help="Path to dataset YAML.")
    parser.add_argument("--model", default="rtdetr-l.pt", help="Base model weights.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="", help="Device, for example 0, cuda:0, cpu. Empty means auto.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--project", default="runs/train", help="Training output directory.")
    parser.add_argument("--name", default="rtdetr_custom", help="Training run name.")
    args = parser.parse_args()

    model = RTDETR(args.model)
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
    }
    if args.device:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
