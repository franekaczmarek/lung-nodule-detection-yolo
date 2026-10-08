"""Train a YOLO nodule detector on the dataset built by the preprocessing step.

Usage:
    python -m lung_nodule_detection.train --data data/yolo/dataset.yaml \\
        --weights yolov10l.pt --name yolov10l
"""

import argparse
from pathlib import Path

# Hyperparameters used for the thesis runs. Anything not listed uses the Ultralytics default.
HYPERPARAMETERS = {
    "optimizer": "SGD",
    "single_cls": True,
    "patience": 20,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 0.05,
    "cls": 0.5,
    "iou": 0.2,
    # Augmentations applied on the fly by Ultralytics, on top of augment.py.
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "multi_scale": False,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data", type=Path, required=True, help="dataset.yaml written by the preprocessing step"
    )
    parser.add_argument(
        "--weights", default="yolov10l.pt", help="pretrained weights, or a model .yaml to train from scratch"
    )
    parser.add_argument("--name", default="yolov10l", help="run name inside --project")
    parser.add_argument("--project", type=Path, default=Path("runs"), help="folder for training results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--imgsz", type=int, default=502, help="input size; the canvas width from preprocessing"
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        **HYPERPARAMETERS,
    )


if __name__ == "__main__":
    main()
