"""
Trains YOLO to detect guns and knives using the data built by prepare_datasets.py.

We don't train a network from nothing. yolov8s.pt has already been trained on COCO,
a large dataset of 80 everyday classes (people, cars, cups and so on), so its early
layers already know how to pick out edges, textures and the outlines of objects.
Training it on our images, which is called fine-tuning, swaps its 80-class output for
our 2 classes and adjusts the whole network towards telling guns and knives apart.
That takes a fraction of the data and time that starting from scratch would.

    python train.py                                # everything, default settings
    python train.py --fraction 0.05 --epochs 3     # quick check that it all works
    python train.py --data data/kaggle.yaml        # train on a single source

The full dataset is around 118k training images, which is a lot for a laptop. On this
Mac (M5) training got through about 16 images a second, so one epoch over everything
takes roughly two hours, plus about ten minutes to check the 15k validation images.
Fifty epochs would mean days. Start with a small --fraction to make sure everything
runs, then either train on a fraction of the data or do the full run on a machine with
an NVIDIA GPU (Google Colab, or a university GPU server), which is many times faster.

Each run is saved in runs/<name>/. The file you want afterwards is
runs/<name>/weights/best.pt, which you give to main.py with --weights. If a folder
with that name already exists, YOLO adds a number (weapons2, weapons3, ...) rather
than overwriting it.
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def pick_device():
    """
    Use the fastest hardware available. "0" means the first NVIDIA GPU (e.g. on
    Colab), "mps" is the GPU in an Apple Silicon Mac, and the CPU is the slow
    fallback.
    """
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on the weapon datasets")
    parser.add_argument("--data", default="data/weapons.yaml",
                        help="dataset config from prepare_datasets.py")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="weights to start from")
    parser.add_argument("--epochs", type=int, default=50,
                        help="how many full passes over the training images")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="images are resized to this many pixels across before going into the network")
    parser.add_argument("--batch", type=int, default=16,
                        help="images processed together in one step; lower it if you run out of memory")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="share of the training images to use, e.g. 0.1 for 10%%")
    parser.add_argument("--name", default="weapons",
                        help="name of the folder in runs/ to save this run in")
    args = parser.parse_args()

    model = YOLO(args.model)

    # A few of these settings deserve a word of explanation:
    #
    # epochs: an epoch is one pass over every training image. After each one YOLO
    #   checks how well the model does on the validation images, which it never
    #   trains on, and saves the best version so far as best.pt.
    #
    # patience: if the validation score hasn't improved for this many epochs in a
    #   row, training stops early. Past that point the model is usually just
    #   memorising the training images (overfitting), not getting any better.
    #
    # imgsz: bigger images make small objects (a knife across a room) easier to
    #   spot but are slower to train. 640 is what yolov8s was originally trained at.
    #
    # YOLO also applies its own random changes to the training images as it goes
    # (flips, colour shifts, scaling, stitching four images into one), so the
    # model sees a slightly different version of each image every epoch. That's
    # separate from, and on top of, any augmentation a dataset already came with.
    model.train(
        data=str(ROOT / args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        patience=10,
        device=pick_device(),
        project=str(ROOT / "runs"),
        name=args.name,
    )


if __name__ == "__main__":
    main()
