"""
Live object detection from the webcam using YOLO.

Each frame from the camera goes through a YOLO model, which hands back a list of
boxes: where it thinks an object is, which class it thinks the object belongs to,
and how confident it is, as a score from 0 to 1. We draw those boxes onto the
frame and show it in a window, over and over, which gives a live video.

By default this uses yolov8s.pt, which was trained on the COCO dataset. It knows 80
everyday classes (person, car, phone, cup and so on) but nothing about guns or
knives. Once train.py has produced a weapon model, point --weights at it instead:

    python main.py --weights runs/weapons/weights/best.pt

Press q in the video window to quit.
"""
import argparse

import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import colors


def draw_detections(frame, result):
    # result.names maps class numbers to names, e.g. {0: "gun", 1: "knife"}.
    class_names = result.names

    for box in result.boxes:
        # xyxy describes the box by two corners, top-left (x1, y1) and bottom-right
        # (x2, y2), measured in pixels. YOLO returns them as decimal numbers inside a
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        colour = colors(cls, bgr=True)

        # The tracker gives each object an ID and tries to keep it the same from one
        # frame to the next, so you can tell "the same gun" apart from "a new gun".
        label = f"{class_names[cls]} {conf:.2f}"
        if box.id is not None:
            label = f"ID {int(box.id[0])} {label}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # The label goes on a solid block of the box's colour so it can still be read
        # against a busy background. 

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1, th + baseline + 4)
        cv2.rectangle(frame, (x1, ty - th - baseline - 4), (x1 + tw + 4, ty), colour, -1)
        cv2.putText(frame, label, (x1 + 2, ty - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description="Live webcam object detection with YOLO")
    parser.add_argument("--weights", default="yolov8s.pt",
                        help="model weights; point this at your trained best.pt later")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--conf", type=float, default=0.4, help="minimum confidence to show a box")
    args = parser.parse_args()

    # Apple Silicon Macs have a GPU that PyTorch can use through "mps" (Apple's Metal
    # framework). It's several times faster than the CPU, which matters when every
    # frame has to be processed live. On other machines we fall back to the CPU.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    yolo = YOLO(args.weights)

    # Camera 0 is normally the built-in webcam; any external cameras are 1, 2, ...
    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        raise SystemExit(f"Could not open camera {args.camera} "
                         "(check the terminal has camera permission in System Settings)")
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            result = yolo.track(frame, persist=True, conf=args.conf,
                                device=device, verbose=False)[0]
            draw_detections(frame, result)

            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
