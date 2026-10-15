from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

MODEL = "runs/segment/runs/dual_fdi_seg_v1/weights/best.pt"
SOURCE = "datasets/processed/dual_yolo_seg/test/images"
OUTPUT = Path("results/clean_test_predictions")

CONF = 0.25
ALPHA = 0.35

OUTPUT.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL)

# Fixed palette only for visualization
palette = [
    (255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 180, 80),
    (220, 80, 255), (80, 255, 220), (255, 120, 180), (180, 255, 80),
]

results = model.predict(
    source=SOURCE,
    imgsz=640,
    conf=CONF,
    device="mps",
    stream=True,
    verbose=False,
)

for result in results:
    image = result.orig_img.copy()
    overlay = image.copy()

    if result.masks is not None and result.boxes is not None:

        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        h, w = image.shape[:2]

        for mask, cls, conf in zip(masks, classes, confs):

            mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

            binary = mask > 0.5
            color = palette[cls % len(palette)]

            overlay[binary] = color

            # Find mask centroid for FDI label
            ys, xs = np.where(binary)

            if len(xs) == 0:
                continue

            cx = int(xs.mean())
            cy = int(ys.mean())

            fdi = model.names[cls]

            # Small readable FDI label only
            cv2.putText(
                image,
                str(fdi),
                (cx - 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # Blend masks with original OPG
    image = cv2.addWeighted(
        overlay,
        ALPHA,
        image,
        1.0 - ALPHA,
        0
    )

    # Re-draw FDI after blending so labels remain sharp
    if result.masks is not None and result.boxes is not None:

        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        h, w = image.shape[:2]

        for mask, cls in zip(masks, classes):

            mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

            ys, xs = np.where(mask > 0.5)

            if len(xs) == 0:
                continue

            cx = int(xs.mean())
            cy = int(ys.mean())

            cv2.putText(
                image,
                str(model.names[cls]),
                (cx - 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    filename = Path(result.path).stem + ".png"
    cv2.imwrite(str(OUTPUT / filename), image)

print("Saved clean predictions to:")
print(OUTPUT.resolve())
