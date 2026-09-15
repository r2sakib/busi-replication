"""
Turn a trained detector into per-image predictions, then optionally fuse with MedSAM.

This writes the preds.json contract. Nothing downstream ever touches a .pt file,
which is what lets you build Table 4 at hour 6 from runs that are still going.

    python -m src.predict --run results/yolov10b_B --data /content/busi_yolo
    python -m src.predict --run results/yolov10b_B --data /content/busi_yolo \
        --medsam-cache /content/medsam_cache

The paper never states how a detector becomes an image-level classifier. The only
reading consistent with its own Table 4 is:

  Config A/B  image label = class of the highest-CONFIDENCE box
  Config C/D  image label = class of the highest-FUSION box, fusion = conf * iou_pred

CHECKPOINT: under this reading, Config C comes out byte-identical to Config A for
yolov10n and yolo11s, which is exactly what the paper reports. That happens because
multiplying every box by a positive scalar cannot change the argmax unless boxes of
DIFFERENT classes are competing. If your Config C differs from Config A on those two
variants, your fusion is reading a different quantity -- fix that before running
anything else.
"""
import argparse, json
from pathlib import Path

CLASSES = ["benign", "malignant", "normal"]


def truth_from_labels(labels_dir: Path, stem: str) -> int:
    line = (labels_dir / f"{stem}.txt").read_text().strip().split("\n")[0]
    return int(line.split()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path, help="dir containing weights/best.pt")
    ap.add_argument("--data", required=True, type=Path, help="busi_yolo root")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--medsam-cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    from ultralytics import YOLO
    meta = json.loads((a.run / "metrics.json").read_text()) if (a.run / "metrics.json").exists() else {}
    imgsz = a.imgsz or meta.get("hp", {}).get("imgsz", 640)

    model = YOLO(str(a.run / "weights" / "best.pt"))
    img_dir = a.data / a.split / "images"
    lab_dir = a.data / a.split / "labels"
    paths = sorted(img_dir.glob("*.png"))

    sc = None
    if a.medsam_cache:
        from src.medsam import Scorer
        sc = Scorer(a.medsam_cache)

    preds, n_empty = {}, 0
    for p in paths:
        r = model.predict(str(p), conf=a.conf, iou=a.iou, imgsz=imgsz,
                          max_det=300, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy().tolist()
        confs = r.boxes.conf.cpu().numpy().tolist()
        clss = [int(c) for c in r.boxes.cls.cpu().numpy().tolist()]
        if not boxes:
            n_empty += 1
        ious = None
        if sc is not None and boxes:
            ious, _ = sc.score(p.stem, boxes)
        preds[p.stem] = {"true": truth_from_labels(lab_dir, p.stem),
                         "boxes": boxes, "confs": confs, "clss": clss, "ious": ious}

    out = a.out or (a.run / ("preds_fused.json" if sc else "preds.json"))
    out.write_text(json.dumps({"classes": CLASSES, "split": a.split,
                               "imgsz": imgsz, "conf_thr": a.conf,
                               "n_images": len(preds), "n_empty": n_empty,
                               "fused": sc is not None, "preds": preds}, indent=2))
    print(f"wrote {out}  ({len(preds)} images, {n_empty} with zero detections)")


if __name__ == "__main__":
    main()
