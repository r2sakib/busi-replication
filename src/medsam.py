"""
MedSAM: embedding cache, box-prompted scoring, and the section 4.3 standalone eval.

The ViT-B image encoder is the only expensive part and it does not depend on the
box prompt, so we run it once per image and cache the embedding. Every later
fusion pass over any YOLO variant is then decoder-only and takes seconds.

Runs fine on CPU for 117 test images (~10 min). Do it in a CPU lane and save
your GPU quota for training.

  # once, ~10 min CPU
  python -m src.medsam cache --images /content/busi_yolo/test/images --out /content/medsam_cache

  # section 4.3: oracle upper bound using ground-truth boxes as prompts
  python -m src.medsam standalone --data /content/busi_yolo --cache /content/medsam_cache \
      --src /content/Dataset_BUSI_with_GT --out results/medsam_standalone.json

IMPORTANT -- two different quantities are both called "IoU" in the paper:
  * section 4.3 reports TRUE IoU against ground-truth masks
    (their Dice 0.9396 -> IoU 0.886 checks out exactly)
  * the fusion score uses SAM's PREDICTED-IoU head, which is a self-estimate
    (Fig. 3 shows "IoU=0.7698 | Dice=0.0000" on a normal image -- impossible for
     true IoU, so that figure is reading the predicted head)
This module returns both and never conflates them.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_ID = "wanglab/medsam-vit-base"
CLASSES = ["benign", "malignant", "normal"]


def find_busi_class_dir(src: Path, cls: str) -> Path:
    """Resolve <busi_root>/<cls> regardless of how the Kaggle zip was nested."""
    src = Path(src)
    if (src / cls).is_dir():
        return src / cls
    for d in sorted(src.rglob(cls)):
        if d.is_dir() and any(d.glob("*_mask*.png")):
            return d
    raise SystemExit(f"no '{cls}' folder with masks found under {src}")


def load(device=None):
    from transformers import SamModel, SamProcessor
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SamModel.from_pretrained(MODEL_ID).to(device).eval()
    proc = SamProcessor.from_pretrained(MODEL_ID)
    return model, proc, device


# ---------------------------------------------------------------- cache
@torch.no_grad()
def build_cache(images: Path, out: Path, device=None):
    model, proc, device = load(device)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted([p for p in images.iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")])
    meta = {}
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        inp = proc(img, return_tensors="pt").to(device)
        emb = model.get_image_embeddings(inp["pixel_values"])
        np.save(out / f"{p.stem}.npy", emb.cpu().numpy().astype(np.float16))
        meta[p.stem] = {"original_size": [int(x) for x in inp["original_sizes"][0].tolist()],
                        "reshaped_size": [int(x) for x in inp["reshaped_input_sizes"][0].tolist()],
                        "file": p.name}
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(paths)}")
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"cached {len(paths)} embeddings -> {out}")


class Scorer:
    """Decoder-only scoring against the cached embeddings."""

    def __init__(self, cache: Path, device=None):
        self.model, self.proc, self.device = load(device)
        self.cache = Path(cache)
        self.meta = json.loads((self.cache / "meta.json").read_text())

    @torch.no_grad()
    def score(self, stem, boxes_xyxy, want_masks=False):
        """boxes in ORIGINAL image pixel coords. Returns (iou_pred[], masks|None)."""
        if not boxes_xyxy:
            return [], None
        m = self.meta[stem]
        emb = torch.from_numpy(np.load(self.cache / f"{stem}.npy")).float().to(self.device)
        oh, ow = m["original_size"]
        rh, rw = m["reshaped_size"]
        sx, sy = rw / ow, rh / oh
        scaled = [[[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes_xyxy]]
        bt = torch.tensor(scaled, dtype=torch.float32, device=self.device)

        out = self.model(image_embeddings=emb, input_boxes=bt, multimask_output=False)
        iou = out.iou_scores.squeeze().reshape(-1).cpu().numpy().tolist()
        masks = None
        if want_masks:
            masks = self.proc.image_processor.post_process_masks(
                out.pred_masks.cpu(),
                torch.tensor([[oh, ow]]),
                torch.tensor([[rh, rw]]))[0].numpy()
        return iou, masks


# ---------------------------------------------------------------- section 4.3
def standalone(data: Path, cache: Path, src: Path, out: Path):
    """Oracle upper bound: prompt with ground-truth boxes, score against GT masks.

    Normal images are excluded, exactly as the paper says, because they have no
    lesion mask and Dice is undefined.
    """
    import cv2
    sc = Scorer(cache)
    rows = []
    for lab in sorted((data / "test" / "labels").glob("*.txt")):
        stem = lab.stem
        img_p = data / "test" / "images" / f"{stem}.png"
        img = cv2.imread(str(img_p))
        h, w = img.shape[:2]
        recs = [l.split() for l in lab.read_text().split("\n") if l.strip()]
        cid = int(recs[0][0])
        if CLASSES[cid] == "normal":
            continue
        boxes = []
        for r in recs:
            cx, cy, bw, bh = (float(v) for v in r[1:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
        iou_pred, masks = sc.score(stem, boxes, want_masks=True)

        gt = np.zeros((h, w), bool)
        for mp in sorted((find_busi_class_dir(src, CLASSES[cid])).glob(f"{stem}_mask*.png")):
            g = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if g.shape[:2] != (h, w):
                g = cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST)
            gt |= g > 127
        pred = np.zeros((h, w), bool)
        for mk in masks:
            pred |= mk.reshape(-1, h, w)[0] > 0

        inter = (pred & gt).sum()
        union = (pred | gt).sum()
        rows.append({"stem": stem, "cls": CLASSES[cid],
                     "dice": float(2 * inter / (pred.sum() + gt.sum() + 1e-9)),
                     "iou_true": float(inter / (union + 1e-9)),
                     "iou_pred_head": float(np.mean(iou_pred))})

    res = {"per_image": rows, "summary": {}}
    for c in ("benign", "malignant"):
        sub = [r for r in rows if r["cls"] == c]
        res["summary"][c] = {
            "n": len(sub),
            "dice_mean": float(np.mean([r["dice"] for r in sub])),
            "dice_std": float(np.std([r["dice"] for r in sub])),
            "iou_true_mean": float(np.mean([r["iou_true"] for r in sub])),
            "iou_true_std": float(np.std([r["iou_true"] for r in sub])),
            "iou_pred_head_mean": float(np.mean([r["iou_pred_head"] for r in sub])),
        }
    res["summary"]["global"] = {
        "dice_mean": float(np.mean([r["dice"] for r in rows])),
        "dice_std": float(np.std([r["dice"] for r in rows])),
        "iou_true_mean": float(np.mean([r["iou_true"] for r in rows])),
    }
    res["paper_targets"] = {
        "benign": {"dice": "0.9396 +/- 0.0348", "iou": "0.8879 +/- 0.0580"},
        "malignant": {"dice": "0.9178 +/- 0.0460", "iou": "0.8510 +/- 0.0734"},
        "global": {"dice": "0.9326 +/- 0.0398", "iou": "~0.876"},
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache");      c.add_argument("--images", required=True, type=Path); c.add_argument("--out", required=True, type=Path)
    s = sub.add_parser("standalone"); s.add_argument("--data", required=True, type=Path); s.add_argument("--cache", required=True, type=Path); s.add_argument("--src", required=True, type=Path); s.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if a.cmd == "cache":
        build_cache(a.images, a.out)
    else:
        standalone(a.data, a.cache, a.src, a.out)
