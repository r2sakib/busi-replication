"""
Build the YOLO-format BUSI dataset exactly once, then ship the zip to every lane.

Run this on a CPU runtime. Never regenerate it per-lane: if two lanes build the
split independently you are comparing models trained on different data and the
ablation table becomes meaningless.

    python -m src.build_dataset --src /content/Dataset_BUSI_with_GT --out /content/busi_yolo

Produces:
    busi_yolo/{train,val,test}/{images,labels}/
    busi_yolo/data.yaml
    busi_yolo/manifest.csv      per-image split assignment + phash
    busi_yolo/dedup_report.csv  near-duplicate groups (the paper never checks this)
    busi_yolo.zip
"""
import argparse, csv, hashlib, json, os, random, re, shutil, zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["benign", "malignant", "normal"]
CLS_ID = {c: i for i, c in enumerate(CLASSES)}
TARGET_PER_CLASS = 306          # section 3.1: 918 training images, balanced
MIN_CONTOUR_AREA = 50


# ---------------------------------------------------------------- discovery
def find_root(start: Path) -> Path:
    """Locate the folder that actually holds benign/ malignant/ normal/.

    Kaggle re-uploads of BUSI nest it differently (sometimes
    Dataset_BUSI_with_GT/, sometimes one level deeper, sometimes not at all),
    so search rather than hard-coding a path.
    """
    if all((start / c).is_dir() for c in CLASSES):
        return start
    for d in sorted(start.rglob("*")):
        if d.is_dir() and all((d / c).is_dir() for c in CLASSES):
            print(f"auto-located BUSI root: {d}")
            return d
    raise SystemExit(
        f"could not find a folder containing {CLASSES} under {start}.\n"
        f"Run:  find {start} -maxdepth 3 -type d\n"
        f"and pass the right path to --src.")


def discover(src: Path):
    """Return {stem: {'cls':, 'img': Path, 'masks': [Path,...]}}."""
    items = {}
    for cls in CLASSES:
        d = src / cls
        if not d.is_dir():
            raise SystemExit(f"missing class dir: {d}")
        for p in sorted(d.glob("*.png")):
            if "_mask" in p.stem:
                continue
            items[p.stem] = {"cls": cls, "img": p, "masks": []}
        for p in sorted(d.glob("*_mask*.png")):
            base = re.sub(r"_mask(_\d+)?$", "", p.stem)
            if base in items:
                items[base]["masks"].append(p)
    return items


# ---------------------------------------------------------------- labels
def boxes_from_masks(rec, w, h):
    """Contour -> normalized YOLO boxes. Normal images get one full-image box."""
    cid = CLS_ID[rec["cls"]]
    if rec["cls"] == "normal":
        return [(cid, 0.5, 0.5, 1.0, 1.0)]
    out = []
    for mp in rec["masks"]:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            out.append((cid, (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
    return out


# ---------------------------------------------------------------- dedup
def phash(img, size=16):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8]
    return "".join("1" if v > np.median(d[1:]) else "0" for v in d.flatten())


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


# ---------------------------------------------------------------- augmentation
def aug_once(img, boxes, rng):
    """Section 3.1 augmentation, bbox-aware. boxes are normalized (cid,cx,cy,w,h)."""
    h, w = img.shape[:2]
    out = img.copy()
    bb = [list(b) for b in boxes]

    if rng.random() < 0.5:                                    # horizontal flip
        out = cv2.flip(out, 1)
        for b in bb:
            b[1] = 1.0 - b[1]
    if rng.random() < 0.5:                                    # vertical flip
        out = cv2.flip(out, 0)
        for b in bb:
            b[2] = 1.0 - b[2]

    ang = rng.uniform(-15, 15)                                # rotation +/-15 deg
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    for b in bb:
        cx, cy, bw, bh = b[1] * w, b[2] * h, b[3] * w, b[4] * h
        corners = np.array([[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2],
                            [cx + bw / 2, cy + bh / 2], [cx - bw / 2, cy + bh / 2]])
        ones = np.hstack([corners, np.ones((4, 1))])
        rot = ones @ M.T
        x0, y0 = rot.min(0)
        x1, y1 = rot.max(0)
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(w), x1), min(float(h), y1)
        b[1], b[2] = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        b[3], b[4] = (x1 - x0) / w, (y1 - y0) / h

    alpha = rng.uniform(0.85, 1.15)                           # contrast
    beta = rng.uniform(-20, 20)                               # brightness
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if rng.random() < 0.5:                                    # gaussian noise
        noise = rng.normal(0, rng.uniform(3, 10), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    bb = [b for b in bb if b[3] > 0.01 and b[4] > 0.01]
    return out, bb


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dedup-threshold", type=int, default=6,
                    help="max phash hamming distance to call two images duplicates")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    nprng = np.random.default_rng(args.seed)
    root = find_root(args.src)
    items = discover(root)
    print(f"discovered {len(items)} images")
    if not 700 < len(items) < 820:
        print(f"  !! expected ~780 images for BUSI, got {len(items)}. "
              f"Wrong dataset version? Check before continuing.")

    # ---- read once, compute labels + phash
    recs = []
    for stem, rec in items.items():
        img = cv2.imread(str(rec["img"]))
        if img is None:
            print(f"  SKIP unreadable {stem}")
            continue
        if rec["cls"] != "normal" and not rec["masks"]:
            print(f"  SKIP no mask {stem}")
            continue
        h, w = img.shape[:2]
        boxes = boxes_from_masks(rec, w, h)
        if not boxes:
            print(f"  SKIP empty mask {stem}")
            continue
        recs.append({"stem": stem, "cls": rec["cls"], "path": rec["img"],
                     "boxes": boxes, "phash": phash(img), "wh": (w, h)})
    print(f"kept {len(recs)} after cleaning")
    for c in CLASSES:
        n = sum(r["cls"] == c for r in recs)
        print(f"  {c:10s} {n:4d}  ({n/len(recs)*100:.1f}%)")

    # ---- near-duplicate report (BUSI is known to contain repeats; paper ignores this)
    dup_rows, seen = [], []
    for r in recs:
        for s in seen:
            d = hamming(r["phash"], s["phash"])
            if d <= args.dedup_threshold:
                dup_rows.append({"a": s["stem"], "a_cls": s["cls"],
                                 "b": r["stem"], "b_cls": r["cls"], "hamming": d})
        seen.append(r)
    print(f"near-duplicate pairs at hamming<={args.dedup_threshold}: {len(dup_rows)}")

    # ---- stratified 70/15/15, seed 42
    by_cls = defaultdict(list)
    for r in recs:
        by_cls[r["cls"]].append(r)
    split = {}
    for c, rs in by_cls.items():
        rs = sorted(rs, key=lambda x: x["stem"])
        rng.shuffle(rs)
        n = len(rs)
        n_tr, n_va = int(round(0.70 * n)), int(round(0.15 * n))
        for i, r in enumerate(rs):
            split[r["stem"]] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    counts = defaultdict(lambda: defaultdict(int))
    for r in recs:
        counts[split[r["stem"]]][r["cls"]] += 1
    for s in ("train", "val", "test"):
        print(f"  {s:5s} " + "  ".join(f"{c}={counts[s][c]}" for c in CLASSES)
              + f"  total={sum(counts[s].values())}")

    # ---- leakage check: duplicates straddling the train/test boundary
    leak = [d for d in dup_rows
            if {split[d["a"]], split[d["b"]]} & {"test"} and split[d["a"]] != split[d["b"]]]
    if leak:
        print(f"  !! {len(leak)} duplicate pairs straddle the test boundary "
              f"-- this inflates test accuracy. See dedup_report.csv")

    # ---- write
    for s in ("train", "val", "test"):
        (args.out / s / "images").mkdir(parents=True, exist_ok=True)
        (args.out / s / "labels").mkdir(parents=True, exist_ok=True)

    def write(stem, s, img, boxes):
        cv2.imwrite(str(args.out / s / "images" / f"{stem}.png"), img)
        with open(args.out / s / "labels" / f"{stem}.txt", "w") as f:
            for cid, cx, cy, bw, bh in boxes:
                f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    train_by_cls = defaultdict(list)
    for r in recs:
        s = split[r["stem"]]
        img = cv2.imread(str(r["path"]))
        write(r["stem"], s, img, r["boxes"])
        if s == "train":
            train_by_cls[r["cls"]].append((r["stem"], img, r["boxes"]))

    # ---- oversample train only, to 306/class (never touch val/test)
    for c, pool in train_by_cls.items():
        need = TARGET_PER_CLASS - len(pool)
        if need <= 0:
            print(f"  {c}: {len(pool)} train originals, no oversampling "
                  f"(paper caps at {TARGET_PER_CLASS}; extras kept)")
            continue
        for i in range(need):
            stem, img, boxes = pool[i % len(pool)]
            a_img, a_box = aug_once(img, boxes, nprng)
            if a_box:
                write(f"{stem}__aug{i}", "train", a_img, a_box)
        print(f"  {c}: {len(pool)} -> {len(pool) + need} train images")

    # ---- artifacts
    with open(args.out / "manifest.csv", "w", newline="") as f:
        wri = csv.DictWriter(f, ["stem", "cls", "split", "phash", "w", "h", "n_boxes"])
        wri.writeheader()
        for r in recs:
            wri.writerow({"stem": r["stem"], "cls": r["cls"], "split": split[r["stem"]],
                          "phash": r["phash"], "w": r["wh"][0], "h": r["wh"][1],
                          "n_boxes": len(r["boxes"])})
    with open(args.out / "dedup_report.csv", "w", newline="") as f:
        wri = csv.DictWriter(f, ["a", "a_cls", "b", "b_cls", "hamming"])
        wri.writeheader()
        wri.writerows(dup_rows)

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: train/images\nval: val/images\n"
        f"test: test/images\nnc: 3\nnames: {CLASSES}\n")

    zpath = args.out.parent / f"{args.out.name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in args.out.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(args.out.parent))
    md5 = hashlib.md5(zpath.read_bytes()).hexdigest()
    print(f"\nwrote {zpath}  md5={md5}")
    print("EVERY LANE MUST USE THIS EXACT MD5.")


if __name__ == "__main__":
    main()
