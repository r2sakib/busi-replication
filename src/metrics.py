"""
Everything downstream of preds.json. No torch, no ultralytics, runs anywhere.

    python -m src.metrics --preds results/yolov10b_B/preds.json --mode yolo
    python -m src.metrics --preds results/yolov10b_B/preds_fused.json --mode fusion
"""
import argparse, json
from pathlib import Path

import numpy as np

CLASSES = ["benign", "malignant", "normal"]
FALLBACK = 2  # zero detections -> call it normal; the rate is reported


def image_scores(rec, mode):
    """Per-class score vector for one image, plus the argmax label."""
    s = np.zeros(len(CLASSES))
    boxes_present = bool(rec["boxes"])
    if boxes_present:
        ious = rec.get("ious")
        for i, (c, cf) in enumerate(zip(rec["clss"], rec["confs"])):
            v = cf if mode == "yolo" else cf * (ious[i] if ious else 1.0)
            s[c] = max(s[c], v)
    if s.sum() == 0:
        s[FALLBACK] = 1e-9
        return s, FALLBACK, False
    return s, int(np.argmax(s)), boxes_present


def confusion(y, p, k=3):
    m = np.zeros((k, k), int)
    for a, b in zip(y, p):
        m[a, b] += 1
    return m


def prf(m):
    out = []
    for i in range(m.shape[0]):
        tp = m[i, i]
        prec = tp / m[:, i].sum() if m[:, i].sum() else 0.0
        rec = tp / m[i, :].sum() if m[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append({"precision": prec, "recall": rec, "f1": f1})
    return out


def auc_ovr(y, S, cls):
    """One-vs-rest AUC by rank statistic. No sklearn dependency."""
    s = S[:, cls]
    pos = s[np.array(y) == cls]
    neg = s[np.array(y) != cls]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks over ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        idx = np.where(allv == v)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def evaluate(preds_path, mode, n_boot=1000, seed=42):
    d = json.loads(Path(preds_path).read_text())
    stems = sorted(d["preds"])
    y, p, S, had_box = [], [], [], []
    for st in stems:
        rec = d["preds"][st]
        s, lab, hb = image_scores(rec, mode)
        y.append(rec["true"]); p.append(lab); S.append(s); had_box.append(hb)
    y, p, S = np.array(y), np.array(p), np.vstack(S)

    m = confusion(y, p)
    per = prf(m)
    res = {
        "mode": mode,
        "n": len(y),
        "n_no_detection": int(len(y) - sum(had_box)),
        "accuracy": float((y == p).mean()),
        "n_correct": int((y == p).sum()),
        "f1_macro": float(np.mean([c["f1"] for c in per])),
        "confusion": m.tolist(),
        "per_class": {CLASSES[i]: {k: float(v) for k, v in per[i].items()}
                      for i in range(3)},
        "auc": {CLASSES[i]: auc_ovr(y, S, i) for i in range(3)},
    }

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        boot.append(np.mean([c["f1"] for c in prf(confusion(y[idx], p[idx]))]))
    res["f1_macro_ci95"] = [float(np.percentile(boot, 2.5)),
                            float(np.percentile(boot, 97.5))]
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--mode", default="yolo", choices=["yolo", "fusion"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = evaluate(a.preds, a.mode)
    print(json.dumps(r, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=2))
