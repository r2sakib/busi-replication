"""
Assemble Tables 3 and 4 from whatever runs have landed so far, with a delta column
against the published values. Safe to run at any time; missing runs show as "--".

    python -m src.tables --results results --out report
"""
import argparse, json
from pathlib import Path

import yaml

from src.metrics import evaluate

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x",
         "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]
ABL = ["yolov10b", "yolov10n", "yolo11s"]


def md(rows, header):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (x + 2) for x in w) + "|"]
                     + [line(r) for r in rows])


def load(run: Path, fused: bool):
    f = run / ("preds_fused.json" if fused else "preds.json")
    if not f.exists():
        return None
    return evaluate(f, "fusion" if fused else "yolo")


def fmt(v, ref=None):
    if v is None:
        return "--"
    s = f"{v:.4f}"
    if ref is not None:
        s += f" ({v-ref:+.4f})"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", type=Path)
    ap.add_argument("--out", default="report", type=Path)
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "variants.yaml").read_text())
    a.out.mkdir(parents=True, exist_ok=True)

    # ---------------- Table 3
    rows = []
    for v in ORDER:
        run = a.results / f"{v}_B"
        r = load(run, fused=False)
        pt = cfg["paper_table3"][v]
        mj = json.loads((run / "metrics.json").read_text()) if (run / "metrics.json").exists() else {}
        if r is None:
            rows.append([v] + ["--"] * 7)
            continue
        rows.append([v,
                     fmt(r["accuracy"], pt["acc"]),
                     fmt(r["f1_macro"], pt["f1_macro"]),
                     fmt(r["per_class"]["benign"]["f1"], pt["f1_benign"]),
                     fmt(r["per_class"]["malignant"]["f1"], pt["f1_malignant"]),
                     fmt(r["per_class"]["normal"]["f1"], pt["f1_normal"]),
                     fmt(mj.get("map50"), pt["map50"]),
                     f"{mj.get('wall_min','--')} / {mj.get('epochs_run','--')}ep"])
    t3 = md(rows, ["Model", "Accuracy", "F1 macro", "F1 benign", "F1 malig",
                   "F1 normal", "mAP@0.5", "Time/Epochs"])

    # ---------------- Table 4
    rows4 = []
    for v in ABL:
        for cfg_name, run_name, fused in [("A: YOLO only", f"{v}_A", False),
                                          ("B: +Optuna", f"{v}_B", False),
                                          ("C: +MedSAM", f"{v}_A", True),
                                          ("D: full pipeline", f"{v}_B", True)]:
            r = load(a.results / run_name, fused=fused)
            key = cfg_name[0]
            pt = cfg["paper_table4"][v][key]
            if r is None:
                rows4.append([v, cfg_name] + ["--"] * 5)
                continue
            rows4.append([v, cfg_name,
                          fmt(r["accuracy"], pt["acc"]),
                          fmt(r["f1_macro"], pt["f1_macro"]),
                          fmt(r["per_class"]["malignant"]["f1"], pt["f1_malignant"]),
                          f"[{r['f1_macro_ci95'][0]:.4f}-{r['f1_macro_ci95'][1]:.4f}]",
                          r["n_no_detection"]])
    t4 = md(rows4, ["Model", "Config", "Acc.", "F1 macro", "F1 malig",
                    "CI95 F1 macro", "no-det"])

    # ---------------- the checkpoint that validates your fusion implementation
    checks = []
    for v in ("yolov10n", "yolo11s"):
        A = load(a.results / f"{v}_A", False)
        C = load(a.results / f"{v}_A", True)
        if A and C:
            same = abs(A["f1_macro"] - C["f1_macro"]) < 1e-9 and A["accuracy"] == C["accuracy"]
            checks.append(f"- {v}: Config C {'==' if same else '!='} Config A "
                          f"{'(matches paper, fusion implementation validated)' if same else '(DIVERGES from paper -- inspect fusion)'}")

    body = ("# Replication report\n\nValues are mine, `(delta)` is mine minus the paper's.\n\n"
            "## Table 3\n\n" + t3 + "\n\n## Table 4 (ablation)\n\n" + t4 +
            "\n\n## Fusion implementation checkpoint\n\n" + ("\n".join(checks) or "- pending") + "\n")
    (a.out / "report.md").write_text(body)
    print(body)


if __name__ == "__main__":
    main()
