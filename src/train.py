"""
Train one variant under one config. Idempotent and disconnect-safe.

    python -m src.train --variant yolov10b --config B

Re-running after a Colab disconnect is a no-op if the run already finished, and
resumes from the synced last.pt otherwise. You should be able to mash the same
cell repeatedly without thinking.

Config A = Ultralytics stock hyperparameters (paper's "YOLO only" baseline)
Config B = Table 2 hyperparameters (paper's "+Optuna")
Configs C and D are inference-only; see src/fuse.py.
"""
import argparse, json, os, shutil, time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def sync(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("weights/last.pt", "weights/best.pt", "results.csv", "args.yaml"):
        p = src / name
        if p.exists():
            q = dst / name
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--config", required=True, choices=["A", "B"])
    ap.add_argument("--data", default="/content/busi_yolo/data.yaml")
    ap.add_argument("--project", default="/content/runs")
    ap.add_argument("--mirror", default="/content/drive/MyDrive/busi-repro/results",
                    help="durable storage; survives runtime death")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--sync-every", type=int, default=5)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "variants.yaml").read_text())
    if args.variant not in cfg["variants"]:
        raise SystemExit(f"unknown variant {args.variant}")

    hp = dict(cfg["common"])
    hp.update(cfg["defaults"] if args.config == "A"
              else {k: v for k, v in cfg["variants"][args.variant].items()
                    if k in ("lr0", "momentum", "weight_decay", "optimizer", "imgsz", "batch")})
    hp["seed"] = args.seed
    if args.epochs:
        hp["epochs"] = args.epochs

    tag = f"{args.variant}_{args.config}" + (f"_s{args.seed}" if args.seed != 42 else "")
    local = Path(args.project) / tag
    mirror = Path(args.mirror) / tag

    # ---- already finished?
    if (mirror / "metrics.json").exists():
        print(f"[{tag}] already complete in mirror, nothing to do.")
        return

    # ---- pull any partial run back down so Ultralytics can resume
    resume = False
    if (mirror / "weights" / "last.pt").exists():
        local.mkdir(parents=True, exist_ok=True)
        (local / "weights").mkdir(exist_ok=True)
        for name in ("weights/last.pt", "results.csv", "args.yaml"):
            if (mirror / name).exists():
                shutil.copy2(mirror / name, local / name)
        resume = True
        print(f"[{tag}] resuming from mirrored last.pt")

    from ultralytics import YOLO
    import ultralytics
    print(f"ultralytics {ultralytics.__version__}")

    model = YOLO(str(local / "weights" / "last.pt") if resume else f"{args.variant}.pt")

    # periodic sync so a disconnect costs at most --sync-every epochs
    state = {"t0": time.time()}

    def on_epoch_end(trainer):
        if trainer.epoch % args.sync_every == 0:
            sync(Path(trainer.save_dir), mirror)

    model.add_callback("on_fit_epoch_end", on_epoch_end)

    model.train(data=args.data, project=args.project, name=tag, exist_ok=True,
                resume=resume, plots=True, val=True, **hp)

    elapsed = (time.time() - state["t0"]) / 60.0

    # ---- detection metrics on the held-out test split (not val)
    m = YOLO(str(local / "weights" / "best.pt"))
    res = m.val(data=args.data, split="test", imgsz=hp["imgsz"], plots=False)
    out = {"variant": args.variant, "config": args.config, "seed": args.seed,
           "hp": hp, "wall_min": round(elapsed, 1),
           "map50": float(res.box.map50), "map5095": float(res.box.map),
           "epochs_run": int(getattr(model.trainer, "epoch", -1)) + 1}
    (local / "metrics.json").write_text(json.dumps(out, indent=2))
    sync(local, mirror)
    shutil.copy2(local / "metrics.json", mirror / "metrics.json")
    print(json.dumps(out, indent=2))
    print(f"[{tag}] done in {elapsed:.1f} min -> {mirror}")


if __name__ == "__main__":
    main()
