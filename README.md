# BUSI replication runbook

Replication of Boudraa et al. (2026), *Intelligence-Based Medicine* 15:100434.
Target: accuracy 0.8803 (103/117 test images), macro F1 0.8821, malignant F1 0.8254.

## The one thing that decides whether this fits in 40 hours

**Do not rerun the Optuna search.** The paper describes 30 TPE trials x 50 epochs
x 11 variants = 16,500 training epochs, which is >150 T4-hours. Table 2 already
publishes the winning hyperparameters. Every result in the paper is reproducible
by *training with those values*. The search is a claim about provenance, not an
input. Run a reduced search after the deadline as a separate experiment.

Total compute for the full 11-variant reproduction plus the 3-variant ablation:
**~12 T4-hours**, which is 3-5 hours of wall time across 4 lanes. Your bottleneck
is engineering time on the fusion definition, not GPU time.

## Setup (once, ~20 min, CPU only)

```bash
pip install ultralytics==8.3.40 transformers opencv-python-headless pyyaml
python -m src.build_dataset --src /path/Dataset_BUSI_with_GT --out /content/busi_yolo
# -> note the md5. Publish busi_yolo.zip as a GitHub release asset.
python -m src.medsam cache --images /content/busi_yolo/test/images --out medsam_cache
```

Every lane downloads the *same zip* and asserts the *same md5*. If two lanes
regenerate the split independently, the ablation table is comparing models
trained on different data and means nothing.

## Gates — do not proceed past a red gate

**Gate 1 (hour 1).** 5-epoch smoke test loads `yolov10b.pt` and completes.
YOLOv10 support has migrated between the THU-MIG repo and Ultralytics more than
once. Find this out at hour 1, not hour 20.

**Gate 2 (hour 4).** Run Config C for `yolov10n` and `yolo11s`. They must come
out **byte-identical to Config A**. That is what the paper's own Table 4 reports,
and it happens because multiplying every box by a positive scalar cannot change
the argmax unless boxes of different classes are competing. If your Config C
differs, your fusion is reading a different quantity. `src/tables.py` prints this
check automatically.

**Gate 3 (hour 6).** `yolov10b_B` accuracy within ~0.03 of 0.8803. Expect
+/-1-3 test images (~1.5% accuracy) of run-to-run variance from CUDA
nondeterminism. Report mean over 3 seeds for yolov10b or you will spend hours
chasing noise.

## What you'll find (I checked the arithmetic)

The headline numbers are internally consistent. Fig. 6's confusion matrix
(57/6/3, 5/26/0, 0/0/20) reproduces Table 3's yolov10b row exactly; `src/metrics.py`
is unit-tested against it and returns 0.8803 / 0.8821 / 0.8254 / CI [0.811, 0.939]
vs. the paper's [0.8156, 0.9348].

The MedSAM claim does not survive the same check:

| variant  | B (+Optuna) | D (full pipeline) | effect of MedSAM |
|----------|-------------|-------------------|------------------|
| yolov10b | 0.8821      | 0.8821            | exactly zero     |
| yolov10n | 0.8713      | 0.8617            | **-0.0096**      |
| yolo11s  | 0.8556      | 0.8465            | **-0.0091**      |

MedSAM does nothing or actively hurts in all three cases, yet the abstract claims
"independent and complementary contributions of both." Reproducing this honestly
is your contribution.

Two different quantities are both called "IoU". Section 4.3 reports true IoU
against ground-truth masks (Dice 0.9396 -> IoU 0.886 checks out). The fusion score
uses SAM's predicted-IoU *head*: Fig. 3 shows "IoU=0.7698 | Dice=0.0000" on a
normal image, impossible for true IoU. `src/medsam.py` returns both separately and
never conflates them.

## Landmines

1. **BUSI duplicates.** The dataset is known to contain repeats. `build_dataset.py`
   runs a perceptual-hash sweep and flags duplicate pairs straddling the test
   boundary. The paper never checks this. If leakage exists, 88% is inflated.
2. **The normal class gets a full-image box.** That makes it trivially detectable
   and is why normal recall is 100% and AUC 0.992. Do not present it as a strength.
3. **Table 2 contradicts section 3.3.** imgsz=416 for yolov10l is not in the stated
   search space {512, 640, 800}. Table 2 wins because Table 2 produced Table 3.
4. **Provenance errors to correct.** BUSI is from Baheya Hospital, Cairo
   (Al-Dhabyani et al. 2020), not Shiraz. Ultralytics ships COCO-pretrained
   detection weights, not ImageNet.
5. **Table 3's Time column is non-monotonic** in model size (v10n 58.8 min >
   v10m 28.3 min). Early stopping fired at different epochs. Those numbers carry
   no information; `train.py` logs actual epochs run instead.
6. **BUS-UCLM** comes from Mendeley Data with a different mask convention.
   Convert it in the CPU lane at hour 2, not at hour 30.

## Schedule (4 GPU lanes + 1 CPU lane)

| Hours | L0 (CPU)                         | L1                  | L2              | L3              | L4              |
|-------|----------------------------------|---------------------|-----------------|-----------------|-----------------|
| 0-1   | build dataset, publish zip       | Gate 1 smoke test   | —               | —               | —               |
| 1-2   | MedSAM cache, sec 4.3            | v10b A              | v10n A          | 11s A           | v10x B          |
| 2-4   | BUS-UCLM conversion              | v10b B              | v10n B, 11x B   | 11s B, v10l B   | (v10x cont.)    |
| 4-6   | **Gates 2 & 3**, Table 4         | v10b seeds 7, 1337  | (11x cont.)     | 11l B           | v10m, 11m B     |
| 6-8   | Table 3, Figs 5/6, Table 5       | BUS-UCLM inference  | —               | —               | v10s, 11n B     |
| 8-16  | failure analysis, writeup        | buffer / reruns     | buffer          | buffer          | buffer          |
| 16-30 | slides, delta tables             | reduced Optuna (P3) | buffer          | buffer          | buffer          |
| 30-40 | **reserved for things breaking — schedule nothing** |    |                 |                 |                 |

Everything after hour 8 is bonus. The 4x margin is the plan, not slack.
