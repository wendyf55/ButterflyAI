# general_feeding_model_eval.py
# evaluates the general feeding model against monarch labels

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report, f1_score)

MONARCH_DIR = Path("data/Monarch_images")  # run from monarch/
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(MONARCH_DIR / "Monarch_image_labels.csv")
df = df[df["Label"].isin(["Feeding", "Non_feeding"])].copy()   # drop any stray/blank rows
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df = df.dropna(subset=["score"])

# Ground truth: Feeding = 1, Non_feeding = 0
y_true = (df["Label"] == "Feeding").astype(int).values
n = len(df); n_feed = int(y_true.sum()); n_non = n - n_feed
print(f"Images with label + score : {n}")
print(f"Feeding : {n_feed} ({n_feed/n:.1%})")
print(f"Non_feeding : {n_non} ({n_non/n:.1%})")
print(f"Majority-class baseline : {max(n_feed, n_non)/n:.4f} accuracy\n")

# Orientation of the raw score: does high score mean Feeding or Non_feeding? 
# we don't actually know, going in, whether a high score from the model means "feeding" or "non-feeding."
auc_raw = roc_auc_score(y_true, df["score"].values)

flip = auc_raw < 0.5
# flip = auc_raw < 0.5. so if the raw AUC comes out below 0.5, 
# # we've caught that the score is oriented toward non-feeding, and we set a flag to correct it.

#figure out which direction the score points, correct it once, and report threshold-independent quality (AUC and AP)
score = (1 - df["score"].values) if flip else df["score"].values
auc = max(auc_raw, 1 - auc_raw)
ap  = average_precision_score(y_true, score)
print(f"AUC (score vs Feeding) : {auc:.4f}   [raw {auc_raw:.4f}, "
      f"{'FLIPPED: high raw score = Non_feeding' if flip else 'high raw score = Feeding'}]")
print(f"Average precision (PR) : {ap:.4f}\n")

def report(y_true, y_pred, title):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / len(y_true)
    print(f"{title}   accuracy {acc:.4f}")
    print(f"                   pred Non_feeding   pred Feeding")
    print(f"true Non_feeding      {tn:6d}          {fp:6d}")
    print(f"true Feeding          {fn:6d}          {tp:6d}")
    print(classification_report(y_true, y_pred,
          target_names=["Non_feeding", "Feeding"], digits=3, zero_division=0))
    return acc, tp, fp, fn, tn

# 1) What the pipeline ACTUALLY did: its stored 0/1 prediction column
#    (aligned to Feeding=1; flip if the score was flipped)
stored = df["prediction"].astype(int).values
stored = (1 - stored) if flip else stored
report(y_true, stored, "Stored pipeline prediction (as-shipped, thr=0.1)")

# 2) Re-derive from the score at a couple of thresholds
for thr in (0.1, 0.5):
    report(y_true, (score > thr).astype(int), f"Score thresholded at {thr}")

# 3) Best threshold by F1
grid = np.linspace(0.01, 0.99, 99)
f1s = [f1_score(y_true, (score > t).astype(int), zero_division=0) for t in grid]
best_i = int(np.argmax(f1s)); best_t = grid[best_i]
print(f"Best-F1 threshold : {best_t:.2f}  (F1 = {f1s[best_i]:.4f})")
report(y_true, (score > best_t).astype(int), f"Score at best-F1 threshold {best_t:.2f}")

#  Save a compact summary 
summary = pd.DataFrame({"threshold": grid, "f1_feeding": f1s})
summary.to_csv(RESULTS_DIR / "model2_transfer_f1_by_threshold.csv", index=False)
with open(RESULTS_DIR / "model2_transfer_eval.md", "w") as f:
    f.write(f"# Model 2 (general feeding model) — transfer to monarchs\n\n")
    f.write(f"- N = {n} ({n_feed} Feeding / {n_non} Non_feeding)\n")
    f.write(f"- AUC = {auc:.4f} (raw {auc_raw:.4f}{', flipped' if flip else ''})\n")
    f.write(f"- Average precision = {ap:.4f}\n")
    f.write(f"- Majority-class baseline accuracy = {max(n_feed,n_non)/n:.4f}\n")
    f.write(f"- Best-F1 threshold = {best_t:.2f} (F1 = {f1s[best_i]:.4f})\n")
print(f"\nSaved: results/model2_transfer_eval.md and results/model2_transfer_f1_by_threshold.csv")