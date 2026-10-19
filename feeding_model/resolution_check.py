"""script to check whether the model does use a resolution shortcut - all non-feeding photos are 
500px, so future 500px photos may indicate to the model that it's more likely to be feeding. 

The iNat and Ontario test photos are 500 px, and that's where the model misses the most feeding.

Diagnostic only: nothing here trains or tunes a model, and the test sets are not used.

Test: the fold-1 model from xval_2026-10-08_1032_folds1 and its held-out real photos (never trained on).
Score each photo as it is, then again after shrinking it. Same photo, same model; only the size changes.

Decision rule (agreed 2026-10-08, before seeing results), feeding recall at 500 px vs original:
  drop >= 5 points (and a smaller drop at 1000 px) -> shortcut supported
  drop < 2 points -> no shortcut
  in between -> weak effect, note it and move on
"""

from pathlib import Path
from datetime import datetime
import os
import json
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy.stats import binomtest
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

##############################################################################################
# Settings

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
FOLD_RUN = MODULE_DIR / "results" / "xval_2026-10-08_1032_folds1"
MODEL_PATH = FOLD_RUN / "model_fold1.keras"
RUN_DIR = MODULE_DIR / "results" / f"resolution_check_{datetime.now():%Y-%m-%d_%H%M}"

IMG_SIZE = (224, 224) # same as training; flow_from_dataframe resizes with nearest-neighbour (its default)
BATCH_SIZE = 32
CLASSES = ['F', 'N'] # F=0, N=1, score = P(non-feeding)
PREDICTION_THRESHOLD = 0.5

# Shrunk copies: long side in px (None = original size, only re-saved as JPEG, to separate size from re-compression).
# Shrinking uses PIL's Lanczos filter and JPEG quality 90, roughly what iNat does. Images already smaller are not enlarged.
CONDITIONS = {"jpeg_q90": None, "1000px": 1000, "500px": 500}
JPEG_QUALITY = 90
SMALL_LONG_SIDE = 600  # "free check": dev pool photos already this small or smaller

##############################################################################################
# Data: held-out real photos of fold 1, with the scores the cross-validation saved

os.chdir(PROJECT_ROOT) # image paths in predictions.csv are relative to the repo root
RUN_DIR.mkdir(parents=True, exist_ok=False)   # never overwrite an earlier run

preds = pd.read_csv(FOLD_RUN / "predictions.csv")
real = preds[preds["photo_type"] == "real"].reset_index(drop=True)
assert real["filename"].map(lambda p: Path(p).name).is_unique, "two photos share a file name; copies would collide"

def long_side(path):
    with Image.open(path) as im:   # reads the header only
        return max(im.size)

real["long_side"] = real["filename"].map(long_side)
print(f"{len(real)} real photos ({(real['label'] == 'F').sum()} F, {(real['label'] == 'N').sum()} N), "
      f"{(real['long_side'] <= SMALL_LONG_SIDE).sum()} already <= {SMALL_LONG_SIDE} px")

##############################################################################################
# Shrunk copies (originals are never touched)

def make_copy(src, dst, max_side):
    with Image.open(src) as im:
        im = im.convert("RGB")
        if max_side is not None and max(im.size) > max_side:
            scale = max_side / max(im.size)
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
        im.save(dst, "JPEG", quality=JPEG_QUALITY)

##############################################################################################
# Scoring, the same way as the validation fold in Final_feeding_model_xval.py

model = tf.keras.models.load_model(MODEL_PATH)
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

def score(paths):
    df = pd.DataFrame({"path": list(paths), "label": real["label"].values})
    gen = datagen.flow_from_dataframe(df, x_col="path", y_col="label", classes=CLASSES,
                                      target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                      class_mode="binary", shuffle=False)
    # flow_from_dataframe silently skips images it can't find; then scores would no longer line up
    assert len(gen.classes) == len(df), "some images were not found"
    return model.predict(gen, verbose=0).ravel()

rows = []
for cond in ["original", *CONDITIONS]:
    if cond == "original":
        paths = real["filename"]
    else:
        out_dir = RUN_DIR / f"images_{cond}"
        out_dir.mkdir()
        paths = []
        for src in real["filename"]:
            dst = out_dir / Path(src).name
            make_copy(src, dst, CONDITIONS[cond])
            paths.append(str(dst))
    print(f"Scoring {cond} ...")
    s = score(paths)
    rows.append(real[["filename", "label", "long_side"]].assign(condition=cond, score=s))

scores_long = pd.concat(rows, ignore_index=True)
scores_long["y_true"] = (scores_long["label"] == "N").astype(int)
scores_long["y_pred"] = (scores_long["score"] > PREDICTION_THRESHOLD).astype(int)
scores_long["correct"] = scores_long["y_true"] == scores_long["y_pred"]
scores_long.to_csv(RUN_DIR / "scores_long.csv", index=False)

# Check: the originals must score exactly as in the cross-validation run (same model, same loading)
orig = scores_long[scores_long["condition"] == "original"].set_index("filename")
max_diff = (orig["score"] - real.set_index("filename")["score"]).abs().max()
print(f"\nOriginals vs saved cross-validation scores: max difference {max_diff:.2e} (should be ~0)")
if max_diff > 1e-4:
    print("WARNING: the originals don't reproduce the saved scores; the comparison below is not reliable")

##############################################################################################
# Summary per condition x label: recall, scores, and photos that flip vs the original (McNemar)

summary = []
for cond, d in scores_long.groupby("condition", sort=False):
    d = d.set_index("filename")
    for label in CLASSES:
        dl = d[d["label"] == label]
        o = orig.loc[dl.index]
        right_to_wrong = int((o["correct"] & ~dl["correct"]).sum())
        wrong_to_right = int((~o["correct"] & dl["correct"]).sum())
        flips = right_to_wrong + wrong_to_right
        summary.append({
            "condition": cond, "label": label, "n": len(dl),
            "recall": dl["correct"].mean(),                       # recall of this class
            "median_score": dl["score"].median(),                 # median P(non-feeding)
            "median_change": (dl["score"] - o["score"]).median(), # per photo, vs original
            "right_to_wrong": right_to_wrong, "wrong_to_right": wrong_to_right,
            "mcnemar_p": binomtest(right_to_wrong, flips, 0.5).pvalue if flips else np.nan,
        })
summary = pd.DataFrame(summary)
summary.to_csv(RUN_DIR / "summary.csv", index=False)
print("\n" + summary.to_string(index=False, float_format="%.3f"))

recall_F = summary.set_index(["condition", "label"])["recall"].xs("F", level="label")
drop_1000 = 100 * (recall_F["original"] - recall_F["1000px"])
drop_500 = 100 * (recall_F["original"] - recall_F["500px"])
if drop_500 >= 5 and drop_1000 < drop_500:
    verdict = "shortcut supported"
elif drop_500 < 2:
    verdict = "no shortcut"
else:
    verdict = "weak effect"
print(f"\nFeeding recall drop vs original: 1000 px {drop_1000:+.1f} points, 500 px {drop_500:+.1f} points -> {verdict}")

##############################################################################################
# Free check: photos that were already small, scored as they are (only a hint: few photos)

small = (orig.assign(small=orig["long_side"] <= SMALL_LONG_SIDE)
             .groupby(["label", "small"])["correct"].agg(n="size", accuracy="mean").reset_index())
small.to_csv(RUN_DIR / "small_photos_check.csv", index=False)
print(f"\nAlready-small photos (<= {SMALL_LONG_SIDE} px) vs the rest, original images:")
print(small.to_string(index=False, float_format="%.3f"))

##############################################################################################
# Run record

run_info = {
    "script": Path(__file__).name,
    "finished": datetime.now().isoformat(timespec="seconds"),
    "model": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
    "model_md5": hashlib.md5(MODEL_PATH.read_bytes()).hexdigest(),
    "fold_run": str(FOLD_RUN.relative_to(PROJECT_ROOT)),
    "n_photos": len(real),
    "conditions": CONDITIONS, "jpeg_quality": JPEG_QUALITY, "resample": "LANCZOS",
    "img_size": IMG_SIZE, "threshold": PREDICTION_THRESHOLD,
    "originals_max_score_diff": float(max_diff),
    "feeding_recall_drop_points": {"1000px": drop_1000, "500px": drop_500},
    "verdict": verdict,
    "tensorflow": tf.__version__,
}
(RUN_DIR / "run_info.json").write_text(json.dumps(run_info, indent=2))
print(f"\nSaved to {RUN_DIR}")