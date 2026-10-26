# Plant ID multiclass model: 5-fold cross validation
#
#estimates how well the multiclass training recipe works (mean ± SD over 5 folds).
# The fold models are thrown away. The final model is trained on all the data with the same
# recipe in a separate script, and only that model is scored on the test sets (gold, mixed plants).
#
# Recipe = the July 2025 multiclass model (read from Multiclass_detectron_july22.keras):
#   ResNet50 -> average pooling -> Dropout 0.5 -> Dense 512 ReLU -> Dropout 0.5 -> Dense 10 softmax,
#   only the conv5 block trainable, Adam with learning rate 1e-5.
# Changes from how the old models were trained:
#   - folds come from dev_pool.csv (grouped: a plain flower photo and its composites share a fold)
#   - a fresh model every fold
#   - fixed number of epochs, nothing chosen on the validation fold (no early stopping)
#   - every validation image's 10 scores are saved, and results are reported separately
#     for composites and plain photos
#
# Run from the repo root:  python plant_id_model/Plant_multiclass_xval.py

import os
import sys
import json
import random
import hashlib
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


###############################################################
# Settings

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent  # image paths in dev_pool.csv are relative to the repo root
TRAINING_CSV = PROJECT_ROOT / "data" / "splits" / "plant" / "dev_pool.csv"
RESULTS_DIR = MODULE_DIR / "results"  # every run gets its own folder in here

USE_KINDS = ["composite", "plain"]  # which dev-pool images to use
SEED_VALUE = 321
GENERATOR_SEED = 123
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
K_FOLDS = 5  # folds are fixed in dev_pool.csv (column `fold`, 0-4)
EPOCHS = 10    # fixed; the final model uses the same number
LEARNING_RATE = 1e-5   # same as the July model
DROPOUT_RATE = 0.5
DENSE_UNITS = 512
TRAINABLE_BLOCK = "conv5_block"   # only this part of ResNet50 is trained

# Fixed class order (alphabetical). Passing this to the generators means output 0 is always
# Achillea, output 1 is always Anaphalis, and so on, in every run and every script.
SPECIES = [
    "Achillea_millefolium",
    "Anaphalis_margaritacea",
    "Apocynum_androsaemifolium",
    "Asclepias_speciosa",
    "Cirsium_arvense",
    "Ericameria_nauseosa",
    "Leucanthemum_vulgare",
    "Medicago_sativa",
    "Sisymbrium_loeselii",
    "Tripleurospermum_inodorum",
]

# Quick pipeline check (set to False for the real run)
SMOKE_TEST = False     # True = 2 folds, 2 epochs, 8 images per species per fold. The numbers mean nothing
SMOKE_FOLDS = 2
SMOKE_EPOCHS = 2
SMOKE_IMAGES = 8

if SMOKE_TEST:
    K_FOLDS = SMOKE_FOLDS
    EPOCHS = SMOKE_EPOCHS


###############################################################
# Run folder and log

run_name = f"multiclass_xval_{datetime.now():%Y-%m-%d_%H%M}" + ("_smoke" if SMOKE_TEST else "")
RUN_DIR = RESULTS_DIR / run_name
RUN_DIR.mkdir(parents=True, exist_ok=False)   # never overwrite an earlier run

os.chdir(PROJECT_ROOT)   # so the relative image paths in the CSV work


class Tee:
    """Send everything printed to the console AND to log.txt (same as the feeding script)."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for stream in self.streams:
            stream.write(text)
    def flush(self):
        for stream in self.streams:
            stream.flush()
    def __getattr__(self, name):
        return getattr(self.streams[0], name)

sys.stdout = Tee(sys.__stdout__, open(RUN_DIR / "log.txt", "w"))
print(f"Run folder: {RUN_DIR}")


###############################################################
# Load the data

all_df = pd.read_csv(TRAINING_CSV)
all_df = all_df[all_df["kind"].isin(USE_KINDS)].reset_index(drop=True)

if SMOKE_TEST:
    # the same few images every time: the first SMOKE_IMAGES of each fold x species
    # half composites, half plain photos, so both kinds go through the pipeline
    all_df = all_df.groupby(["fold", "label", "kind"]).head(SMOKE_IMAGES // 2).reset_index(drop=True)
    print(f"SMOKE TEST: {len(all_df)} images, {K_FOLDS} folds, {EPOCHS} epochs. The numbers mean nothing")

print(f"{len(all_df)} images")
print(all_df.groupby(["label", "kind"]).size().unstack())


###############################################################
# Run record: code version, software, settings, data file (for reproducibility)

def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()

run_info = {
    "script": Path(__file__).name,
    "started": datetime.now().isoformat(timespec="seconds"),
    "git_commit": git("rev-parse", "HEAD"),
    "git_uncommitted_changes": git("status", "--porcelain", "--untracked-files=no").splitlines(),
    "python": platform.python_version(),
    "tensorflow": tf.__version__,
    "machine": platform.platform(),
    "gpus": [gpu.name for gpu in tf.config.get_visible_devices("GPU")],
    "config": {"use_kinds": USE_KINDS, "seed": SEED_VALUE, "img_size": [IMG_HEIGHT, IMG_WIDTH],
               "batch_size": BATCH_SIZE, "epochs": EPOCHS, "k_folds": K_FOLDS,
               "learning_rate": LEARNING_RATE, "dropout": DROPOUT_RATE, "dense_units": DENSE_UNITS,
               "trainable_block": TRAINABLE_BLOCK, "species": SPECIES, "smoke_test": SMOKE_TEST},
    "data": {"training_csv": str(TRAINING_CSV.relative_to(PROJECT_ROOT)),
             "training_csv_md5": md5(TRAINING_CSV),
             "n_images": len(all_df)},
}

def save_run_info():
    with open(RUN_DIR / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

save_run_info()

if not run_info["gpus"]:
    print("WARNING: no GPU visible, training on the CPU")


###############################################################
# Helper functions

def set_seeds():
    # Reset all random number generators, so every fold starts from the same place
    os.environ["PYTHONHASHSEED"] = str(SEED_VALUE)
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    tf.random.set_seed(SEED_VALUE)


def build_model():
    # Same architecture as the July 2025 multiclass model
    base_model = ResNet50(include_top=False, weights="imagenet")
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(DENSE_UNITS, activation="relu")(x)
    x = Dropout(DROPOUT_RATE)(x)
    outputs = Dense(len(SPECIES), activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    # Freeze everything except the last block of ResNet50
    for layer in base_model.layers:
        layer.trainable = TRAINABLE_BLOCK in layer.name

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def score_fold(fold, pred_df):
    # Accuracy, macro recall and per-species recall: for all images, composites only, plain only
    rows = []
    for subset in ["all", "composite", "plain"]:
        if subset == "all":
            d = pred_df
        else:
            d = pred_df[pred_df["kind"] == subset]
        if len(d) == 0:
            continue

        row = {"fold": fold, "subset": subset, "n": len(d),
               "accuracy": (d["label"] == d["predicted"]).mean()}
        recalls = []
        for species in SPECIES:
            species_rows = d[d["label"] == species]
            recall = (species_rows["predicted"] == species).mean()   # share of this species called correctly
            row["recall_" + species] = recall
            recalls.append(recall)
        row["macro_recall"] = np.mean(recalls)   # average over species, so big species don't dominate
        rows.append(row)
    return rows


###############################################################
# Image generators
# Training folds get the same augmentation as the old plant scripts (flip + shear)
# Validation images are not augmented.

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   horizontal_flip=True,
                                   shear_range=0.2)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


###############################################################
# Cross-validation loop: one pass per fold (build, train, predict, score)

all_predictions = []   # one row per validation image, all folds
fold_metrics = []  # one row per fold x subset

for fold in range(K_FOLDS):
    print(f"\n===== Fold {fold + 1}/{K_FOLDS} =====")

    # Fresh start for each fold, so folds are independent
    tf.keras.backend.clear_session()
    set_seeds()
    model = build_model()

    # Validation = this fold, training = the other 4
    train_df = all_df[all_df["fold"] != fold]
    val_df = all_df[all_df["fold"] == fold].reset_index(drop=True)
    print(f"Training on {len(train_df)} images, validating on {len(val_df)}")

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col="image_path",
        y_col="label",
        classes=SPECIES,               # fixed class order
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=GENERATOR_SEED,
    )
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col="image_path",
        y_col="label",
        classes=SPECIES,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,   # keep the order, so predictions line up with val_df
    )

    # flow_from_dataframe silently skips missing images, which would misalign the predictions
    assert len(val_generator.classes) == len(val_df), "some validation images were not found"

    history = model.fit(train_generator,
                        epochs=EPOCHS,
                        validation_data=val_generator, # only for watching the curves, nothing is chosen on it
                        verbose=2)   # one line per epoch
    pd.DataFrame(history.history).rename_axis("epoch").to_csv(RUN_DIR / f"history_fold{fold + 1}.csv")

    # Predict every validation image with the last-epoch weights
    scores = model.predict(val_generator, verbose=0)    # shape (n_images, 10)

    fold_pred = val_df[["image_path", "label", "kind", "group", "fold"]].copy()
    for i, species in enumerate(SPECIES):
        fold_pred["score_" + species] = scores[:, i]
    fold_pred["predicted"] = [SPECIES[i] for i in scores.argmax(axis=1)]
    fold_pred["top_score"] = scores.max(axis=1)   # for choosing an "other plant" cut-off later
    all_predictions.append(fold_pred)

    # Save after every fold, so a crash keeps the finished folds
    pd.concat(all_predictions).to_csv(RUN_DIR / "predictions.csv", index=False)

    this_fold_rows = score_fold(fold + 1, fold_pred)   # one row per subset: all, composite, plain
    fold_metrics.extend(this_fold_rows)
    pd.DataFrame(fold_metrics).to_csv(RUN_DIR / "metrics_per_fold.csv", index=False)

    m = this_fold_rows[0]   # the first row is always "all images"
    print(f"Fold {fold + 1} done. Accuracy {m['accuracy']:.4f}, macro recall {m['macro_recall']:.4f}")


###############################################################
# Summary over the folds (mean and SD)

metrics_df = pd.DataFrame(fold_metrics)
metric_cols = ["accuracy", "macro_recall"] + ["recall_" + s for s in SPECIES]
summary = metrics_df.groupby("subset", sort=False)[metric_cols].agg(["mean", "std"])
summary.to_csv(RUN_DIR / "metrics_summary.csv")

print(f"\nCross-validation results ({K_FOLDS} folds, mean ± SD):")
for subset in summary.index:
    print(f"\n  {subset}")
    for col in metric_cols:
        mean = summary.loc[subset, (col, "mean")]
        sd = summary.loc[subset, (col, "std")]
        print(f"    {col:<38} {mean:.4f} ± {sd:.4f}")

# Confusion matrix over all folds (rows = true species, columns = predicted)
preds = pd.concat(all_predictions, ignore_index=True)   # renumber rows 0..n-1 (each fold starts at 0)
confusion = pd.crosstab(preds["label"], preds["predicted"]).reindex(index=SPECIES, columns=SPECIES, fill_value=0)
confusion.to_csv(RUN_DIR / "confusion_matrix.csv")

run_info["finished"] = datetime.now().isoformat(timespec="seconds")
save_run_info()
print(f"\nSaved to {RUN_DIR}")