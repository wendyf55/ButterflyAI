# score_old_models.py
#
#sccores the OLD plant models on real photos, before retraining anything.
# This is a diagnosis only: these models will be replaced, because their training and
# validation folds could share a flower, and each OVR file is only the fold-5 model.
#
# Models scored:
#   - the 10 one-vs-rest (OVR) models in models/OVR_models/ (one yes/no model per species)
#   - the July 2025 multiclass model (one model, 10 outputs)
#
# Test sets:
#   - gold:          2,162 real photos, every one is one of the 10 species
#   - mixed_plants:  138 real photos, 24 are one of the 10 species, 114 are other plants
#   - sanity:        20 composites per species from the dev pool. The old models probably saw
#                    these in training, so they should score close to 100%. If they don't,
#                    the label order or image preprocessing is wrong and the other numbers
#                    can't be trusted.
#
# Run from the repo root:  python plant_id_model/score_old_models.py

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import roc_auc_score


#Settings

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent   # image paths in the split CSVs are relative to the repo root
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "plant"
OVR_MODELS_DIR = MODULE_DIR / "models" / "OVR_models"
MULTICLASS_MODEL_FILE = MODULE_DIR / "models" / "Multiclass_detectron_july22.keras"
RESULTS_DIR = MODULE_DIR / "results" / "score_old_models_2026-10-20"

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
THRESHOLD = 0.5  # an OVR model "says yes" when its score is at least this
SANITY_PER_SPECIES = 20
SEED = 321

# The 10 species, in alphabetical order
# This is also the order of the multiclass model's 10 outputs, because flow_from_dataframe
# numbers the classes alphabetically. (The old spelling "Sisybirum" sorts into the same
# place as "Sisymbrium", so the order didn't change.)
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

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def ovr_model_file(species):
    # The old Sisymbrium model file uses the misspelling "Sisybirum"
    file_species = species.replace("Sisymbrium", "Sisybirum")
    return OVR_MODELS_DIR / f"ovr_model_{file_species}.keras"


def load_images(image_paths):
    # Load every image, resized to 224x224.
    # "nearest" resizing matches training (it's ImageDataGenerator's default
    images = []
    for path in image_paths:
        img = load_img(PROJECT_ROOT / path, target_size=(IMG_HEIGHT, IMG_WIDTH), interpolation="nearest")
        images.append(img_to_array(img).astype(np.uint8))   # uint8 keeps memory use low
    return np.array(images)


def predict(model, images):
    # Run the model in batches. preprocess_input is the same ResNet50 preprocessing used in training.
    all_scores = []
    for start in range(0, len(images), BATCH_SIZE):
        batch = images[start:start + BATCH_SIZE].astype(np.float32)
        batch = preprocess_input(batch)
        all_scores.append(model.predict(batch, verbose=0))
    return np.concatenate(all_scores)


# Step 1: Load test sets

gold = pd.read_csv(SPLITS_DIR / "test_gold.csv")
mixed = pd.read_csv(SPLITS_DIR / "test_mixed_plants.csv")
dev_pool = pd.read_csv(SPLITS_DIR / "dev_pool.csv")

# Sanity set: 20 random composites of each species, 200 total
# check that the scoring script is set up correctly
sanity_parts = []
for species in SPECIES:
    composites = dev_pool[(dev_pool["label"] == species) & (dev_pool["kind"] == "composite")]
    sanity_parts.append(composites.sample(SANITY_PER_SPECIES, random_state=SEED))
sanity = pd.concat(sanity_parts).reset_index(drop=True)

test_sets = {"gold": gold, "mixed_plants": mixed, "sanity": sanity}

for name, df in test_sets.items():
    print(f"{name}: {len(df)} images")


# Step 2: load the images

images = {}
for name, df in test_sets.items():
    print(f"Loading images for {name}...")
    images[name] = load_images(df["image_path"])


# 3. score with the 10 ovr models
# Each OVR model gives one score per photo: how likely the photo is its species.
# Saved as a column "ovr_<species>" in each test set.

for species in SPECIES:
    start_time = time.time()
    model = load_model(ovr_model_file(species), compile=False)
    for name in test_sets:
        scores = predict(model, images[name])          # shape (n_photos, 1)
        test_sets[name]["ovr_" + species] = scores[:, 0]
    tf.keras.backend.clear_session()                    # free memory before the next model
    print(f"OVR {species}: done in {time.time() - start_time:.0f} s")


# 4. score with multiclass model (code to generate this model missing currently, 10/20)
# The multiclass model gives 10 scores per photo (one per species, adding up to 1).
# Saved as columns "mc_<species>".

model = load_model(MULTICLASS_MODEL_FILE, compile=False)
for name in test_sets:
    scores = predict(model, images[name])              # shape (n_photos, 10)
    for i, species in enumerate(SPECIES):
        test_sets[name]["mc_" + species] = scores[:, i]
print("Multiclass: done")


#5. Add summary columns and save per-photo predictions

ovr_columns = ["ovr_" + s for s in SPECIES]
mc_columns = ["mc_" + s for s in SPECIES]

for name, df in test_sets.items():
    # OVR: the species whose model gave the highest score, and how many models said yes
    df["ovr_top1"] = df[ovr_columns].idxmax(axis=1).str.replace("ovr_", "")
    df["ovr_max"] = df[ovr_columns].max(axis=1)
    df["ovr_n_yes"] = (df[ovr_columns] >= THRESHOLD).sum(axis=1)
    # Multiclass: the species with the highest score, and that score
    df["mc_top1"] = df[mc_columns].idxmax(axis=1).str.replace("mc_", "")
    df["mc_max"] = df[mc_columns].max(axis=1)

    df.to_csv(RESULTS_DIR / f"predictions_{name}.csv", index=False)


# 6. calculate Metrics
# All results go into a list of lines, which is printed and saved as metrics.md

lines = ["# Old plant models on real photos", ""]

# this 'sanity check'
sanity_ovr_acc = (sanity["label"] == sanity["ovr_top1"]).mean()
sanity_mc_acc = (sanity["label"] == sanity["mc_top1"]).mean()
lines.append(f"Sanity (composites, probably seen in training): OVR {sanity_ovr_acc:.3f}, multiclass {sanity_mc_acc:.3f}")
lines.append("")

#Gold test
# For OVR, the predicted species is the one whose model scored highest.
baseline = gold["label"].value_counts(normalize=True).iloc[0]   # always guessing the most common species
gold_ovr_acc = (gold["label"] == gold["ovr_top1"]).mean()
gold_mc_acc = (gold["label"] == gold["mc_top1"]).mean()
no_model_says_yes = (gold["ovr_n_yes"] == 0).mean()
two_or_more_say_yes = (gold["ovr_n_yes"] >= 2).mean()

lines.append(f"## Gold ({len(gold)} real photos)")
lines.append("")
lines.append(f"- top-1 accuracy: OVR {gold_ovr_acc:.3f}, multiclass {gold_mc_acc:.3f} (always-most-common-species baseline {baseline:.3f})")
lines.append(f"- OVR: no model says yes on {no_model_says_yes:.1%} of photos; 2 or more say yes on {two_or_more_say_yes:.1%}")
lines.append("")
lines.append("Per species. 'top-1 recall' = share of that species' photos given the right species.")
lines.append("'binary' columns = that one OVR model on its own, yes/no at the threshold.")
lines.append("")
lines.append("| species | n | OVR top-1 recall | MC top-1 recall | OVR binary recall | OVR binary precision | OVR AUC |")
lines.append("|---|---|---|---|---|---|---|")

ovr_recalls = []
mc_recalls = []
for species in SPECIES:
    rows = gold[gold["label"] == species]
    ovr_recall = (rows["ovr_top1"] == species).mean()
    mc_recall = (rows["mc_top1"] == species).mean()
    ovr_recalls.append(ovr_recall)
    mc_recalls.append(mc_recall)

    # This species' OVR model on its own
    is_species = gold["label"] == species
    says_yes = gold["ovr_" + species] >= THRESHOLD
    binary_recall = says_yes[is_species].mean()
    if says_yes.sum() > 0:
        binary_precision = is_species[says_yes].mean()
    else:
        binary_precision = float("nan")
    auc = roc_auc_score(is_species, gold["ovr_" + species])

    lines.append(f"| {species} | {len(rows)} | {ovr_recall:.3f} | {mc_recall:.3f} | {binary_recall:.3f} | {binary_precision:.3f} | {auc:.3f} |")

lines.append("")
lines.append(f"Macro recall (average of the per-species recalls; chance = 0.100): OVR {np.mean(ovr_recalls):.3f}, multiclass {np.mean(mc_recalls):.3f}")
lines.append("")

# Mixed plants test set
# Can the models tell "one of our 10 species" from "some other plant"?
target = mixed[mixed["is_target"] == True]
other = mixed[mixed["is_target"] == False]

target_ovr_acc = (target["label"] == target["ovr_top1"]).mean()
target_mc_acc = (target["label"] == target["mc_top1"]).mean()
other_rejected = (other["ovr_n_yes"] == 0).mean()   # correct answer for these photos: no model says yes

lines.append(f"## Mixed plants ({len(target)} target species, {len(other)} other plants)")
lines.append("")
lines.append(f"- target photos given the right species: OVR {target_ovr_acc:.3f}, multiclass {target_mc_acc:.3f}")
lines.append(f"- other-plant photos where no OVR model says yes (correct): {other_rejected:.3f}")
lines.append(f"- average number of OVR models saying yes on an other-plant photo: {other['ovr_n_yes'].mean():.2f}")
lines.append(f"- multiclass top score, median: target {target['mc_max'].median():.3f}, other {other['mc_max'].median():.3f}")
lines.append("  (if 'other' is much lower than 'target', a cut-off on this score could reject other plants)")
lines.append("- which species the OVR models pick for other-plant photos:")
for species, count in other["ovr_top1"].value_counts().items():
    lines.append(f"  - {species}: {count}")

# Save and print
(RESULTS_DIR / "metrics.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))


# 7. Run record
# Which models, settings and versions produced these numbers

model_files = [ovr_model_file(s) for s in SPECIES] + [MULTICLASS_MODEL_FILE]
run_record = {
    "date": time.strftime("%Y-%m-%d %H:%M"),
    "tensorflow_version": tf.__version__,
    "threshold": THRESHOLD,
    "image_size": [IMG_HEIGHT, IMG_WIDTH],
    "resize": "nearest",
    "n_images": {name: len(df) for name, df in test_sets.items()},
    "models": [
        {"file": f.name,
         "bytes": f.stat().st_size,
         "modified": time.strftime("%Y-%m-%d", time.localtime(f.stat().st_mtime))}
        for f in model_files
    ],
}
with open(RESULTS_DIR / "run_record.json", "w") as f:
    json.dump(run_record, f, indent=2)

print(f"\nSaved to {RESULTS_DIR}")