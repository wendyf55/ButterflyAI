"""Blind error review, step 1: draw feeding test photos the final model missed and photos it caught,
and hide which is which. Diagnostic only: it reads the saved test predictions; nothing is trained,
scored or chosen here.

Per source (inat_2024, ontario): N_PER_GROUP feeding photos the model called non-feeding (missed) and
N_PER_GROUP it called feeding (caught), drawn at random. They are shuffled together and copied to
photos/001.jpeg, 002.jpeg, ... so the file names say nothing about the outcome.
I can then tag each photo in tags.csv; the answers are in KEY_do_not_open_until_tagged.csv.
"""

from pathlib import Path
import shutil
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
TEST_DIR = MODULE_DIR / "results" / "test_feeding_final_real_and_super_2026-10-01"
OUT_DIR = MODULE_DIR / "results" / "error_review_2026-10-19"
SOURCES = {"inat_2024": TEST_DIR / "predictions_test1.csv",     # source name -> predictions file
           "ontario": TEST_DIR / "predictions_ontario.csv"}
N_PER_GROUP = 25        # per source x (missed / caught) -> 100 photos
SEED = 2026
TAGS = ["tiny_butterfly", "busy_scene", "mouthparts_not_visible", "unusual_view",
        "blurry_or_dark", "not_on_flower"]
GRADCAM_N, GRADCAM_SEED = 8, 0 

##############################################################################################

OUT_DIR.mkdir(parents=True, exist_ok=False)   # never overwrite an earlier draw
photo_dir = OUT_DIR / "photos"
photo_dir.mkdir()

# Leave out the Ontario photos already seen (with their outcome) in the Grad-CAM sheets
on = pd.read_csv(SOURCES["ontario"])
seen = set()
for outcome in [on["y_pred"] == 0, on["y_pred"] == 1]:
    group = on[(on["y_true"] == 0) & outcome]
    seen |= set(group.sample(min(GRADCAM_N, len(group)), random_state=GRADCAM_SEED)["image_path"])

picked = []
for source, csv in SOURCES.items():
    d = pd.read_csv(csv)
    d = d[(d["source"] == source) & (d["label"] == "F") & ~d["image_path"].isin(seen)]
    for missed, group in [(True, d[d["y_pred"] == 1]), (False, d[d["y_pred"] == 0])]:
        picked.append(group.sample(N_PER_GROUP, random_state=SEED).assign(missed=missed))

key = pd.concat(picked).sample(frac=1, random_state=SEED).reset_index(drop=True)   # shuffle
key.insert(0, "id", [f"{i:03d}" for i in range(1, len(key) + 1)])
assert key["image_path"].is_unique

for r in key.itertuples():
    src = PROJECT_ROOT / r.image_path
    shutil.copy2(src, photo_dir / f"{r.id}{src.suffix}")

key[["id", "image_path", "source", "label", "score", "missed"]].to_csv(
    OUT_DIR / "KEY_do_not_open_until_tagged.csv", index=False)

tags = pd.DataFrame({"id": key["id"]})
for t in TAGS:
    tags[t] = ""          # 1 = yes, leave blank = no
tags["my_call"] = ""      # your own judgement: F / N / unsure
tags["notes"] = ""
tags.to_csv(OUT_DIR / "tags.csv", index=False)

print(f"{len(key)} photos in {photo_dir}")
print(f"Left out {len(seen)} Ontario photos already seen in the Grad-CAM sheets")
print(f"Tag them in {OUT_DIR / 'tags.csv'} (don't open the KEY file until you're done)")