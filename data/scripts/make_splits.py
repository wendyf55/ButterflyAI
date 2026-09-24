"""
make_splits.py -- build every split CSV in data/splits/ from data/images/ + data/metadata/.

    python data/scripts/make_splits.py

Writes (image_path is relative to the repo root):
  data/splits/feeding/dev_pool.csv        training + validation, with a `fold` column (0-4)
  data/splits/feeding/test1.csv           BIMBY-2024 + gold feeding labels, with a `source` column
  data/splits/feeding/test2_ontario.csv   Ontario
  data/splits/plant/dev_pool.csv          composites + plain flowers, with a `fold` column
  data/splits/plant/test_gold.csv         gold plant labels (only the 10 target species)
  data/splits/plant/test_mixed_plants.csv BIMBY-2024 photos with a plant ID, mostly NOT a target species
  data/splits/monarch/dev_pool.csv        monarch photos, with a `fold` column
  data/splits/removed.csv                 every row left out of a split, and why

Rules (see data/README.md):
  - Rows are grouped so related images always share a fold: a superimposed image with its source
    photo, a plain flower with its composites, and byte-identical files with each other.
  - Test rows are dropped if the image is also in that model's development pool.
"""
import hashlib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
IMG = DATA / "images"
META = DATA / "metadata"
OUT = DATA / "splits"
N_FOLDS = 5
SEED = 42

SAME_SESSION = {"68Dxllirdq.jpg", "esR4jtAvxb.jpg", "hm7H0EPpUq.jpg", "NALL3X6ZZE.jpg"}  # Block B photos
PLANT_NAMES = {"Circium_arvense": "Cirsium_arvense",
               "Sisybirum_loeselii": "Sisymbrium_loeselii",
               "Sisymbrium loeselii": "Sisymbrium_loeselii"}
# plant IDs that could be one of the 10 target species (genus/tribe-level) -> left out of test_mixed_plants
PLANT_AMBIGUOUS = {"Apocynum", "Achillea", "Astereae", "Tracheophyta"}

_hash_cache = {}
removed = []   # rows dropped from any split


def md5(rel_path):
    if rel_path not in _hash_cache:
        p = REPO / rel_path
        _hash_cache[rel_path] = hashlib.md5(p.read_bytes()).hexdigest() if p.is_file() else None
    return _hash_cache[rel_path]


def rel(folder, name):
    return str((IMG / folder / str(name)).relative_to(REPO))


def plant_name(s):
    s = str(s).strip()
    return PLANT_NAMES.get(s, s.replace(" ", "_"))


def drop(df, mask, split, why):
    """Remove rows where mask is True, remembering them in `removed`."""
    gone = df[mask].copy()
    if len(gone):
        gone["split"], gone["reason"] = split, why
        removed.append(gone)
    return df[~mask].copy()


def clean(df, split, forbidden_hashes=frozenset(), forbidden_reason=""):
    """Drop missing images, images in forbidden_hashes, and repeated images."""
    df["hash"] = df["image_path"].map(md5)
    df = drop(df, df["hash"].isna(), split, "missing_image")
    df = drop(df, df["hash"].isin(forbidden_hashes), split, forbidden_reason)
    df = drop(df, df["hash"].duplicated(), split, "duplicate")
    return df


def merge_groups(df):
    """Byte-identical files join the same group (first group seen wins)."""
    first = df.groupby("hash")["group"].transform("first")
    return df.assign(group=first)


def add_folds(df, strat_col):
    df = df.reset_index(drop=True)
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, idx) in enumerate(sgkf.split(df, df[strat_col], groups=df["group"])):
        df.loc[idx, "fold"] = k
    return df


def save(df, name, cols):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(path, index=False)
    print(f"  {name}: {len(df)} rows")


# ---------------------------------------------------------------- feeding
print("feeding")
pool = pd.read_csv(META / "bimby_collection" / "DataFilenamesRedo.csv")
pool["image_path"] = [rel("bimby_superimposed" if t == "super" else "bimby_real", f)
                      for f, t in zip(pool["filename"], pool["photo_type"])]
real_stems = set(pool.loc[pool["photo_type"] == "real", "filename"].str.rsplit(".", n=1).str[0])
def feeding_group(f):
    if f.startswith("superimposed_"):
        stem = f[len("superimposed_"):].rsplit(".", 1)[0]
        return stem if stem in real_stems else f
    return f.rsplit(".", 1)[0]
pool["group"] = pool["filename"].map(feeding_group)
pool["hash"] = pool["image_path"].map(md5)
pool = merge_groups(pool)
pool["strat"] = pool["label"] + "_" + pool["photo_type"]
pool = add_folds(pool, "strat")
save(pool, "feeding/dev_pool.csv", ["image_path", "label", "photo_type", "group", "fold"])
feeding_hashes = set(pool["hash"])

bimby = pd.read_csv(META / "bimby2024" / "BC2024_goldstandard_appendfilenames2.csv")
bimby["source"] = bimby["place_state_name"].eq("British Columbia").map(
    {True: "inat_2024", False: "bimby_collection"})
bimby["image_path"] = bimby["FileName"].map(lambda f: rel("bimby2024", f))
bimby["label"] = bimby["feeding_status"]
gold = pd.read_csv(META / "joint_gold" / "Joint_gold_standard.csv")
gold["source"] = "inat_2024_gold"
gold["image_path"] = gold["FileName"].map(lambda f: rel("joint_gold", f))
gold["label"] = gold["field.dsf.butterfly.activity"].map({"Feeding": "F", "Nonfeeding": "NF"})
cols = ["image_path", "label", "source", "image_url", "plant_scientific_name", "FileName"]
t1 = pd.concat([bimby[cols], gold[cols]], ignore_index=True)
t1 = drop(t1, ~t1["label"].isin(["F", "NF"]), "feeding/test1", "bad_label")
t1 = drop(t1, t1["FileName"].isin(SAME_SESSION), "feeding/test1", "same_session")
t1 = clean(t1, "feeding/test1", feeding_hashes, "in_dev_pool")
save(t1, "feeding/test1.csv", ["image_path", "label", "source", "image_url", "plant_scientific_name"])

ont = pd.read_csv(META / "ontario" / "Ontario_urls.csv")
ont["image_path"] = ont["FileName"].map(lambda f: rel("ontario", f))
ont["label"] = ont["Label"]
ont = drop(ont, ~ont["label"].isin(["F", "NF"]), "feeding/test2_ontario", "bad_label")
ont = clean(ont, "feeding/test2_ontario", feeding_hashes, "in_dev_pool")
save(ont, "feeding/test2_ontario.csv", ["image_path", "label"])

# ---------------------------------------------------------------- plant
print("plant")
arch = META / "plant_archive_batches"
comp = pd.concat([
    pd.read_csv(META / "plantID_detectron_composites" / "specified_flower_photos_detectron.csv")
      .rename(columns={"Filename": "file"})[["file", "Label"]],
    pd.read_csv(arch / "Cirsium_detectron_new.csv").rename(columns={"FileName": "file"})[["file", "Label"]],
    pd.read_csv(arch / "Sisymbrium_detectron_new.csv").rename(columns={"FileName": "file"})[["file", "Label"]],
], keys=["orig", "v2", "v2"]).reset_index(level=0).rename(columns={"level_0": "batch"})
comp["file"] = [f if b == "orig" else f.rsplit(".", 1)[0] + "_v2." + f.rsplit(".", 1)[1]
                for f, b in zip(comp["file"], comp["batch"])]
comp["image_path"] = comp["file"].map(lambda f: rel("plantID_detectron_composites", f))
comp["kind"] = "composite"

plain = pd.concat([
    pd.read_csv(META / "plant_plain_flowers" / "flower_only.csv")[["FileName", "Label"]],
    pd.read_csv(arch / "Cirsium_subset_new.csv")[["FileName"]].assign(Label="Cirsium_arvense"),
    pd.read_csv(arch / "Sisymbrium_subset_new.csv")[["FileName"]].assign(Label="Sisymbrium_loeselii"),
], ignore_index=True).rename(columns={"FileName": "file"})
plain["image_path"] = plain["file"].map(lambda f: rel("plant_plain_flowers", f))
plain["kind"] = "plain"

gold_plant = pd.read_csv(META / "joint_gold" / "Joint_gold_standard.csv")
gold_plant["image_path"] = gold_plant["FileName"].map(lambda f: rel("joint_gold", f))
gold_plant["label"] = gold_plant["plant_scientific_name"].map(plant_name)
gold_plant = clean(gold_plant, "plant/test_gold")

pp = pd.concat([comp, plain], ignore_index=True)
pp["label"] = pp["Label"].map(plant_name)
pp["group"] = pp["file"].str.replace(r"^aug_", "", regex=True).str.replace(r"(_v2)?\.\w+$", "", regex=True)
pp = drop(pp, pp["image_path"].duplicated(), "plant/dev_pool", "listed_twice")
pp = clean(pp, "plant/dev_pool", set(gold_plant["hash"]), "in_test_set")   # never train on a test image
pp = merge_groups(pp)
pp = add_folds(pp, "label")
save(pp, "plant/dev_pool.csv", ["image_path", "label", "kind", "group", "fold"])
save(gold_plant, "plant/test_gold.csv", ["image_path", "label"])

# BIMBY-2024 photos whose plant was identified: a few are target species, most are other plants,
# so this checks how often each one-vs-rest model says "yes" to the wrong plant
mixed = pd.read_csv(META / "bimby2024" / "BC2024_goldstandard_appendfilenames2.csv")
mixed = mixed[mixed["plant_scientific_name"].fillna("").str.strip() != ""].copy()
mixed["plant_scientific_name"] = mixed["plant_scientific_name"].str.split().str.join(" ")
mixed["image_path"] = mixed["FileName"].map(lambda f: rel("bimby2024", f))
mixed["label"] = mixed["plant_scientific_name"].map(plant_name)
mixed["is_target"] = mixed["label"].isin(set(pp["label"]))
mixed["taxon_rank"] = mixed["plant_scientific_name"].str.contains(" ").map({True: "species", False: "higher"})
mixed = drop(mixed, mixed["plant_scientific_name"].isin(PLANT_AMBIGUOUS), "plant/test_mixed_plants", "ambiguous_taxon")
mixed = clean(mixed, "plant/test_mixed_plants", set(pp["hash"]), "in_dev_pool")
save(mixed, "plant/test_mixed_plants.csv", ["image_path", "plant_scientific_name", "label", "is_target", "taxon_rank"])

# ---------------------------------------------------------------- monarch
print("monarch")
mon = pd.read_csv(META / "monarch" / "Monarch_image_labels.csv")
mon["image_path"] = mon["FileName"].map(lambda f: rel("monarch", f))
mon["label"] = mon["Label"]
mon["group"] = mon["FileName"].str.rsplit(".", n=1).str[0]
mon["hash"] = mon["image_path"].map(md5)
mon = drop(mon, mon["hash"].isna(), "monarch/dev_pool", "missing_image")
mon = merge_groups(mon)
mon = add_folds(mon, "label")
save(mon, "monarch/dev_pool.csv", ["image_path", "label", "group", "fold"])

# ---------------------------------------------------------------- removed rows
rem = pd.concat(removed, ignore_index=True)
rem["image_path"] = rem["image_path"].fillna("")
rem[["split", "reason", "image_path"] + [c for c in ["label", "source", "file", "FileName"] if c in rem]] \
    .to_csv(OUT / "removed.csv", index=False)
print(f"\nremoved.csv: {len(rem)} rows")
print(rem.groupby(["split", "reason"]).size().to_string())
