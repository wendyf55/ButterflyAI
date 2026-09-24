"""
move_data.py -- move datasets into data/, one batch at a time.

Usage (from anywhere):
    python data/scripts/move_data.py 1_test_sets          # dry run: prints what would happen
    python data/scripts/move_data.py 1_test_sets --run    # actually moves the files

Rules:
  - Never overwrites. If a file with the same name already exists at the destination:
      identical content  -> left where it is, logged as "identical_exists"
      different content  -> left where it is, logged as "CONFLICT" (nothing is lost)
  - Nothing is deleted. Emptied source folders are left for you to remove.
  - Every move is appended to data/move_log.csv (src, dst, status), so it can be undone.
"""
import csv, fnmatch, hashlib, sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
LOG_CSV = DATA / "move_log.csv"
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
ARCHIVE = "archive/Species_specific_tests_of_image_download"
OLD = "data/superseded"   # old copies and retired files, kept for you to delete later
SKIP = {".DS_Store"}

# (source, what, destination[, options])
#   what    = "images" (image files in the folder), "folder" (every file in the folder,
#             except .DS_Store and editor lock files) or "file" (one file)
#   options = optional {"include": [patterns], "exclude": [patterns]} on file names,
#             and/or {"suffix": "_v2"} to rename files on the way (aug_1.png -> aug_1_v2.png)
BATCHES = {
    "1_test_sets": [
        ("test_data/2024_BIMBY_appended", "images", "data/images/bimby2024"),
        ("test_data/2024_BIMBY_appended/BC2024_goldstandard_appendfilenames2.csv", "file", "data/metadata/bimby2024"),
        ("test_data/Bimby_2024_feeding_unsure.xlsx", "file", "data/metadata/bimby2024"),
        ("test_data/Joint_gold_standard", "images", "data/images/joint_gold"),
        ("test_data/Joint_gold_standard/Joint_gold_standard.csv", "file", "data/metadata/joint_gold"),
        ("test_data/Ontario_2024", "images", "data/images/ontario"),
        ("test_data/Ontario_2024/Ontario_urls.csv", "file", "data/metadata/ontario"),
        ("test_data/Ontario_2024/Ontario_F.csv", "file", "data/metadata/ontario"),
        ("test_data/Ontario_2024/Ontario_NF.csv", "file", "data/metadata/ontario"),
        ("test_data/Ontario_data.csv", "file", "data/metadata/ontario"),
    ],
    "2_feeding_pool": [
        ("feeding_model/data/Final_Feeding_Images", "images", "data/images/bimby_superimposed",
         {"include": ["superimposed_*"]}),
        ("feeding_model/data/Final_Feeding_Images", "images", "data/images/bimby_real",
         {"exclude": ["superimposed_*", "107.jpg"]}),   # 107.jpg: unlabelled, not in any CSV
        ("feeding_model/data/Final_Feeding_Images/DataFilenamesRedo.csv", "file", "data/metadata/bimby_collection"),
        ("feeding_model/data/Final_Feeding_Images/Redo_data/READme for feeding images.rtf", "file", "data/metadata/bimby_collection"),
    ],
    "3_plant": [
        ("plant_id_model/data/specified_flower_photos_detectron_ALL", "images", "data/images/plantID_detectron_composites"),
        ("plant_id_model/data/specified_flower_photos_detectron_ALL/specified_flower_photos_detectron.csv", "file", "data/metadata/plantID_detectron_composites"),
        ("plant_id_model/data/plant_data_specified", "images", "data/images/plant_plain_flowers"),
        ("plant_id_model/data/plant_data_specified/flower_only.csv", "file", "data/metadata/plant_plain_flowers"),
        # archive extras: 26 plain flowers are already in plant_data_specified (identical -> left in archive)
        (f"{ARCHIVE}/Cirsium_subset_new", "images", "data/images/plant_plain_flowers"),
        (f"{ARCHIVE}/Sisymbrium_subset_new", "images", "data/images/plant_plain_flowers"),
        # archive composites re-use flowers from detectron_ALL with a different butterfly -> rename to *_v2
        (f"{ARCHIVE}/Cirsium_detectron_new", "images", "data/images/plantID_detectron_composites", {"suffix": "_v2"}),
        (f"{ARCHIVE}/Sisymbrium_detectron_new", "images", "data/images/plantID_detectron_composites", {"suffix": "_v2"}),
        (f"{ARCHIVE}/Cirsium_subset_new/Cirsium_subset_new.csv", "file", "data/metadata/plant_archive_batches"),
        (f"{ARCHIVE}/Sisymbrium_subset_new/Sisymbrium_subset_new.csv", "file", "data/metadata/plant_archive_batches"),
        (f"{ARCHIVE}/Cirsium_detectron_new/Cirsium_detectron_new.csv", "file", "data/metadata/plant_archive_batches"),
        (f"{ARCHIVE}/Sisymbrium_detectron_new/Sisymbrium_detectron_new.csv", "file", "data/metadata/plant_archive_batches"),
    ],
    "4_monarch": [
        ("monarch/data/Monarch_images", "images", "data/images/monarch"),
        ("monarch/data/Monarch_images/Monarch_images.csv", "file", "data/metadata/monarch"),
        ("monarch/data/Monarch_images.csv", "file", "data/metadata/monarch"),   # identical copy -> left behind
        ("monarch/data/Monarch_images/Monarch_image_predictions.csv", "file", "data/metadata/monarch"),
        ("monarch/data/Monarch_images/Monarch_non_feeding_ls.csv", "file", "data/metadata/monarch"),
        ("monarch/data/Monarch_images/Monarch_image_labels.csv", "file", "data/metadata/monarch"),
    ],
    "5_superseded": [
        # Block B's original list -> keep with the BIMBY-2024 metadata
        ("test_data/2024_BIMBY/BIMBY_redo.csv", "file", "data/metadata/bimby2024"),
        # everything else keeps its old path under data/superseded/
        ("feeding_model/data/F", "folder", f"{OLD}/feeding_model/data/F"),
        ("feeding_model/data/Final_Feeding_Images", "folder", f"{OLD}/feeding_model/data/Final_Feeding_Images",
         {"include": ["DataFilenamesRedo_*.csv", "107.jpg"]}),
        ("test_data/2024_BIMBY", "folder", f"{OLD}/test_data/2024_BIMBY"),
        ("test_data/2024_BIMBY_appended", "folder", f"{OLD}/test_data/2024_BIMBY_appended"),
        ("test_data/Gold_standard_research_grade", "folder", f"{OLD}/test_data/Gold_standard_research_grade"),
        ("test_data/Joint_gold_standard", "folder", f"{OLD}/test_data/Joint_gold_standard"),
        ("test_data/Joint_model_csvs.R", "file", f"{OLD}/test_data"),
        ("test_data/make_bimby2024_test_clean.py", "file", f"{OLD}/test_data"),
        ("monarch/data/Monarch_images.csv", "file", f"{OLD}/monarch/data"),
        (f"{ARCHIVE}/Cirsium_subset_new", "folder", f"{OLD}/{ARCHIVE}/Cirsium_subset_new"),
        (f"{ARCHIVE}/Sisymbrium_subset_new", "folder", f"{OLD}/{ARCHIVE}/Sisymbrium_subset_new"),
        (f"{ARCHIVE}/Sisymbrium_detectron_new", "folder", f"{OLD}/{ARCHIVE}/Sisymbrium_detectron_new"),
    ],
}


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def files_for(src, what, options):
    if what == "file":
        return [src] if src.is_file() else []
    if not src.is_dir():
        return []
    files = [p for p in src.iterdir() if p.is_file()]
    if what == "images":
        files = [p for p in files if p.suffix.lower() in IMAGE_EXT]
    else:  # "folder"
        files = [p for p in files if p.name not in SKIP and not p.name.startswith(".~lock")]
    inc, exc = options.get("include"), options.get("exclude", [])
    if inc:
        files = [p for p in files if any(fnmatch.fnmatch(p.name, pat) for pat in inc)]
    files = [p for p in files if not any(fnmatch.fnmatch(p.name, pat) for pat in exc)]
    return sorted(files)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in BATCHES:
        sys.exit(f"usage: move_data.py <batch> [--run]   batches: {', '.join(BATCHES)}")
    batch, run = sys.argv[1], "--run" in sys.argv
    log = []
    print(("RUNNING " if run else "DRY RUN ") + batch + "\n")
    for src_rel, what, dst_rel, *rest in BATCHES[batch]:
        src, dst_dir = REPO / src_rel, REPO / dst_rel
        options = rest[0] if rest else {}
        files = files_for(src, what, options)
        suffix = options.get("suffix", "")
        counts = {"moved": 0, "identical_exists": 0, "CONFLICT": 0}
        for f in files:
            dst = dst_dir / (f.stem + suffix + f.suffix)
            if dst.exists():
                status = "identical_exists" if md5(dst) == md5(f) else "CONFLICT"
            else:
                status = "moved"
                if run:
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    f.rename(dst)          # same disk -> instant, no copy
            counts[status] += 1
            log.append([date.today(), batch, str(f.relative_to(REPO)), str(dst.relative_to(REPO)), status])
        label = "would move" if not run else "moved"
        extra = "".join(f", {v} {k}" for k, v in counts.items() if k != "moved" and v)
        print(f"{src_rel}  ->  {dst_rel}\n    {label} {counts['moved']}{extra}"
              + ("" if files else "   (SOURCE NOT FOUND / EMPTY)"))
    if run:
        DATA.mkdir(exist_ok=True)
        new = not LOG_CSV.exists()
        with open(LOG_CSV, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["date", "batch", "src", "dst", "status"])
            w.writerows(log)
        print(f"\nLogged {len(log)} rows to {LOG_CSV.relative_to(REPO)}")
    else:
        print("\nNothing was moved. Add --run to do it.")


if __name__ == "__main__":
    main()