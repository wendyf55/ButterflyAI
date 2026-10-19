"""Make shrunk copies of the feeding images (long side <= LONG_SIDE px) for the image-size experiment.

Photos larger than LONG_SIDE are shrunk with PIL's Lanczos filter and saved as JPEG quality 90
(roughly what iNat does for its 500 px "medium" photos; same as feeding_model/resolution_check.py).
Photos already LONG_SIDE or smaller are copied byte for byte (no extra re-compression).
Originals are never touched. Re-running skips copies that already exist.
"""

from pathlib import Path
import shutil
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "feeding"
CSVS = [SPLITS_DIR / "dev_pool.csv", SPLITS_DIR / "test1.csv", SPLITS_DIR / "test2_ontario.csv"]
LONG_SIDE = 500
JPEG_QUALITY = 90
SRC_PREFIX = "data/images/"
DST_PREFIX = f"data/images_{LONG_SIDE}px/"
LOG_CSV = PROJECT_ROOT / "data" / f"images_{LONG_SIDE}px_log.csv"

paths = sorted(set(pd.concat([pd.read_csv(c)["image_path"] for c in CSVS])))
assert all(p.startswith(SRC_PREFIX) for p in paths), "an image_path is outside data/images/"
print(f"{len(paths)} images")

log = []
for i, rel in enumerate(paths, start=1):
    src = PROJECT_ROOT / rel
    dst = PROJECT_ROOT / rel.replace(SRC_PREFIX, DST_PREFIX, 1)
    if not src.exists():
        log.append({"image_path": rel, "action": "missing"})
        continue
    with Image.open(src) as im:
        size = im.size
    if dst.exists():
        action = "exists"
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if max(size) <= LONG_SIDE:
            shutil.copy2(src, dst)
            action = "copied"
        else:
            with Image.open(src) as im:
                im = im.convert("RGB")
                scale = LONG_SIDE / max(im.size)
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
                im.save(dst, "JPEG", quality=JPEG_QUALITY)
            action = "shrunk"
    log.append({"image_path": rel, "width": size[0], "height": size[1], "action": action})
    if i % 1000 == 0:
        print(f"{i}/{len(paths)}")

log = pd.DataFrame(log)
log.to_csv(LOG_CSV, index=False)
print(log["action"].value_counts().to_string())
print(f"Log: {LOG_CSV}")