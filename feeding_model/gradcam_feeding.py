"""Grad-CAM for the final feeding model: which parts of a photo push the score toward
feeding / non-feeding. Diagnostic only: uses test-set photos, nothing here chooses or tunes a model."""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input

REPO = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO / "feeding_model/models/feeding_final_real_and_super_2026-10-01.keras"
TEST_DIR = REPO / "feeding_model/results/test_feeding_final_real_and_super_2026-10-01"
DEV_POOL = REPO / "data/splits/feeding/dev_pool.csv"
OUT_DIR = REPO / f"feeding_model/results/gradcam_{datetime.now():%Y-%m-%d}"
LAST_CONV = "conv5_block3_out"   # last ResNet50 layer with a spatial (7x7) layout
IMG_SIZE = (224, 224)
INTERPOLATION = "nearest"        # same resize as flow_from_dataframe in training and testing
N_PER_GROUP = 8
SEED = 0

##############################################################################################
# Model: also return the conv5 output and the input to the last Dense layer, so we can
# work with the logit (pre-sigmoid). Gradients of the sigmoid vanish when P is ~0 or ~1.
model = tf.keras.models.load_model(MODEL_PATH)
last_dense = model.layers[-1]
grad_model = tf.keras.Model(model.inputs, [model.get_layer(LAST_CONV).output, last_dense.input])
W, b = last_dense.kernel, last_dense.bias

def load(path):
    img = tf.keras.utils.load_img(REPO / path, target_size=IMG_SIZE, interpolation=INTERPOLATION)
    arr = tf.keras.utils.img_to_array(img)
    return arr.astype("uint8"), preprocess_input(arr.copy())[None]

def gradcam(x):
    """Signed Grad-CAM (7x7): > 0 pushes toward non-feeding, < 0 toward feeding. Also returns P(non-feeding)."""
    with tf.GradientTape() as tape:
        conv, h = grad_model(x, training=False)
        z = (tf.matmul(h, W) + b)[0, 0]          # logit of P(non-feeding); F=0, N=1
    grads = tape.gradient(z, conv)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(conv[0] * weights, axis=-1).numpy()
    return cam, float(tf.sigmoid(z))

def upsample(cam):
    return tf.image.resize(cam[..., None], IMG_SIZE, method="bilinear").numpy()[..., 0]

##############################################################################################
# Groups of photos to look at
t1 = pd.read_csv(TEST_DIR / "predictions_test1.csv")
on = pd.read_csv(TEST_DIR / "predictions_ontario.csv")
dev = pd.read_csv(DEV_POOL)
groups = {
    "ontario_feeding_correct":       on[(on.y_true == 0) & (on.y_pred == 0)],
    "ontario_feeding_missed":        on[(on.y_true == 0) & (on.y_pred == 1)],
    "inat2024_nonfeeding_correct":   t1[(t1.source == "inat_2024") & (t1.y_true == 1) & (t1.y_pred == 1)],
    "inat2024_nonfeeding_called_F":  t1[(t1.source == "inat_2024") & (t1.y_true == 1) & (t1.y_pred == 0)],
    "bimby_collection_NF_called_F":  t1[(t1.source == "bimby_collection") & (t1.y_true == 1) & (t1.y_pred == 0)],
    "composites_TRAINING_images":    dev[dev.photo_type == "super"],
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
for name, df in groups.items():
    rows = df.sample(min(N_PER_GROUP, len(df)), random_state=SEED)
    fig, axes = plt.subplots(len(rows), 2, figsize=(6.5, 3.1 * len(rows)), squeeze=False)
    for ax_row, r in zip(axes, rows.itertuples()):
        img, x = load(r.image_path)
        cam, p = gradcam(x)
        lim = np.abs(cam).max() or 1.0             # same scale both ways: weak evidence looks weak
        ax_row[0].imshow(img)
        ax_row[0].set_title(f"{Path(r.image_path).name}\nlabel {r.label}, P(N) = {p:.2f}", fontsize=8)
        ax_row[1].imshow(img)
        ax_row[1].imshow(upsample(cam), cmap="RdBu", alpha=0.5, vmin=-lim, vmax=lim)
        ax_row[1].set_title("red = toward feeding, blue = toward non-feeding", fontsize=7)
        for ax in ax_row:
            ax.axis("off")
    fig.suptitle(name, fontsize=11)
    fig.savefig(OUT_DIR / f"{name}.jpg", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}.jpg ({len(rows)} photos)")