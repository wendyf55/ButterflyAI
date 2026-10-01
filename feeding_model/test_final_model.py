#Test the final feeding model ONCE on the two feeding test sets
#  Test 1: BIMBY-2024 clean + gold's feeding labels (data/splits/feeding/test1.csv), reported overall AND per source
#  Test 2: Ontario 2024 (data/splits/feeding/test2_ontario.csv), out of region
#Nothing here is used to choose or change the model. The output folder can only be made once per model,
#so a model can't be quietly re-tested after changes.

##############################################################################################
# Imports

import tensorflow as tf
import os
import sys
import json
import hashlib
import platform
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
import sklearn
from pathlib import Path

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, log_loss

##############################################################################################
# Settings

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "feeding"
MODELS_DIR = MODULE_DIR / "models"
RESULTS_DIR = MODULE_DIR / "results"
MODEL_PATH = MODELS_DIR / "feeding_final_real_and_super_2026-10-01.keras" 
TEST_SETS = {'test1': SPLITS_DIR / "test1.csv",
             'ontario': SPLITS_DIR / "test2_ontario.csv"}
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
PREDICTION_THRESHOLD = 0.5
CLASSES = ['F', 'N']           # F=0, N=1, so the model's output = P(non-feeding); same as training
LABEL_MAP = {'NF': 'N'}        # the test CSVs write non-feeding as NF

RUN_DIR = RESULTS_DIR / f"test_{MODEL_PATH.stem}"

##############################################################################################
# Run folder and log

if not MODEL_PATH.exists():
    sys.exit(f"Model not found: {MODEL_PATH}")
RUN_DIR.mkdir(parents=True, exist_ok=False)   # fails if this model was already tested: test once

os.chdir(PROJECT_ROOT)

class Tee:
    """Write everything to the console and to a log file."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for stream in self.streams:
            stream.write(text)
    def flush(self):
        for stream in self.streams:
            stream.flush()
    def __getattr__(self, name):   # isatty(), encoding, ... come from the console
        return getattr(self.streams[0], name)

sys.stdout = Tee(sys.__stdout__, open(RUN_DIR / "log.txt", "w"))
print(f"Testing {MODEL_PATH}\nResults in {RUN_DIR}")

##############################################################################################
# Run record

def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()

run_info = {
    'script': Path(__file__).name,
    'started': datetime.now().isoformat(timespec='seconds'),
    'git_commit': git('rev-parse', 'HEAD'),
    'git_uncommitted_changes': git('status', '--porcelain', '--untracked-files=no').splitlines(),
    'tensorflow': tf.__version__,
    'machine': platform.platform(),
    'gpus': [gpu.name for gpu in tf.config.get_visible_devices('GPU')],
    'model_file': str(MODEL_PATH.relative_to(PROJECT_ROOT)),
    'model_md5': md5(MODEL_PATH),
    'config': {'img_size': [IMG_HEIGHT, IMG_WIDTH], 'threshold': PREDICTION_THRESHOLD, 'classes': CLASSES},
    'test_sets': {name: {'csv': str(path.relative_to(PROJECT_ROOT)), 'md5': md5(path)} for name, path in TEST_SETS.items()},
}
def save_run_info():
    (RUN_DIR / 'run_info.json').write_text(json.dumps(run_info, indent=2))
save_run_info()

##############################################################################################
# Metrics (same as the xval: both classes, NaN when a metric can't be computed)

def compute_metrics(df):
    """Accuracy, loss, per-class precision/recall/F1 and the confusion matrix counts for one set of predictions."""
    y_true, y_pred = df['y_true'].values, df['y_pred'].values
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    predicted = np.bincount(y_pred, minlength=2)
    row = {'n': len(df),
           'accuracy': accuracy_score(y_true, y_pred),
           'loss': log_loss(y_true, df['score'].values, labels=[0, 1])}
    for i, c in enumerate(CLASSES):
        row[f'n_{c}'] = support[i]
        row[f'precision_{c}'] = precision[i] if predicted[i] > 0 else np.nan
        row[f'recall_{c}'] = recall[i] if support[i] > 0 else np.nan
        row[f'f1_{c}'] = f1[i] if (predicted[i] > 0 and support[i] > 0) else np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])   # rows = true, columns = predicted
    for i, true_c in enumerate(CLASSES):
        for j, pred_c in enumerate(CLASSES):
            row[f'true_{true_c}_pred_{pred_c}'] = cm[i, j]
    return row

##############################################################################################
# Predict every test image

model = tf.keras.models.load_model(MODEL_PATH)
datagen = ImageDataGenerator(preprocessing_function = preprocess_input)   # same preprocessing as validation in the xval

all_metrics = []
for name, csv_path in TEST_SETS.items():
    df = pd.read_csv(csv_path)
    df['label'] = df['label'].replace(LABEL_MAP)
    assert set(df['label']) <= set(CLASSES), f"unexpected labels in {csv_path}: {set(df['label'])}"
    if 'source' not in df:
        df['source'] = name

    generator = datagen.flow_from_dataframe(
        df,
        x_col='image_path',
        y_col='label',
        classes = CLASSES,
        target_size = (IMG_HEIGHT, IMG_WIDTH),
        batch_size = BATCH_SIZE,
        class_mode = 'binary',
        shuffle = False
    )
    assert len(generator.classes) == len(df), f"some images in {csv_path} were not found"

    scores = model.predict(generator, verbose=0).ravel()   # P(non-feeding)
    pred = df[['image_path', 'label', 'source']].copy()
    pred['y_true'] = generator.classes
    pred['score'] = scores
    pred['y_pred'] = (scores > PREDICTION_THRESHOLD).astype(int)
    pred.to_csv(RUN_DIR / f"predictions_{name}.csv", index=False)

    # overall, then per source (Test 1 has three sources; always report them separately)
    all_metrics.append({'test_set': name, 'subset': 'overall', **compute_metrics(pred)})
    if pred['source'].nunique() > 1:
        for source, part in pred.groupby('source'):
            all_metrics.append({'test_set': name, 'subset': source, **compute_metrics(part)})

##############################################################################################
# Results

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(RUN_DIR / "metrics.csv", index=False)

cols = ['test_set', 'subset', 'n', 'n_F', 'n_N', 'accuracy', 'precision_F', 'recall_F', 'precision_N', 'recall_N']
print("\nTest results (F = feeding, N = non-feeding):")
print(metrics_df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

run_info['finished'] = datetime.now().isoformat(timespec='seconds')
save_run_info()
print(f"\nSaved to {RUN_DIR}")