#Final feeding model cross validation

#Create the final feeding - non-feeding model 
#Make sure to verify training header

#GOAL = get 5 fold cross validated result and get standard deviation of the model metrics 
#The xval estimates how well the training RECIPE works (mean ± SD over folds). The fold models are
#thrown away; the final model is trained on all the data with the same recipe (Final_feeding_model_train_on_ALL.py)

###############################################################
# imports

import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import random
import pandas as pd
from pathlib import Path
import sys
import json
import hashlib
import platform
import subprocess
from datetime import datetime
import sklearn

from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, log_loss
import pandas as pd

###################################################################################
# Settings

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "feeding"
RESULTS_DIR = MODULE_DIR / "results" # CHANGED: every run gets its own folder in here (no fold models are saved any more)
TRAINING_CSV = SPLITS_DIR / "dev_pool.csv"
SEED_VALUE = 321
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
VAL_BATCH_SIZE = 32   # CHANGED from 1: same predictions (no dropout / fixed BatchNorm at prediction time), much faster
GENERATOR_SEED = 123
K_FOLDS = 5   # folds are fixed in dev_pool.csv (column `fold`, 0-4)
EPOCHS = 10  #fixed, same as the final model; nothing is chosen on the validation fold (no early stopping / best checkpoint)
PREDICTION_THRESHOLD = 0.5
DROPOUT_RATE = 0.5
DENSE_UNITS = 512
TRAINABLE_BLOCK = "conv5_block"   # same as Final_feeding_model_train_on_ALL.py
CLASSES = ['F', 'N']   # CHANGED: fixed class order -> F=0, N=1, so the sigmoid output = P(non-feeding)
EXCLUDED_CSV = SPLITS_DIR / "excluded.csv"   # 17 label-conflict images (identical pairs labelled F and N)
DROP_UNTRACEABLE_SUPER = True   # superimposed images with no source photo in the pool (103); off once J clears them
#just not sure about origin or data leakage issues since we can't trace these

# Quick checks (both False for the real run)
SMOKE_TEST = False   # True = pipeline check on a few images: 2 folds, 3 epochs, a few minutes. The numbers mean nothing
SMOKE_IMAGES = 16    # smoke test: images per fold x label x photo_type (16 real F + 16 real N + 16 super N = 48 per fold)
SMOKE_FOLDS = 2
SMOKE_EPOCHS = 3
FORCE_CPU = False    # True = hide the GPU and train on the CPU (for the GPU-vs-CPU check)

if SMOKE_TEST:
    K_FOLDS = SMOKE_FOLDS
    EPOCHS = SMOKE_EPOCHS
if FORCE_CPU:
    tf.config.set_visible_devices([], 'GPU')   # must happen before TensorFlow touches the GPU

##############################################################################################
# Run folder and log

# CHANGED: run folder + everything printed also goes to log.txt
RUN_DIR = RESULTS_DIR / (f"xval_{datetime.now():%Y-%m-%d_%H%M}" + ("_smoke" if SMOKE_TEST else "") + ("_cpu" if FORCE_CPU else ""))
RUN_DIR.mkdir(parents=True, exist_ok=False)   # never overwrite an earlier run

#data/images/bimby_real and data/images/bimby_superimposed contain the 2023 BC BIMBY feeding and non-feeding photos and the superimposed photos
#image paths in the split CSVs are relative to the repo root (see data/README.md)
os.chdir(PROJECT_ROOT)

cwd = os.getcwd()
cwd

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
print(f"Run folder: {RUN_DIR}")

##############################################################################################
# Load the data 

# The ResNet50 model expects images to be 224x224, so we set those values here
img_height, img_width = (IMG_HEIGHT, IMG_WIDTH)
batch_size = BATCH_SIZE

full_df = pd.read_csv(TRAINING_CSV)

# superimposed images whose group has no real photo in the pool = no traceable source photo
real_groups = set(full_df.loc[full_df['photo_type'] == 'real', 'group'])
untraceable = (full_df['photo_type'] == 'super') & ~full_df['group'].isin(real_groups)
excluded = full_df['image_path'].isin(pd.read_csv(EXCLUDED_CSV)['image_path'])

drop = excluded | (untraceable if DROP_UNTRACEABLE_SUPER else False)
print(f"Dropping {excluded.sum()} excluded + {(untraceable & ~excluded).sum() if DROP_UNTRACEABLE_SUPER else 0} untraceable superimposed; keeping {(~drop).sum()} of {len(full_df)}")
full_df = full_df[~drop].reset_index(drop=True)

full_df = full_df.rename(columns={'image_path': 'filename'})  # image_path is relative to the repo root

#run code with just real images (all_df, name is misleading, watch out!)
#all_df = full_df[full_df['photo_type'] == 'real']
all_df = full_df

if SMOKE_TEST:   # same few images every time (first SMOKE_IMAGES of each fold x label x photo_type)
    all_df = all_df.groupby(['fold', 'label', 'photo_type']).head(SMOKE_IMAGES).reset_index(drop=True)
    print(f"SMOKE TEST: {len(all_df)} images, {K_FOLDS} folds, {EPOCHS} epochs. The numbers mean nothing")

##############################################################################################
# Folds (fixed in dev_pool.csv, grouped and stratified)

#K-fold cross validation
#folds come from dev_pool.csv: grouped (a superimposed image shares a fold with its source photo)
#and balanced on label x photo_type (made by data/scripts/make_splits.py)

folds = all_df['fold'].values
fold_splits = [(np.where(folds != k)[0], np.where(folds == k)[0]) for k in range(K_FOLDS)]

# CHANGED: the LabelEncoder is gone. The generators get classes=CLASSES, so F=0 and N=1 everywhere
# (the old comment "F=1" was wrong: alphabetical order already made F=0, so the old precision/recall/F1 were for N)

##############################################################################################
# Run record (run_info.json will contain code version, software, GPU, settings, data files)

# CHANGED: record of the run (code version, software, GPU, settings, exact data files)
# good for reproducibility
def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()

run_info = {
    'script': Path(__file__).name,
    'started': datetime.now().isoformat(timespec='seconds'),
    'git_commit': git('rev-parse', 'HEAD'),
    'git_uncommitted_changes': git('status', '--porcelain', '--untracked-files=no').splitlines(),   # recorded, not blocked
    'python': platform.python_version(),
    'tensorflow': tf.__version__,
    'keras': tf.keras.__version__,
    'numpy': np.__version__,
    'pandas': pd.__version__,
    'scikit_learn': sklearn.__version__,
    'machine': platform.platform(),
    'gpus': [gpu.name for gpu in tf.config.get_visible_devices('GPU')],   # empty when FORCE_CPU
    'config': {'seed': SEED_VALUE, 'img_size': [IMG_HEIGHT, IMG_WIDTH], 'batch_size': BATCH_SIZE,
               'epochs': EPOCHS, 'k_folds': K_FOLDS, 'threshold': PREDICTION_THRESHOLD,
               'dropout': DROPOUT_RATE, 'dense_units': DENSE_UNITS, 'trainable_block': TRAINABLE_BLOCK,
               'classes': CLASSES, 'drop_untraceable_super': DROP_UNTRACEABLE_SUPER,
               'smoke_test': SMOKE_TEST, 'force_cpu': FORCE_CPU},
    'data': {'training_csv': str(TRAINING_CSV.relative_to(PROJECT_ROOT)), 'training_csv_md5': md5(TRAINING_CSV),
             'excluded_csv': str(EXCLUDED_CSV.relative_to(PROJECT_ROOT)), 'excluded_csv_md5': md5(EXCLUDED_CSV),
             'n_images': len(all_df),
             'counts': {f'{photo_type}_{label}': int(n) for (photo_type, label), n in all_df.groupby(['photo_type', 'label']).size().items()}},
}

def save_run_info():
    (RUN_DIR / 'run_info.json').write_text(json.dumps(run_info, indent=2))

save_run_info()

if not run_info['gpus'] and not FORCE_CPU:
    print("WARNING: no GPU visible, training on the CPU")
if run_info['git_uncommitted_changes']:
    print(f"Note: uncommitted changes: {run_info['git_uncommitted_changes']}")

##############################################################################################
# Metrics: for both classes: all / real only / superimposed only

# CHANGED: metrics for BOTH classes. Also used for the real-only and super-only subsets
def compute_metrics(df):
    """Accuracy, loss, per-class precision/recall/F1 and the confusion matrix counts for one set of predictions.
    A metric that can't be computed (e.g. recall of F on the super-only subset, which has no F) is NaN."""
    y_true, y_pred = df['y_true'].values, df['y_pred'].values
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    predicted = np.bincount(y_pred, minlength=2)
    row = {'n': len(df),
           'accuracy': accuracy_score(y_true, y_pred),
           'loss': log_loss(y_true, df['score'].values, labels=[0, 1])}
    for i, c in enumerate(CLASSES):
        row[f'precision_{c}'] = precision[i] if predicted[i] > 0 else np.nan
        row[f'recall_{c}'] = recall[i] if support[i] > 0 else np.nan
        row[f'f1_{c}'] = f1[i] if (predicted[i] > 0 and support[i] > 0) else np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])   # rows = true, columns = predicted
    for i, true_c in enumerate(CLASSES):
        for j, pred_c in enumerate(CLASSES):
            row[f'true_{true_c}_pred_{pred_c}'] = cm[i, j]
    return row

SUBSETS = {'all': lambda df: df,
           'real': lambda df: df[df['photo_type'] == 'real'],
           'super': lambda df: df[df['photo_type'] == 'super']}

##############################################################################################
# Model pieces, seeds, and image generators

#CREATE MODEL
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import random #TODO: move these up to the top

#set seed so its always the same

os.environ['PYTHONHASHSEED']=str(SEED_VALUE)
 
random.seed(SEED_VALUE)

np.random.seed(SEED_VALUE)

tf.random.set_seed(SEED_VALUE)

all_predictions = []   # CHANGED: one row per validation image, all folds
fold_metrics = []      # CHANGED: one row per fold x subset

# Example of a simple data generator setup
datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

# CHANGED: same augmentation as Final_feeding_model_train_on_ALL.py for the training folds (validation stays un-augmented)
train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                   horizontal_flip = True,
                                   shear_range = 0.2
                                  )

##############################################################################################
# Cross validation loop - one pass per 80/20 fold that builds, trains, predicts, and scores

for fold, (train_idx, val_idx) in enumerate(fold_splits, start=1):   # CHANGED: fold counter added
    print(f"\n Fold {fold}/{K_FOLDS} ")

    # CHANGED: reset before each fold so folds are independent
    tf.keras.backend.clear_session()
    os.environ['PYTHONHASHSEED']=str(SEED_VALUE)
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    tf.random.set_seed(SEED_VALUE)

    # CHANGED: build a FRESH model each fold (this block was originally outside the loop, so every fold
    # kept training the previous fold's model). Same architecture as Final_feeding_model_train_on_ALL.py
    # (the "final" model): BatchNorm head, only the conv5 block trainable.
    base_model = ResNet50(include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(DROPOUT_RATE)(x)

    x = Dense(DENSE_UNITS)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(DROPOUT_RATE)(x)

    predictions = Dense(1, activation='sigmoid')(x)

    # previous head (whole ResNet50 trainable, no BatchNorm), kept for reference:
    #x = GlobalAveragePooling2D()(x)
    #x = Dropout(DROPOUT_RATE)(x)
    #x = Dense(DENSE_UNITS, activation='relu')(x)
    #x = Dropout(DROPOUT_RATE)(x)
    #predictions = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs = base_model.input, outputs = predictions)

    for layer in base_model.layers:
        if TRAINABLE_BLOCK in layer.name: #tune just the final block ~15-20 layers
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # CHANGED: no EarlyStopping / ModelCheckpoint. The final model can't use a validation set, so the folds
    # train the same fixed number of epochs and are scored with the last-epoch weights.

    # CHANGED: take the rows straight from all_df, so photo_type/group/fold travel with each image
    # (needed for the real-only / super-only metrics and predictions.csv)
    train_df = all_df.iloc[train_idx]
    val_df = all_df.iloc[val_idx].reset_index(drop=True)

    # Create generators
    # CHANGED: classes=CLASSES pins F=0, N=1
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filename',
        y_col='label',
        classes = CLASSES,
        target_size = (img_height, img_width),
        batch_size = batch_size,
        class_mode = 'binary',
        seed = GENERATOR_SEED
    )

    val_generator = datagen.flow_from_dataframe(
        val_df,
        x_col='filename',
        y_col='label',
        classes = CLASSES,
        target_size = (img_height, img_width),
        batch_size = VAL_BATCH_SIZE,
        class_mode = 'binary',
        shuffle = False,
        seed = GENERATOR_SEED
    )

    # flow_from_dataframe silently skips images it can't find; then predictions would no longer line up with val_df
    assert len(val_generator.classes) == len(val_df), "some validation images were not found"

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        verbose=2   # CHANGED: one line per epoch (readable in log.txt)
        )

    # CHANGED: training curves, for looking at only (nothing is chosen from them)
    pd.DataFrame(history.history).rename_axis('epoch').to_csv(RUN_DIR / f"history_fold{fold}.csv")

    # Predict every validation image and save
    # last-epoch weights; score = sigmoid output = P(non-feeding)
    scores = model.predict(val_generator, verbose=0).ravel()

    fold_pred = val_df[['filename', 'label', 'photo_type', 'group', 'fold']].copy()
    fold_pred['y_true'] = val_generator.classes
    fold_pred['score'] = scores
    fold_pred['y_pred'] = (scores > PREDICTION_THRESHOLD).astype(int)
    all_predictions.append(fold_pred)
    # CHANGED: saved after every fold, so a crash keeps the finished folds
    pd.concat(all_predictions).to_csv(RUN_DIR / "predictions.csv", index=False)

    # Score this fold (all / real / super)
    for subset, select in SUBSETS.items():
        fold_metrics.append({'fold': fold, 'subset': subset, **compute_metrics(select(fold_pred))})
    pd.DataFrame(fold_metrics).to_csv(RUN_DIR / "metrics_per_fold.csv", index=False)

    m = fold_metrics[-3]   # this fold, all images
    print(f"Fold {fold} completed (all images). Accuracy: {m['accuracy']:.4f}, Loss: {m['loss']:.4f} | "
          f"F: precision {m['precision_F']:.4f}, recall {m['recall_F']:.4f}, F1 {m['f1_F']:.4f} | "
          f"N: precision {m['precision_N']:.4f}, recall {m['recall_N']:.4f}, F1 {m['f1_N']:.4f}")

##########################################################################################################
# Summary over the folds (gives mean and SD) and finish the run record

metrics_df = pd.DataFrame(fold_metrics)
metric_cols = ['accuracy', 'loss'] + [f'{m}_{c}' for c in CLASSES for m in ('precision', 'recall', 'f1')]
summary = metrics_df.groupby('subset', sort=False)[metric_cols].agg(['mean', 'std'])
summary.to_csv(RUN_DIR / "metrics_summary.csv")

print(f"\nCross-Validation Results ({K_FOLDS} folds, mean ± SD; F = feeding, N = non-feeding):")
for subset in SUBSETS:
    print(f"\n  {subset} (n per fold: {metrics_df.loc[metrics_df['subset'] == subset, 'n'].tolist()})")
    for col in metric_cols:
        mean, sd = summary.loc[subset, (col, 'mean')], summary.loc[subset, (col, 'std')]
        if not np.isnan(mean):
            print(f"    {col:<12} {mean:.4f} ± {sd:.4f}")

run_info['finished'] = datetime.now().isoformat(timespec='seconds')
save_run_info()
print(f"\nSaved to {RUN_DIR}")