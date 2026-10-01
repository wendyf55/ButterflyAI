#Final feeding model, train on all of the available 2023 data
 
#Create the final feeding - non-feeding model
#Same data filter and same recipe as Final_feeding_model_xval.py: the xval estimates how well this recipe works,
#this script trains the one model we keep. No validation set: nothing here is chosen on data.
#Test it ONCE with test_final_model.py
 
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
import random
import pandas as pd
import sklearn
from pathlib import Path
 
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model

##############################################################################################
# Settings (recipe values must match Final_feeding_model_xval.py)
 
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "feeding"
MODELS_DIR = MODULE_DIR / "models"
RESULTS_DIR = MODULE_DIR / "results"
TRAINING_CSV = SPLITS_DIR / "dev_pool.csv"   # the whole development pool (all 5 folds)
EXCLUDED_CSV = SPLITS_DIR / "excluded.csv"   # 17 label-conflict images (identical pairs labelled F and N)
DROP_UNTRACEABLE_SUPER = True   # superimposed images with no source photo in the pool (103); same as the xval
TRAIN_ON_REAL_ONLY = False      # CHANGED: a setting instead of a commented-out line (which saved to the same file name)
 
SEED_VALUE = 321
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
GENERATOR_SEED = 123
EPOCHS = 10
DROPOUT_RATE = 0.5
DENSE_UNITS = 512
TRAINABLE_BLOCK = "conv5_block"
CLASSES = ['F', 'N']   # F=0, N=1, so the sigmoid output = P(non-feeding)
 
# CHANGED: a NEW model file every time; the old production model (REAL_AND_SUPER_Final_feeding_model_unfrozen.keras)
# is never overwritten
MODEL_NAME = f"feeding_final_{'real_only' if TRAIN_ON_REAL_ONLY else 'real_and_super'}_{datetime.now():%Y-%m-%d}"
MODEL_OUTPUT = MODELS_DIR / f"{MODEL_NAME}.keras"
RUN_DIR = RESULTS_DIR / f"train_{MODEL_NAME}"   # run record: run_info.json, history.csv, log.txt

##############################################################################################
# Run folder and log
 
if MODEL_OUTPUT.exists():
    sys.exit(f"{MODEL_OUTPUT} already exists. Rename or move it first (models are never overwritten)")
RUN_DIR.mkdir(parents=True, exist_ok=False)
 
#image paths in the split CSVs are relative to the repo root (see data/README.md)
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
print(f"Run folder: {RUN_DIR}")
print(f"Model will be saved to: {MODEL_OUTPUT}")

##############################################################################################
# Load the data (same filter as the xval)
 
full_df = pd.read_csv(TRAINING_CSV)
 
# superimposed images whose group has no real photo in the pool = no traceable source photo
real_groups = set(full_df.loc[full_df['photo_type'] == 'real', 'group'])
untraceable = (full_df['photo_type'] == 'super') & ~full_df['group'].isin(real_groups)
excluded = full_df['image_path'].isin(pd.read_csv(EXCLUDED_CSV)['image_path'])
 
drop = excluded | (untraceable if DROP_UNTRACEABLE_SUPER else False)
print(f"Dropping {excluded.sum()} excluded + {(untraceable & ~excluded).sum() if DROP_UNTRACEABLE_SUPER else 0} untraceable superimposed; keeping {(~drop).sum()} of {len(full_df)}")
full_df = full_df[~drop].reset_index(drop=True)
 
full_df = full_df.rename(columns={'image_path': 'filename'})  # image_path is relative to the repo root
 
train_df = full_df
if TRAIN_ON_REAL_ONLY:
    train_df = train_df[train_df['photo_type'] == 'real'].reset_index(drop=True)
print(f"Training on {len(train_df)} images: {train_df.groupby(['photo_type', 'label']).size().to_dict()}")

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
    'python': platform.python_version(),
    'tensorflow': tf.__version__,
    'keras': tf.keras.__version__,
    'numpy': np.__version__,
    'pandas': pd.__version__,
    'scikit_learn': sklearn.__version__,
    'machine': platform.platform(),
    'gpus': [gpu.name for gpu in tf.config.get_visible_devices('GPU')],
    'config': {'seed': SEED_VALUE, 'img_size': [IMG_HEIGHT, IMG_WIDTH], 'batch_size': BATCH_SIZE,
               'epochs': EPOCHS, 'dropout': DROPOUT_RATE, 'dense_units': DENSE_UNITS,
               'trainable_block': TRAINABLE_BLOCK, 'classes': CLASSES,
               'drop_untraceable_super': DROP_UNTRACEABLE_SUPER, 'train_on_real_only': TRAIN_ON_REAL_ONLY},
    'data': {'training_csv': str(TRAINING_CSV.relative_to(PROJECT_ROOT)), 'training_csv_md5': md5(TRAINING_CSV),
             'excluded_csv': str(EXCLUDED_CSV.relative_to(PROJECT_ROOT)), 'excluded_csv_md5': md5(EXCLUDED_CSV),
             'n_images': len(train_df),
             'counts': {f'{photo_type}_{label}': int(n) for (photo_type, label), n in train_df.groupby(['photo_type', 'label']).size().items()}},
    'model_file': str(MODEL_OUTPUT.relative_to(PROJECT_ROOT)),
}
def save_run_info():
    (RUN_DIR / 'run_info.json').write_text(json.dumps(run_info, indent=2))
save_run_info()
if not run_info['gpus']:
    print("WARNING: no GPU visible, training on the CPU")
if run_info['git_uncommitted_changes']:
    print(f"Note: uncommitted changes: {run_info['git_uncommitted_changes']}")
 

##############################################################################################
# Model (same architecture as the xval: BatchNorm head, only the conv5 block trainable)
 
#set seed so its always the same
os.environ['PYTHONHASHSEED']=str(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)
 
base_model = ResNet50(include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(DROPOUT_RATE)(x)
 
x = Dense(DENSE_UNITS)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(DROPOUT_RATE)(x)
 
predictions = Dense(1, activation='sigmoid')(x)
 
model = Model(inputs = base_model.input, outputs = predictions)
 
for layer in base_model.layers:
    if TRAINABLE_BLOCK in layer.name: #tune just the final block
        layer.trainable = True
    else:
        layer.trainable = False
 
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
##############################################################################################
# Train (fixed EPOCHS, no validation, same augmentation as the xval training folds)
 
train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                   horizontal_flip = True,
                                   shear_range = 0.2
                                  )
 
train_generator = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    x_col = 'filename',
    y_col = 'label',
    classes = CLASSES,   # CHANGED: pins F=0, N=1
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE,
    class_mode = 'binary',
    seed = GENERATOR_SEED,
    shuffle = True
)
# flow_from_dataframe silently skips images it can't find
assert train_generator.samples == len(train_df), "some training images were not found"
print("Training class indices: ", train_generator.class_indices)
 
history = model.fit(train_generator, epochs = EPOCHS, verbose = 2)
pd.DataFrame(history.history).rename_axis('epoch').to_csv(RUN_DIR / "history.csv")
 
##############################################################################################
# Save the model + finish the run record
 
model.save(str(MODEL_OUTPUT))
run_info['model_md5'] = md5(MODEL_OUTPUT)
run_info['finished'] = datetime.now().isoformat(timespec='seconds')
save_run_info()
print(f"\nSaved model to {MODEL_OUTPUT}")
print(f"Run record in {RUN_DIR}")