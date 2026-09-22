#Train a monarch feeding model

#Create the feeding - non-feeding model for monarchs using 5-fold cross validation
# this is not the final model necessarily (final model would be trained on all images)
# but a cross0validated model gives mean and SD accuracy
#Make sure to verify training header

#GOAL = get 5 fold cross validated result and get standard deviation of the model metrics 

# updates WF made, Sept. 22nd: 

# 1. PER-FOLD MODEL RESET
# The original built 1 model outside the fold loop and reused it across all 5 folds 
# (the `load_weights` reset line was commented out). So each fold kept
# training on top of the previous fold's weights, and by fold 5 the model had
# effectively already seen most of the data. That is why the original per-fold
# accuracy climbed monotonically (0.84 -> 0.98) and why the 0.906 mean was
# optimistically biased. Here a fresh model is built inside the loop each fold,
# with clear_session() + reseed, so folds are independent and comparable.
#
# 2. tf.data INPUT PIPELINE (replaces ImageDataGenerator).
# Keras 3 removed ImageDataGenerator, so the original import no longer exists. The replacement
# is a tf.data pipeline with parallel JPEG decode + prefetch, which also keeps
# the Metal GPU fed instead of bottlenecking on single-thread PIL loading.
#
# 3. EXPLICIT LABEL ENCODING: Feeding = 1, Non_feeding = 0.
# The original let flow_from_dataframe assign class indices alphabetically,
# which makes 'Feeding' = 0 and 'Non_feeding' = 1. sklearn's precision/recall
# then defaulted to pos_label=1 = Non_feeding, so the reported "precision/recall"
# were actually for the NON-feeding class.
#
# 4. PER-FOLD CHECKPOINTS. The original ModelCheckpoint wrote a single filename,
# so each fold overwrote the last. Each fold now saves its own file, and
# EarlyStopping(restore_best_weights=True) guarantees the evaluated model is the
# best-val-loss epoch (the original evaluated the last in-memory epoch, which
# was not necessarily the checkpointed one).
#
# 5. No os.chdir needed, pipeline uses absolute image paths
#
# Run in the GPU env:  conda activate butterflyai_gpu
# (tensorflow==2.19.1 + tensorflow-metal==1.2.0)

######################################################################################

import os
import sys
import numpy as np
# from matplotlib import pyplot as plt #unused import
import random
import pandas as pd

from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#from tensorflow.keras.preprocessing.image import ImageDataGenerator #CHANGED: removed in keras 3
#from tensorflow.keras.regularizers import l2    # CHANGED: only used by commented-out layers

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.model_selection import train_test_split

####################################################################################

seed_value= 321

#The folder Monarch contains API downloaded monarch images labeled as feeding or non-feeding by Julie Aug 10-11, 2026

MODULE_DIR = Path(__file__).resolve().parent
MONARCH_DIR = MODULE_DIR / "data" / "Monarch_images"
MODELS_DIR = MODULE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RESULTS_DIR = MODULE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

class Tee:
    """Write everything printed to the console to a log file as well."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()          # flush so the log updates live during training
    def flush(self):
        for s in self.streams:
            s.flush()

_logfile = open(RESULTS_DIR / "Monarch_xval_output_fixed.txt", "w")
sys.stdout = Tee(sys.__stdout__, _logfile)
sys.stderr = Tee(sys.__stderr__, _logfile)

# os.chdir(MONARCH_DIR) #not needed - tf.data ppipeline uses absolute paths

# The ResNet50 model expects images to be 224x224, so we set those values here
img_height, img_width = (224,224)
batch_size = 32

#Monarch_image_labels.csv is in the Monarch_images folder and has the feeding status and whether the image is real or not
all_df = pd.read_csv(MONARCH_DIR / 'Monarch_image_labels.csv')
all_df = all_df[all_df['Label'].isin(['Feeding', 'Non_feeding'])].reset_index(drop=True)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# CHANGED: numeric labels, Feeding=1 / Non_feeding=0
y = (all_df['Label'].values == 'Feeding').astype('int32')
#y = all_df['Label'].values #this was the original

#from sklearn.preprocessing import LabelEncoder
#labelencoder = LabelEncoder()
#Y = labelencoder.fit_transform(y) # F=1 and B=0 # Y was computed but never used

X = all_df['FileName'].values

#set seed so its always the same

os.environ['PYTHONHASHSEED']=str(seed_value)
 
random.seed(seed_value)

np.random.seed(seed_value)

tf.random.set_seed(seed_value)

###############################################################################################
# tf.data helpers (replace ImageDataGenerator)

AUTOTUNE = tf.data.AUTOTUNE

def _load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])   # -> float32, still 0..255
    img = preprocess_input(img) # ResNet50 (caffe) preprocessing
    return img, label

def make_dataset(filenames, labels, training):
    paths = [str(MONARCH_DIR / f) for f in filenames]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels.astype('float32')))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=123, reshuffle_each_iteration=True)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

accuracy_per_fold = []
loss_per_fold = []
precision_per_fold = []
recall_per_fold = []
f1_per_fold = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):   # CHANGED: fold counter added
    print(f"\n Fold {fold}/5 ")

    # CHANGED: reset before each fold so folds are independent
    tf.keras.backend.clear_session()
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # CHANGED: build a FRESH model each fold (this block was originally outside the loop)
    base_model = ResNet50(include_top = False, weights = 'imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs = base_model.input, outputs = predictions)
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(str(MODELS_DIR / f'Monarch_feeding_xval_fold{fold}.keras'),
                                 monitor='val_loss', save_best_only=True)   # CHANGED: per-fold name

    # CHANGED: datasets instead of flow_from_dataframe generators
    train_ds = make_dataset(X_train, y_train, training=True)
    val_ds = make_dataset(X_val, y_val, training=False)   # unshuffled: order preserved for metrics

    history = model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        callbacks=[early_stopping, checkpoint]
    )

    #check accuracy metrics

    # Evaluate model on validation data
    val_predictions = model.predict(val_ds)
    val_predictions = (val_predictions > 0.5).astype(int)

    val_labels = y_val   # CHANGED: use this fold's held-out labels (val_ds is unshuffled)

    accuracy = accuracy_score(val_labels, val_predictions)
    precision = precision_score(val_labels, val_predictions, zero_division=0)   # pos_label=1 = Feeding
    recall = recall_score(val_labels, val_predictions, zero_division=0)
    f1 = f1_score(val_labels, val_predictions, zero_division=0)
    loss = model.evaluate(val_ds, verbose=0)[0]

    accuracy_per_fold.append(accuracy)
    precision_per_fold.append(precision)
    recall_per_fold.append(recall)
    f1_per_fold.append(f1)
    loss_per_fold.append(loss)

    print(f'Fold completed. Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Loss: {loss:.4f}')

# Convert lists to numpy arrays for easier computation
accuracy_per_fold = np.array(accuracy_per_fold)
precision_per_fold = np.array(precision_per_fold)
recall_per_fold = np.array(recall_per_fold)
f1_per_fold = np.array(f1_per_fold)
loss_per_fold = np.array(loss_per_fold)

# Print averages and standard deviations
print("Cross-Validation Results (5 folds):")
print(f"Accuracy: {accuracy_per_fold.mean():.4f} ± {accuracy_per_fold.std():.4f}")
print(f"Precision: {precision_per_fold.mean():.4f} ± {precision_per_fold.std():.4f}")
print(f"Recall: {recall_per_fold.mean():.4f} ± {recall_per_fold.std():.4f}")
print(f"F1 Score: {f1_per_fold.mean():.4f} ± {f1_per_fold.std():.4f}")
print(f"Loss: {loss_per_fold.mean():.4f} ± {loss_per_fold.std():.4f}")

# The code in my tutorial had a few additional layers, but I would try this out too. Sometimes simpler is better
#base_model = ResNet50(include_top = False, weights = 'imagenet')
#x = base_model.output

#add layers, but could overfit

#x = Dense(256, activation='relu', kernel_regularizer=l2(0.1))(x)

#change from 0.1 to 0.2 to 0.5 to reduce overfitting
#x = Dropout(0.1)(x)
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.5)(x)
# x = Dense(512, activation='relu')(x)
# x = Dropout(0.5)(x)
# predictions = Dense(1, activation = 'sigmoid')(x)

#base_model = ResNet50(include_top=False, weights='imagenet')
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#x = Dropout(0.5)(x)

#x = Dense(512)(x)
#x = BatchNormalization()(x)
#x = Activation('relu')(x)
#x = Dropout(0.5)(x)

#predictions = Dense(1, activation='sigmoid')(x)

# model = Model(inputs = base_model.input, outputs = predictions)
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# accuracy_per_fold = []
# loss_per_fold = []
# precision_per_fold = []
# recall_per_fold = []
# f1_per_fold = []

# # Example of a simple data generator setup
# datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

# checkpoint = ModelCheckpoint(str(MODELS_DIR / 'Monarch_feeding_train_xval.keras'), monitor='val_loss', save_best_only=True)


# for train_idx, val_idx in kfold.split(X):
#     X_train, X_val = X[train_idx], X[val_idx]
#     y_train, y_val = y[train_idx], y[val_idx]

#     # Create dataframes for training and validation
#     train_df = pd.DataFrame({'FileName': X_train, 'Label': y_train})
#     val_df = pd.DataFrame({'FileName': X_val, 'Label': y_val})

#     # Create generators
#     train_generator = datagen.flow_from_dataframe(
#         train_df,
#         x_col='FileName',
#         y_col='Label',
#         target_size = (img_height, img_width),
#         batch_size = batch_size,
#         class_mode = 'binary',
#         seed = 123
#     )

#     val_generator = datagen.flow_from_dataframe(
#         val_df,
#         x_col='FileName',
#         y_col='Label',
#         target_size = (img_height, img_width),
#         batch_size = 1,
#         class_mode = 'binary',
#         shuffle = False,
#         seed = 123
#     )

#     history = model.fit(
#         train_generator,
#         epochs=10,
#         validation_data=val_generator,
#         callbacks=[early_stopping, checkpoint]
#     )

#     #check accuracy metrics

#     # Evaluate model on validation data
#     val_predictions = model.predict(val_generator)
#     val_predictions = (val_predictions > 0.5).astype(int)

#     val_labels = val_generator.classes

#     accuracy = accuracy_score(val_labels, val_predictions)
#     precision = precision_score(val_labels, val_predictions)
#     recall = recall_score(val_labels, val_predictions)
#     f1 = f1_score(val_labels, val_predictions)
#     loss = model.evaluate(val_generator, verbose=0)[0]

#     accuracy_per_fold.append(accuracy)
#     precision_per_fold.append(precision)
#     recall_per_fold.append(recall)
#     f1_per_fold.append(f1)
#     loss_per_fold.append(loss)

#     print(f'Fold completed. Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Loss: {loss:.4f}')

#     tf.keras.backend.clear_session()


#     # Reload model weights for the next fold
#     # ask why this line is included
#     #model.load_weights('best_weights_ResNet50_binary_superimposed2.keras')


# # Convert lists to numpy arrays for easier computation
# accuracy_per_fold = np.array(accuracy_per_fold)
# precision_per_fold = np.array(precision_per_fold)
# recall_per_fold = np.array(recall_per_fold)
# f1_per_fold = np.array(f1_per_fold)
# loss_per_fold = np.array(loss_per_fold)

# # Print averages and standard deviations
# print("Cross-Validation Results (5 folds):")
# print(f"Accuracy: {accuracy_per_fold.mean():.4f} ± {accuracy_per_fold.std():.4f}")
# print(f"Precision: {precision_per_fold.mean():.4f} ± {precision_per_fold.std():.4f}")
# print(f"Recall: {recall_per_fold.mean():.4f} ± {recall_per_fold.std():.4f}")
# print(f"F1 Score: {f1_per_fold.mean():.4f} ± {f1_per_fold.std():.4f}")
# print(f"Loss: {loss_per_fold.mean():.4f} ± {loss_per_fold.std():.4f}")
