#Final feeding model

#Create the final feeding - non-feeding model 
#Make sure to verify training header

#GOAL = get 5 fold cross validated result and get standard deviation of the model metrics 

import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import random
import pandas as pd
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "feeding"
MODELS_DIR = MODULE_DIR / "models"
TRAINING_CSV = SPLITS_DIR / "dev_pool.csv"
MODEL_CHECKPOINT = MODELS_DIR / "FINAL_Real_Feeding_unfrozen_xval_augmented.keras"
SEED_VALUE = 321
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
VAL_BATCH_SIZE = 1
GENERATOR_SEED = 123
K_FOLDS = 5   # folds are fixed in dev_pool.csv (column `fold`, 0-4)
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 10
PREDICTION_THRESHOLD = 0.5
DROPOUT_RATE = 0.5
DENSE_UNITS = 512
TRAINABLE_BLOCK = "conv5_block"   # same as Final_feeding_model_train_on_ALL.py

from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#data/images/bimby_real and data/images/bimby_superimposed contain the 2023 BC BIMBY feeding and non-feeding photos and the superimposed photos
#image paths in the split CSVs are relative to the repo root (see data/README.md)
os.chdir(PROJECT_ROOT)

cwd = os.getcwd()
cwd

# The ResNet50 model expects images to be 224x224, so we set those values here
img_height, img_width = (IMG_HEIGHT, IMG_WIDTH)
batch_size = BATCH_SIZE

from sklearn.model_selection import train_test_split
import pandas as pd

#dev_pool.csv is in data/splits/feeding and has the feeding status, whether the image is real or superimposed, and its fold
full_df = pd.read_csv(TRAINING_CSV)
full_df = full_df.rename(columns={'image_path': 'filename'})  # image_path is relative to the repo root

#run code with just real images (all_df, name is misleading, watch out!)
#all_df = full_df[full_df['photo_type'] == 'real']
all_df = full_df

#K-fold cross validation
#folds come from dev_pool.csv: grouped (a superimposed image shares a fold with its source photo)
#and balanced on label x photo_type (made by data/scripts/make_splits.py)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y = all_df['label'].values
folds = all_df['fold'].values
fold_splits = [(np.where(folds != k)[0], np.where(folds == k)[0]) for k in range(K_FOLDS)]

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(y) # F=1 and B=0

X = all_df['filename'].values


#CREATE MODEL
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import random

#set seed so its always the same

os.environ['PYTHONHASHSEED']=str(SEED_VALUE)
 
random.seed(SEED_VALUE)

np.random.seed(SEED_VALUE)

tf.random.set_seed(SEED_VALUE)


accuracy_per_fold = []
loss_per_fold = []
precision_per_fold = []
recall_per_fold = []
f1_per_fold = []

# Example of a simple data generator setup
datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

# CHANGED: same augmentation as Final_feeding_model_train_on_ALL.py for the training folds (validation stays un-augmented)
train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                   horizontal_flip = True,
                                   shear_range = 0.2
                                  )


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

    # CHANGED: new callbacks each fold, and one checkpoint file per fold
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE)
    checkpoint = ModelCheckpoint(str(MODELS_DIR / f'{MODEL_CHECKPOINT.stem}_fold{fold}.keras'),
                                 monitor='val_loss', save_best_only=True)

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Create dataframes for training and validation
    train_df = pd.DataFrame({'filename': X_train, 'label': y_train})
    val_df = pd.DataFrame({'filename': X_val, 'label': y_val})

    # Create generators
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filename',
        y_col='label',
        target_size = (img_height, img_width),
        batch_size = batch_size,
        class_mode = 'binary',
        seed = GENERATOR_SEED
    )

    val_generator = datagen.flow_from_dataframe(
        val_df,
        x_col='filename',
        y_col='label',
        target_size = (img_height, img_width),
        batch_size = VAL_BATCH_SIZE,
        class_mode = 'binary',
        shuffle = False,
        seed = GENERATOR_SEED
    )

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[early_stopping, checkpoint]
    )

    #check accuracy metrics

    # Evaluate model on validation data
    val_predictions = model.predict(val_generator)
    val_predictions = (val_predictions > PREDICTION_THRESHOLD).astype(int)

    val_labels = val_generator.classes

    accuracy = accuracy_score(val_labels, val_predictions)
    precision = precision_score(val_labels, val_predictions)
    recall = recall_score(val_labels, val_predictions)
    f1 = f1_score(val_labels, val_predictions)
    loss = model.evaluate(val_generator, verbose=0)[0]

    accuracy_per_fold.append(accuracy)
    precision_per_fold.append(precision)
    recall_per_fold.append(recall)
    f1_per_fold.append(f1)
    loss_per_fold.append(loss)

    print(f'Fold completed. Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Loss: {loss:.4f}')

    # (the old "reload model weights for the next fold" line is no longer needed: each fold builds a fresh model)


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
