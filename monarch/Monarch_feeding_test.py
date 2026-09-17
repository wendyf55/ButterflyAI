#label monarch images

import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import random
import pandas as pd
import tf_keras as k3
import shutil
from pathlib import Path


from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEEDING_DATA_DIR = PROJECT_ROOT / "Final_Feeding_Images"
MONARCH_DIR = PROJECT_ROOT / "data" / "Monarch_images"
SEED_VALUE = 321
MODEL_FILENAME = "REAL_AND_SUPER_Final_feeding_model_unfrozen.keras"
PREDICTIONS_CSV = MONARCH_DIR / "Monarch_image_predictions.csv"
CLASS0_DIR = MONARCH_DIR / "Class_0"
CLASS1_DIR = MONARCH_DIR / "Class_1"
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
PREDICTION_THRESHOLD = 0.1

model_path = FEEDING_DATA_DIR / MODEL_FILENAME

print(os.path.exists(model_path))
model = load_model(str(model_path))
#model = load_model(FEEDING_DATA_DIR / 'Final_feeding_model_ALL.keras')
#model = load_model(FEEDING_DATA_DIR / 'REAL_AND_SUPER_Final_feeding_model_unfrozen.keras')

#test_df = pd.read_csv('Monarch_images.csv')
test_df = pd.read_csv(PREDICTIONS_CSV)

# Source folder containing your unlabeled images
source_folder = MONARCH_DIR

# Destination folders
class0_folder = CLASS0_DIR
class1_folder = CLASS1_DIR

os.makedirs(class0_folder, exist_ok=True)
os.makedirs(class1_folder, exist_ok=True)

# Move images based on prediction
for _, row in test_df.iterrows():
    filename = row['FileName']
    prediction = row['prediction']

    destination_path = source_folder / filename

    if prediction == 1:
        source_path = class1_folder / filename
    else:
        source_path = class0_folder / filename

    shutil.move(source_path, destination_path)

print("Images moved successfully")

# Preprocessing
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

img_height, img_width = IMG_HEIGHT, IMG_WIDTH
batch_size = BATCH_SIZE

# Generator for unlabeled images
generator = datagen.flow_from_dataframe(
    dataframe= test_df,
    directory=str(MONARCH_DIR),
    x_col='FileName',
    y_col=None,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)

# Get model predictions
preds = model.predict(generator)

# Convert predictions to a 1D array
scores = preds.flatten()

# Binary prediction using 0.5 threshold
predicted_classes = (scores > PREDICTION_THRESHOLD).astype(int)

# Add results to the dataframe
test_df['prediction'] = predicted_classes
test_df['score'] = scores

test_df.to_csv(PREDICTIONS_CSV, encoding='utf-8', index=False)


print("Csv saved")


# Source folder containing your unlabeled images
source_folder = MONARCH_DIR

# Destination folders
class0_folder = CLASS0_DIR
class1_folder = CLASS1_DIR

os.makedirs(class0_folder, exist_ok=True)
os.makedirs(class1_folder, exist_ok=True)

# Move images based on prediction
for _, row in test_df.iterrows():
    filename = row['FileName']
    prediction = row['prediction']

    source_path = source_folder / filename

    if prediction == 1:
        destination_path = class1_folder / filename
    else:
        destination_path = class0_folder / filename

    shutil.move(source_path, destination_path)

print("Images moved successfully")
