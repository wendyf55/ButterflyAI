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


from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

seed_value = 321


os.chdir('/home/jsieg/butterflyAI/Final_Feeding_Images')


model_path = "/mnt/sharedstorage/jsieg/butterflyAI/Final_Feeding_Images/REAL_AND_SUPER_Final_feeding_model_unfrozen.keras"

print(os.path.exists(model_path))
model = load_model(model_path)
#model = load_model(os.path.join('/home/jsieg/butterflyAI/Final_Feeding_Images/Final_feeding_model_ALL.keras'))
#model = load_model(os.path.join('REAL_AND_SUPER_Final_feeding_model_unfrozen.keras'))

#model = k3.models.load_model(os.path.join('/mnt/sharedstorage/jsieg/butterflyAI/Final_Feeding_Images'))


os.chdir('/mnt/sharedstorage/jsieg/butterflyAI/Monarch_images')

#test_df = pd.read_csv('Monarch_images.csv')
test_df = pd.read_csv('Monarch_image_predictions.csv')

# Source folder containing your unlabeled images
source_folder = "/mnt/sharedstorage/jsieg/butterflyAI/Monarch_images"

# Destination folders
class0_folder = "Class_0"
class1_folder = "Class_1"

os.makedirs(class0_folder, exist_ok=True)
os.makedirs(class1_folder, exist_ok=True)

# Move images based on prediction
for _, row in test_df.iterrows():
    filename = row['FileName']
    prediction = row['prediction']

    destination_path = os.path.join(source_folder, filename)

    if prediction == 1:
        source_path = os.path.join(class1_folder, filename)
    else:
        source_path = os.path.join(class0_folder, filename)

    shutil.move(source_path, destination_path)

print("Images moved successfully")

# Preprocessing
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

img_height, img_width = 224, 224
batch_size = 32

# Generator for unlabeled images
generator = datagen.flow_from_dataframe(
    dataframe= test_df,
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
predicted_classes = (scores > 0.1).astype(int)

# Add results to the dataframe
test_df['prediction'] = predicted_classes
test_df['score'] = scores

test_df.to_csv('/mnt/sharedstorage/jsieg/butterflyAI/Monarch_images/Monarch_image_predictions.csv', encoding='utf-8', index=False)


print("Csv saved")


# Source folder containing your unlabeled images
source_folder = "/mnt/sharedstorage/jsieg/butterflyAI/Monarch_images"

# Destination folders
class0_folder = "Class_0"
class1_folder = "Class_1"

os.makedirs(class0_folder, exist_ok=True)
os.makedirs(class1_folder, exist_ok=True)

# Move images based on prediction
for _, row in test_df.iterrows():
    filename = row['FileName']
    prediction = row['prediction']

    source_path = os.path.join(source_folder, filename)

    if prediction == 1:
        destination_path = os.path.join(class1_folder, filename)
    else:
        destination_path = os.path.join(class0_folder, filename)

    shutil.move(source_path, destination_path)

print("Images moved successfully")