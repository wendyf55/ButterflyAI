#Testing step: how does baseline vs augmented model perform on feeding and non-feeding images with flowers?

#80-20 split

#train two versions of the 80 split, one with only real images, one with all images 

#test both models on testing set, which is sorted by real feeding and fake non-feeding (don't care about class balancing?)


import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import random
import pandas as pd
from tensorflow.keras.optimizers import Adam

seed_value= 321 

from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import precision_recall_curve, average_precision_score



#The folder Final_Feeding_Images contains the 2023 BC BIMBY feeding and non-feeding photos as well as the superimposed photos 
os.chdir('Final_Feeding_Images')

cwd = os.getcwd()
cwd

# The ResNet50 model expects images to be 224x224, so we set those values here
img_height, img_width = (224,224)
batch_size = 32

from sklearn.model_selection import train_test_split
import pandas as pd

#DataFilenamesRedo.csv is in the Final_Feeding_Images folder and has the feeding status and whether the image is real or not
full_df = pd.read_csv('DataFilenamesRedo.csv')

stratify_col = (
    full_df["label"].astype(str) + "_" +
    full_df["photo_type"].astype(str)
)

train_df, val_df = train_test_split(
    full_df,
    test_size=0.2,
    random_state=seed_value,
    stratify=stratify_col
)

print(f"Train images: {len(train_df)} | Val images: {len(val_df)}")

#run code with just real images (all_df, name is misleading, watch out!)
train_real_df = train_df[train_df['photo_type'] == 'real']

train_all_df = train_df

val_df = val_df = val_df[
    (val_df["label"] == "F") |
    ((val_df["label"] == "N") & (val_df["photo_type"] == "super"))
].copy()

print(train_real_df.head)
print(train_all_df.head)
print(val_df.head)

train_df = train_real_df #only train on real data

#CREATE MODEL
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import random

#set seed so its always the same

os.environ['PYTHONHASHSEED']=str(seed_value)
 
random.seed(seed_value)

np.random.seed(seed_value)

tf.random.set_seed(seed_value)


# The code in my tutorial had a few additional layers, but I would try this out too. Sometimes simpler is better
base_model = ResNet50(include_top = False, weights = 'imagenet')
x = base_model.output

#add layers, but could overfit

#x = Dense(256, activation='relu', kernel_regularizer=l2(0.1))(x)

#change from 0.1 to 0.2 to 0.5 to reduce overfitting
#x = Dropout(0.1)(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation = 'sigmoid')(x)

#base_model = ResNet50(include_top=False, weights='imagenet')
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#x = Dropout(0.5)(x)

#x = Dense(512)(x)
#x = BatchNormalization()(x)
#x = Activation('relu')(x)
#x = Dropout(0.5)(x)

#predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs = base_model.input, outputs = predictions)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

accuracy_per_fold = []
loss_per_fold = []
precision_per_fold = []
recall_per_fold = []
f1_per_fold = []

# Example of a simple data generator setup
datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

#checkpoint = ModelCheckpoint('FEED_TEST_trained_real.keras', monitor='val_loss', save_best_only=True)

# Create generators
train_generator = datagen.flow_from_dataframe(
        train_df,
        x_col='filename',
        y_col='label',
        target_size = (img_height, img_width),
        batch_size = batch_size,
        class_mode = 'binary',
    )

val_generator = datagen.flow_from_dataframe(
        val_df,
        x_col='filename',
        y_col='label',
        target_size = (img_height, img_width),
        batch_size = 1,
        class_mode = 'binary',
        shuffle = False,
        seed = 123
    )


history = model.fit(
        train_generator,
        epochs=20
    )

#check accuracy metrics
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, f1_score

test_generator = val_generator

np.random.seed(seed_value)
tf.random.set_seed(seed_value)

valid_loss, valid_acc = model.evaluate(test_generator, verbose = 1) 

print(valid_loss, valid_acc)

preds = model.predict(test_generator)


predicted_classes = (preds > 0.5).astype(int).flatten()  # Flatten in case preds is a 2D array

true_classes = test_generator.classes

report = classification_report(true_classes, 
                               predicted_classes, 
                               output_dict=True)

macro_precision = report['macro avg']['precision']
macro_recall = report['macro avg']['recall']
macro_f1_score = report['macro avg']['f1-score']

print(f"Macro Precision: {macro_precision}")
print(f"Macro Recall: {macro_recall}")
print(f"Macro F1-Score: {macro_f1_score}")

conf_matrix = confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)

print("Testing class indices: ", test_generator.class_indices)

test_df = val_df

# Initialize a counter for false positives and an empty list for filenames and labels
false_positive_count = 0
false_positive_details = []

# Iterate through the true and predicted classes, along with the filenames
for i, (true, pred) in enumerate(zip(true_classes, predicted_classes)):
    if true == 0 and pred == 1:  # Check if actual is 0 (f) and predicted is 1 (nf)
        false_positive_count += 1
        # Get the corresponding filename and label
        filename = test_df.iloc[i]['filename']
        label = test_df.iloc[i]['label'] 
        false_positive_details.append((filename, label))

# Print the manual false positive count
print(f"Manual False Positive Count: {false_positive_count}")

# Print the filenames and labels of the false positives
print("False Positive Filenames and Labels:")
for filename, label in false_positive_details:
    print(f"Filename: {filename}, Label: {label}")


# Initialize a counter for false negatives and an empty list for filenames and labels
false_negative_count = 0
false_negative_details = []

# Iterate through the true and predicted classes, along with the filenames
for i, (true, pred) in enumerate(zip(true_classes, predicted_classes)):
    if true == 1 and pred == 0:  # Check if actual is 1 (nf) and predicted is 0 (f)
        false_negative_count += 1
        # Get the corresponding filename and label
        filename = test_df.iloc[i]['filename']
        label = test_df.iloc[i]['label'] 
        false_negative_details.append((filename, label))

# Print the manual false negative count
print(f"Manual False Negative Count: {false_negative_count}")

# Print the filenames and labels of the false negatives
print("False Negative Filenames and Labels:")
for filename, label in false_negative_details:
    print(f"Filename: {filename}, Label: {label}")


print("Train generator class indices:")
print(train_generator.class_indices)

print("\nTest generator class indices:")
print(test_generator.class_indices)