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

seed_value= 321

from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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

#run code with just real images (all_df, name is misleading, watch out!)
#all_df = full_df[full_df['photo_type'] == 'real']
all_df = full_df

#K-fold cross validation
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

y = all_df['label'].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(y) # F=1 and B=0

X = all_df['filename'].values


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

checkpoint = ModelCheckpoint('FINAL_Real_Feeding_unfrozen_xval_augmented.keras', monitor='val_loss', save_best_only=True)


for train_idx, val_idx in kfold.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Create dataframes for training and validation
    train_df = pd.DataFrame({'filename': X_train, 'label': y_train})
    val_df = pd.DataFrame({'filename': X_val, 'label': y_val})

    # Create generators
    train_generator = datagen.flow_from_dataframe(
        train_df,
        x_col='filename',
        y_col='label',
        target_size = (img_height, img_width),
        batch_size = batch_size,
        class_mode = 'binary',
        seed = 123
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
        epochs=10,
        validation_data=val_generator,
        callbacks=[early_stopping, checkpoint]
    )

    #check accuracy metrics

    # Evaluate model on validation data
    val_predictions = model.predict(val_generator)
    val_predictions = (val_predictions > 0.5).astype(int)

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

    tf.keras.backend.clear_session()


    # Reload model weights for the next fold
    # ask why this line is included
    #model.load_weights('best_weights_ResNet50_binary_superimposed2.keras')


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
