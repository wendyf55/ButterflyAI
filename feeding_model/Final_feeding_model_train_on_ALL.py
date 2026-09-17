#Final feeding model, train on all of the available 2023 data

#Create the final feeding - non-feeding model 
#Make sure to verify training header

import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import random
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEEDING_DATA_DIR = PROJECT_ROOT / "Final_Feeding_Images"
TRAINING_CSV = FEEDING_DATA_DIR / "DataFilenamesRedo_train.csv"
MODEL_OUTPUT = FEEDING_DATA_DIR / "REAL_AND_SUPER_Final_feeding_model_unfrozen.keras"
SEED_VALUE = 321
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
GENERATOR_SEED = 123
EPOCHS = 20
DROPOUT_RATE = 0.5
DENSE_UNITS = 512
TRAINABLE_BLOCK = "conv5_block"

from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#The folder Final_Feeding_Images contains the 2023 BC BIMBY feeding and non-feeding photos as well as the superimposed photos 
os.chdir(FEEDING_DATA_DIR)

cwd = os.getcwd()
cwd

# The ResNet50 model expects images to be 224x224, so we set those values here
img_height, img_width = (IMG_HEIGHT, IMG_WIDTH)
batch_size = BATCH_SIZE

from sklearn.model_selection import train_test_split
import pandas as pd

#DataFilenamesRedo.csv is in the Final_Feeding_Images folder and has the feeding status and whether the image is real or not
full_df = pd.read_csv(TRAINING_CSV)

#run code with just real images (all_df, name is misleading, watch out!)
# train_df = full_df[full_df['photo_type'] == 'real']
train_df = full_df



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


# The code in my tutorial had a few additional layers, but I would try this out too. Sometimes simpler is better
#base_model = ResNet50(include_top = False, weights = 'imagenet')
#x = base_model.output

#add layers, but could overfit

#x = Dense(256, activation='relu', kernel_regularizer=l2(0.1))(x)

#change from 0.1 to 0.2 to 0.5 to reduce overfitting
#x = Dropout(0.1)(x)
#x = GlobalAveragePooling2D()(x)
#x = Dropout(0.5)(x)
#x = Dense(512, activation='relu')(x)
#x = Dropout(0.5)(x)
#predictions = Dense(1, activation = 'sigmoid')(x)


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
        if TRAINABLE_BLOCK in layer.name: #tune just the final block ~15-20 layers
            layer.trainable = True
        else:
            layer.trainable = False


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                   horizontal_flip = True,
                                   shear_range = 0.2
                                  )

train_generator = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    x_col = 'filename',
    y_col = 'label',
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'binary',
    seed = GENERATOR_SEED,
    shuffle = True
)

# Print the number of validated images found
print(f"Number of training images found: {train_generator.samples}")

# Print the number of images of each class for training data
print("Training class indices: ", train_generator.class_indices)
print("Training labels count: ", train_df['label'].value_counts())

history = model.fit(train_generator, 
          epochs = EPOCHS)

model.save(str(MODEL_OUTPUT))
