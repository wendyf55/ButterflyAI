#Final feeding model, train on all of the available 2023 data

#Create the final feeding - non-feeding model 
#Make sure to verify training header

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
full_df = pd.read_csv('DataFilenamesRedo_train.csv')

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

os.environ['PYTHONHASHSEED']=str(seed_value)
 
random.seed(seed_value)

np.random.seed(seed_value)

tf.random.set_seed(seed_value)


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
x = Dropout(0.5)(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs = base_model.input, outputs = predictions)


for layer in base_model.layers:
        if "conv5_block" in layer.name: #tune just the final block ~15-20 layers
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
    seed = 123,
    shuffle = True
)

# Print the number of validated images found
print(f"Number of training images found: {train_generator.samples}")

# Print the number of images of each class for training data
print("Training class indices: ", train_generator.class_indices)
print("Training labels count: ", train_df['label'].value_counts())

history = model.fit(train_generator, 
          epochs = 20)

model.save('REAL_AND_SUPER_Final_feeding_model_unfrozen.keras')


