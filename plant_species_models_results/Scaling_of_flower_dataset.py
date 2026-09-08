# Make a for loop of training the flower dataset and save the macro accuracy and precision using different subsets of data


#Multiclass plant classifier
#Retrain model on all of the filtered flower photos where a detectron butterfly were added
#Train model on all data for testing on gold standard dataset

import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import random
import pandas as pd

import json
import shutil

from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split


cwd = os.getcwd()
cwd

#set seed so its always the same
seed_value= 321

os.environ['PYTHONHASHSEED']=str(seed_value)

random.seed(seed_value)

np.random.seed(seed_value)

tf.random.set_seed(seed_value)

os.chdir('specified_flower_photos_detectron_ALL')

entries = os.listdir()

# The ResNet50 model expects images to be 224x224, so we set those values here
img_height, img_width = (224, 224)
batch_size = 64 #32 - powers of 2

#all_df = pd.read_csv('PlantIDs_API.csv')
all_df = pd.read_csv('specified_flower_photos_detectron.csv')

train_df, test_df = train_test_split(all_df, test_size=0.2, random_state=42)

percentages = [0.05, 0.10, 0.20, 0.50, 0.75, 1.00]

results = []

for p in percentages:
    # sample p% of the dataframe
    new_df = train_df.sample(frac=p, random_state=42)

    train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                   horizontal_flip = True,
                                   shear_range = 0.2
                                  )

    test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input
                                    #rescale=1./255
                                    )

    # # Data generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe= new_df,
        x_col='Filename',
        y_col='Label',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode = 'categorical',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe= test_df,
        x_col='Filename',
        y_col='Label',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode = 'categorical',
        shuffle=False
    )
    # Print the number of validated images found
    print(f"Number of training images found: {train_generator.samples}")

    # Print the number of images of each class for training data
    print("Training class indices: ", train_generator.class_indices)
    print("Training labels count: ", new_df['Label'].value_counts())


    #first model , dropout - 0,5, dense - 512

    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2


    base_model = ResNet50(include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation = 'softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        if "conv5_block" in layer.name: #tune just the final block ~15-20 layers
            layer.trainable = True
        else:
            layer.trainable = False

    # Compile the model (ensuring it's ready for training)
    model.compile(optimizer=Adam(learning_rate=1e-5), loss = 'categorical_crossentropy', metrics = ['accuracy'])


    ###### Set the epochs to however many you want

    cv_results = {}
    confusion_matrices = {}
    accuracy_loss_data = {}
    prediction_val = {}

    #early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    #checkpoint = ModelCheckpoint('Multiclass_detectron_specified_ALL.keras', monitor='val_loss', save_best_only=True)


    from sklearn.utils.class_weight import compute_class_weight

    # Get the class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )

    # Turn into dict for model.fit
    class_weights = dict(enumerate(class_weights))

    # Use in training
    history = model.fit(train_generator,
                        epochs=20,
                        class_weight=class_weights)

    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np

    # Get the model predictions
    preds = model.predict(test_generator)

    # Get the predicted class with the highest probability for each instance
    predicted_classes = np.argmax(preds, axis=1)  # axis=1 to get class index with highest probability for each image

    # True labels from the test generator
    true_classes = test_generator.classes

    class_indices = test_generator.class_indices 

    # Generate the classification report
    report = classification_report(true_classes, 
                                predicted_classes, 
                                output_dict=True)

    class_names = [None] * len(class_indices)
    for class_name, index in class_indices.items():
        class_names[index] = class_name

    # Extract macro-averaged precision, recall, and F1 score
    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    macro_f1_score = report['macro avg']['f1-score']

    # Print the metrics
    print(f"Macro Precision: {macro_precision}")
    print(f"Macro Recall: {macro_recall}")
    print(f"Macro F1-Score: {macro_f1_score}")

    # Generate and print the confusion matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    print(conf_matrix)

    # Print per-class metrics
    print("\nPer-Class Metrics:")
    print(f"{'Class':<35} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 80)

    for idx, class_name in enumerate(class_names):
        if str(idx) in report:
            class_report = report[str(idx)]
            print(f"{class_name:<35} "
                f"{class_report['precision']:>10.4f} "
                f"{class_report['recall']:>10.4f} "
                f"{class_report['f1-score']:>10.4f} "
                f"{int(class_report['support']):>10}")


    results.append({
        "percentage": int(p * 100),
        "precision": macro_precision
    })


results_df = pd.DataFrame(results)
results_df.to_csv("Scaling_test_results.csv", index=False)
