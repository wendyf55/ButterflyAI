#PLANT OVR WITH CROSS VALIDATION

# Get 5 fold accuracy for OVR with new compiled flower photos 
import os
import numpy as np
from matplotlib import pyplot as plt
import random
import pandas as pd

import tensorflow as tf

import json
import shutil

from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import defaultdict

#does tf detect the GPU?
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Devices: ", tf.config.list_physical_devices())

cwd = os.getcwd()
cwd

os.chdir('plant_data_specified')


entries = os.listdir()

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import random
import os
import numpy as np
#set seed so its always the same
seed_value= 321

os.environ['PYTHONHASHSEED']=str(seed_value)

random.seed(seed_value)

np.random.seed(seed_value)

tf.random.set_seed(seed_value)

#first model , dropout - 0,5, dense - 512
def create_ovr_model():

  base_model = ResNet50(include_top=False, weights='imagenet')
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.5)(x)
  x = Dense(512)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.5)(x)
  predictions = Dense(1, activation='sigmoid')(x)
  model = Model(inputs=base_model.input, outputs=predictions)

  

  # Compile the model (ensuring it's ready for training)
  model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

  return model

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample

#
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix



import numpy as np

# Parameters
img_height, img_width = (224, 224)
batch_size = 64 #32 - powers of 2
k_folds = 5  # Number of folds for cross-validation

# Load dataset
all_df = pd.read_csv('Flower_only_specified.csv')

# Data generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    shear_range=0.2
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Dictionary to store performance metrics
cv_results = {}
confusion_matrices = {}
accuracy_loss_data = {}

for class_index, class_name in enumerate(all_df['Label'].unique()):
    
    # Create binary labels for the current class
    all_df_copy = all_df.copy()
    all_df_copy['BinaryLabel'] = (all_df_copy['Label'] == class_name).astype(int)
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_metrics = []
    #conf_matrix = {}
    conf_metrics = {}
    class_reports = {}
    class_accuracy_loss = {}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_df_copy, all_df_copy['BinaryLabel'])):
        print(f"Training Fold {fold+1}/{k_folds} for class: {class_name}")

        # Split data
        train_df, val_df = all_df_copy.iloc[train_idx], all_df_copy.iloc[val_idx]
        
        # Handle class imbalance (resampling)
        majority_train = train_df[train_df['BinaryLabel'] == 0]
        minority_train = train_df[train_df['BinaryLabel'] == 1]

        if len(majority_train) > len(minority_train):
            majority_train_resampled = resample(
                majority_train, replace=False, n_samples=len(minority_train), random_state=42
            )
        else:
            majority_train_resampled = majority_train

        balanced_train = pd.concat([majority_train_resampled, minority_train]).sample(frac=1, random_state=42)

         # Print class distribution after resampling
        # print("\nAfter resampling:")
        # print("Balanced Train class distribution:")
        # print(balanced_train['BinaryLabel'].value_counts())

        # Convert labels to string format for the generators
        balanced_train['BinaryLabel'] = balanced_train['BinaryLabel'].astype(str)

        #VALIDATION
        # Handle class imbalance in testing set
        majority_val = val_df[val_df['BinaryLabel'] == 0]
        minority_val = val_df[val_df['BinaryLabel'] == 1]

        if len(majority_val) > len(minority_val):
            majority_val_resampled = resample(
                majority_val, replace=False, n_samples=len(minority_val), random_state=42
            )
            print("val resampled")
        else:
            majority_val_resampled = majority_val
            print("val not resampled")
 
        balanced_val = pd.concat([majority_val_resampled, minority_val])
        balanced_val = balanced_val.sample(frac=1, random_state=42).reset_index(drop=True)

        # Convert labels to string for generators
        
        balanced_val['BinaryLabel'] = balanced_val['BinaryLabel'].astype(str)

        #Print number of images
        class_counts = balanced_val['BinaryLabel'].value_counts()
        print(f"Val class counts = {class_counts}")


        # # Data generators
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=balanced_train,
            x_col='Filename',
            y_col='BinaryLabel',
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True
        )

        val_generator = test_datagen.flow_from_dataframe(
            dataframe=balanced_val,
            x_col='Filename',
            y_col='BinaryLabel',
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )

        # Print class distribution after resampling
        print(f"After resampling (Fold {fold+1}, Class {class_name}):")
        print(balanced_train['BinaryLabel'].value_counts().to_dict())

        # Create and train model
        model = create_ovr_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            train_generator,
            epochs= 20,
            validation_data=val_generator,
            callbacks=[early_stopping]
        )
        
               # Create a safe file-friendly name
        safe_name = class_name.replace(" ", "_").replace("/", "_")

        # Save model for this class (OVR classifier)
        model.save(f"ovr_model_{safe_name}.keras")
        print(f"Saved OVR model for class: {class_name} ➜ ovr_model_{safe_name}.keras")

        # Store accuracy and loss data
        class_accuracy_loss[f'fold_{fold+1}'] = {
            'val_loss': history.history['val_loss'],
            'val_accuracy': history.history['val_accuracy']
        }

        # Evaluate model on validation set
        val_loss, val_acc = model.evaluate(val_generator)

        # Ensure y_true matches val_generator's actual labels
        y_true = np.array(val_generator.classes)
        print(f"Shape of y_true: {y_true.shape}")

        # Ensure model processes all samples
        steps_per_epoch = int(np.ceil(len(val_generator.filenames) / batch_size))
        y_pred_probs = model.predict(val_generator, steps=steps_per_epoch, verbose=1)
        y_pred_probs = y_pred_probs.ravel()

        print(f"Shape of y_pred_probs: {y_pred_probs.shape}")

        # Ensure lengths match
        assert len(y_true) == len(y_pred_probs), f"Mismatch: y_true={len(y_true)}, y_pred_probs={len(y_pred_probs)}"

        # Compute precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_probs)

        # Get model predictions
        #y_true = val_df['BinaryLabel'].astype(int).values  # True labels
        #y_pred_probs = model.predict(val_generator)  # Predicted probabilities
        #best_threshold_index = np.argmax(precisions[:-1] * recalls[:-1])  # Remove last element
        #best_threshold = thresholds[best_threshold_index]

        #print(f"Optimal threshold: {best_threshold}")
        
        # Apply new threshold
        #y_pred = (y_pred_probs > best_threshold).astype(int)
        
        y_pred = (y_pred_probs > 0.5).astype(int)  # Convert probabilities to binary predictions

        # Compute precision, recall, and F1-score
        report = classification_report(y_true, y_pred, output_dict=True)

        fold_index = len(fold_metrics) + 1 
        #conf_matrix[f'fold_{fold_index}'] = confusion_matrix(y_true, y_pred)

        # Convert confusion matrix to list before storing
        conf_matrix = confusion_matrix(y_true, y_pred).tolist()
        conf_metrics[f'fold_{fold+1}'] = conf_matrix

        #class_reports[f'fold_{fold+1}'] = report
        
        #conf_metrics[f'fold_{fold+1}'] = confusion_matrix(y_true, y_pred)

        # Store fold metrics
        fold_metrics.append({
            'val_loss': val_loss,
            'val_acc': val_acc,
            'precision': report['1']['precision'],  # Precision for the positive class
            'recall': report['1']['recall'],        # Recall for the positive class
            'f1_score': report['1']['f1-score']     # F1-score for the positive class
        })

        from tensorflow.keras import backend as K
        import gc

        K.clear_session()
        gc.collect()
    
    # Store results for the class
    cv_results[class_name] = fold_metrics
    confusion_matrices[class_name] = conf_metrics
    accuracy_loss_data[class_name] = class_accuracy_loss

# Save dictionaries as JSON files
with open('aug4_cv_results_ovr_xval.json', 'w') as f:
    json.dump(cv_results, f, indent=4)

with open('aug4_confusion_matrices_ovr_xval.json', 'w') as f:
    json.dump(confusion_matrices, f, indent=4)

with open('aug4_accuracy_loss_data_ovr_xval.json', 'w') as f:
    json.dump(accuracy_loss_data, f, indent=4)

os.chdir('/home/jsieg/butterflyAI')


entries = os.listdir()

for class_name, metrics in cv_results.items():
    avg_loss = np.mean([m['val_loss'] for m in metrics])
    avg_acc = np.mean([m['val_acc'] for m in metrics])
    avg_precision = np.mean([m['precision'] for m in metrics])
    avg_recall = np.mean([m['recall'] for m in metrics])
    avg_f1 = np.mean([m['f1_score'] for m in metrics])

    std_loss = np.std([m['val_loss'] for m in metrics])
    std_acc = np.std([m['val_acc'] for m in metrics])
    std_precision = np.std([m['precision'] for m in metrics])
    std_recall = np.std([m['recall'] for m in metrics])
    std_f1 = np.std([m['f1_score'] for m in metrics])

    print(f"Class: {class_name}")
    print(f"  Avg Val Loss: {avg_loss:.4f} ± {std_loss:.4f}")
    print(f"  Avg Val Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"  Avg Precision: {avg_precision:.4f} ± {std_precision:.4f}")
    print(f"  Avg Recall: {avg_recall:.4f} ± {std_recall:.4f}")
    print(f"  Avg F1-score: {avg_f1:.4f} ± {std_f1:.4f}")
    print("-" * 40)

# After training, print the confusion matrix for each fold
print("\nConfusion Matrices for Each Fold:")
for fold, matrix in conf_matrix.items():
    print(f"\n{fold}:")
    print(matrix)
