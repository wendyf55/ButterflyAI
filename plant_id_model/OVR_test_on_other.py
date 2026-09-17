## Test gold standard dataset on photos of "other" species using OVR
# use the 10 ovr models in the folder "OVR Model"

import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import random
import pandas as pd

import csv

import json
import shutil
from pathlib import Path


from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
 
from tensorflow.keras.models import load_model

from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = PROJECT_ROOT / "BC2024_plant_otheronly.csv"
TEST_IMAGE_DIR = PROJECT_ROOT / "BC2024_plant"
MODEL_DIR = PROJECT_ROOT / "OVR_models"
PREDICTIONS_DIR = PROJECT_ROOT / "Other_conundrum"
METRICS_JSON = PREDICTIONS_DIR / "ovr_model_metrics.json"
METRICS_CSV = PREDICTIONS_DIR / "ovr_model_metrics.csv"
SEED_VALUE = 871
MODEL_PREFIX = "ovr_model_"
MODEL_SUFFIX = ".keras"
PREDICTION_FILE_PREFIX = "ovr_"
PREDICTION_FILE_SUFFIX = "_preds.json"
PREDICTION_THRESHOLD = 0.5
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32


def normalize_species_name(species_name):
    species_name = str(species_name).replace("_", " ").strip()
    return " ".join(species_name.split())

#set seed so its always the same
#321

os.environ['PYTHONHASHSEED']=str(SEED_VALUE)

random.seed(SEED_VALUE)

np.random.seed(SEED_VALUE)

tf.random.set_seed(SEED_VALUE)


test_df = pd.read_csv(TEST_CSV)
test_df = test_df.copy()
test_df["normalized_plant_scientific_name"] = test_df["plant_scientific_name"].apply(normalize_species_name)
test_df = test_df[test_df["normalized_plant_scientific_name"].str.contains(" ", na=False)]

print(test_df.head())




# List all .keras files
PREDICTIONS_DIR.mkdir(exist_ok=True)
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(MODEL_SUFFIX)]
model_metrics = []

for model_file in model_files:

    # Extract class name from filename:
    # "ovr_model_Achillea_millefolium.keras" → "Achillea_millefolium"
    class_name = model_file.replace(MODEL_PREFIX, "").replace(MODEL_SUFFIX, "")
    target_species = normalize_species_name(class_name)

    print(f"Loading model for class: {target_species}")

    # Load model
    model = load_model(str(MODEL_DIR / model_file))

    binary_test_df = test_df.copy()
    binary_test_df["BinaryLabel"] = (
        binary_test_df["normalized_plant_scientific_name"] == target_species
    ).astype(int)

    print(binary_test_df.head())


    test_datagen = ImageDataGenerator(preprocessing_function =  preprocess_input
                                 )

    # The ResNet50 model expects images to be 224x224, so we set those values here
    img_height, img_width = (IMG_HEIGHT, IMG_WIDTH)
    batch_size = BATCH_SIZE

    # Same as above but for your test dataset
    test_generator = test_datagen.flow_from_dataframe(
    dataframe = binary_test_df,
    directory = str(TEST_IMAGE_DIR),
    x_col = "FileName",
    y_col = 'BinaryLabel',
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'raw',
    shuffle = False
    )

    # Print the number of validated images found
    print(f"Number of testing images found: {test_generator.samples}")


    preds = model.predict(test_generator, verbose=1).ravel()
    
    # Convert probabilities to predicted class: 0 or 1
    predicted_classes = (preds >= PREDICTION_THRESHOLD).astype(int)
    # Confidence is the sigmoid probability for the target plant species.
    predicted_confidences = preds.copy()
    validated_test_df = binary_test_df.set_index("FileName").loc[test_generator.filenames].reset_index()
    true_classes = validated_test_df["BinaryLabel"].to_numpy()
    assert len(true_classes) == len(predicted_classes), (
        f"Mismatch: y_true={len(true_classes)}, y_pred={len(predicted_classes)}"
    )
    
    class_confidence_dict = defaultdict(list)
    
    for cls, conf in zip(predicted_classes, predicted_confidences):
        class_confidence_dict[cls].append(conf)
        
    # Compute mean confidence per predicted class
    for cls in sorted(class_confidence_dict.keys()):
        mean_conf = np.mean(class_confidence_dict[cls])
        count = len(class_confidence_dict[cls])
        print(f"Predicted Class {cls}: {count} images, Mean confidence = {mean_conf:.4f}")  

    conf_matrix = confusion_matrix(true_classes, predicted_classes, labels=[0, 1])
    report = classification_report(
        true_classes,
        predicted_classes,
        labels=[0, 1],
        target_names=["other_plants", target_species],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model_file": model_file,
        "target_species": target_species,
        "threshold": PREDICTION_THRESHOLD,
        "n_images": int(len(true_classes)),
        "n_target_images": int(true_classes.sum()),
        "n_other_images": int(len(true_classes) - true_classes.sum()),
        "predicted_target_images": int(predicted_classes.sum()),
        "predicted_other_images": int(len(predicted_classes) - predicted_classes.sum()),
        "accuracy": float(accuracy_score(true_classes, predicted_classes)),
        "precision": float(precision_score(true_classes, predicted_classes, zero_division=0)),
        "recall": float(recall_score(true_classes, predicted_classes, zero_division=0)),
        "f1_score": float(f1_score(true_classes, predicted_classes, zero_division=0)),
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": report,
    }
    model_metrics.append(metrics)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 score: {metrics['f1_score']:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")
        
    # Save per-image prediction to a list of dicts (using filenames
    predictions_list = []
    for filename, true_cls, cls, conf, plant_name in zip(
        test_generator.filenames,
        true_classes,
        predicted_classes,
        predicted_confidences,
        validated_test_df["plant_scientific_name"],
    ):
        predictions_list.append({
            "filename": filename,
            "plant_scientific_name": plant_name,
            "true_class": int(true_cls),
            "predicted_class": int(cls),     # ensure JSON serializable
            "confidence": float(conf)        # ensure JSON serializable
            })
            
    print(f"Total predictions stored: {len(predictions_list)}")
    
    save_path = PREDICTIONS_DIR / f"{PREDICTION_FILE_PREFIX}{class_name}{PREDICTION_FILE_SUFFIX}"
    
    # Save to JSON file

    with open(save_path, "w") as f:
        json.dump(predictions_list, f, indent=4)
    
    print(f"Predictions saved to {save_path}")
    
    
with open(METRICS_JSON, "w") as f:
    json.dump(model_metrics, f, indent=4)

metrics_df = pd.DataFrame([
    {
        "model_file": metrics["model_file"],
        "target_species": metrics["target_species"],
        "threshold": metrics["threshold"],
        "n_images": metrics["n_images"],
        "n_target_images": metrics["n_target_images"],
        "n_other_images": metrics["n_other_images"],
        "predicted_target_images": metrics["predicted_target_images"],
        "predicted_other_images": metrics["predicted_other_images"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "confusion_matrix": metrics["confusion_matrix"],
    }
    for metrics in model_metrics
])
metrics_df.to_csv(METRICS_CSV, index=False)

print(f"Metrics saved to {METRICS_JSON}")
print(f"Metrics summary saved to {METRICS_CSV}")
