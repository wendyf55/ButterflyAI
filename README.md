# ButterflyAI

Code for the Tseng lab butterfly AI project training a CNN to identify butterfly feeding behaviour and plants they are feeding on.

Research Questions:

- Identifying butterfly feeding behaviour: feeding vs. non-feeding.
- Identifying the plant species a butterfly is feeding on.

Most modeling scripts use TensorFlow/Keras transfer learning with ImageNet-pretrained ResNet50.

## To Do

- fix all hardcoded paths
- find and extract hardcoded config values, like TAXON_ID, start_page
- find other examples of ecology and ml projects like this: check data included, how results gathered (notebook? .md?), repo organization
- separate models and outputs by feeding/non feeding classifiers and plant species identification in feeding pics

## Repository Contents

- `*.py`: scripts for data collection, labeling, training, and evaluation.
- `Monarch_images/`: downloaded monarch photos plus monarch prediction/label CSVs.
- `Monarch_images.csv`: top-level copy of the monarch image manifest.
- `*.txt`: captured console output from model training/evaluation runs.
- `.gitignore`: ignores saved Keras model files (`*.keras`).

Expected training/test image folders referenced by scripts, but not present in this checkout:

- `Final_Feeding_Images/`
- `plant_data_specified/`
- `specified_flower_photos_detectron_ALL/`
- `BC2024_plant/`
- `OVR_models/`
- `Other_conundrum/`

## Script

### `Monarch_API_call.py`

Downloads occurrences from the iNaturalist observations API.

- Data used: iNaturalist API results for a configured taxon: `TAXON_ID = 48662`, nickname `Monarch`, `quality_grade=research`, `place_id=97394`, and months June/July.
- queries pages of iNaturalist observations, keeps research-grade observations, and stores the accumulated JSON metadata.
- Results go to: `Monarch_images_test_set/Monarch_metadata_testset.json`.

### `Monarch_download_from_API.py`

Downloads image files from an iNaturalist metadata JSON file.

- Data used: metadata JSON at `Monarch_images/Monarch_metadata.json`; each observation's first `observation_photos` item.
- What it does: converts iNaturalist photo URLs from `square` to `medium`, downloads each image from the iNaturalist open-data S3 bucket, and records the image filename/species pair.
- Results go to: downloaded JPEGs in `Monarch_images/`; appended manifest rows in `Monarch_images.csv`.
- The script appends to `Monarch_images.csv`, so reruns can duplicate rows unless the file is cleaned first.

### `Monarch_feeding_test.py`

Uses a trained feeding/non-feeding model to predict labels for monarch images.

- Model used: saved Keras feeding classifier at `/mnt/sharedstorage/jsieg/butterflyAI/Final_Feeding_Images/REAL_AND_SUPER_Final_feeding_model_unfrozen.keras`.
- Data used: `Monarch_images/Monarch_image_predictions.csv`, with images expected in `Monarch_images/`.
- What it does: loads the feeding model, generates feeding/non-feeding scores for monarch images, writes `prediction` and `score` columns, then sorts image files into `Class_0/` and `Class_1/` folders.
- Results go to: `Monarch_images/Monarch_image_predictions.csv`; image files moved into `Monarch_images/Class_0/` and `Monarch_images/Class_1/`.
- Notes: the classification threshold is `score > 0.1`. The script first tries to move images from `Class_0/` or `Class_1/` back into the main folder based on existing predictions, then predicts and moves them again.

### `label_monarchs.py`

Creates manually corrected monarch feeding labels after reviewing model-sorted images.

- Data used: `Monarch_images/Monarch_image_predictions.csv` plus `Monarch_images/Monarch_non_feeding_ls.csv`, a manually curated list of non-feeding filenames.
- What it does: adds a `Label` column: filenames in `Monarch_non_feeding_ls.csv` become `Non_feeding`; all others become `Feeding`.
- Results go to: `Monarch_images/Monarch_image_labels.csv`.
- The checked-in `Monarch_image_labels.csv` has columns `FileName, Species, prediction, score, Label`.

### `Final_feeding_model_train_on_ALL.py`

Trains the final binary feeding/non-feeding classifier on all available labeled feeding data.

- Model used: ImageNet-pretrained ResNet50 backbone (`include_top=False`) with global average pooling, dropout, a 512-unit dense layer, batch normalization, ReLU, more dropout, and a 1-unit sigmoid output.
- Data used: `Final_Feeding_Images/DataFilenamesRedo_train.csv`; images are read from `Final_Feeding_Images/`.
- What it does: trains a binary classifier for feeding (`F`) vs. non-feeding (`N`) using `ImageDataGenerator` with ResNet50 preprocessing, horizontal flips, and shear augmentation.
- Results go to: `Final_Feeding_Images/REAL_AND_SUPER_Final_feeding_model_unfrozen.keras`.
- Notes: only ResNet50 layers with `"conv5_block"` in their name are trainable; earlier backbone layers are frozen. The script uses all rows in `DataFilenamesRedo_train.csv`; a commented line can restrict training to `photo_type == "real"`.

### `Final_feeding_model_xval.py`

- Model used: ImageNet-pretrained ResNet50 backbone with global average pooling, dropout, a 512-unit ReLU dense layer, dropout, and a 1-unit sigmoid output.
- Data used: `Final_Feeding_Images/DataFilenamesRedo.csv`; images are read from `Final_Feeding_Images/`.
- What it does: performs 5-fold KFold cross-validation, trains the binary feeding classifier, predicts each validation fold, and prints accuracy, precision, recall, F1, and loss.
- Results go to: console output; best model checkpoint path is `Final_Feeding_Images/FINAL_Real_Feeding_unfrozen_xval_augmented.keras`.
- Existing captured output: `Monarch_xval_output.txt` reports 5-fold averages of accuracy `0.9055 +/- 0.0448`, precision `0.8963 +/- 0.0557`, recall `0.9681 +/- 0.0250`, F1 `0.9299 +/- 0.0331`, and loss `0.3530 +/- 0.1574`.
- the model is created once before the fold loop and `clear_session()` is called after each fold, but the model is not rebuilt or reloaded for each fold. That means folds may not be fully independent as written.

### `Feeding_model_only_with_flower_comparison.py`

Compares feeding-model performance when training on real images only vs. all images, then testing on images with flowers.

- Model used: ImageNet-pretrained ResNet50 binary classifier with global average pooling, dropout, 512-unit ReLU dense layer, dropout, and 1-unit sigmoid output.
- Data used: `Final_Feeding_Images/DataFilenamesRedo.csv`; images are read from `Final_Feeding_Images/`.
- What it does: creates an 80/20 stratified split using both `label` and `photo_type`, then filters validation data to real feeding images plus superimposed non-feeding images. In the checked-in script, `train_df = train_real_df`, so it trains only on real images.
- Results go to: console output only.
- Existing captured outputs:
  - `Output_feeding_flower_only_real_test.txt`: real-only training result, macro precision `0.6989`, macro recall `0.5399`, macro F1 `0.5192`.
  - `Output_feeding_flower_only_ALL_test.txt`: all-image training result, macro precision `0.8792`, macro recall `0.8772`, macro F1 `0.8782`, with 24 manual false positives and 25 manual false negatives.
- So end conclusion here: when trained only on real images, the feeding model does not handle “butterfly on flower but not feeding” very well, it's biased toward calling all flower-present images feeding.
- all image model: after training with the superimposed non-feeding examples included, it correctly identified 107/132 fake non-feeding images, while still getting 403/427 real feeding images right.

### `Final_plant_OVR_xval.py`

Trains one-vs-rest plant classifiers with 5-fold cross-validation.

- Model used: one binary ImageNet-pretrained ResNet50 classifier per plant class. Each classifier has global average pooling, dropout, a 512-unit dense layer, batch normalization, ReLU, dropout, and a 1-unit sigmoid output.
- Data used: `plant_data_specified/Flower_only_specified.csv`; images are read from `plant_data_specified/`. Expected columns include `Filename` and `Label`.
- What it does: for each unique plant label, creates a binary target for that plant vs. all other plants, balances the training and validation folds by downsampling the majority class, trains a 5-fold stratified cross-validation model, and records validation metrics and confusion matrices.
- Results go to:
  - `plant_data_specified/ovr_model_<plant_name>.keras`
  - `plant_data_specified/aug4_cv_results_ovr_xval.json`
  - `plant_data_specified/aug4_confusion_matrices_ovr_xval.json`
  - `plant_data_specified/aug4_accuracy_loss_data_ovr_xval.json`
- Notes: OVR means one-vs-rest: one binary classifier is trained per plant species.

### `OVR_test_on_other.py`

Applies saved one-vs-rest plant models to a gold-standard dataset of other butterfly species.

- Model used: every `.keras` file in `butterflyAI/OVR_models`; these are expected to be the `ovr_model_<plant_name>.keras` models from `Final_plant_OVR_xval.py`.
- Data used: `BC2024_plant_otheronly.csv`; images are read from `BC2024_plant/`. Expected columns include `FileName` and `plant_scientific_name`.
- What it does: loops through each OVR model, loads the model, builds a binary test label for that plant model, predicts class/confidence for each image, prints confidence summaries, and saves per-image predictions.
- Results go to: `Other_conundrum/ovr_<class_name>_preds.json`.
- Notes: the binary label line currently uses `plant_scientific_name != class_name`. Because training uses positive class `1` for the target plant, this may invert the expected labels during evaluation unless the saved model convention is intentionally different. Also, model filenames replace spaces with underscores, while `plant_scientific_name` may contain spaces; normalize names before relying on this comparison.

### `Scaling_of_flower_dataset.py`

Tests how plant-classifier performance changes as the training dataset size changes.

- Model used: ImageNet-pretrained ResNet50 multiclass classifier with global average pooling, dropout, a 512-unit ReLU dense layer, dropout, and a 10-unit softmax output.
- Data used: `specified_flower_photos_detectron_ALL/specified_flower_photos_detectron.csv`; images are read from `specified_flower_photos_detectron_ALL/`. Expected columns include `Filename` and `Label`.
- What it does: makes an 80/20 train/test split, trains separate multiclass models using 5%, 10%, 20%, 50%, 75%, and 100% of the training split, evaluates each on the same held-out test split, and prints macro/per-class metrics.
- Results go to: `specified_flower_photos_detectron_ALL/Scaling_test_results.csv`.
- Notes: the model output is hard-coded to 10 classes (`Dense(10, softmax)`), so the input CSV must contain exactly 10 plant labels unless the script is changed. Only macro precision is saved to CSV, even though macro recall/F1 and confusion matrices are printed.

## Data Files In This Checkout

- `Monarch_images/Monarch_images.csv` and top-level `Monarch_images.csv`: manifests with `FileName,Species`; 8,018 monarch image rows plus header.
- `Monarch_images/Monarch_image_predictions.csv`: monarch manifest plus model `prediction` and `score`.
- `Monarch_images/Monarch_non_feeding_ls.csv`: manually curated list of filenames considered non-feeding.
- `Monarch_images/Monarch_image_labels.csv`: monarch manifest plus final `Label` assigned from the manual non-feeding list.
- `Monarch_xval_output.txt`: captured feeding-model cross-validation run.
- `Output_feeding_flower_only_real_test.txt`: captured real-only feeding-model comparison run.
- `Output_feeding_flower_only_ALL_test.txt`: captured all-data feeding-model comparison run.
- `Output_feeding_test_on_BIMBY2024.txt`: captured evaluation on a BIMBY 2024 feeding/non-feeding dataset. The corresponding script is not currently checked in.

## Typical Workflow

1. Download monarch metadata with `Monarch_API_call.py`.
2. Download monarch images and write a manifest with `Monarch_download_from_API.py`.
3. Train the final feeding classifier with `Final_feeding_model_train_on_ALL.py`.
4. Use `Monarch_feeding_test.py` to score/sort monarch images.
5. Manually curate non-feeding images and write final labels with `label_monarchs.py`.
6. Evaluate feeding-model behavior with `Final_feeding_model_xval.py` and `Feeding_model_only_with_flower_comparison.py`.
7. Train plant classifiers with `Final_plant_OVR_xval.py` or study multiclass scaling with `Scaling_of_flower_dataset.py`.
8. Apply saved plant OVR models to other-species data with `OVR_test_on_other.py`.

## Environment Notes

The scripts use:

- `tensorflow` / `keras`
- `tf_keras` in `Monarch_feeding_test.py`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `requests`

No `requirements.txt` or environment file is currently checked in. Reproducibility would improve a lot if the next cleanup adds one and replaces hard-coded absolute paths with configurable paths.
