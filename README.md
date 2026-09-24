# ButterflyAI

Code for the Tseng lab butterfly AI project training a CNN to identify butterfly feeding behaviour and plants they are feeding on.

Research Questions:

- Identifying butterfly feeding behaviour: feeding vs. non-feeding.
- Identifying the plant species a butterfly is feeding on.

Most modeling scripts use TensorFlow/Keras transfer learning with ImageNet-pretrained ResNet50.

## Missing data — BC2024_plant/

`plant_id_model/OVR_test_on_other.py` expects `BC2024_plant/` (test images) and
`BC2024_plant_otheronly.csv`, used to test the plant-ID OVR models on the held-out
2024 BC set. These files are not in this checkout and were not on the project hard
drive, so this cross-dataset test cannot be run until they are recovered (re-download
from iNaturalist, or locate another copy). Training and cross-validation for the
plant-ID models are unaffected.

**Replaced (2026-09-24):** `data/splits/plant/test_mixed_plants.csv` (138 BIMBY-2024 photos with a plant ID, built by `data/scripts/make_splits.py`) is used instead; `OVR_test_on_other.py` now points at it.

## Hard Drive README.md

This hard drive contains all of the files for the butterlfyAI project

- the GPU (datascience UBC) and the GitHub <https://github.com/JulieSieg/ButterflyAI/tree/main> contain subsets of this data
- all monarch related files can be found on the GitHub except for the models (.keras files) and the images (.jpg) which are exclusively on this hard drive. GitHub caps the number of files to 1000 so the monarch images on the GitHub are a subset of the whole dataset

Each folder on this hard drive should contain its own READme file. Please refer to those for details for each model.

The Monarch_models folder contains the code and images required for training a feeding model to identify feeding and non-feeding behaviour in monarchs

The Feeding_models contains the original feeding non-feeding models, training data, and code for testing the efficacy of copy paste data augmentation on BIMBY 2023 data (all butterfly species)

The Plant_ID_Models contains the photos and code for training OVRs on 10 plant species, including with just flowers and with detectron images (for more info email <julie.sieg5678@gmail.com>). Multiclass code is not on this hard drive as it was deleted from the GPU in Aug2026 and has yet to be recovered. I plan to upload it to the GitHub when I either recover it or rewrite it.

The Testing_datasets have the hold out test sets from BIMBY2024 (BC) and Ontario, including the gold standard test set that has both plant and feeding labels.

Detectron folder contains the datasets for cropping butterflies but not the code. This code will be made available on the GitHub once/if its recovered, if not I’ll rewrite it (email me)

butterflyAI contains miscellaneous code from the GPU as of Aug 2026

Disregard Old_feeding_model_ipynbs as all code has been updated and turned into .py files rather than Jupyter notebook formats

For any questions please contact me at <Julie.sieg5678@gmail.com>

## To Do

Data is now organized in `data/` — see `data/README.md` for which images are used for training, validation and testing, and which script reads which split. The full audit behind it is in the Claude project doc `data-splits-audit.md`.

### Next

### Blocking model reruns (settle these first, so every model is rerun only once)

Reruns wait until the training data is final. Plan and known script bugs: Claude project doc `rerun-plan.md`.

- Feeding (all feeding reruns): Julie's answers on the 102 superimposed images with no traceable source photo and on the label conflicts (see Qs for J). The 7 byte-identical pairs labelled both F and N (17 images incl. superimposed copies) will be left out via `data/splits/feeding/excluded.csv` (drafted, not yet added)
- Monarch: decide whether superimposed monarch images will be added to training — if yes, rerun only after they exist
- Plant: find a machine for the OVR cross-validation (~50 models, roughly 30–45 h) — likely the UBC datascience GPU; too big for the laptop
- Before the first real run on the Mac: add the run-record code (saves settings, git commit, log, per-image predictions and metrics for every run), fix the script bugs listed in `rerun-plan.md`, and do the GPU-vs-CPU smoke test (tensorflow-metal has reported silent training bugs)
- Pin the GPU environment: `conda env export > environment-gpu.yml` (the unpinned `environment.yml` would install a TensorFlow version that breaks tensorflow-metal)

### Rerun (results affected by data leakage or bugs)

Feeding

- `Final_feeding_model_xval.py` cross-validation results — the model was reused across folds and was a different architecture from the final model (both fixed; just rerun). Decided: both scripts train a fixed 10 epochs, no early stopping
- `REAL_AND_SUPER_Final_feeding_model_unfrozen.keras` (production, also used by the monarch pipeline) and `REAL_ONLY_Final_feeding_model_unfrozen.keras` — trained on the old ungrouped 80/20 split. `Final_feeding_model_train_on_ALL.py` now trains on the whole dev pool (4,750 images)
- `results/Output_feeding_flower_only_real_test.txt` / `_ALL_test.txt` (the 107/132 numbers below) — from a split where a superimposed image and its source photo could land on opposite sides. Rerun the two comparison scripts
- `results/Output_feeding_test_on_BIMBY2024.txt` (85.4%) and `results/model_catalogue_evaluation.csv/.md` — the old BIMBY-2024 test set contained ~1,950 training images and the old gold set overlapped it. Rerun `Model_Catalogue_Evaluation.ipynb` (now reports Test 1 overall + per source, and Ontario)

Plant

- OVR models in `models/OVR_models/` and the `aug4_*` cross-validation JSONs — the same flower (archive re-pastes) could be in both training and validation. Rerun `Final_plant_OVR_xval.py` (writes to `models/OVR_models_xval/` and `results/`)
- `Scaling_test_results.csv` — random split; rerun on the new pool (grouped fold 0 held out)
- run `OVR_test_on_other.py` on the new mixed-plants test set (it reads `models/OVR_models/`; point `MODEL_DIR` at `models/OVR_models_xval/` once the OVR models are retrained)

Monarch

- `results/Monarch_xval_output_fixed.txt` — plain KFold, 13 duplicate image pairs could straddle folds (minor). Rerun with the grouped folds in `data/splits/monarch/dev_pool.csv`
- `results/model2_transfer_eval.md` — the monarch labels were made by reviewing this same model's predictions, so the result is circular. Needs an independently labelled monarch test set

### Housekeeping

- old `.keras` models still sit in `feeding_model/data/Final_Feeding_Images/` and `plant_id_model/data/*/` — move them into `models/` (and update `DATA_MODELS_DIR` in the catalogue notebook)
- the Script section below still describes old paths — `data/README.md` has the current ones

### Qs for J

- where the `bimby_collection` photos (BIMBY-2024 Block B, `data/metadata/bimby2024/BIMBY_redo.csv`) came from
- the 102 superimposed images with no traceable source photo (e.g. `superimposed_1008.jpeg`)
- which photos the butterflies in the plant composites were cut from
- label conflicts: 7 duplicate image pairs in the feeding dev pool have different labels, and 66 images were labelled differently in BIMBY-2024 than in training

### Longer term

- find other examples of ecology and ml projects like this: check data included, how results gathered (notebook? .md?), repo organization
- rewrite multiclass code mentioned above
- superimposed images for the monarch models
- independently labelled monarch test set
- expanding the list of 10 species
- model tuning with parameters

### Plant ID Model TODOs

- the OVR script fine tunes layers; the scaling script does not
- the scaling script COULD be used to create a multiclass model if 100% of the training data was used; right now, I don't have that model

## Complete

- separate models and outputs by feeding/non feeding classifiers and plant species identification in feeding pics
- reconcile script paths to the new repo layout (`feeding_model/`, `monarch/`, `plant_id_model/` — each with `data/`, `models/`, `results/`)
- fix hardcoded paths
- create an environment
- find and extract hardcoded config values, like TAXON_ID, start_page
- sort out training, testing, and validation sets (2026-09-24): all data moved into `data/` (`images/`, `metadata/`, `splits/`, `superseded/`) with `data/README.md`; `data/scripts/make_splits.py` builds grouped 5-fold dev pools + test sets for every model
- removed the ~1,950 training images from the BIMBY-2024 test set; feeding Test 1 = BIMBY-2024 + gold feeding labels (reported per source), Test 2 = Ontario
- plant: flower_only photos and the archive Cirsium/Sisymbrium batches added to the training pool, species names normalized, duplicate flowers grouped, Joint gold standard is now the plant test set
- `Final_feeding_model_xval.py`: builds a fresh model every fold (it used to keep training the previous fold's model), with the same architecture and training augmentation as `Final_feeding_model_train_on_ALL.py` (BatchNorm head, only `conv5_block` trainable); one checkpoint file per fold
- all scripts and `Model_Catalogue_Evaluation.ipynb` point at `data/splits/`; `Monarch_feeding_test.py` writes a prediction CSV instead of moving images
- built the plant "mixed plants" test set (`data/splits/plant/test_mixed_plants.csv`, 138 BIMBY-2024 photos: 24 target species, 114 other plants) to replace the missing `BC2024_plant_otheronly.csv`; `OVR_test_on_other.py` points at it

## Repository Contents

- `*.py`: scripts for data collection, labeling, training, and evaluation.
- `data/Monarch_images/`: downloaded monarch photos plus monarch prediction/label CSVs.
- `data/Monarch_images.csv`: top-level copy of the monarch image manifest.
- `feeding_non_feeding_models_results/results/*.txt`: captured console output from model training/evaluation runs.
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
- Results go to: `data/Monarch_images/Monarch_metadata_testset.json`.

### `Monarch_download_from_API.py`

Downloads image files from an iNaturalist metadata JSON file.

- Data used: metadata JSON at `data/Monarch_images/Monarch_metadata.json`; each observation's first `observation_photos` item.
- What it does: converts iNaturalist photo URLs from `square` to `medium`, downloads each image from the iNaturalist open-data S3 bucket, and records the image filename/species pair.
- Results go to: downloaded JPEGs in `data/Monarch_images/`; appended manifest rows in `data/Monarch_images.csv`.
- The script appends to `data/Monarch_images.csv`, so reruns can duplicate rows unless the file is cleaned first.

### `Monarch_feeding_test.py`

Uses a trained feeding/non-feeding model to predict labels for monarch images.

- Model used: saved Keras feeding classifier at `Final_Feeding_Images/REAL_AND_SUPER_Final_feeding_model_unfrozen.keras`.
- Data used: `data/Monarch_images/Monarch_image_predictions.csv`, with images expected in `data/Monarch_images/`.
- What it does: loads the feeding model, generates feeding/non-feeding scores for monarch images, writes `prediction` and `score` columns, then sorts image files into `Class_0/` and `Class_1/` folders.
- Results go to: `data/Monarch_images/Monarch_image_predictions.csv`; image files moved into `data/Monarch_images/Class_0/` and `data/Monarch_images/Class_1/`.
- Notes: the classification threshold is `score > 0.1`. The script first tries to move images from `Class_0/` or `Class_1/` back into the main folder based on existing predictions, then predicts and moves them again.

### `label_monarchs.py`

Creates manually corrected monarch feeding labels after reviewing model-sorted images.

- Data used: `data/Monarch_images/Monarch_image_predictions.csv` plus `data/Monarch_images/Monarch_non_feeding_ls.csv`, a manually curated list of non-feeding filenames.
- What it does: adds a `Label` column: filenames in `Monarch_non_feeding_ls.csv` become `Non_feeding`; all others become `Feeding`.
- Results go to: `data/Monarch_images/Monarch_image_labels.csv`.
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
- ~~the model is created once before the fold loop…~~ fixed 2026-09-24: a fresh model is built every fold, using the same architecture as `Final_feeding_model_train_on_ALL.py` (BatchNorm head, only `conv5_block` trainable, same training augmentation). Checkpoints are saved per fold (`…_fold1.keras` … `_fold5.keras`).

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

- Model used: every `.keras` file in `OVR_models/`; these are expected to be the `ovr_model_<plant_name>.keras` models from `Final_plant_OVR_xval.py`.
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

- `data/Monarch_images/Monarch_images.csv` and `data/Monarch_images.csv`: manifests with `FileName,Species`; 8,018 monarch image rows plus header.
- `data/Monarch_images/Monarch_image_predictions.csv`: monarch manifest plus model `prediction` and `score`.
- `data/Monarch_images/Monarch_non_feeding_ls.csv`: manually curated list of filenames considered non-feeding.
- `data/Monarch_images/Monarch_image_labels.csv`: monarch manifest plus final `Label` assigned from the manual non-feeding list.
- `feeding_non_feeding_models_results/results/Monarch_xval_output.txt`: captured feeding-model cross-validation run.
- `feeding_non_feeding_models_results/results/Output_feeding_flower_only_real_test.txt`: captured real-only feeding-model comparison run.
- `feeding_non_feeding_models_results/results/Output_feeding_flower_only_ALL_test.txt`: captured all-data feeding-model comparison run.
- `feeding_non_feeding_models_results/results/Output_feeding_test_on_BIMBY2024.txt`: captured evaluation on a BIMBY 2024 feeding/non-feeding dataset. The corresponding script is not currently checked in.

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

To create the environment:

mamba env create -f environment.yml
mamba activate butterflyai
