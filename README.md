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

## Project history

The project and most of its files come from JS. In September 2026 WF took it over from JS's project hard drive; the scripts and data were reorganized into this repository (see `data/README.md` for where every dataset went). Notes from the hard drive's README:

- The UBC datascience GPU and JS's GitHub (<https://github.com/JulieSieg/ButterflyAI/tree/main>) hold subsets of the project files.
- Monarch models (`.keras`) and images were only on the hard drive. GitHub caps the number of files, so the GitHub copy has a subset of the monarch images.
- The multiclass plant-ID code was deleted from the GPU in Aug 2026 and has not been recovered.
- The Detectron2 code used to cut butterflies out of photos was not on the hard drive (only its output images).
- JS's contact: <julie.sieg5678@gmail.com>. JS may not be available for questions, so open questions are settled with working decisions (see "Qs for JS" and `data/README.md`).

## To Do

### Feeding model — where it stands (2026-10-01)

Final model: `feeding_model/models/feeding_final_real_and_super_2026-10-01.keras` (trained on all 4,630 dev-pool images, real + superimposed). Run records in `feeding_model/results/`.

| | Accuracy | Feeding recall | Non-feeding recall |
| --- | --- | --- | --- |
| Cross-validation, real photos (`xval_2026-09-25_1422`) | 0.966 ± 0.003 | 0.961 | 0.971 |
| Test 1 overall (6,353) | 0.818 | 0.785 | 0.877 |
| — `inat_2024` (2,884) | 0.829 | 0.644 | 0.904 |
| — `inat_2024_gold` (2,162; only 21 non-feeding) | 0.870 | 0.873 | — |
| — `bimby_collection` (1,307) | 0.705 | 0.725 | 0.559 |
| Test 2 Ontario (2,785) | 0.775 | 0.618 | 0.900 |

- big drop from cross-validation to the test sets (photos from other collections, years and regions)
- the model misses feeding on new photos (Test 1: 886 feeding called non-feeding vs 273 the other way), so it would undercount feeding
- the test sets have now been looked at once: choose any further changes on cross-validation, and report new test results as a second look

### Next

- Test whether the superimposed images help on real photos of butterflies sitting on flowers without feeding (the case they were made for). So far we only know that a model trained without them labels most *composites* "feeding" (30% right vs 93% with them); on real non-feeding photos the two models are tied (non-feeding recall 0.977 real-only vs 0.971 real+super). To do: tag which real non-feeding photos in the dev pool show a butterfly sitting on a flower, then compare the two models on just those photos. Both models' predictions for every image are already saved (`feeding_model/results/xval_2026-10-01_1119/predictions.csv` = real+super, `xval_2026-10-01_1043_real_only/predictions.csv` = real-only), so only the tags are needed, no retraining
- Error review (optional): look through a sample of the missed feeding photos per test source (tiny butterfly? blurry? mislabelled?). Diagnosis only; note that it uses the test sets
- Take the `bimby_collection` result (see Qs for JS): it's the worst test source (0.705, non-feeding recall 0.559) although it's supposedly the same collection as training, which points to a labelling or source difference
- Repoint `monarch/Monarch_feeding_test.py`, `monarch/general_feeding_model_eval.py` and `Model_Catalogue_Evaluation.ipynb` at the new model (they still use `REAL_AND_SUPER_Final_feeding_model_unfrozen.keras`)

### Feeding model — improving it (choose on cross-validation; one ~36 min run each)

- Training stability: validation accuracy dips mid-training (fold 1 down to 0.58) before recovering by epoch 10; the final model has no validation set to catch a bad last epoch. Fix drafted but not applied: freeze ResNet's BatchNorm layers, learning rate 1e-4, cosine decay, optional head-only warm-up
- Closing the gap to new photos: stronger augmentation (random crop/zoom, colour), bilinear resizing instead of nearest-neighbour (every script that scores images must then match), more varied training photos (e.g. iNat-style, lower resolution)
- Optional: repeat the best recipe with 2–3 seeds
- Move the shared code (data filter, metrics, run record) into one module: it's copied across `Final_feeding_model_xval.py`, `Final_feeding_model_train_on_ALL.py` and `test_final_model.py`, so the recipe can drift

### Blocking model reruns (settle these first, so every model is rerun only once)

- Feeding: JS's answers on the 103 superimposed images with no traceable source photo (dropped for now via `DROP_UNTRACEABLE_SUPER`; add back and rerun if cleared) and on the label conflicts
- Monarch: decide whether superimposed monarch images will be added to training — if yes, rerun only after they exist

### Rerun (results affected by data leakage or bugs)

Feeding

- `results/Output_feeding_flower_only_real_test.txt` / `_ALL_test.txt` (the 107/132 numbers below): superseded by the cross-validation real-only vs real+super comparison; the two comparison scripts can be archived
- `results/Output_feeding_test_on_BIMBY2024.txt` (85.4%) and `results/model_catalogue_evaluation.csv/.md`: the old BIMBY-2024 test set contained ~1,950 training images. Superseded by `test_feeding_final_real_and_super_2026-10-01`. If the catalogue notebook is rerun: it flips a model's labels based on test accuracy, ranks ~30 models on the test sets (so it can't give an honest number for the winner), and doesn't save per-image scores

Plant

- OVR models in `models/OVR_models/` and the `aug4_*` cross-validation JSONs — the same flower (archive re-pastes) could be in both training and validation. Rerun `Final_plant_OVR_xval.py` (writes to `models/OVR_models_xval/` and `results/`). Fix first: model file overwritten each fold (only fold 5 kept); final print loop crashes (`conf_matrix` is a list, use `conf_metrics`); restore_best_weights on the validation fold; metrics on downsampled validation; whole ResNet trainable (the scaling script trains conv5 only)
- add a script that evaluates the plant models on `data/splits/plant/test_gold.csv` (none exists)
- `Scaling_test_results.csv` — random split; rerun on the new pool (grouped fold 0 held out). Fix first: saves only macro precision, no model saved, head differs from OVR, no check that all 10 classes are in each subsample
- run `OVR_test_on_other.py` on the new mixed-plants test set (point `MODEL_DIR` at `models/OVR_models_xval/` once the OVR models are retrained)

Monarch

- `results/Monarch_xval_output_fixed.txt` — plain KFold, 13 duplicate image pairs could straddle folds (minor). Rerun with the grouped folds in `data/splits/monarch/dev_pool.csv`. Fix first: whole ResNet trainable, Adam 1e-3, no augmentation, simpler head than the feeding model; restore_best_weights on the validation fold; nothing saved beyond the log
- `results/model2_transfer_eval.md` — the monarch labels were made by reviewing this same model's predictions, so the result is circular. Needs an independently labelled monarch test set
- `Monarch_feeding_test.py` overwrites `data/metadata/monarch/Monarch_image_predictions.csv` (git-ignored, the record the monarch labels were built from) — write to a new file

### Housekeeping

- the monarch and plant entries in the Script section below still describe old paths — `data/README.md` has the current paths
- archive `Feeding_model_only_with_flower_comparison.py` / `_augmentedversion.py` (superseded by the real-only cross-validation)

### Qs for JS

- where the `bimby_collection` photos (BIMBY-2024 Block B, `data/metadata/bimby2024/BIMBY_redo.csv`) came from and how they were labelled (worst test source: 0.705)
- the 103 superimposed images with no traceable source photo (e.g. `superimposed_1008.jpeg`)
- which photos the butterflies in the plant composites were cut from
- label conflicts: 7 duplicate image pairs in the feeding dev pool have different labels, and 66 images were labelled differently in BIMBY-2024 than in training
- 112 photos in `data/images/bimby2024/` are not listed in any CSV (not used by any split): what are they?

### Longer term

- find other examples of ecology and ml projects like this: check data included, how results gathered (notebook? .md?), repo organization
- rewrite multiclass code mentioned above
- superimposed images for the monarch models
- independently labelled monarch test set
- expanding the list of 10 species
- model tuning with parameters (see "Feeding model — improving it")
- update the numbers in the paper, unfortunately it has to be done
- rewrite multiclass plant id code
- OSR models - incorpoate for plant id

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
- `Final_feeding_model_xval.py`: builds a fresh model every fold (it used to keep training the previous fold's model), with the same architecture and training augmentation as `Final_feeding_model_train_on_ALL.py` (BatchNorm head, only `conv5_block` trainable)
- all scripts and `Model_Catalogue_Evaluation.ipynb` point at `data/splits/`; `Monarch_feeding_test.py` writes a prediction CSV instead of moving images
- feeding training data settled (2026-09-25): the 17 label-conflict images excluded (`data/splits/feeding/excluded.csv`), the 103 untraceable superimposed images dropped for now; dev pool = 4,630 images
- `Final_feeding_model_xval.py` reworked (2026-09-25): fixed 10 epochs with nothing chosen on validation, F=0/N=1 pinned, metrics for both classes and for real / superimposed separately, every run saved to its own folder (predictions, metrics, training curves, run record, log), `SMOKE_TEST` / `FORCE_CPU` / `TRAIN_ON_REAL_ONLY` flags
- GPU environment pinned in `environment-gpu.yml`; Mac GPU checked against the CPU (same training); reruns reproduce exactly
- first clean feeding cross-validation (`xval_2026-09-25_1422`): real photos 0.966 ± 0.003
- real-only vs real+superimposed (2026-10-01): tied on real photos (0.969 vs 0.966, p = 0.33); final model = real+super, by the rule set before testing
- `Final_feeding_model_train_on_ALL.py` rewritten (same filter and recipe as the xval, run record, never overwrites a model) and final model trained (2026-10-01)
- `test_final_model.py` added; final model tested once on Test 1 (per source) and Ontario (2026-10-01)
- old `.keras` files moved out of the data folders into `models/`
- docs updated (2026-10-02): `data/README.md` brought up to date, `feeding_model/README.md` added, feeding entries in the Script section rewritten, project history rewritten

## Repository layout

```
ButterflyAI/
├── README.md              this file: status, to-do, project history
├── data/                  all images, original metadata and the train/test splits for every model
│                          → data/README.md (single source of truth for the data)
├── feeding_model/         feeding vs non-feeding model: scripts, models/, results/
│                          → feeding_model/README.md
├── monarch/               feeding model for monarchs only
├── plant_id_model/        10-species plant ID model
├── archive/               older scripts kept for reference
├── docs/                  notes; PDFs and spreadsheets stay local (git-ignored)
├── environment.yml        CPU environment (butterflyai)
└── environment-gpu.yml    pinned Mac GPU environment (butterflyai_gpu)
```

Images, `.keras` models and `data/metadata/` are not in git (see `.gitignore`); they live on the local copy.

## Script

The feeding entries below are current. The monarch and plant entries describe the scripts before the data was moved into `data/`, so their paths are out of date; `data/README.md` has the current inputs and outputs of every script.

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

### `Final_feeding_model_xval.py`

Estimates how well the feeding-model recipe works, with 5-fold cross-validation.

- Data used: `data/splits/feeding/dev_pool.csv` (the `fold` column, 0–4; folds are grouped so a superimposed image stays with its source photo). Leaves out the 17 label-conflict photos in `data/splits/feeding/excluded.csv` and, while `DROP_UNTRACEABLE_SUPER = True`, the 103 superimposed images with no source photo in the pool: 4,630 images.
- Model: ImageNet ResNet50, only `conv5_block` trainable; head = global average pooling, dropout 0.5, dense 512, BatchNorm, ReLU, dropout 0.5, 1 sigmoid unit. Adam 1e-3, batch 32, 10 epochs, horizontal flip + shear 0.2, threshold 0.5.
- Labels: F = 0, N = 1, so the output is P(non-feeding). Metrics are reported for both classes, and for real and superimposed images separately.
- Results go to: a new folder per run, `feeding_model/results/xval_<date>_<time>/` (predictions, per-fold and summary metrics, training curves, run record, log). Fold models are not saved.
- Settings: `TRAIN_ON_REAL_ONLY`, `SMOKE_TEST` (quick pipeline check), `FORCE_CPU`.

### `Final_feeding_model_train_on_ALL.py`

Trains the one feeding model we keep, with the same data filter and recipe as the cross-validation, on all 4,630 images.

- Results go to: a new model file `feeding_model/models/feeding_final_<real_and_super|real_only>_<date>.keras` (stops if the file exists; models are never overwritten) and a run record in `feeding_model/results/train_<model name>/`.
- No validation set: nothing is chosen on data.

### `test_final_model.py`

Tests the final model once on Feeding Test 1 (`data/splits/feeding/test1.csv`, overall and per `source`) and Test 2 (`data/splits/feeding/test2_ontario.csv`).

- Results go to: `feeding_model/results/test_<model name>/` (metrics, per-image predictions, run record, log). The folder can only be made once per model.

### `Feeding_model_only_with_flower_comparison.py` / `Feeding_only_with_flower_comparison_augmentedversion.py`

Superseded by the real-only vs real+superimposed cross-validation (`xval_2026-10-01_1043_real_only` vs `xval_2026-10-01_1119`); to be archived. They hold out dev-pool fold 0 and score real feeding photos + superimposed non-feeding photos. Their old outputs (`results/Output_feeding_flower_only_*.txt`, the 107/132 numbers) came from an ungrouped 80/20 split, where most of the 132 superimposed test images were made from photos in the training split.

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

## Environment Notes

CPU environment:

mamba env create -f environment.yml
mamba activate butterflyai

Mac GPU environment (pinned versions; used for the 2026 feeding runs):

mamba env create -f environment-gpu.yml
mamba activate butterflyai_gpu
