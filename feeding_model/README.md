# Feeding model

Classifies a butterfly photo as feeding (F) or non-feeding (N). ImageNet ResNet50, fine-tuned on the BIMBY-BC2023 photos plus superimposed (CPDA) non-feeding composites. All data comes from `data/splits/feeding/` (see `data/README.md`).

**Label convention:** F = 0, N = 1, so the model's output is P(non-feeding) and a score above 0.5 means "non-feeding". This was also true in JS's original scripts (Keras sorts the labels alphabetically), so any old single-class precision or recall is for non-feeding. Always report both classes, labelled by name.

## Current results (final model, 2026-10-01)

Model: `models/feeding_final_real_and_super_2026-10-01.keras`, trained on all 4,630 pool images (2,130 real F, 1,947 real N, 553 superimposed N).

| Set | Images | Accuracy | F recall | N recall | Run folder |
| --- | --- | --- | --- | --- | --- |
| Cross-validation, real photos | 4,077 | 0.966 ± 0.003 | 0.961 | 0.971 | `results/xval_2026-09-25_1422` |
| Cross-validation, superimposed | 553 | 0.934 ± 0.033 | — | 0.934 | same |
| Test 1, overall | 6,353 | 0.818 | 0.785 | 0.877 | `results/test_feeding_final_real_and_super_2026-10-01` |
| Test 1, `inat_2024` | 2,884 | 0.829 | 0.644 | 0.904 | same |
| Test 1, `inat_2024_gold` (21 N) | 2,162 | 0.870 | 0.873 | — | same |
| Test 1, `bimby_collection` | 1,307 | 0.705 | 0.725 | 0.559 | same |
| Test 2, Ontario | 2,785 | 0.775 | 0.618 | 0.900 | same |

- Accuracy drops from 0.966 in cross-validation to 0.818 / 0.775 on new photos, and on new photos the model mostly misses feeding (Test 1: 886 F called N, 273 N called F).
- Real-only vs real+superimposed (2026-10-01): tied on real photos (0.969 ± 0.009 vs 0.966 ± 0.003, McNemar p = 0.33), but the real-only model gets only 30.5% of composites right (it calls them feeding) vs 93.4%. The final model is real+superimposed, by the rule set before testing.
- The test sets have been used once. Choose any further change on cross-validation and report new test results as a second look.
- Recall on groups with very few photos (21 N in gold, 152 N in `bimby_collection`) is unreliable.

## Scripts

| Script | What it does | Reads | Writes |
| --- | --- | --- | --- |
| `Final_feeding_model_xval.py` | 5-fold cross-validation of the recipe; fold models are thrown away | `dev_pool.csv` minus `excluded.csv` | `results/xval_<date>_<time>/` |
| `Final_feeding_model_train_on_ALL.py` | trains the model we keep, same filter and recipe | same, all folds | `models/feeding_final_<…>_<date>.keras` + `results/train_<model>/` |
| `test_final_model.py` | tests one model once, Test 1 per source + Ontario | `test1.csv`, `test2_ontario.csv` | `results/test_<model>/` |
| `Feeding_model_only_with_flower_comparison.py`, `…_augmentedversion.py` | old real-only vs all-images comparison | `dev_pool.csv`, fold 0 held out | console. Superseded by the two 10-01 cross-validation runs; to be archived |
| `Model_Catalogue_Evaluation.ipynb` | scores every saved `.keras` model on the test sets | `test1.csv`, `test2_ontario.csv` | `results/model_catalogue_evaluation.*`. Old: ranks ~30 models on the test sets, so it can't give an honest number for the winner |

**Recipe** (the same in the cross-validation and final training): 224 × 224 input, only `conv5_block` trainable, head = global average pooling → dropout 0.5 → dense 512 → BatchNorm → ReLU → dropout 0.5 → 1 sigmoid unit; Adam 1e-3, batch 32, 10 epochs, horizontal flip + shear 0.2, no early stopping, threshold 0.5.

**Settings at the top of the scripts:** `DROP_UNTRACEABLE_SUPER` (drop the 103 superimposed images with no source photo in the pool), `TRAIN_ON_REAL_ONLY`, `SMOKE_TEST` (quick pipeline check, numbers mean nothing), `FORCE_CPU`.

**Run:** `mamba activate butterflyai_gpu`, then `python feeding_model/<script>.py`. Cross-validation takes about 36 min and final training about 7 min on an Apple M4. Every run records its git commit, package versions and data checksums in `run_info.json`; reruns on the same machine reproduce exactly.

## Models (`models/`, not in git)

- `feeding_final_real_and_super_2026-10-01.keras` — **current model**.
- `REAL_AND_SUPER_Final_feeding_model_unfrozen.keras` — JS's production model (Aug 2026), trained on the old ungrouped 80/20 training split. `monarch/Monarch_feeding_test.py`, `monarch/general_feeding_model_eval.py` and the catalogue notebook still load it.
- Everything else (`ResNet50_*`, `Real_Feeding_*`, `Final_feeding_model_ALL*`, …) — JS's 2024–2025 experiments, kept for reference.

## Results (`results/`)

| Folder / file | What it is |
| --- | --- |
| `xval_2026-09-25_1422` | first clean cross-validation, real + superimposed |
| `xval_2026-10-01_1043_real_only` | same folds, trained on real photos only |
| `xval_2026-10-01_1119` | rerun of `xval_2026-09-25_1422` on committed code; every number identical |
| `train_feeding_final_real_and_super_2026-10-01` | run record of the final model |
| `test_feeding_final_real_and_super_2026-10-01` | the one test of the final model (metrics + per-image predictions) |
| `Output_feeding_flower_only_real_test.txt`, `_ALL_test.txt` | **superseded.** JS's comparison runs on an ungrouped 80/20 split (559 photos: 427 real F, 132 composites); most of those composites were made from photos in the training split |
| `Output_feeding_test_on_BIMBY2024.txt`, `model_catalogue_evaluation.csv/.md` | **superseded.** Their "bimby2024" set contains ~1,950 training photos, and the catalogue ranks models on the test sets |

`READme for feeding images.rtf` is JS's original note on this folder; its paths and model names are out of date.
