# ButterflyAI — data splits audit (2026-09-24)

Overlaps were checked by file-content hash (MD5), not filename, across every image folder in the repo.

## 1. Image types and where they live

| Image type | Folder | Count | Labels |
|---|---|---|---|
| Real butterfly photos, feeding / non-feeding (BC BIMBY 2023) | `feeding_model/data/Final_Feeding_Images` (`photo_type=real`) | 4,091 | 2,137 F / 1,954 N |
| Superimposed feeding composites (`superimposed_*`) | same folder (`photo_type=super`) | 659 | all N. 557 are built from a real N photo in the same folder; 102 have no traceable source |
| Flower-only photos (no butterfly) | `plant_id_model/data/plant_data_specified` | 5,000 | 500 per plant species |
| Detectron composites (butterfly pasted onto a flower photo) | `plant_id_model/data/specified_flower_photos_detectron_ALL` (`aug_*.png`) | 13,378 | 10 plant species. These are **superimposed**, just not named that way |
| Plain monarch photos (iNat research grade) | `monarch/data/Monarch_images` | 8,018 | 2,843 Feeding / 5,175 Non_feeding, made by sorting with the production model and then reviewing by hand |
| Joint gold standard (real, feeding + plant labels) | `test_data/Joint_gold_standard` | 2,166 | 2,145 Feeding / 21 Nonfeeding |
| BC 2024 BIMBY + "appended" set | `test_data/2024_BIMBY_appended` | 6,156 rows | 3,908 F / 2,248 NF |
| Unused | `test_data/Gold_standard_research_grade` (older gold), `test_data/2024_BIMBY`, `test_data/Ontario_2024`, `feeding_model/data/F` (exact copy of Ontario F) | | not read by any script |

## 2. What each script uses as train / validation / test

### Feeding model
| Script | Train | Validation | Test |
|---|---|---|---|
| `Final_feeding_model_train_on_ALL.py` → **production** `REAL_AND_SUPER_…` | `_train.csv`: real + super. The model file (Aug 25) predates the grouped split (Sep 18), so it was trained on what is now `_train.leaky.csv` | none | none in script |
| `Final_feeding_model_xval.py` | 4/5 of full CSV (real + super) | 1/5 fold, **includes super**; checkpoints on val_loss | none |
| `Feeding_model_only_with_flower_comparison.py` / `…augmentedversion.py` | grouped 80%: real-only or real+super | 20%, filtered to **real F + super N only** (real N dropped) | same set (reported as "test") |
| `Model_Catalogue_Evaluation.ipynb` | — | **gold** (used to rank ~30 models, so it acts as validation) | **bimby2024** |

### Plant ID model
| Script | Train | Validation / test |
|---|---|---|
| `Final_plant_OVR_xval.py` | Points to `Flower_only_specified.csv`, which doesn't exist, and to a wrong path. The saved results and models use the detectron CSV's class names and order (`Sisybirum_loeselii`, `Ericameria` first), so they were **very likely trained on detectron composites** | stratified k-fold folds of the same data, so **superimposed** |
| `Scaling_of_flower_dataset.py` | random 80% of detectron composites | random 20% of detectron composites, **all superimposed** |
| `OVR_test_on_other.py` | — | real BC2024 plant set; the data is missing, so it can't run |

Unused: the flower-only photos, and the gold standard's plant labels. **No plant model is currently evaluated on real butterfly-on-flower photos.**

### Monarch
| Script | Train | Validation / test |
|---|---|---|
| `Monarch_feeding_train_xval.py` | 4/5 of monarch photos | 1/5 fold (plain `KFold`); no held-out test |
| `general_feeding_model_eval.py` | — | Scores the production feeding model against monarch labels that were made by reviewing that same model's sorting, so the evaluation is **circular** |

No superimposed images are used for monarchs (listed as a TODO).

## 3. Problems found

1. **The bimby2024 "test" set contains training images.** 1,945 of its 6,153 images (32%) are also in `Final_Feeding_Images`, and 1,590 of them were in the production model's training CSV. All come from the "appended" half (the BIMBY_redo rows: 1,914 F + 36 NF). The bimby2024 scores in the catalogue are inflated.
2. **Gold and bimby2024 aren't independent.** 2,241 of gold's 2,269 non-feeding images are the same files as bimby2024's non-feeding images (built that way in `Joint_model_csvs.R`).
3. **Gold overlaps training a little:** 36 images, 22 of them in the train CSV. Of those 36, 30 are labelled N in gold but F in training.
4. **Labels conflict across datasets:** 66 images labelled differently in bimby2024 vs training. `Final_Feeding_Images` also has 18 duplicate image pairs; 7 have conflicting labels and 6 are split across train/test.
5. **The production feeding model was trained on the leaky split**, where 90 superimposed test images had their real twin in train. This only affects the internal 80/20 numbers.
6. **The superimposed images in the plant model aren't labelled as such.** Its only working evaluations are on composites.
7. **Monarch k-fold is not grouped.** 13 duplicate image pairs could straddle folds.
8. **Gold is used for model selection**, so the repo currently has no untouched feeding test set.
