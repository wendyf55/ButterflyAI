"""
Regenerate leakage-free train/test split CSVs and audit the old ones.

- Backs up the existing (leaky) DataFilenamesRedo_{train,test}.csv to *.leaky.csv
- Writes new grouped, stratified DataFilenamesRedo_{train,test}.csv (twins kept together)
- Prints a before/after leakage audit and asserts zero cross-split twins.

Run:  python regenerate_grouped_splits.py

Claude helped with writing this script. 
Code edited and reviewed by me - WF
"""
import shutil
from pathlib import Path
import pandas as pd

from leakage_safe_split import (
    add_group_column, grouped_train_test_split, verify_no_leakage,
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "Final_Feeding_Images"
FULL = DATA_DIR / "DataFilenamesRedo.csv"
TRAIN = DATA_DIR / "DataFilenamesRedo_train.csv"
TEST = DATA_DIR / "DataFilenamesRedo_test.csv"
SEED = 321
TEST_SIZE = 0.2


def straddling_groups(train_df, test_df):
    """Groups that appear on BOTH sides, using a GLOBAL identity assigned on the
    union (so a real in test and its composite in train share one group).
    """
    tr = train_df.copy(); tr["_side"] = "train"
    te = test_df.copy(); te["_side"] = "test"
    both = add_group_column(pd.concat([tr, te], ignore_index=True))
    per = both.groupby("group")["_side"].nunique()
    return set(per[per > 1].index)


def audit(train_df, test_df, tag):
    overlap = straddling_groups(train_df, test_df)
    print(f"--- {tag} ---")
    print(f"  train rows={len(train_df)}  test rows={len(test_df)}")
    print(f"  source groups straddling the split (leakage): {len(overlap)}")
    for col in ("label", "photo_type"):
        if col in train_df.columns:
            print(f"  train {col}: {train_df[col].value_counts().to_dict()}")
            print(f"  test  {col}: {test_df[col].value_counts().to_dict()}")
    return len(overlap)


def main():
    full = pd.read_csv(FULL)

    if TRAIN.exists() and TEST.exists():
        print("=" * 60)
        audit(pd.read_csv(TRAIN), pd.read_csv(TEST),
              "BEFORE (existing row-wise split)")

    full = add_group_column(full)
    train_df, test_df = grouped_train_test_split(full, test_size=TEST_SIZE, seed=SEED)
    verify_no_leakage(train_df, test_df)

    print("=" * 60)
    n = audit(train_df, test_df, "AFTER (grouped split)")
    assert n == 0, "grouped split still leaks — investigate"

    for p in (TRAIN, TEST):
        if p.exists():
            bak = p.with_suffix(".leaky.csv")
            if not bak.exists():
                shutil.copy2(p, bak)
                print(f"backed up {p.name} -> {bak.name}")

    train_df.drop(columns=["group"]).to_csv(TRAIN, index=False)
    test_df.drop(columns=["group"]).to_csv(TEST, index=False)
    print(f"\nwrote leakage-free {TRAIN.name} and {TEST.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
