"""
leakage_safe_split.py

Group-aware split for the feeding / non-feeding dataset.

Each superimposed image is built by cutting the butterfly out of a real photo
and pasting it onto a flower background. The composite is named after its source, like:

real photo qwer.jpg
composite superimposed_qwe.jpg   (SAME individual butterfly)

If the split is made by randomly partitioning ROWS (train_test_split / KFold on
filenames), the real image and its superimposed twin can land on opposite sides of
the train/test line. The model sees that individual butterfly during
training and is scored on it at test time -> optimistic, inflated metrics
(esp non-feeding recall, which the superimposition experiment exists to
measure).
 
this script splits on source identity, not row, so twins are together on one side of the
split.

Class balance is preserved with StratifiedGroupKFold. By default it stratifies
on BOTH `label` and `photo_type` (so feeding/non-feeding AND real/super stay
balanced across the split)

Help from Claude generating parts of this script
reviewed by me - WF
"""

import re
import pandas as pd

_EXT_RE = re.compile(r"\.(jpg|jpeg|png)$", re.IGNORECASE)
_SUPER_RE = re.compile(r"^superimposed_(.+?)\.(?:jpg|jpeg|png)$", re.IGNORECASE)

# Balance feeding/non-feeding AND real/superimposed by default, matching the
# original row-wise stratify=label+photo_type behaviour.
DEFAULT_STRATIFY = ("label", "photo_type")


def source_group(filename):
    """Identity key tying a real image and all composites made from it together.

    'V2qjJokvPD.jpg'              -> 'V2qjJokvPD'
    'superimposed_V2qjJokvPD.jpg'-> 'V2qjJokvPD'          (same group as its source)
    'superimposed_1008.jpeg'     -> 'superimposed_1008'   (no real twin -> own group)
    """
    fn = str(filename)
    m = _SUPER_RE.match(fn)
    if m:
        return m.group(1)
    return _EXT_RE.sub("", fn)


def add_group_column(df, filename_col="filename", group_col="group"):
    """Add a group column. Composites whose stem matches no real image are
    given a unique singleton group so they go to either side."""
    df = df.copy()
    real_stems = {
        _EXT_RE.sub("", str(fn))
        for fn in df[filename_col]
        if not str(fn).lower().startswith("superimposed_")
    }

    def key(fn):
        fn = str(fn)
        m = _SUPER_RE.match(fn)
        if m:
            stem = m.group(1)
            # traceable composite -> share its real source's group;
            # numeric/orphan composite -> its own unique group.
            return stem if stem in real_stems else f"__solo__{fn}"
        return _EXT_RE.sub("", fn)

    df[group_col] = df[filename_col].map(key)
    return df


def stratify_key(df, stratify_cols=DEFAULT_STRATIFY):
    """Build a single categorical target from one or more columns (only those
    present), e.g. 'N_super', 'F_real'. Used to stratify the grouped split."""
    cols = [c for c in stratify_cols if c in df.columns]
    if not cols:
        raise ValueError(f"none of stratify_cols={stratify_cols} are in the dataframe")
    key = df[cols[0]].astype(str)
    for c in cols[1:]:
        key = key + "_" + df[c].astype(str)
    return key.values


def grouped_train_test_split(df, test_size=0.2, seed=321,
                             stratify_cols=DEFAULT_STRATIFY, group_col="group"):
    """Group-aware, stratified 80/20-style split.

    Guarantees no group (individual butterfly) appears on both sides, while
    keeping label AND photo_type balanced. Returns (train_df, test_df)."""
    from sklearn.model_selection import StratifiedGroupKFold

    if group_col not in df.columns:
        df = add_group_column(df)

    y = stratify_key(df, stratify_cols)
    n_splits = max(2, round(1 / test_size))  # test_size=0.2 -> 5 folds -> ~20% test
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df[group_col].values, y,
                                          groups=df[group_col].values))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def grouped_kfold(df, n_splits=5, seed=42,
                  stratify_cols=DEFAULT_STRATIFY, group_col="group"):
    """Yield (train_df, val_df) for group-aware, stratified K-fold CV."""
    from sklearn.model_selection import StratifiedGroupKFold

    if group_col not in df.columns:
        df = add_group_column(df)
    y = stratify_key(df, stratify_cols)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in sgkf.split(df[group_col].values, y, groups=df[group_col].values):
        yield df.iloc[tr].copy(), df.iloc[va].copy()


def verify_no_leakage(train_df, test_df, group_col="group"):
    """Raise if any group straddles the split. Returns the offending groups."""
    if group_col not in train_df.columns:
        train_df = add_group_column(train_df)
        test_df = add_group_column(test_df)
    overlap = set(train_df[group_col]) & set(test_df[group_col])
    if overlap:
        raise AssertionError(
            f"LEAKAGE: {len(overlap)} source group(s) appear in both splits: "
            f"{sorted(list(overlap))[:10]}{'...' if len(overlap) > 10 else ''}"
        )
    return overlap
