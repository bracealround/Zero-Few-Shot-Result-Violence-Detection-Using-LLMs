import pandas as pd
import numpy as np
from collections import Counter

# ── 1. Load ──────────────────────────────────────────────────────────────────
train_df = pd.read_csv("data/train.csv")

val_df = pd.read_csv("data/dev.csv")

print("=== Dataset Overview ===")
print(f"Total rows:     {len(train_df)}")
print(f"Columns:        {list(train_df.columns)}")
print(f"Missing values:\n{train_df.isnull().sum()}")

# ── 2. Label mapping ─────────────────────────────────────────────────────────
label_map = {
    0: "Non-Violence",
    1: "Passive Violence",
    2: "Direct Violence"
}
train_df["label_name"] = train_df["label"].map(label_map)

print("\n=== Class Distribution ===")
counts = train_df["label_name"].value_counts()
for label, count in counts.items():
    pct = count / len(train_df) * 100
    print(f"  {label:<20} {count:>5} ({pct:.1f}%)")

# ── 3. Basic text stats ───────────────────────────────────────────────────────
train_df["text_length"] = train_df["text"].astype(str).apply(len)
print("\n=== Text Length Stats ===")
print(train_df["text_length"].describe().round(1))

# ── 4. Clean: drop nulls and duplicates ──────────────────────────────────────
before = len(train_df)
train_df = train_df.dropna(subset=["text", "label"])
train_df = train_df.drop_duplicates(subset=["text"])
train_df["text"] = train_df["text"].astype(str).str.strip()
print(f"\n=== Cleaning ===")
print(f"Rows before: {before}, after: {len(train_df)} (removed {before - len(train_df)})")

# ── 5. Split into train/test (if no separate test file) ───────────────────────
from sklearn.model_selection import train_test_split

train_data = train_df.copy()

# Clean val set too
val_df = val_df.dropna(subset=["text", "label"])
val_df = val_df.drop_duplicates(subset=["text"])
val_df["text"] = val_df["text"].astype(str).str.strip()
val_df["label_name"] = val_df["label"].map(label_map)
test_data = val_df.copy()

print(f"\n=== Train/Test Split ===")
print(f"Train: {len(train_data)} rows")
print(f"Test:  {len(test_data)} rows")

print("\nTrain distribution:")
for label, count in train_data["label_name"].value_counts().items():
    print(f"  {label:<20} {count}")

print("\nTest distribution:")
for label, count in test_data["label_name"].value_counts().items():
    print(f"  {label:<20} {count}")

# ── 6. Build few-shot pool (from train set only!) ─────────────────────────────
# Take N examples per class for few-shot prompting later
N_SHOTS = 3   # we'll use 3 examples per class = 9 total examples

few_shot_pool = train_data.groupby("label").apply(
    lambda x: x.sample(n=N_SHOTS, random_state=42)
).reset_index(drop=True)

print(f"\n=== Few-Shot Pool ===")
print(f"Total examples: {len(few_shot_pool)} ({N_SHOTS} per class)")
print(few_shot_pool[["text", "label_name"]])

# ── 7. Save everything ────────────────────────────────────────────────────────
train_data.to_csv("data/train_split.csv", index=False)
test_data.to_csv("data/test_split.csv", index=False)
few_shot_pool.to_csv("data/few_shot_pool.csv", index=False)

print("\n=== Saved ===")
print("  data/train_split.csv")
print("  data/test_split.csv")
print("  data/few_shot_pool.csv")
print("\nPhase 2 complete! ✅")