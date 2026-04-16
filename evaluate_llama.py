import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

LABEL_NAMES = ["Non-Violence", "Passive Violence", "Direct Violence"]

def evaluate(results_file, title, cmap="Oranges"):
    df = pd.read_csv(results_file)

    failed = (df["predicted_label"] == -1).sum()
    if failed > 0:
        print(f"Warning: Dropping {failed} failed predictions")
    df = df[df["predicted_label"] != -1]

    y_true = df["label"]
    y_pred = df["predicted_label"]

    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    print(f"Total evaluated: {len(df)}")
    print(f"Accuracy:        {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1:        {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1:     {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"\n--- Per-Class Report ---")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    out_file = f"results/{title.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {out_file}")

# ── All runs — add/comment out as results become available ───────────────────
RUNS = [
    ("results/zero_shot_gpt4_predictions.csv",    "Zero-Shot GPT-4o",      "Blues"),
    ("results/few_shot_gpt4_predictions.csv",     "Few-Shot GPT-4o",       "Blues"),
    ("results/zero_shot_llama_predictions.csv",   "Zero-Shot LLaMA 3.1 8B","Oranges"),
    ("results/zero_shot_mistral_predictions.csv", "Zero-Shot Mistral 7B",  "Greens"),
    ("results/few_shot_mistral_predictions.csv",  "Few-Shot Mistral 7B",   "Greens"),
    # Uncomment when LLaMA few-shot is available:
    # ("results/few_shot_llama_predictions.csv",  "Few-Shot LLaMA 3.1 8B", "Oranges"),
]

for fname, title, cmap in RUNS:
    try:
        evaluate(fname, title, cmap)
    except FileNotFoundError:
        print(f"\nSkipping '{title}' — file not found: {fname}")

# ── Summary comparison table ──────────────────────────────────────────────────
print(f"\n{'='*75}")
print(f" FULL COMPARISON — All LLM Prompting Results")
print(f"{'='*75}")
print(f"{'Model':<30} {'Accuracy':>9} {'Macro F1':>9} {'Direct F1':>10} {'Passive F1':>11} {'Non-V F1':>9}")
print(f"{'-'*75}")

for fname, label, _ in RUNS:
    try:
        df = pd.read_csv(fname)
        df = df[df["predicted_label"] != -1]
        y_true, y_pred = df["label"], df["predicted_label"]

        acc     = accuracy_score(y_true, y_pred)
        macro   = f1_score(y_true, y_pred, average="macro")
        per_cls = f1_score(y_true, y_pred, average=None, labels=[2, 1, 0])
        direct_f1, passive_f1, nonv_f1 = per_cls

        print(f"{label:<30} {acc:>9.4f} {macro:>9.4f} {direct_f1:>10.4f} {passive_f1:>11.4f} {nonv_f1:>9.4f}")
    except FileNotFoundError:
        print(f"{label:<30} {'(not yet run)':>50}")
