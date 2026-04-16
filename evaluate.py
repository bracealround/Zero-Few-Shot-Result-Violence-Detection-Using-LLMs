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
matplotlib.use("Agg")   # no display needed on server

LABEL_NAMES = ["Non-Violence", "Passive Violence", "Direct Violence"]

def evaluate(results_file, title):
    df = pd.read_csv(results_file)

    # Drop failed predictions
    failed = df[df["predicted_label"] == -1]
    if len(failed) > 0:
        print(f"⚠️  Dropping {len(failed)} failed predictions")
    df = df[df["predicted_label"] != -1]

    y_true = df["label"]
    y_pred = df["predicted_label"]

    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    print(f"Total evaluated: {len(df)}")
    print(f"\nAccuracy:        {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1:        {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1:     {f1_score(y_true, y_pred, average='weighted'):.4f}")

    print(f"\n--- Per-Class Report ---")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    out_file = f"results/{title.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {out_file}")

# Run evaluation
evaluate(
    "results/zero_shot_gpt4_predictions.csv",
    "Zero-Shot GPT-4o"
)