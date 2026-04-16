"""
generate_figures.py
Produces all report figures for the Zero-Shot vs Few-Shot LLM paper.

Figures generated:
  1. Confusion matrices for every LLM condition
  2. Macro F1 bar chart — baselines + all LLMs
  3. Per-class F1 grouped bar chart — LLM conditions only
  4. Zero-shot vs Few-shot delta chart — GPT-4o and Mistral
  5. Error analysis — misclassification heatmap per model
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, f1_score, classification_report
)
import os

os.makedirs("results/figures", exist_ok=True)

LABEL_NAMES  = ["Non-Violence", "Passive Violence", "Direct Violence"]
LABEL_SHORT  = ["Non-V", "Passive", "Direct"]
CLASSES      = [0, 1, 2]

# ── Published baseline results (validation set, from Saha et al. 2023) ────────
BASELINES = {
    "Unigram SVM":    {"Direct": 0.444, "Passive": 0.629, "Non-V": 0.776, "Macro": 0.616},
    "Char-2-gram":    {"Direct": 0.490, "Passive": 0.630, "Non-V": 0.788, "Macro": 0.636},
    "Char-3-gram":    {"Direct": 0.505, "Passive": 0.662, "Non-V": 0.803, "Macro": 0.657},
    "DistilBERT":     {"Direct": 0.548, "Passive": 0.637, "Non-V": 0.771, "Macro": 0.652},
    "MBERT":          {"Direct": 0.601, "Passive": 0.643, "Non-V": 0.802, "Macro": 0.682},
    "XLM-RoBERTa":   {"Direct": 0.670, "Passive": 0.704, "Non-V": 0.729, "Macro": 0.729},
    "BanglaBERT":     {"Direct": 0.754, "Passive": 0.764, "Non-V": 0.846, "Macro": 0.788},
}

# ── LLM result files ──────────────────────────────────────────────────────────
LLM_RUNS = [
    ("results/zero_shot_gpt4_predictions.csv",    "Zero-Shot\nGPT-4o",      "#4C72B0"),
    ("results/few_shot_gpt4_predictions.csv",     "Few-Shot\nGPT-4o",       "#4C72B0"),
    ("results/zero_shot_llama_predictions.csv",   "Zero-Shot\nLLaMA 3.1 8B","#DD8452"),
    ("results/few_shot_llama_predictions.csv",    "Few-Shot\nLLaMA 3.1 8B", "#DD8452"),
    ("results/zero_shot_mistral_predictions.csv", "Zero-Shot\nMistral 7B",  "#55A868"),
    ("results/few_shot_mistral_predictions.csv",  "Few-Shot\nMistral 7B",   "#55A868"),
]

def load_results(fpath):
    df = pd.read_csv(fpath)
    df = df[df["predicted_label"] != -1]
    return df["label"].values, df["predicted_label"].values

def per_class_f1(y_true, y_pred):
    scores = f1_score(y_true, y_pred, average=None, labels=CLASSES, zero_division=0)
    return {"Direct": scores[2], "Passive": scores[1], "Non-V": scores[0]}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Confusion matrices for each LLM condition
# ─────────────────────────────────────────────────────────────────────────────
CMAPS = {
    "GPT-4o":    "Blues",
    "LLaMA":     "Oranges",
    "Mistral":   "Greens",
}

def get_cmap(label):
    for key, cmap in CMAPS.items():
        if key in label:
            return cmap
    return "Blues"

print("Figure 1: Confusion matrices ...")
for fpath, label, _ in LLM_RUNS:
    try:
        y_true, y_pred = load_results(fpath)
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap=get_cmap(label),
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
        ax.set_title(label.replace("\n", " "), fontsize=13, fontweight="bold")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()
        fname = label.replace("\n", "_").replace(" ", "_").replace("-", "").lower()
        out = f"results/figures/cm_{fname}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved {out}")
    except FileNotFoundError:
        print(f"  Skipping {label} — file not found")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Macro F1 comparison: baselines + all LLMs
# ─────────────────────────────────────────────────────────────────────────────
print("\nFigure 2: Macro F1 bar chart ...")

baseline_names  = list(BASELINES.keys())
baseline_macros = [v["Macro"] for v in BASELINES.values()]

llm_names, llm_macros, llm_colors = [], [], []
for fpath, label, color in LLM_RUNS:
    try:
        y_true, y_pred = load_results(fpath)
        macro = f1_score(y_true, y_pred, average="macro")
        llm_names.append(label.replace("\n", " "))
        llm_macros.append(macro)
        llm_colors.append(color)
    except FileNotFoundError:
        pass

all_names  = baseline_names + llm_names
all_macros = baseline_macros + llm_macros
all_colors = ["#B0B0B0"] * len(baseline_names) + llm_colors

fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(range(len(all_names)), all_macros, color=all_colors, edgecolor="white", linewidth=0.8)

# Value labels on bars
for bar, val in zip(bars, all_macros):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(range(len(all_names)))
ax.set_xticklabels(all_names, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Macro F1", fontsize=11)
ax.set_title("Macro F1 Comparison: Baselines vs LLM Prompting", fontsize=13, fontweight="bold")
ax.set_ylim(0, 0.95)
ax.axvline(len(baseline_names) - 0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
ax.text(len(baseline_names) - 0.5 - 0.1, 0.91, "Baselines", ha="right", fontsize=9, color="gray")
ax.text(len(baseline_names) - 0.5 + 0.1, 0.91, "LLM Prompting", ha="left", fontsize=9, color="gray")

legend_patches = [
    mpatches.Patch(color="#B0B0B0", label="Baselines (val set)"),
    mpatches.Patch(color="#4C72B0", label="GPT-4o"),
    mpatches.Patch(color="#DD8452", label="LLaMA 3.1 8B"),
    mpatches.Patch(color="#55A868", label="Mistral 7B"),
]
ax.legend(handles=legend_patches, fontsize=9, loc="upper left")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/macro_f1_comparison.png", dpi=150)
plt.close()
print("  Saved results/figures/macro_f1_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Per-class F1 grouped bar chart (LLMs only)
# ─────────────────────────────────────────────────────────────────────────────
print("\nFigure 3: Per-class F1 grouped bar chart ...")

llm_labels, direct_f1s, passive_f1s, nonv_f1s = [], [], [], []
for fpath, label, _ in LLM_RUNS:
    try:
        y_true, y_pred = load_results(fpath)
        pc = per_class_f1(y_true, y_pred)
        llm_labels.append(label.replace("\n", " "))
        direct_f1s.append(pc["Direct"])
        passive_f1s.append(pc["Passive"])
        nonv_f1s.append(pc["Non-V"])
    except FileNotFoundError:
        pass

x = np.arange(len(llm_labels))
width = 0.26

fig, ax = plt.subplots(figsize=(11, 5))
b1 = ax.bar(x - width, direct_f1s,  width, label="Direct Violence",  color="#E74C3C", alpha=0.85)
b2 = ax.bar(x,         passive_f1s, width, label="Passive Violence", color="#F39C12", alpha=0.85)
b3 = ax.bar(x + width, nonv_f1s,   width, label="Non-Violence",     color="#27AE60", alpha=0.85)

for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(llm_labels, fontsize=9.5)
ax.set_ylabel("F1 Score", fontsize=11)
ax.set_title("Per-Class F1 Score by Model and Prompting Strategy", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.0)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/per_class_f1.png", dpi=150)
plt.close()
print("  Saved results/figures/per_class_f1.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Zero-shot vs Few-shot delta (GPT-4o and Mistral)
# ─────────────────────────────────────────────────────────────────────────────
print("\nFigure 4: Zero-shot vs Few-shot delta ...")

PAIRS = [
    ("results/zero_shot_gpt4_predictions.csv",    "results/few_shot_gpt4_predictions.csv",    "GPT-4o",       "#4C72B0"),
    ("results/zero_shot_llama_predictions.csv",   "results/few_shot_llama_predictions.csv",   "LLaMA 3.1 8B", "#DD8452"),
    ("results/zero_shot_mistral_predictions.csv", "results/few_shot_mistral_predictions.csv", "Mistral 7B",   "#55A868"),
]

fig, axes = plt.subplots(1, len(PAIRS), figsize=(15, 5), sharey=False)
metrics = ["Direct", "Passive", "Non-V", "Macro"]
x = np.arange(len(metrics))
width = 0.35

for ax, (zpath, fpath, model_name, color) in zip(axes, PAIRS):
    try:
        yz_true, yz_pred = load_results(zpath)
        yf_true, yf_pred = load_results(fpath)

        zpc = per_class_f1(yz_true, yz_pred)
        fpc = per_class_f1(yf_true, yf_pred)
        zmacro = f1_score(yz_true, yz_pred, average="macro")
        fmacro = f1_score(yf_true, yf_pred, average="macro")

        z_scores = [zpc["Direct"], zpc["Passive"], zpc["Non-V"], zmacro]
        f_scores = [fpc["Direct"], fpc["Passive"], fpc["Non-V"], fmacro]

        bz = ax.bar(x - width/2, z_scores, width, label="Zero-Shot", color=color, alpha=0.5)
        bf = ax.bar(x + width/2, f_scores, width, label="Few-Shot",  color=color, alpha=0.9)

        # Delta annotations
        for xi, (zs, fs) in enumerate(zip(z_scores, f_scores)):
            delta = fs - zs
            sign  = "+" if delta >= 0 else ""
            ax.text(xi, max(zs, fs) + 0.015, f"{sign}{delta:.2f}",
                    ha="center", va="bottom", fontsize=8,
                    color="#27AE60" if delta >= 0 else "#E74C3C", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_title(model_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("F1 Score", fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    except FileNotFoundError:
        ax.set_title(f"{model_name} (missing data)")

fig.suptitle("Zero-Shot vs Few-Shot F1 Improvement", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/figures/zeroshot_vs_fewshot_delta.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved results/figures/zeroshot_vs_fewshot_delta.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Error analysis: misclassification heatmap per model
# ─────────────────────────────────────────────────────────────────────────────
print("\nFigure 5: Error analysis heatmap ...")

error_data = {}
for fpath, label, _ in LLM_RUNS:
    try:
        y_true, y_pred = load_results(fpath)
        # Only misclassified
        mask = y_true != y_pred
        yt_err = y_true[mask]
        yp_err = y_pred[mask]

        # For each true class, count what it was predicted as
        row = {}
        for true_cls in range(3):
            for pred_cls in range(3):
                if true_cls == pred_cls:
                    continue
                key = f"{LABEL_SHORT[true_cls]}→{LABEL_SHORT[pred_cls]}"
                count = ((yt_err == true_cls) & (yp_err == pred_cls)).sum()
                row[key] = count
        error_data[label.replace("\n", " ")] = row
    except FileNotFoundError:
        pass

error_df = pd.DataFrame(error_data).T.fillna(0).astype(int)

# Sort columns by total errors
col_order = error_df.sum().sort_values(ascending=False).index
error_df = error_df[col_order]

fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(error_df, annot=True, fmt="d", cmap="YlOrRd",
            linewidths=0.5, linecolor="white", ax=ax)
ax.set_title("Error Analysis: Misclassification Counts per Model\n(True → Predicted, errors only)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Error Type (True → Predicted)", fontsize=10)
ax.set_ylabel("Model", fontsize=10)
ax.tick_params(axis="x", rotation=30)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.savefig("results/figures/error_analysis.png", dpi=150)
plt.close()
print("  Saved results/figures/error_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — 6-panel confusion matrices (Zero/Few-Shot x GPT-4o/LLaMA/Mistral)
# ─────────────────────────────────────────────────────────────────────────────
print("\nFigure 6: 6-panel confusion matrices ...")

PANEL_RUNS = [
    ("results/zero_shot_gpt4_predictions.csv",    "Zero-Shot GPT-4o",       "Blues"),
    ("results/few_shot_gpt4_predictions.csv",     "Few-Shot GPT-4o",        "Blues"),
    ("results/zero_shot_llama_predictions.csv",   "Zero-Shot LLaMA 3.1 8B", "Oranges"),
    ("results/few_shot_llama_predictions.csv",    "Few-Shot LLaMA 3.1 8B",  "Oranges"),
    ("results/zero_shot_mistral_predictions.csv", "Zero-Shot Mistral 7B",   "Greens"),
    ("results/few_shot_mistral_predictions.csv",  "Few-Shot Mistral 7B",    "Greens"),
]

LABEL_SHORT_CM = ["Non-V", "Passive V", "Direct V"]

fig, axes = plt.subplots(3, 2, figsize=(12, 15))
fig.suptitle("Confusion Matrices: GPT-4o (top), LLaMA 3.1 8B (middle), Mistral 7B (bottom)",
             fontsize=13, fontweight="bold", y=1.01)

for ax, (fpath, title, cmap) in zip(axes.flatten(), PANEL_RUNS):
    try:
        y_true, y_pred = load_results(fpath)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                    xticklabels=LABEL_SHORT_CM, yticklabels=LABEL_SHORT_CM,
                    ax=ax, cbar=False)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
    except FileNotFoundError:
        ax.set_title(f"{title} (missing)")
        ax.axis("off")

plt.tight_layout()
plt.savefig("results/figures/confusion_matrices_4panel.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved results/figures/confusion_matrices_4panel.png")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print(" All figures saved to results/figures/")
print("="*55)
print(f"  cm_*.png                    — Confusion matrices")
print(f"  macro_f1_comparison.png     — Main result bar chart")
print(f"  per_class_f1.png            — Per-class grouped bars")
print(f"  zeroshot_vs_fewshot_delta.png — Zero vs few-shot gain")
print(f"  error_analysis.png          — Misclassification heatmap")
