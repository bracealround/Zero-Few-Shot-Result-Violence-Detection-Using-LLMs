# Zero-Shot vs Few-Shot LLM Prompting for Bangla Violence Detection

Benchmarking zero-shot and few-shot prompting strategies across GPT-4o, LLaMA 3.1 8B, and Mistral 7B on a Bangla social media violence detection dataset. Models classify text into three categories: **Non-Violence**, **Passive Violence**, and **Direct Violence**.

## Results

| Model | Strategy | Accuracy | Macro F1 | Direct F1 | Passive F1 | Non-V F1 |
|---|---|---|---|---|---|---|
| GPT-4o | Zero-Shot | 0.7023 | 0.6559 | 0.5117 | 0.6890 | 0.7670 |
| GPT-4o | Few-Shot | **0.7617** | **0.7202** | 0.6246 | 0.7108 | 0.8253 |
| LLaMA 3.1 8B | Zero-Shot | 0.6391 | 0.5893 | 0.4424 | 0.6019 | 0.7236 |
| LLaMA 3.1 8B | Few-Shot | 0.6436 | — | — | — | — |
| Mistral 7B | Zero-Shot | 0.5820 | 0.4310 | 0.1370 | 0.4455 | 0.7106 |
| Mistral 7B | Few-Shot | 0.5759 | 0.5310 | 0.3969 | 0.5320 | 0.6641 |

Supervised baselines from Saha et al. (2023) for reference: BanglaBERT achieves **Macro F1 = 0.788**.

## Dataset

- `data/train_split.csv` — training split
- `data/test_split.csv` — test split (1,330 samples)
- `data/few_shot_pool.csv` — 9 labeled examples used for few-shot prompting (3 per class)
- `data/dev.csv` — development set

Labels: `0` = Non-Violence, `1` = Passive Violence, `2` = Direct Violence

## Project Structure

```
.
├── data/                        # Dataset splits
├── results/
│   ├── figures/                 # All generated figures
│   └── *_predictions.csv        # Model prediction outputs
├── zero_shot_gpt4.py            # GPT-4o zero-shot inference
├── few_shot_gpt4.py             # GPT-4o few-shot inference
├── zero_shot_llama.py           # LLaMA 3.1 8B zero-shot inference
├── few_shot_llama.py            # LLaMA 3.1 8B few-shot inference
├── zero_shot_mistral.py         # Mistral 7B zero-shot inference
├── few_shot_mistral.py          # Mistral 7B few-shot inference
├── evaluate.py                  # Evaluate zero-shot results
├── evaluate_few_shot.py         # Evaluate few-shot results
├── evaluate_llama.py            # Full summary across all models
├── generate_figures.py          # Generate all paper figures
├── data_prep.py                 # Dataset preparation
└── rerun_failed.py              # Retry failed predictions
```

## Setup

```bash
pip install transformers torch pandas scikit-learn matplotlib seaborn openai
```

**For LLaMA 3.1 8B** — requires a HuggingFace account with access granted to `meta-llama/Meta-Llama-3.1-8B-Instruct`:

```bash
export HF_TOKEN=hf_your_token_here
```

**For GPT-4o:**

```bash
export OPENAI_API_KEY=sk-your_key_here
```

## Running Inference

```bash
# GPT-4o
python zero_shot_gpt4.py
python few_shot_gpt4.py

# LLaMA 3.1 8B (downloads ~16GB on first run)
python zero_shot_llama.py
python few_shot_llama.py

# Mistral 7B
python zero_shot_mistral.py
python few_shot_mistral.py
```

## Evaluation & Figures

```bash
# Print metrics for all models
python evaluate_llama.py

# Regenerate all figures
python generate_figures.py
```

Figures are saved to `results/figures/`:
- `confusion_matrices_4panel.png` — 6-panel confusion matrices for all models
- `macro_f1_comparison.png` — Macro F1 vs supervised baselines
- `per_class_f1.png` — Per-class F1 grouped by model
- `zeroshot_vs_fewshot_delta.png` — Zero-shot vs few-shot gain per model
- `error_analysis.png` — Misclassification heatmap

## Reference

Saha et al. (2023). *VioLi: Violence Detection in Bangla Text.* ACM SIGIR.
