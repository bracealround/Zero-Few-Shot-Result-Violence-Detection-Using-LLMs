import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID    = "mistralai/Mistral-7B-Instruct-v0.2"
INPUT_FILE  = "data/test_split.csv"
OUTPUT_FILE = "results/zero_shot_mistral_predictions.csv"

LABEL_MAP = {
    0: "Non-Violence",
    1: "Passive Violence",
    2: "Direct Violence"
}

os.makedirs("results", exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} ...")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name} | {round(p.total_memory / 1e9, 1)} GB VRAM")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print("Model loaded.\n")

# ── Prompt ────────────────────────────────────────────────────────────────────
# Mistral-Instruct does not support a system role — prepend instructions to the user turn
SYSTEM_PROMPT = """You are an expert content moderator specializing in Bangla social media text.
Your task is to classify Bangla text into exactly one of three categories:

0 - Non-Violence: Text that does not incite or contain any form of violence or abuse.
1 - Passive Violence: Text that contains abuse, insults, or degrading language targeting a person or community.
2 - Direct Violence: Text that directly calls for physical violence, killing, destruction, or attacks against a person or community.

Rules:
- Respond with ONLY a single digit: 0, 1, or 2.
- Do not explain your answer.
- Do not add any other text.

"""

def classify_zero_shot(text):
    messages = [
        {"role": "user", "content": SYSTEM_PROMPT + f"Classify this Bangla text:\n\n{text}"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    for char in raw:
        if char in ["0", "1", "2"]:
            return int(char)
    return -1

# ── Main loop ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} test samples")
print(f"Using model: {MODEL_ID}\n")

predictions = []
failed = 0

for i, row in df.iterrows():
    pred = classify_zero_shot(row["text"])
    predictions.append(pred)

    if pred == -1:
        failed += 1

    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{len(df)} | Failed so far: {failed}", flush=True)

# ── Save results ──────────────────────────────────────────────────────────────
df["predicted_label"]      = predictions
df["predicted_label_name"] = df["predicted_label"].map(LABEL_MAP)
df["true_label_name"]      = df["label"].map(LABEL_MAP)
df["correct"]              = df["label"] == df["predicted_label"]

df.to_csv(OUTPUT_FILE, index=False)

valid = df[df["predicted_label"] != -1]
print(f"\n=== Zero-Shot Mistral 7B Results ===")
print(f"Total samples:   {len(df)}")
print(f"Failed/unparsed: {failed}")
print(f"Accuracy:        {valid['correct'].mean():.4f}")
print(f"\nResults saved to {OUTPUT_FILE}")
