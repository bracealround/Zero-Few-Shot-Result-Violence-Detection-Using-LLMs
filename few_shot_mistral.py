import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID      = "mistralai/Mistral-7B-Instruct-v0.2"
INPUT_FILE    = "data/test_split.csv"
FEW_SHOT_FILE = "data/few_shot_pool.csv"
OUTPUT_FILE   = "results/few_shot_mistral_predictions.csv"

LABEL_MAP = {
    0: "Non-Violence",
    1: "Passive Violence",
    2: "Direct Violence"
}

os.makedirs("results", exist_ok=True)

# ── Load few-shot examples ────────────────────────────────────────────────────
few_shot_df = pd.read_csv(FEW_SHOT_FILE)
print(f"Loaded {len(few_shot_df)} few-shot examples:")
print(few_shot_df[["text", "label"]].to_string())
print()

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
Your task is to classify Bangla text into exactly one of three categories using this decision framework:

Step 1: Does the post call for or justify any form of violence against a person or community?
  - If NO → go to Step 4
  - If YES → go to Step 2

Step 2: Does it call for DIRECT violence (killing, attacking, raping, deportation, forceful conversion)?
  - If YES → go to Step 3
  - If NO → Label = 1 (Passive Violence)

Step 3: Does it explicitly call for Kill/Attack or physical harm against a person or community?
  - If YES → Label = 2 (Direct Violence)
  - If NO → Label = 2 (Direct Violence) [repression/deportation also counts]

Step 4: Does it express social rights, protests, or non-violent discussion?
  - If YES or NO → Label = 0 (Non-Violence)

Category definitions:
0 - Non-Violence: Text that does not incite violence. Includes peaceful discussion, social rights, general opinions.
1 - Passive Violence: Derogatory language, insults, abuse, slurs targeting a person or community. Justification of violence.
2 - Direct Violence: Explicit calls for killing, attacking, rape, vandalism, deportation, or forceful conversion.

Rules:
- Respond with ONLY a single digit: 0, 1, or 2.
- Do not explain your answer.
- Do not add any other text.

"""

# ── Build few-shot message list ───────────────────────────────────────────────
def build_few_shot_messages(text):
    messages = []

    # First user turn includes the full system prompt + first example
    first_row = few_shot_df.iloc[0]
    messages.append({
        "role": "user",
        "content": SYSTEM_PROMPT + f"Classify this Bangla text:\n\n{first_row['text']}"
    })
    messages.append({
        "role": "assistant",
        "content": str(int(first_row["label"]))
    })

    # Remaining few-shot examples as plain user/assistant turns
    for _, row in few_shot_df.iloc[1:].iterrows():
        messages.append({
            "role": "user",
            "content": f"Classify this Bangla text:\n\n{row['text']}"
        })
        messages.append({
            "role": "assistant",
            "content": str(int(row["label"]))
        })

    # Actual query
    messages.append({
        "role": "user",
        "content": f"Classify this Bangla text:\n\n{text}"
    })
    return messages

# ── Classification function ───────────────────────────────────────────────────
def classify_few_shot(text):
    messages = build_few_shot_messages(text)

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
print(f"Using model: {MODEL_ID} with {len(few_shot_df)} few-shot examples\n")

predictions = []
failed = 0

for i, row in df.iterrows():
    pred = classify_few_shot(row["text"])
    predictions.append(pred)

    if pred == -1:
        failed += 1
        print(f"  Warning: Failed on row {i}", flush=True)

    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{len(df)} | Failed so far: {failed}", flush=True)

# ── Save results ──────────────────────────────────────────────────────────────
df["predicted_label"]      = predictions
df["predicted_label_name"] = df["predicted_label"].map(LABEL_MAP)
df["true_label_name"]      = df["label"].map(LABEL_MAP)
df["correct"]              = df["label"] == df["predicted_label"]

df.to_csv(OUTPUT_FILE, index=False)

valid = df[df["predicted_label"] != -1]
print(f"\n=== Few-Shot Mistral 7B Results ===")
print(f"Total samples:   {len(df)}")
print(f"Failed/unparsed: {failed}")
print(f"Accuracy:        {valid['correct'].mean():.4f}")
print(f"\nResults saved to {OUTPUT_FILE}")
