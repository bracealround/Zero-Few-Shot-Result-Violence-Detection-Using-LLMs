import os
import time
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Config ────────────────────────────────────────────────────────────────────
MODEL         = "gpt-4o"
INPUT_FILE    = "data/test_split.csv"
FEW_SHOT_FILE = "data/few_shot_pool.csv"
OUTPUT_FILE   = "results/few_shot_gpt4_predictions.csv"
SLEEP_SEC     = 0.5

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

# ── System prompt (enhanced with paper's decision tree) ───────────────────────
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
- Do not add any other text."""

# ── Build few-shot messages ───────────────────────────────────────────────────
def build_few_shot_messages(text):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add few-shot examples as alternating user/assistant turns
    for _, row in few_shot_df.iterrows():
        label_name = LABEL_MAP[int(row["label"])]
        messages.append({
            "role": "user",
            "content": f"Classify this Bangla text:\n\n{row['text']}"
        })
        messages.append({
            "role": "assistant",
            "content": str(int(row["label"]))   # just the digit
        })

    # Add the actual query
    messages.append({
        "role": "user",
        "content": f"Classify this Bangla text:\n\n{text}"
    })

    return messages

# ── Classification function ───────────────────────────────────────────────────
def classify_few_shot(text):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=build_few_shot_messages(text),
            temperature=0,
            max_tokens=5
        )
        raw = response.choices[0].message.content.strip()
        for char in raw:
            if char in ["0", "1", "2"]:
                return int(char)
        return -1

    except Exception as e:
        print(f"  API error: {e}")
        time.sleep(5)   # wait longer on error before retrying
        return -1

# ── Main loop ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} test samples")
print(f"Using model: {MODEL} with {len(few_shot_df)} few-shot examples\n")

predictions = []
failed = 0

for i, row in df.iterrows():
    pred = classify_few_shot(row["text"])
    predictions.append(pred)

    if pred == -1:
        failed += 1
        print(f"  ⚠️  Failed on row {i}")

    if (i + 1) % 100 == 0:
        done = i + 1
        print(f"  Processed {done}/{len(df)} | Failed so far: {failed}")

    time.sleep(SLEEP_SEC)

# ── Save results ──────────────────────────────────────────────────────────────
df["predicted_label"]      = predictions
df["predicted_label_name"] = df["predicted_label"].map(LABEL_MAP)
df["true_label_name"]      = df["label"].map(LABEL_MAP)
df["correct"]              = df["label"] == df["predicted_label"]

df.to_csv(OUTPUT_FILE, index=False)

valid = df[df["predicted_label"] != -1]
print(f"\n=== Few-Shot GPT-4o Results ===")
print(f"Total samples:   {len(df)}")
print(f"Failed/unparsed: {failed}")
print(f"Accuracy:        {valid['correct'].mean():.4f}")
print(f"Results saved to {OUTPUT_FILE}")