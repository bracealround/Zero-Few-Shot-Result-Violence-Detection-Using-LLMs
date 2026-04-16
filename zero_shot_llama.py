import os
import time
import pandas as pd
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "llama-3.1-8b-instant"     # LLaMA 3.1 8B on Groq
INPUT_FILE = "data/test_split.csv"
OUTPUT_FILE = "results/zero_shot_llama_predictions.csv"
SLEEP_SEC  = 2.1                        # Groq free tier: 30 req/min → 2s gap is safe

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

LABEL_MAP = {
    0: "Non-Violence",
    1: "Passive Violence",
    2: "Direct Violence"
}

os.makedirs("results", exist_ok=True)

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert content moderator specializing in Bangla social media text.
Your task is to classify Bangla text into exactly one of three categories:

0 - Non-Violence: Text that does not incite or contain any form of violence or abuse.
1 - Passive Violence: Text that contains abuse, insults, or degrading language targeting a person or community.
2 - Direct Violence: Text that directly calls for physical violence, killing, destruction, or attacks against a person or community.

Rules:
- Respond with ONLY a single digit: 0, 1, or 2.
- Do not explain your answer.
- Do not add any other text."""

def classify_zero_shot(text):
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Classify this Bangla text:\n\n{text}"},
            ],
            temperature=0,
            max_tokens=5,
        )
        raw = response.choices[0].message.content.strip()
        for char in raw:
            if char in ["0", "1", "2"]:
                return int(char)
        return -1
    except Exception as e:
        print(f"  API error: {e}")
        time.sleep(10)
        return -1

# ── Main loop ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} test samples")
print(f"Using model: {MODEL_ID} via Groq API\n")

predictions = []
failed = 0

for i, row in df.iterrows():
    pred = classify_zero_shot(row["text"])
    predictions.append(pred)

    if pred == -1:
        failed += 1

    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{len(df)} | Failed so far: {failed}")

    time.sleep(SLEEP_SEC)

# ── Save results ──────────────────────────────────────────────────────────────
df["predicted_label"]      = predictions
df["predicted_label_name"] = df["predicted_label"].map(LABEL_MAP)
df["true_label_name"]      = df["label"].map(LABEL_MAP)
df["correct"]              = df["label"] == df["predicted_label"]

df.to_csv(OUTPUT_FILE, index=False)

valid = df[df["predicted_label"] != -1]
print(f"\n=== Zero-Shot LLaMA 3 8B Results ===")
print(f"Total samples:   {len(df)}")
print(f"Failed/unparsed: {failed}")
print(f"Accuracy:        {valid['correct'].mean():.4f}")
print(f"\nResults saved to {OUTPUT_FILE}")
