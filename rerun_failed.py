import os
import time
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL         = "gpt-4o"
RESULTS_FILE  = "results/few_shot_gpt4_predictions.csv"
FEW_SHOT_FILE = "data/few_shot_pool.csv"
SLEEP_SEC     = 3.0

LABEL_MAP = {0: "Non-Violence", 1: "Passive Violence", 2: "Direct Violence"}

few_shot_df = pd.read_csv(FEW_SHOT_FILE)
df = pd.read_csv(RESULTS_FILE)

# Find only failed rows
failed_idx = df[df["predicted_label"] == -1].index.tolist()
print(f"Found {len(failed_idx)} failed rows to rerun")

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
  - If NO → Label = 2 (Direct Violence)

Step 4: Does it express social rights, protests, or non-violent discussion?
  - If YES or NO → Label = 0 (Non-Violence)

0 - Non-Violence: Text that does not incite violence.
1 - Passive Violence: Derogatory language, insults targeting a person or community.
2 - Direct Violence: Explicit calls for killing, attacking, rape, vandalism, deportation.

Rules:
- Respond with ONLY a single digit: 0, 1, or 2.
- Do not explain your answer."""

def build_few_shot_messages(text):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for _, row in few_shot_df.iterrows():
        messages.append({"role": "user", "content": f"Classify this Bangla text:\n\n{row['text']}"})
        messages.append({"role": "assistant", "content": str(int(row["label"]))})
    messages.append({"role": "user", "content": f"Classify this Bangla text:\n\n{text}"})
    return messages

def classify_with_retry(text, max_retries=5):
    for attempt in range(max_retries):
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
            if "429" in str(e):
                wait = 2 ** attempt
                print(f"  Rate limit, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Error: {e}")
                return -1
    return -1

# Rerun only failed rows
for i, idx in enumerate(failed_idx):
    text = df.loc[idx, "text"]
    pred = classify_with_retry(text)
    df.loc[idx, "predicted_label"] = pred
    df.loc[idx, "predicted_label_name"] = LABEL_MAP.get(pred, "unknown")
    df.loc[idx, "correct"] = df.loc[idx, "label"] == pred
    print(f"  Row {idx}: predicted {pred}")
    time.sleep(SLEEP_SEC)

# Save back
df.to_csv(RESULTS_FILE, index=False)
still_failed = (df["predicted_label"] == -1).sum()
print(f"\nDone! Still failed: {still_failed}")
print(f"Results saved to {RESULTS_FILE}")