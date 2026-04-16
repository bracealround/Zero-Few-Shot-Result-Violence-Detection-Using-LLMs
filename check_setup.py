import torch
import transformers
import openai
import sklearn
import os

print("=== Environment Sanity Check ===")
print(f"PyTorch:        {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} | {round(props.total_memory / 1e9, 1)} GB VRAM")
print(f"Transformers:   {transformers.__version__}")
print(f"Scikit-learn:   {sklearn.__version__}")
print(f"OpenAI key:     {'✓ Set' if os.getenv('OPENAI_API_KEY') else '✗ MISSING'}")
print(f"Mistral key:    {'✓ Set' if os.getenv('MISTRAL_API_KEY') else '✗ MISSING'}")
print("================================")