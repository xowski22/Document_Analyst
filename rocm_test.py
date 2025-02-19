import torch
import transformers
from transformers import AutoTokenizer, AutoModel

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.get_device_name(0)}")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

text = "Test transformers installation."
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model(**inputs)

print("Test completed successfully!")