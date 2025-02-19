from transformers import AutoTokenizer, AutoModel

class BaseTransformerModel:
    def __init__(self, model_name:str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def preprocess(self, text:str):
        return self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)