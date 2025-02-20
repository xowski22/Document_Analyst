from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

from src.models.base_model import BaseTransformerModel


class DocumentSummarizer(BaseTransformerModel):
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda:" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def summarize(self, text: str, max_length: int = 130, min_length:int = 30):
        inputs = self.tokenizer.encode("summarize: " + text,
                                       return_tensors="pt",
                                       truncation=True,
                                       max_length=1024)
        inputs = inputs.to(self.device)

        summary_ids = self.model.generate(inputs,
                                          max_length=max_length,
                                          min_length=min_length,
                                          num_beams=4,
                                          length_penalty=2.0,
                                          early_stopping=True)

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary