from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

from src.models.base_model import BaseTransformerModel


class DocumentSummarizer():
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Divides text into smaller chunks"""

        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) +1

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks
