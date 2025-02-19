from transformers import AutoModelForSeq2SeqLM

from src.models.base_model import BaseTransformerModel


class DocumentSummarizer(BaseTransformerModel):
    def __init__(self):
        super().__init__("facebook/bart-large-cnn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    def generate_summary(self, text:str) -> str:
        inputs = self.preprocess(text)
        summary_ids = self.model.generate(["input_ids"])
        return self.tokenizer.decode(summary_ids[0])