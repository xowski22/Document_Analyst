import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class QuestionAnswerer:
    def __init__(self, model_name="deepset/roberta-base-squad2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def answer_question(self, question: str, context: str ) -> str:
        """
        Answer a question based on the given context.

        Args:
            question (str): The question to answer.
            context (str): The context to find the answer in
        Returns:

        """
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
            stride=128,
            return_overflowing_tokens=True
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        offset_mapping = inputs["offset_mapping"]
        sequence_ids = inputs.sequence_ids()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start_idx = torch.argmax(start_logits).item()
        end_idx = torch.argmax(end_logits).item()

        if (
            sequence_ids[start_idx] != 1 or
            sequence_ids[end_idx] != 1 or
            end_idx < start_idx
        ):
            return "Unable to find answer."

        start_char = offset_mapping[start_idx][0]
        end_char = offset_mapping[end_idx][1]

        answer = context[start_char:end_char]

        if not answer or len(answer) > 100:
            return "Unable to find answer."

        return answer.strip()

