import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import logging


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
        try:

            if not question or not context:
                return "Unable to find answer: Missing question or context."

            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
                stride=128,
                # return_overflowing_tokens=True
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            start_idx = torch.argmax(start_logits).item()
            end_idx = torch.argmax(end_logits).item()

            if end_idx < start_idx:
                end_idx = min(start_idx + 10, len(input_ids[0]) - 1)


            answer_tokens = input_ids[0][start_idx:end_idx+1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

            answer = answer.strip()

            if not answer or len(answer) < 1 or len(answer) > 100:
                return "Unable to find answer."

            return answer
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error processing question: {str(e)}"