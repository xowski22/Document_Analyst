import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
from pathlib import Path
import json

class ModelBenchmark:
    def __init__(self, save_dir="benchmarks"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.results = {}

    def benchmark_summarization_model(self, model_name, sample_texts, batch_sizes=None, num_runs=5, max_length=130):
        if batch_sizes is None:
            batch_sizes = [1, 2, 4]

        """Benchmark summarization model performance with different batch sizes"""

        print(f"Benchmarking summarization model on {model_name}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.to(self.device)

            initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            results = {}

            for batch_size in batch_sizes:
                if batch_size > len(sample_texts):
                    continue



        except Exception as e:
            print(f"Error benchmarking {model_name}: {str(e)}")
            return {"error": str(e)}