import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from nltk.book import texts
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
from pathlib import Path
import json

class ModelBenchmark:
    def __init__(self, save_dir="benchmarks"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.results = {}

    def benchmark_summarization_model(self, model_name, sample_texts, batch_sizes=[1, 2, 4], num_runs=5, max_length=130):

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
                batch_text = sample_texts[:batch_size]
                times = []

                inputs= tokenizer(
                    ["summarize " + text for text in batch_text],
                    return_tensors = "pt",
                    padding = True,
                    truncation = True,
                    max_length = 1024
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                _ = model.generate(**inputs, max_length=max_length)
                for _ in range(num_runs):
                    start_time = time.time()
                    inputs = tokenizer(
                        ["summarize: "+ text for text in batch_text],
                        return_tensors = "pt",
                        padding = True,
                        truncation = True,
                        max_length = 1024
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    _ = model.generate(**inputs, max_length=max_length)
                    end_time = time.time()
                    times.append(end_time - start_time)

                avg_time = sum(times) / len(times)
                results[f"batch_size_{batch_size}"] = {
                    "avg_time_seconds": avg_time,
                    "throughput": batch_size / avg_time
                }

            peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

            memory_usage = peak_memory - initial_memory

            model_results = {
                "model_name": model_name,
                "device": str(self.device),
                "memory_usage": memory_usage,
                "batch_results": results
            }

            self.results[f"summarization_{model_name.replace('/','_')}"] = model_results
            return model_results


        except Exception as e:
            print(f"Error benchmarking {model_name}: {str(e)}")
            return {"error": str(e)}

    def benchmark_qa_model(self, model_name, questions, contexts, batch_sizes=[1, 2, 4], num_runs=5):
        """Benchmark a question answering model with different batch sizes"""

        print(f"Benchmarking qa model on {model_name}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            model.to(self.device)

            initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            results = {}

            for batch_size in batch_sizes:
                if batch_size > len(questions) or batch_size > len(contexts):
                    continue

                batch_questions = questions[:batch_size]
                batch_contexts = contexts[:batch_size]
                times = []

                inputs = tokenizer(
                    batch_questions,
                    batch_contexts,
                    return_tensors = "pt",
                    padding = True,
                    truncation = True,
                    max_length = 512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                _ = model(**inputs)

                for _ in range(num_runs):
                    start_time = time.time()
                    inputs = tokenizer(
                        batch_questions,
                        batch_contexts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    _ = model(**inputs)
                    end_time = time.time()
                    times.append(end_time - start_time)

                avg_time = sum(times) / len(times)
                results[f"batch_size_{batch_size}"] = {
                    "avg_time_seconds": avg_time,
                    "throughput": batch_size / avg_time
                }

            peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            memory_usage = peak_memory - initial_memory

            model_results = {
                "model_name": model_name,
                "device": str(self.device),
                "memory_usage": memory_usage,
                "batch_results": results
            }

            self.results[f"qa_{model_name.replace('/','_')}"] = model_results
            return model_results

        except Exception as e:
            print(f"Error benchmarking {model_name}: {str(e)}")

    def save_results(self, filename="benchmark_results.json"):
        """Save benchmark results to a json file"""

        with open(self.save_dir/filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {self.save_dir/filename}")

    def plot_results(self,model_type="summarization", filename="benchmark_results.json"):
        """Plot benchmark results"""

        data = []

        for k, res in self.results.items():
            if k.startswith(model_type) and "error" not in res:
                model_name = res["model_name"].split("/")[-1]

                for batch_size, batch_res in res["batch_results"].items():
                    data.append({
                        "model_name": model_name,
                        "batch_size": batch_size.split("_")[-1],
                        "latency": batch_res["avg_time_seconds"],
                        "throughput": batch_res["throughput"]
                    })

        if not data:
            print("No results found")
            return

        df = pd.DataFrame(data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))


