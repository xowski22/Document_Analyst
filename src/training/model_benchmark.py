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

