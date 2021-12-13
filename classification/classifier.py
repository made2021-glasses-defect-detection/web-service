import torch
import numpy as np

class Evaluator:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        return torch.load(model_path, map_location=torch.device('cpu'))

    def predict(self, data):
        return self.model(data.unsqueeze(0)).item()
