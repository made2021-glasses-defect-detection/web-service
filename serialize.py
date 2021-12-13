import json
import torch
import numpy as np

def torch_to_json(tensor):
    return json.dumps(tensor.numpy().tolist())

def json_to_torch(data, dtype=torch.float):
    return torch.tensor(np.array(json.loads(data)), dtype=dtype)
