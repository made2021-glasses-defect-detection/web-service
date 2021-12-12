import json

def torch_to_json(tensor):
    return json.dumps(tensor.numpy().tolist())

def json_to_torch(data):
    return torch.tensor(np.array(json.loads(data)))
