import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

DATA_DIR = "images"
MODEL_PATH = "efficientnet-b0.pch"
THRESHOLD = 0.5

class ImageData(Dataset):
    def __init__(self, df, transform):
        super().__init__()
        self.df = df.reset_index()
        self.data_dir = DATA_DIR
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):       
        img_path = self.df.path[index]
        label = self.df.glasses[index]       
        
        image = np.array(Image.open(img_path).convert('RGB'))
        image = self.transform(image)
        
        return image, label

class Evaluator:
	def __init__(self):
		self.model = self.load_model()
		self.transformations = self.load_transformations()

	def load_model(self):
		model = EfficientNet.from_name('efficientnet-b0')
		num_ftrs = model._fc.in_features
		model._fc = nn.Linear(num_ftrs, 1)
		model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
		model.eval()
		return model

	def load_transformations(self):
		return transforms.Compose([
		    transforms.ToPILImage(), 
		    transforms.Resize((224, 224)),
		    transforms.ToTensor()]
		)

	def load_data(self, image_path):
		df = pd.DataFrame({"path": [image_path], "glasses": 0})		
		test_data = ImageData(df=df, transform=self.transformations)
		test_loader = DataLoader(dataset=test_data, shuffle=False)
		# import pdb; pdb.set_trace()
		_i, (data, _target) = next(enumerate(test_loader))
		return data

	def predict(self, image_path):
		data = self.load_data(image_path)
		output = self.model(data)
		pred = torch.sigmoid(output)
		return float(pred[0][0]) > THRESHOLD

