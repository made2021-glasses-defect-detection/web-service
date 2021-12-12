import os
from segmentor import Evaluator

GLASS_IMAGE_PATH = "01.png"
MODEL_PATH = "unet_resnet34_whole.pth"
IMG_DIR = "."

def test_with_glasses_image():
  e = Evaluator(MODEL_PATH, IMG_DIR)
  predicted, image_pred_path = e.predict(GLASS_IMAGE_PATH)

  assert os.path.isfile(image_pred_path)
