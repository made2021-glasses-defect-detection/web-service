import cv2
import numpy as np
import torch
import os

from PIL import Image
from uuid import uuid4

IMAGE_SIZE = (480, 320)
MASK_TRESHOLD = 0.97

class Evaluator:
  def __init__(self, model_path, img_dir):
    self.model = self.load_model(model_path)
    self.img_dir = img_dir

  def load_model(self, model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    return model

  def predict(self, image_path):
    image = self.load_scale_image(image_path)

    with torch.no_grad():
        predicted = self.model(image.unsqueeze(0))[0]  # CHW.
        image_pred_path = self.save_image_and_mask(predicted, MASK_TRESHOLD, self.img_dir)
        predicted = (predicted > MASK_TRESHOLD).float()
        return predicted, image_pred_path

  def mask_from_prediction(self, predicted, treshold):
    predicted = (predicted > treshold).float()
    return torch.cat((torch.zeros_like(predicted[0]).unsqueeze(0), predicted), dim=0)

  def save_image_and_mask(self, predicted, treshold, img_dir):
    c = predicted.shape[0]

    if c == 1:
        predicted = torch.nn.functional.logsigmoid(predicted)
        predicted_labels = predicted[0] > np.log(0.5)
    else:
        predicted = torch.sigmoid(predicted)
        predicted = self.mask_from_prediction(predicted, treshold)
        predicted_labels = (predicted * 255).permute(1, 2, 0)

    predicted_labels = predicted_labels.cpu().numpy().astype(np.uint8)

    image_pred = Image.fromarray(predicted_labels)
    image_pred_path = os.path.join(img_dir, "segmented-" + str(uuid4()) + ".png")
    image_pred.save(image_pred_path)
    return image_pred_path

  def image_to_numpy(image):
    """Convert Torch tensor to Numpy image."""
    if isinstance(image, np.ndarray):
      return image

    if image.dtype == torch.float32:
      image = image - image.min()
      image = 255 * image / max(1e-6, image.max())

    return image.to(torch.uint8).permute(1, 2, 0).cpu().numpy()

  def image_to_torch(self, image, device=None):
    """Convert Numpy image to Torch tensor."""
    if isinstance(image, torch.Tensor):
      return image

    if image.shape[-1] == 4:
      image = image[..., :3]

    result = torch.from_numpy(np.ascontiguousarray(image))

    if device is not None:
      result = result.to(device)

    result = result.permute(2, 0, 1)
    return result

  def load_scale_image(self, img_name, image_size=IMAGE_SIZE):
    image = Image.open(img_name)
    image = np.asarray(image) / 255.
    scaler = Scaler(image_size)
    image = scaler(image)
    image = self.image_to_torch(image).float()
    return image

class Scaler(object):
  def __init__(self, size, grayscale=False):
    self._size = size
    self._grayscale = grayscale
    self._padding = 0

  def set_padding(self, padding):
    self._padding = padding

  def __call__(self, image):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    grayscale = self._grayscale
    rw, rh = self._size
    p = self._padding
    h, w, c = image.shape
    scale_x = rw / w
    scale_y = rh / h
    scale = min(scale_x, scale_y)
    sw = int(scale * w)
    sh = int(scale * h)
    offset_x = p + (rw - sw) // 2
    offset_y = p + (rh - sh) // 2
    # Используем zeros на случай маленьких изображений.
    # TODO: фикс индексирования для маленьких изображений.
    result = np.zeros((rh + 2 * p, rw + 2 * p, c), dtype=image.dtype)
    cv2.resize(image, (sw, sh),
               interpolation=cv2.INTER_NEAREST if grayscale else cv2.INTER_AREA,
               dst=result[offset_y:offset_y + sh, offset_x:offset_x + sw])

    # Отразить.
    result[offset_y:offset_y + sh, :offset_x] = result[offset_y:offset_y + sh, offset_x:2 * offset_x][:, ::-1]
    offset_end = result.shape[1] - offset_x - sw
    result[offset_y:offset_y + sh, offset_x + sw:] = result[offset_y:offset_y + sh, sw + offset_x - offset_end:sw + offset_x][:, ::-1]

    result[:offset_y] = result[offset_y:2 * offset_y][::-1]
    offset_end = result.shape[0] - offset_y - sh
    result[offset_y + sh:] = result[sh + offset_y - offset_end:sh + offset_y][::-1]
    return result
