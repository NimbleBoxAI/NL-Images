import torch
import torchvision
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, transforms, functional as TF

MEAN = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(-1, 1, 1)
STD = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(-1, 1, 1)


def get_image_grid(images):
  # preprocess images
  image_size = (224, 224)
  image_preprocess = Compose([
    Resize(image_size, interpolation=Image.BICUBIC),
    ToTensor()
  ])
  images = [image_preprocess(img) for img in images]

  # stack into a grid and return
  image_stack = torch.tensor(np.stack(images))
  image_grid = torchvision.utils.make_grid(image_stack, nrow=5)
  transform = transforms.ToPILImage()
  image_grid = transform(image_grid)

  return image_grid


def get_similarity_heatmap(scores, images, text, transpose_flag):
  count_images = len(images)
  count_text = len(text)
  scores = np.round(scores, 2)
  scores = scores.T if transpose_flag else scores

  # create the figure
  fig = plt.figure()
  for i, image in enumerate(images):
    plt.imshow(np.asarray(image), extent=(i, i + 1.0, -1.0, -0.2), origin="lower")
  sns.heatmap(scores, annot=scores, cbar_kws={'label': 'Probaility'}, cmap='viridis')
  plt.xticks([])
  plt.yticks(np.arange(count_text) + 0.5, text, rotation=0, fontsize=10)
  plt.xlabel('Images')
  plt.ylabel('Text')
  plt.xlim([0.0, count_images + 0.5])
  plt.ylim([count_text + 0.5, -1.0])
  plt.title('Predictions', fontweight='bold')

  return fig


def prepare_images(images, out_res, device):
  all_image = []
  for img in images:
    # PNGs are RGBA and JPGs are RGB, fix at RGB
    img = img.convert('RGB')
    res = min(img.size)
    out = TF.center_crop(img, (res, res))
    out = TF.resize(out, (out_res, out_res))
    out = TF.to_tensor(out).unsqueeze(0)
    out = (out - MEAN) / STD
    all_image.append(out)
  return torch.cat(all_image, dim = 0).to(device)
