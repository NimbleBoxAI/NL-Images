import io
import os
import numpy as np
from PIL import Image
import xxhash
import torch
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, transforms, functional as TF
import seaborn as sns
import matplotlib.pyplot as plt

def get_image_grid(images):

  image_size = (224, 224)

  image_preprocess = Compose([
    Resize(image_size, interpolation=Image.BICUBIC),
    ToTensor()
  ])

  images = [image_preprocess(img) for img in images]

  image_stack = torch.tensor(np.stack(images))

  image_grid = torchvision.utils.make_grid(image_stack, nrow=5)

  transform = transforms.ToPILImage()
  image_grid = transform(image_grid)

  return image_grid


def get_output(scores, images, text, transpose_flag):

  count_images = len(images)
  count_text = len(text)

  scores = np.round(scores.cpu().numpy(), 2)
  scores = scores.T if transpose_flag else scores

  fig = plt.figure()

  for i, image in enumerate(images):
    plt.imshow(np.asarray(image), extent=(
      i, i + 1.0, -1.0, -0.2), origin="lower")

  sns.heatmap(scores, annot=scores, cbar_kws={
    'label': 'Probaility'}, cmap='viridis')

  plt.xticks([])
  plt.yticks(np.arange(count_text) + 0.5, text, rotation=0, fontsize=10)
  plt.xlabel('Images')
  plt.ylabel('Text')
  plt.xlim([0.0, count_images + 0.5])
  plt.ylim([count_text + 0.5, -1.0])
  plt.title('Predictions', fontweight='bold')

  return fig


def preprocess_images(images, input_resolution, device):

  image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
  image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)

  image_preprocess = Compose([
    Resize(input_resolution, interpolation=Image.BICUBIC),
    CenterCrop(input_resolution),
    ToTensor()
  ])

  preprocessed_images = [image_preprocess(img) for img in images]

  normalized_images = torch.tensor(np.stack(preprocessed_images)).to(device)
  normalized_images -= image_mean[:, None, None]
  normalized_images /= image_std[:, None, None]

  return normalized_images


def preprocess_text(text, tokenizer, context_length, device):

  text_tokens = [tokenizer.encode("This is " + t) for t in text]

  text_input = torch.zeros(
    len(text_tokens), context_length, dtype=torch.long)
  start_token = tokenizer.encoder['<|startoftext|>']
  end_token = tokenizer.encoder['<|endoftext|>']

  for i, tokens in enumerate(text_tokens):
    tokens = [start_token] + tokens + [end_token]
    text_input[i, : len(tokens)] = torch.tensor(tokens)

  preprocessed_text = text_input.to(device)

  return preprocessed_text


def similarity_score(image_features, text_features, transpose_flag):

  image_features /= image_features.norm(dim=-1, keepdim=True)
  text_features /= text_features.norm(dim=-1, keepdim=True)

  if transpose_flag:
    similarity_matrix = image_features @ text_features.T
  else:
    similarity_matrix = text_features @ image_features.T

  probs = (100.0 * similarity_matrix).softmax(dim=-1)

  return probs


def get_images(images):

  final_images = {}

  for image in images:

    img = Image.open(image, mode = 'r')
    buffer = io.BytesIO()
    img.save(buffer, format = img.format)
    img = buffer.getvalue()

    key = str(xxhash.xxh64(img).hexdigest()) + '.png'

    dir_path = os.path.dirname(os.path.realpath(__file__))
    cache_folder_path = os.path.join(dir_path, '.cache_images')
    cache_file_path = os.path.join(cache_folder_path, key)

    if not os.path.exists(cache_folder_path):
      os.mkdir(cache_folder_path)

    if key in os.listdir(cache_folder_path):

      image = Image.open(cache_file_path).convert('RGB')
      final_images[key] = image

    else:

      image = Image.open(image).convert('RGB')
      image.save(cache_file_path)
      final_images[key] = image

  images = list(final_images.values())
  image_grid = get_image_grid(images)

  return images, image_grid


def prepare_images(images, out_res):
  all_image = []
  for img in images:
    # PNGs are RGBA and JPGs are RGB, fix at RGB
    img = img.convert('RGB')
    res = min(img.size)
    out = TF.center_crop(img, (res, res))
    out = TF.resize(img, (out_res, out_res))
    out = TF.to_tensor(out).unsqueeze(0)
    all_image.append(out)
  return torch.cat(all_image, dim = 0)
