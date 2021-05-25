# from OpenAI CLIP sourcecode: https://github.com/openai/CLIP/blob/main/clip/clip.py
# released under: MIT License
# Modified by NimbleBox.ai

# files with a bunch of helper functions
try:
  from daily import *
except ImportError as e:
  import requests
  x = requests.get(
    "https://gist.githubusercontent.com/yashbonde/62df9d16858a43775c22a6af00a8d707/raw/0764da94f5e243b2bca983a94d5d6a4e4a7eb28a/daily.py"
  )
  with open("daily.py", "wb") as f:
    f.write(x.content)
  from daily import *

import os
import subprocess

from PIL import Image
import torch

import numpy as np
import pickle

from clip.model import build_model
from clip.tokenizer import SimpleTokenizer
from clip.utils import preprocess_images, preprocess_text, similarity_score, get_output, prepare_images

# fixed
_MODELS = {
  "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
  "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
  "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
  "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}
_VOCAB_PATH = 'https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz'

# functions
class CLIP:
  def __init__(
    self,
    image_model = "RN50",
    model_cache_folder = ".cache_model",
    image_cache_folder = ".cache_images",
    jit = False,
  ):
    """CLIP model wrapper.

    Args:
      image_model (str, optional): Model name, one of `"RN50", "RN101", "RN50x4", "ViT-B/32"`.
        Defaults to "RN50".
      model_cache_folder (str, optional): folder with weights and vocab file.
        Defaults to ".cache_model".
      image_cache_folder (str, optional): folder with `latents.npy` and `image_keys.p`.
        Defaults to ".cache_images".
      jit (bool, optional): [BUG] to load the model as jit script.
        Defaults to False.
    """

    # note that this is a simple wget call to get the files, this not take into
    # account the corruption of file that can happen during the transfer by checksum
    # if you find any issue during download we recommend looking at the source code:
    # https://github.com/openai/CLIP/blob/main/clip/clip.py
    mcache = os.path.join(folder(__file__), model_cache_folder)
    os.makedirs(mcache, exist_ok=True)
    model_path = os.path.join(mcache, f"{image_model}.pt")
    vocab_path = os.path.join(mcache, f"bpe_simple_vocab_16e6.txt.gz")

    if not os.path.exists(model_path):
      subprocess.call(['wget', _MODELS[image_model], '-O', model_path])
    if not os.path.exists(vocab_path):
      subprocess.call(['wget', _VOCAB_PATH, '-O', vocab_path, ])

    # the model was saved with CUDA and so it cannot be loaded directly on CPU
    # note: this is why we are using self.device tag for this. Moreover when loading
    # the model, if this has JIT, then there is a huge problem where the subroutines
    # still contains the torchscript code:
    # ```
    # _0 = self.visual input = torch.to(image, torch.device("cuda:0"), 5, False, False, None)
    # ```
    # note the torch.device("cuda:0"), under such a situation we will need to manually
    # build the model using build_model() method.
    self.device = torch.device("cuda:0") if (torch.cuda.is_available() and jit) else "cpu"
    model = torch.jit.load(model_path, map_location = self.device).eval()
    if not jit:
      self.model = build_model(model.state_dict()).to(self.device)
    else:
      self.model = model
    self.input_resolution = self.model.image_resolution
    self.context_length = self.model.context_length
    self.tokenizer = SimpleTokenizer(vocab_path)

    # now we check if there already exists a cache folder and if there is we will load
    # the embeddings for images as well
    icache = os.path.join(folder(__file__), image_cache_folder)
    os.makedirs(icache, exist_ok=True)
    emb_path = os.path.join(icache, "latents.npy")
    f_keys = os.path.join(icache, "image_keys.p")
    if not os.path.exists(emb_path):
      print("Embeddings path not found, upload images to create embeddings")
      emb = None
      keys = {}
    else:
      emb = np.load(emb_path)
      with open(f_keys, "rb") as f: 
        keys = pickle.load(f)
      print("Loaded", emb.shape, "embeddings")
      print("Loaded", len(keys), "keys")

    self.icache = icache
    self.emb_path = emb_path
    self.f_keys = f_keys
    self.emb = emb # emb mat
    self.keys = keys # hash: idx
    self.idx_keys = {v:k for k, v in keys.items()}


  def upate_emb(self, all_i: list, all_h: list, all_emb: list):
    """Update the embeddings, keys and cache images

    Args:
        all_i (list): list 
        all_h (list): [description]
        all_emb (list): [description]
    """
    # update the keys
    self.keys.update({k:i+len(self.keys) for i,k in enumerate(all_h)})
    self.idx_keys = {v:k for k, v in self.keys.items()}

    # cache the images -> copy from source (i) to target (t)
    for _hash, img in zip(all_h, all_i):
      # i is `UploadedFile` object thus i.name
      t = os.path.join(self.icache, _hash + ".png")
      img.save(t)

    # update the embeddings
    if self.emb is not None:
      self.emb = np.vstack([self.emb, *all_emb])
    else:
      self.emb = np.vstack(all_emb)

    # update the cached files
    with open(self.f_keys, "wb") as f:
      pickle.dump(self.keys, f)
    np.save(self.emb_path, self.emb)


  @property
  def n_images(self):
    return self.emb.shape[0] if self.emb is not None else 0

  
  @torch.no_grad()
  def text_to_image_similarity(self, images: list, text: list, transpose_flag: bool):
    """This is the implementation of n-text to m-images similarity checking
    just like how CLIP was intended to be used.

    Args:
      images (list): list of image files
      text (list): list of text strings
      transpose_flag (bool): image first or text first priority boolean

    Returns:
      plt.figure: heatmap for the similarity scores
    """
    input_images = preprocess_images(images, self.input_resolution, self.device)
    input_text = preprocess_text(text, self.tokenizer, self.context_length, self.device)
    image_features = self.model.encode_image(input_images)
    text_features = self.model.encode_text(input_text)
    result = similarity_score(image_features, text_features, transpose_flag)
    output = get_output(result, images, text, transpose_flag)

    return output

  @torch.no_grad()
  def text_search(self, text: str, n: int) -> list:
    """search through images based on the input text

    Args:
      text (str): text string for searching
      n (int): number of results to return

    Returns:
      images (list): return list of iamge
    """
    # get the text features
    input_tokens = [
      self.tokenizer.encoder['<|startoftext|>'],
      *self.tokenizer.encode(text),
      self.tokenizer.encoder['<|endoftext|>']
    ]
    input_tokens = input_tokens + [0 for _ in range(self.context_length - len(input_tokens))]
    input_tokens = torch.Tensor(input_tokens).long().unsqueeze(0)
    text_features = self.model.encode_text(input_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.numpy()

    # score using dot-product and load the images requires
    # note that we shift the selection by 1 because 0 is the image itself
    img_idx = np.argsort(text_features @ self.emb.T)[0][::-1][1:n+1]
    hash_idx = [self.idx_keys[x] for x in img_idx]
    images = []
    for x in hash_idx:
      fp = os.path.join(self.icache, f"{x}.png")
      images.append(Image.open(fp))
    return images


  @torch.no_grad()
  def visual_search(self, image: str, n: int) -> list:
    """CLIP.visual encoder model gives out embeddings that can be used for visual
    similarity. Note: this does not return the perfect similarity but visual similarity.

    Args:
      image (str): path to input image
      n (int): number of results to return

    Returns:
      images (list): returns list of images
    """
    # load image and get the embeddings
    image = Image.open(image)
    _hash = Hashlib.sha256(image.tobytes())
    if _hash not in self.keys:
      out = prepare_images([image], self.input_resolution).to(self.device)
      out = self.model.encode_image(out).cpu()
      out /= out.norm(dim=-1, keepdim=True)
      out = out.numpy()
      # this looks like a new image, store it
      self.upate_emb([image], [_hash], [out])
    else:
      out = self.emb[self.keys[_hash]].reshape(1, -1)

    # score using dot-product and load the images requires
    # note that we shift the selection by 1 because 0 is the image itself
    img_idx = np.argsort(out @ self.emb.T)[0][::-1][1:n+1]
    hash_idx = [self.idx_keys[x] for x in img_idx]
    images = []
    for x in hash_idx:
      fp = os.path.join(self.icache, f"{x}.png")
      images.append(Image.open(fp))
    return images


  @torch.no_grad()
  def upload_images(self, images: list) -> list:
    """uploading simply means processing all these images and creating embeddings
    from this that can then be saved in the 

    Args:
      images (list): image to cache

    Returns:
      list: hash objects of all the files
    """
    # get the hashes for new files only
    hashes = [Hashlib.sha256(Image.open(x).tobytes()) for x in images]
    all_i = []; all_h = []
    for i,h in zip(images,hashes):
      if h in self.keys:
        continue
      all_i.append(i); all_h.append(h)
    
    # get the tensors after processing
    opened_images = [Image.open(i) for i in all_i]
    out = prepare_images(opened_images, self.input_resolution)
    out = self.model.encode_image(out)
    out /= out.norm(dim=-1, keepdim=True)
    out = out.numpy()

    # update and store the new images and hashes
    self.upate_emb(opened_images, all_h, out)

    return all_h
