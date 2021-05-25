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
import torch

import numpy as np
import pickle

from clip.model import build_model
from clip.tokenizer import SimpleTokenizer
from clip.utils import preprocess_images, preprocess_text, similarity_score, get_output

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
    os.path.makedirs(icache, exist_ok=True)
    emb_path = os.path.join(icache, "latents.npy")
    f_keys = os.path.join(icache, "image_keys.p")
    if not os.path.exists(emb_path):
      print("Embeddings path not found, upload images to create embeddings")
      emb = None
      keys = None
    else:
      emb = np.load(emb_path)
      with open(f_keys, "rb") as f:
        keys = pickle.load(f)
    self.emb = emb # emb mat
    self.keys = keys # hash
    self.new_embs = {} # hash: emb

  def add_new_emb(self, ):
    pass


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
    with torch.no_grad():
      image_features = self.model.encode_image(input_images)
      text_features = self.model.encode_text(input_text)
    result = similarity_score(image_features, text_features, transpose_flag)
    output = get_output(result, images, text, transpose_flag)

    return output
  
  def text_search(self, text: str, n: int) -> list:
    """[summary]

    Args:
        text (str): [description]

    Returns:
        list: [description]
    """
    pass

  def visual_search(self, image: str, n: int) -> list:
    pass

  def upload_images(self, image: str) -> list:
    """uploading simply means processing all these images and creating embeddings
    from this that can then be saved in the 

    Args:
        image (str): [description]

    Returns:
        list: [description]
    """
    pass
