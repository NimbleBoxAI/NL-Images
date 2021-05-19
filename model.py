import os
import subprocess
import torch

from utils import preprocess_images, preprocess_text, similarity_score, display_output

class CLIP:
  def __init__(self):
    super().__init__()

    model_args = ['wget', '-P', './models', 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt']
    vocab_args = ['wget', '-P', './vocab', 'https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz']

    if not os.path.exists('./models/ViT-B-32.pt'):
      subprocess.call(model_args)

    if not os.path.exists('./vocab/bpe_simple_vocab_16e6.txt.gz'):
      subprocess.call(vocab_args)

    self.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    self.model = torch.jit.load("./models/ViT-B-32.pt").to(self.device).eval()
    self.input_resolution = self.model.input_resolution.item()
    self.context_length = self.model.context_length.item()

  def eval(self, images, text):

    input_images = preprocess_images(images, self.input_resolution, self.device)
    input_text = preprocess_text(text, self.context_length, self.device)

    with torch.no_grad():
      image_features = self.model.encode_image(input_images)
      text_features = self.model.encode_text(input_text)

    result = similarity_score(image_features, text_features)
    output = display_output(result, images, text)

    return output
