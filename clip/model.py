import os
import subprocess
import torch

from clip.tokenizer import SimpleTokenizer
from clip.utils import preprocess_images, preprocess_text, similarity_score, get_output


class CLIP:
  def __init__(self):
    super().__init__()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, 'models')
    vocab_path = os.path.join(dir_path, 'vocab')

    model_args = ['wget', '-P', model_path,
      'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt']
    vocab_args = ['wget', '-P', vocab_path,
      'https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz']

    model_path = os.path.join(model_path, 'ViT-B-32.pt')
    vocab_path = os.path.join(vocab_path, 'bpe_simple_vocab_16e6.txt.gz')

    if not os.path.exists(model_path):
      subprocess.call(model_args)

    if not os.path.exists(vocab_path):
      subprocess.call(vocab_args)

    self.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    self.model = torch.jit.load(model_path).to(self.device).eval()
    self.input_resolution = self.model.input_resolution.item()
    self.context_length = self.model.context_length.item()
    self.tokenizer = SimpleTokenizer(vocab_path)

  def eval(self, images, text, transpose_flag):

    input_images = preprocess_images(images, self.input_resolution, self.device)
    input_text = preprocess_text(text, self.tokenizer, self.context_length, self.device)

    with torch.no_grad():
      image_features = self.model.encode_image(input_images)
      text_features = self.model.encode_text(input_text)


    result = similarity_score(image_features, text_features, transpose_flag)
    output = get_output(result, images, text, transpose_flag)

    return output