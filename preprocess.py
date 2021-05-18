import numpy as np
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from tokenizer import SimpleTokenizer

tokenizer = SimpleTokenizer()

image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

def preprocess_image(image, input_resolution):
    image_preprocess = Compose([
        Resize(input_resolution, interpolation = Image.BICUBIC),
        CenterCrop(input_resolution),
        ToTensor()
    ])

    preprocessed_image = image_preprocess(image)

    return preprocessed_image

def normalize_image(images):
    normalized_images = torch.tensor(np.stack(images)).cuda()
    normalized_images -= image_mean[ : , None, None]
    normalized_images /= image_std[ : , None, None]

    return normalized_images

def preprocess_text(text, context_length):
    text_tokens = [tokenizer.encode("This is " + text)]

    text_input = torch.zeros(len(text_tokens), context_length, dtype = torch.long)
    start_token = tokenizer.encoder['<|startoftext|>']
    end_token = tokenizer.encoder['<|endoftext|>']

    for i, tokens in enumerate(text_tokens):
        tokens = [start_token] + tokens + [end_token]
        text_input[i, : len(tokens)] = torch.tensor(tokens)

    preprocessed_text = text_input.cuda()
    
    return preprocessed_text

def prepare_data(img1, img2, text, input_resolution, context_length):

    img1 = preprocess_image(img1, input_resolution)
    img2 = preprocess_image(img2, input_resolution)

    images = [img1, img2]

    input_images = normalize_image(images)

    input_text = preprocess_text(text, context_length)

    return input_images, input_text