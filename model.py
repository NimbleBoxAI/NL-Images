import os
import numpy as np
from PIL import Image
import skimage
import torch

from preprocess import prepare_data

class CLIP:
    def __init__(self):
        super().__init__()

        self.model = torch.jit.load("./models/clip_model.pt").cuda().eval()
        self.input_resolution = self.model.input_resolution.item()
        self.context_length = self.model.context_length.item()

    def similarity_score(self, image_features, text_features):
        image_features /= image_features.norm(dim = -1, keepdim = True)
        text_features /= text_features.norm(dim = -1, keepdim = True)

        scores = (100.0 * text_features @ image_features.T).softmax(dim = -1)

        probs = {f'Image {i + 1}' : round(s.item(), 2) for i, s in enumerate(scores[0])}

        return probs

    def eval(self, img1, img2, text):

        input_images, input_text = prepare_data(img1, img2, text, self.input_resolution, self.context_length)

        with torch.no_grad():
            image_features = self.model.encode_image(input_images)
            text_features = self.model.encode_text(input_text)

        result = self.similarity_score(image_features, text_features)

        return result


