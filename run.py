import os
from PIL import Image
import streamlit as st

# this caches the output to store the output and not call this function again
# and again preventing time wastage. `allow_output_mutation = True` tells the
# function to not hash the output of this function and we can get away with it
# because no arguments are passed through this.
# https://docs.streamlit.io/en/stable/api.html#streamlit.cache
@st.cache(allow_output_mutation = True)
def get_cross_modal_search_models():
    from model import CLIP

    return {'CLIP' : CLIP()}

# load all the models before the app starts
MODELS = get_cross_modal_search_models()

st.write('''
# Cross Modal Search
There are different models available for performing Cross Modal Search:
- CLIP: CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image. 
''')

model_name = st.sidebar.selectbox(
  'Please select your model',
  ["CLIP", "VSE"]
)

if model_name != "CLIP":
    st.write("Use `CLIP` model!")
    model = MODELS['CLIP']

if model_name == "CLIP":

    st.write("### `CLIP` Model")
    st.write("Please upload 2 images and 1 text of your choice")
    model = MODELS['CLIP']


IMG_SIZE = (224, 224) 

img1 = st.file_uploader("Upload First Image", type = ['png', 'jpg'])
if img1 is not None:
    img1 = Image.open(img1).convert('RGB')
    resized_img1 = img1.resize(IMG_SIZE)
    st.image(resized_img1, caption = 'Uploaded First Image.')
img2 = st.file_uploader("Upload Second Image", type = ['png', 'jpg'])
if img2 is not None:
    img2 = Image.open(img2).convert('RGB')
    resized_img2 = img2.resize(IMG_SIZE)
    st.image(resized_img2, caption = 'Uploaded Second Image.')

default_ = "a person looking at a camera on a tripod"
description = st.text_input("Description", value = default_, key = "description")

if st.button("Evaluate"):
    data = model.eval(img1, img2, description)
    st.write(data)