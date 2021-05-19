from PIL import Image
import streamlit as st

from utils import display_image_grid

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
# NL-Images
There are different models available for performing Cross Modal Search:
- CLIP: CLIP (Contrastive Language-Image Pre-Training) \
is a neural network trained on a variety of (image, text) pairs. \
It can be instructed in natural language to predict the most \
relevant text snippet, given an image.
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
	st.write("Please upload images and write text of your choice")
	st.write("Note: Write each description in a new line")
	model = MODELS['CLIP']

images = st.file_uploader("Images", accept_multiple_files = True, type = ['png', 'jpg'])

if len(images) != 0:
	images = [Image.open(img).convert('RGB') for img in images]
	image_grid = display_image_grid(images)
	st.image(image_grid)

default_ = "a person looking at a camera on a tripod \na apple on the table\na garden of sunflowers"

text = st.text_area("Text", value = default_, key = "Text")
text = text.splitlines()

if st.button("Predict"):
	output = model.eval(images, text)
	st.write(output)