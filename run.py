from PIL import Image
import streamlit as st

from clip.utils import get_image_grid

# this caches the output to store the output and not call this function again
# and again preventing time wastage. `allow_output_mutation = True` tells the
# function to not hash the output of this function and we can get away with it
# because no arguments are passed through this.
# https://docs.streamlit.io/en/stable/api.html#streamlit.cache

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_cross_modal_search_models():
  from clip.model import CLIP

  return {'CLIP': CLIP()}

# load all the models before the app starts
with st.spinner('Downloading and Loading Model with Vocabulary...'):
  MODELS = get_cross_modal_search_models()

st.write('''
# NL-Images
CLIP is used to perform Cross Modal Search:
- CLIP: CLIP (Contrastive Language-Image Pre-Training) \
is a neural network that consists of a image encoder and a \
text encoder. It predicts the similarity between the \
given images and textual descriptions.
''')

model_name = st.sidebar.selectbox(
  'Please select your model',
  ["CLIP"]
)

if model_name != "CLIP":
  st.write("Use `CLIP` model!")
  model = MODELS['CLIP']

if model_name == "CLIP":
  st.write("### `CLIP` Model")
  st.write("Please upload images and write text of your choice")
  st.write("Note: Write each description in a new line")
  model = MODELS['CLIP']

images = st.file_uploader(
  "Images", accept_multiple_files=True, type=['png', 'jpg'])

if len(images) != 0:
  images = [Image.open(img).convert('RGB') for img in images]
  image_grid = get_image_grid(images)
  st.image(image_grid)

default_ = "a person stuck in traffic\na apple on the table\na garden of sunflowers"

text = st.text_area("Text", value=default_, key="Text")
text = text.splitlines()

flag = st.radio('Priority', ['Image', 'Text'])

if len(images) == 1:
	flag = True

elif len(text) == 1:
	flag = False

else:
	flag = True if flag == 'Image' else False


if st.button("Predict"):
  with st.spinner('Predicting...'):
    output = model.eval(images, text, flag)
  st.write(output)