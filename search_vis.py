import streamlit as st

# this caches the output to store the output and not call this function again
# and again preventing time wastage. `allow_output_mutation = True` tells the
# function to not hash the output of this function and we can get away with it
# because no arguments are passed through this.
# https://docs.streamlit.io/en/stable/api.html#streamlit.cache
@st.cache(allow_output_mutation=True, show_spinner=False)
def get_cross_modal_search_models():
  from clip.clip import CLIP
  return CLIP()

# load all the models before the app starts
with st.spinner('Loading Model with Vocabulary ... (might take sometime)'):
  model = get_cross_modal_search_models()

st.write(f'''
# Image Searching App

Find images using text and yes, there's an easter egg.
''')

app_mode = st.sidebar.selectbox(
  'Please select tasks',
  ["Text Search", "Image Search"]
)

st.write('''Upload more images to cache, if you want to add more!''')
images = st.file_uploader("Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if st.button("Upload") and len(images):
  out = model.upload_images(images)
  st.write(out)
  st.write(f'''{model.n_images}''')

# slider to select the number of images to display
n_images = st.slider('Number of images to see', min_value=1, max_value = model.n_images)

if app_mode == "Image Search":
  st.write(f"Upload any image for similarity search. Searching {n_images} images!")
  image = st.file_uploader("Images", accept_multiple_files=False, type=['png', 'jpg', 'jpeg'])
  if st.button("Process") and image:
    out = model.visual_search(image, n_images)
    for x in out:
      st.image(x)

elif app_mode == "Text Search":
  text = st.text_input(f"Add the text to search. Searching {n_images} images!")
  if st.button("Process") and text:
    out = model.text_search(text, n_images)
    for x in out:
      st.image(x)
