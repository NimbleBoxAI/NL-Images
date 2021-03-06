# CLIP

Files:
- `clip.py`: model with CLIP wrapper.
- `model.py`: Model file from OpenAI [source code](https://github.com/openai/CLIP/blob/main/clip/model.py) with slight modifications
- `tokenizer.py`: Simple Tokenizer (Byte Pair Encoding) file
- `utils.py`: Different utility functions for preprocessing and displaying outputs

## Documentation

## `CLIP`

```python
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
```

### `CLIP.upate_emb`

```python
def upate_emb(self, all_i: list, all_h: list, all_emb: list):
  """Update the embeddings, keys and cache images

  Args:
    all_i (list): list of all opened images
    all_h (list): list of all hashes for corresponding all_i[j]
    all_emb (list): list of embeddings for corresponding all_i[j]
  """
```

### `CLIP.text_to_image_similarity`

```python
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
```

### `CLIP.text_search`

```python
def text_search(self, text: str, n: int) -> list:
  """search through images based on the input text

  Args:
    text (str): text string for searching
    n (int): number of results to return

  Returns:
    images (list): return list of images
  """
```

### `CLIP.visual_search`

```python
def visual_search(self, image: str, n: int) -> list:
  """CLIP.visual encoder model gives out embeddings that can be used for visual
  similarity. Note: this does not return the perfect similarity but visual similarity.

  Args:
    image (str): path to input image
    n (int): number of results to return

  Returns:
    images (list): returns list of images
  """
```

### `CLIP.upload_images`

```python
def upload_images(self, images: list) -> list:
  """uploading simply means processing all these images and creating embeddings
  from this that can then be saved in the 

  Args:
    images (list): image to cache

  Returns:
    list: hash objects of all the files
  """
```

### `CLIP.text_to_text_similarity`

```python
def text_to_text_similarity(self, memory: list, query: str, n: int = 10):
  """Text to text similarity for comparing input query to memory.

  Args:
    memory (list): list of strings for memory
    query (str): query string
    n (int, optional): number of items to return. Defaults to 10.

  Returns:
    (list): return list of matching strings from memory
  """
```

### `Tokenizer`

```python
class SimpleTokenizer(object):
  def __init__(self, bpe_path: str = "./vocab/bpe_simple_vocab_16e6.txt.gz"):
```


### `Tokenizer.__call__`

```python
def __call__(
  self, text: Union[str, list],
  context_length: Union[str, list],
  device = "cpu"
) -> torch.Tensor:
  """tokenize the text and return proper tensors

  Args:
      text (Union[str, list]): text as a string or list of strings
      context_length (Union[str, list]): sequence length for transformer
      device (torch.Device): device to load sequence on 

  Returns:
      (torch.Tensor): input ids for the model
  """
```