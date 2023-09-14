import torch , math, random , numpy , copy
from library.toolbox.constants import MAX_NUM_MIX
from library.toolbox.constants import TENSOR_DEVICE_TYPE , TENSOR_DATA_TYPE
choosen_device = TENSOR_DEVICE_TYPE
datatype = TENSOR_DATA_TYPE


# The Operations class is used to store 
# helper functions to the Data.py class
class TextFunctions:

  @staticmethod
  def _text_to_emb_ids(text , tokenizer , is_sdxl , is_sd2 , is_sd1) :
    text = copy.copy(text.lower())
    emb_ids = None
    if is_sdxl : #SDXL detected
        emb_ids = tokenizer.encode(text)
    elif is_sd1 : # SD1.x detected
        emb_ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    elif is_sd2 : # SD2.0 detected
        emb_ids =  tokenizer.encode(text)
    else: emb_ids = None
    return emb_ids # return list of embedding IDs for text


  def __init__(self):
    pass
