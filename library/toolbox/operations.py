import torch , math, random , numpy , copy
from library.toolbox.constants import MAX_NUM_MIX
from library.toolbox.constants import TENSOR_DEVICE_TYPE , TENSOR_DATA_TYPE
from library.toolbox.functions.using_tensors import TensorFunctions
from library.toolbox.functions.using_text import TextFunctions
choosen_device = TENSOR_DEVICE_TYPE
datatype = TENSOR_DATA_TYPE


# The Operations class is used to store 
# helper functions to the Data.py class
class Operations:
  def __init__(self , \
    internal_embs , internal_embs1280 , \
    no_of_internal_embs , no_of_internal_embs1280 , \
    tokenizer , loaded_embs , \
    is_sdxl ,is_sd2 ,  is_sd1):

    self.using_tensors = TensorFunctions()
    
    self.using_text = \
    TextFunctions(\
    internal_embs , internal_embs1280 , \
    no_of_internal_embs , no_of_internal_embs1280 , \
    tokenizer , loaded_embs , \
    is_sdxl ,is_sd2 ,  is_sd1)
