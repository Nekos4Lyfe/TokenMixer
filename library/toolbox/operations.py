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
  def __init__(self):
    Operations.using_tensors = TensorFunctions()
    Operations.using_text = TextFunctions()
