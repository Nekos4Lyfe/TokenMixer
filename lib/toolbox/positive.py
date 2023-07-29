
import gradio as gr
from modules import script_callbacks, shared, sd_hijack
from modules.shared import cmd_opts
from pandas import Index
from pandas.core.groupby.groupby import OutputFrameOrSeries
import torch, os
from modules.textual_inversion.textual_inversion import Embedding
import collections, math, random , numpy
import re #used to parse string to int
import copy
from torch.nn.modules import ConstantPad1d, container

from lib.toolbox.floatlist import FloatList3
from lib.toolbox.intlist import IntList3
from lib.toolbox.boolist import BooList3
from lib.toolbox.stringlist import StringList3

from lib.toolbox.constants import \
MAX_NUM_MIX , SHOW_NUM_MIX , MAX_SIMILAR_EMBS , \
VEC_SHOW_TRESHOLD , VEC_SHOW_PROFILE , SEP_STR , \
SHOW_SIMILARITY_SCORE , ENABLE_GRAPH , GRAPH_VECTOR_LIMIT , \
ENABLE_SHOW_CHECKSUM , REMOVE_ZEROED_VECTORS , EMB_SAVE_EXT 
#-------------------------------------------------------------------------------

class Positive :
#Positive is a class which stores torch tensors as a list
#It also stores functions which modify this tensor list
#The Positive class also contains StringList lists 
#and Boolist lists 

  def get(self,index) :
    output = self.data[index]
    assert not output == None , "Faulty get!"
    return output

  def validate(self, tensor) :
    assert tensor != None , "Null tensor!"
    assert tensor.shape[0] == 1 , "Too many tensors!"
    assert tensor.shape[1] == self.size , "Wrong tensor dim!"

  def place(self, tensor , index) :
    if (tensor != None) :
      self.validate(tensor)
      assert not (index > MAX_NUM_MIX or index < 0) , "Index out of bounds!"
      self.data[index] = tensor.cpu()
      self.isEmpty.place(False , index)
      assert not self.isEmpty.data[index] , "Faulty place!"

  def clear (self , index) :
    assert not index == None , "Index is NoneType!"
    assert not (index > MAX_NUM_MIX or index < 0) ,  "Index out of bounds!"
    self.data[index] = torch.zeros(self.size).unsqueeze(0).cpu()
    assert not self.data[index] == None , "Bad operation"
    self.isEmpty.clear(index)
    self.ID.clear(index)
    self.name.clear(index)
    #self.weight.clear(index)

  def __init__(self , size):
    Positive.size = size
    Positive.origin = torch.zeros(size).unsqueeze(0).cpu()
    Positive.costheta = 0
    Positive.radius = 0
    Positive.randomization = 0
    Positive.interpolation = 0
    Positive.itermax = 1000
    Positive.gain = 1
    Positive.allow_negative_gain = False
    Positive.data = []

    for i in range (MAX_NUM_MIX):
      tmp = torch.zeros(size).unsqueeze(0).cpu()
      Positive.data.append(tmp)
      tmp=None

    Positive.ID = IntList3(0)
    Positive.name = StringList3()
    Positive.isEmpty = BooList3(True)
    Positive.weight = FloatList3(1)
    Positive.strength = 0
#End of Positive class