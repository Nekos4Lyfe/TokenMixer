
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

from lib.toolbox.floatlist import FloatList2
from lib.toolbox.intlist import IntList2
from lib.toolbox.boolist import BooList2
from lib.toolbox.stringlist import StringList2

from lib.toolbox.constants import MAX_NUM_MIX
#-------------------------------------------------------------------------------

class Negative :
#Negative is a class which stores torch tensors as a list
#It also stores functions which modify this tensor list
#The Negative class also contains StringList lists 
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
    self.weight.clear(index)

  def __init__(self , size):
    Negative.size = size
    Negative.origin = torch.zeros(size).unsqueeze(0).cpu()
    Negative.randomization = 0
    Negative.interpolation = 0
    Negative.itermax = 1000
    Negative.gain = 1
    Negative.allow_negative_gain = False
    Negative.data = []

    for i in range (MAX_NUM_MIX):
      tmp = torch.zeros(size).unsqueeze(0).cpu()
      Negative.data.append(tmp)
      tmp=None

    Negative.ID = IntList2(0)
    Negative.name = StringList2()
    Negative.isEmpty = BooList2(True)
    Negative.weight = FloatList2(1)
    Negative.strength = 0
#End of Negative class