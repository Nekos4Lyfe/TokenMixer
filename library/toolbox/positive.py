import gradio as gr
import torch, os
import collections, math, random , numpy
import re #used to parse string to int
import copy
from library.toolbox.constants import MAX_NUM_MIX
from library.toolbox.floatlist import FloatList3
from library.toolbox.intlist import IntList3
from library.toolbox.boolist import BooList3
from library.toolbox.stringlist import StringList3
#-------------------------------------------------------------------------------

from library.toolbox.constants import TENSOR_DEVICE_TYPE , TENSOR_DATA_TYPE
choosen_device = TENSOR_DEVICE_TYPE
datatype = TENSOR_DATA_TYPE

# Check that MPS is available (for MAC users)
#if torch.backends.mps.is_available(): 
#  choosen_device = torch.device("mps")
#else : choosen_device = torch.device("cpu")
######

class Positive :
#self is a class which stores torch tensors as a list
#It also stores functions which modify this tensor list
#The self class also contains StringList lists 
#and Boolist lists 

  def get(self,index) :
    output = self.data[index]
    assert not output == None , "Faulty get!"
    return output

  def validate(self, tensor) :
    assert tensor != None , "Null tensor!"
    assert tensor.shape[0] == 1 , "Too many tensors!"
    assert tensor.shape[1] == self.size , \
    "Wrong tensor dim! Size should be " + str(self.size) + " but input was "
    "size " + str(tensor.shape[1])

  def place(self, tensor , index) :
    if (tensor != None) :
      self.validate(tensor)
      assert not (index > MAX_NUM_MIX or index < 0) , "Index out of bounds!"
      self.data[index] = tensor\
      .to(device = choosen_device , dtype = datatype)
      self.isEmpty.place(False , index)
      assert not self.isEmpty.data[index] , "Faulty place!"

  def clear (self , index) :
    assert not index == None , "Index is NoneType!"
    assert not (index > MAX_NUM_MIX or index < 0) ,  "Index out of bounds!"
    self.data[index] = torch.zeros(self.size).unsqueeze(0)\
      .to(device = choosen_device , dtype = datatype)
    assert not self.data[index] == None , "Bad operation"
    self.isEmpty.clear(index)
    self.ID.clear(index)
    self.name.clear(index)
    #self.weight.clear(index)

  def __init__(self , size):
    self.size = size
    self.origin = torch.zeros(size).unsqueeze(0)\
      .to(device = choosen_device , dtype = datatype)
    self.costheta = 0
    self.radius = 0
    self.randomization = 0
    self.interpolation = 0
    self.itermax = 1000
    self.gain = 1
    self.allow_negative_gain = False
    self.data = []

    for i in range (MAX_NUM_MIX):
      tmp = torch.zeros(size).unsqueeze(0)\
      .to(device = choosen_device , dtype = datatype)
      self.data.append(tmp)
      tmp=None

    self.ID = IntList3(0)
    self.name = StringList3()
    self.isEmpty = BooList3(True)
    self.weight = FloatList3(1)
    self.strength = 0
#End of self class

###### SDXL STUFF #####

#Vector768
from library.toolbox.floatlist import FloatList7
from library.toolbox.intlist import IntList7
from library.toolbox.boolist import BooList7
from library.toolbox.stringlist import StringList7
####
class Positive1280 :
#self is a class which stores torch tensors as a list
#It also stores functions which modify this tensor list
#The self class also contains StringList lists 
#and Boolist lists 

  def get(self,index) :
    output = self.data[index]
    assert not output == None , "Faulty get!"
    return output

  def validate(self, tensor) :
    assert tensor != None , "Null tensor!"
    assert tensor.shape[0] == 1 , "Too many tensors!"
    assert tensor.shape[1] == self.size , \
    "Wrong tensor dim! Size should be " + str(self.size) + " but input was "
    "size " + str(tensor.shape[1])

  def place(self, tensor , index) :
    if (tensor != None) :
      self.validate(tensor)
      assert not (index > MAX_NUM_MIX or index < 0) , "Index out of bounds!"
      self.data[index] = tensor\
      .to(device = choosen_device , dtype = datatype)
      self.isEmpty.place(False , index)
      assert not self.isEmpty.data[index] , "Faulty place!"

  def clear (self , index) :
    assert not index == None , "Index is NoneType!"
    assert not (index > MAX_NUM_MIX or index < 0) ,  "Index out of bounds!"
    self.data[index] = torch.zeros(self.size).unsqueeze(0)\
      .to(device = choosen_device , dtype = datatype)
    assert not self.data[index] == None , "Bad operation"
    self.isEmpty.clear(index)
    self.ID.clear(index)
    self.name.clear(index)
    #self.weight.clear(index)

  def __init__(self , size):
    self.size = size
    self.origin = torch.zeros(size).unsqueeze(0)\
      .to(device = choosen_device , dtype = datatype)
    self.costheta = 0
    self.radius = 0
    self.randomization = 0
    self.interpolation = 0
    self.itermax = 1000
    self.gain = 1
    self.allow_negative_gain = False
    self.data = []

    for i in range (MAX_NUM_MIX):
      tmp = torch.zeros(size).unsqueeze(0)\
      .to(device = choosen_device , dtype = datatype)
      self.data.append(tmp)
      tmp=None

    self.ID = IntList7(0)
    self.name = StringList7()
    self.isEmpty = BooList7(True)
    self.weight = FloatList7(1)
    self.strength = 0
#End of Positive1280 class