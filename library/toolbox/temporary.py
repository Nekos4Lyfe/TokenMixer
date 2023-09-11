import gradio as gr
import torch, os
import collections, math, random , numpy
import re #used to parse string to int
import copy
from library.toolbox.constants import MAX_NUM_MIX

from library.toolbox.floatlist import FloatList4 #done
from library.toolbox.intlist import IntList4 # done
from library.toolbox.boolist import BooList4 # done
from library.toolbox.stringlist import StringList4 # done
from library.toolbox.constants import MAX_NUM_MIX
#-------------------------------------------------------------------------------

# Check that MPS is available (for MAC users)
choosen_device = torch.device("cpu")
#if torch.backends.mps.is_available(): 
#  choosen_device = torch.device("mps")
#else : choosen_device = torch.device("cpu")
######


class Temporary :
#self is a class which stores torch tensors as a list

#This class should be used akin to volatile memory in a cpu , i.e
#as a place to store temporary tensors before saving them 
#in the 'Vector' class.

#It also stores functions which modify this tensor list
#The self class also contains StringList lists 
#and Boolist lists 

  def get(self,index) :
    output = self.data[index]\
        .to(device = choosen_device , dtype = torch.float32)
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
        .to(device = choosen_device , dtype = torch.float32)
      self.isEmpty.place(False , index)
      assert not self.isEmpty.data[index] , "Faulty place!"

  def clear (self , index) :
    assert not index == None , "Index is NoneType!"
    assert not (index > MAX_NUM_MIX or index < 0) ,  "Index out of bounds!"
    self.data[index] = torch.zeros(self.size).unsqueeze(0)\
    \
        .to(device = choosen_device , dtype = torch.float32)
    assert not self.data[index] == None , "Bad operation"
    self.isEmpty.clear(index)
    self.ID.clear(index)
    self.name.clear(index)
    self.weight.clear(index)

  def __init__(self , size):
    self.size = size
    self.origin = (torch.zeros(size)\
        .to(device = choosen_device , dtype = torch.float32)).unsqueeze(0)
    self.randomization = 0
    self.interpolation = 0
    self.itermax = 1000
    self.gain = 1
    self.allow_negative_gain = False
    self.data = []

    for i in range (MAX_NUM_MIX):
      self.data.append((torch.zeros(size)\
      \
        .to(device = choosen_device , dtype = torch.float32)).unsqueeze(0))

    self.ID = IntList4(0)
    self.name = StringList4()
    self.isEmpty = BooList4(True)
    self.weight = FloatList4(1)
    self.strength = 0
#End of self class

###### SDXL STUFF #####

#Temporary1280
from library.toolbox.floatlist import FloatList6
from library.toolbox.intlist import IntList6 
from library.toolbox.boolist import BooList6
from library.toolbox.stringlist import StringList6
####
class Temporary1280 :
#self is a class which stores torch tensors as a list

#This class should be used akin to volatile memory in a cpu , i.e
#as a place to store temporary tensors before saving them 
#in the 'Vector' class.

#It also stores functions which modify this tensor list
#The self class also contains StringList lists 
#and Boolist lists 

  def get(self,index) :
    output = self.data[index]\
        .to(device = choosen_device , dtype = torch.float32)
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
        .to(device = choosen_device , dtype = torch.float32)
      self.isEmpty.place(False , index)
      assert not self.isEmpty.data[index] , "Faulty place!"

  def clear (self , index) :
    assert not index == None , "Index is NoneType!"
    assert not (index > MAX_NUM_MIX or index < 0) ,  "Index out of bounds!"
    self.data[index] = torch.zeros(self.size).unsqueeze(0)\
    .to(device = choosen_device , dtype = torch.float32)
    assert not self.data[index] == None , "Bad operation"
    self.isEmpty.clear(index)
    self.ID.clear(index)
    self.name.clear(index)
    self.weight.clear(index)

  def __init__(self , size):
    self.size = size
    self.origin = (torch.zeros(size)\
    .to(device = choosen_device , dtype = torch.float32)).unsqueeze(0)
    self.randomization = 0
    self.interpolation = 0
    self.itermax = 1000
    self.gain = 1
    self.allow_negative_gain = False
    self.data = []

    for i in range (MAX_NUM_MIX):
      self.data.append((torch.zeros(size)\
      .to(device = choosen_device , dtype = torch.float32)).unsqueeze(0))

    self.ID = IntList6(0)
    self.name = StringList6()
    self.isEmpty = BooList6(True)
    self.weight = FloatList6(1)
    self.strength = 0
#End of self class



