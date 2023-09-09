import gradio as gr
import torch, os
import collections, math, random , numpy
import re #used to parse string to int
import copy
from lib.toolbox.constants import MAX_NUM_MIX

from lib.toolbox.floatlist import FloatList4 #done
from lib.toolbox.intlist import IntList4 # done
from lib.toolbox.boolist import BooList4 # done
from lib.toolbox.stringlist import StringList4 # done
from lib.toolbox.constants import MAX_NUM_MIX
#-------------------------------------------------------------------------------
class Temporary :
#Temporary is a class which stores torch tensors as a list

#This class should be used akin to volatile memory in a cpu , i.e
#as a place to store temporary tensors before saving them 
#in the 'Vector' class.

#It also stores functions which modify this tensor list
#The Temporary class also contains StringList lists 
#and Boolist lists 

  def get(self,index) :
    output = self.data[index].to(device = "cpu" , dtype = torch.float32)
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
      self.data[index] = tensor.to(device = "cpu" , dtype = torch.float32)
      self.isEmpty.place(False , index)
      assert not self.isEmpty.data[index] , "Faulty place!"

  def clear (self , index) :
    assert not index == None , "Index is NoneType!"
    assert not (index > MAX_NUM_MIX or index < 0) ,  "Index out of bounds!"
    self.data[index] = torch.zeros(self.size).unsqueeze(0)\
    .to(device = "cpu" , dtype = torch.float32)
    assert not self.data[index] == None , "Bad operation"
    self.isEmpty.clear(index)
    self.ID.clear(index)
    self.name.clear(index)
    self.weight.clear(index)

  def __init__(self , size):
    Temporary.size = size
    Temporary.origin = (torch.zeros(size).to(device = "cpu" , dtype = torch.float32)).unsqueeze(0)
    Temporary.randomization = 0
    Temporary.interpolation = 0
    Temporary.itermax = 1000
    Temporary.gain = 1
    Temporary.allow_negative_gain = False
    Temporary.data = []

    for i in range (MAX_NUM_MIX):
      Temporary.data.append((torch.zeros(size)\
      .to(device = "cpu" , dtype = torch.float32)).unsqueeze(0))

    Temporary.ID = IntList4(0)
    Temporary.name = StringList4()
    Temporary.isEmpty = BooList4(True)
    Temporary.weight = FloatList4(1)
    Temporary.strength = 0
#End of Temporary class

###### SDXL STUFF #####

#Temporary1280
from lib.toolbox.floatlist import FloatList6
from lib.toolbox.intlist import IntList6 
from lib.toolbox.boolist import BooList6
from lib.toolbox.stringlist import StringList6
####
class Temporary1280 (Temporary) :
  def __init__(self , size):
    super().__init__(size)
    #Temporary768 is used by SDXL to store 1x768 Vectors , while
    #the Vector class is used by SDXL to store 1x1280 Vectors
    self.ID = IntList6(0)
    self.name = StringList6()
    self.isEmpty = BooList6(True)
    self.weight = FloatList6(1)
#End of Temporary768



