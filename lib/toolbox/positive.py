import gradio as gr
import torch, os
import collections, math, random , numpy
import re #used to parse string to int
import copy
from lib.toolbox.constants import MAX_NUM_MIX
from lib.toolbox.floatlist import FloatList3
from lib.toolbox.intlist import IntList3
from lib.toolbox.boolist import BooList3
from lib.toolbox.stringlist import StringList3
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

###### SDXL STUFF #####

#Vector768
from lib.toolbox.floatlist import FloatList7
from lib.toolbox.intlist import IntList7
from lib.toolbox.boolist import BooList5
from lib.toolbox.stringlist import StringList5
####
class Positive768 (Vector) :
  def __init__(self , size):
    super().__init__()
    #Vector768 is used by SDXL to store 1x768 Vectors , while
    #the Vector class is used by SDXL to store 1x1280 Vectors
    self.ID = IntList5(0)
    self.name = StringList5()
    self.isEmpty = BooList5(True)
    self.weight = FloatList5(1)
#End of Positive768