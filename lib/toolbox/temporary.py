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
    Temporary.size = size
    Temporary.origin = torch.zeros(size).unsqueeze(0).cpu()
    Temporary.randomization = 0
    Temporary.interpolation = 0
    Temporary.itermax = 1000
    Temporary.gain = 1
    Temporary.allow_negative_gain = False
    Temporary.data = []

    for i in range (MAX_NUM_MIX):
      tmp = torch.zeros(size).unsqueeze(0).cpu()
      Temporary.data.append(tmp)
      tmp=None

    Temporary.ID = IntList3(0)
    Temporary.name = StringList3()
    Temporary.isEmpty = BooList3(True)
    Temporary.weight = FloatList3(1)
    Temporary.strength = 0
#End of Temporary class