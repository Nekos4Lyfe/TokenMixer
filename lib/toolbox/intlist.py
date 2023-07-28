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

from lib.toolbox.constants import MAX_NUM_MIX
#-------------------------------------------------------------------------------

class IntList :
  #The class IntList stores ID values as a list
  #It also stores functions which modify this list  
      def validate(self, integer) :
        assert integer != None , "integer is NoneType!"
        assert isinstance(integer , int) , "Not a int!"

      def check(self, index) :
        assert not index == None , "Index is NoneType!"
        assert not (index > MAX_NUM_MIX or index < 0) , "Index out of bounds!"

      def __init__(self , default) :
        self.validate(default)
        IntList.default = default
        IntList.data = []
        for i in range (MAX_NUM_MIX):
          IntList.data.append(default)

      def get(self,index) :
        self.check(index)
        return self.data[index]

      def place(self , integer , index) :
          tmp = math.floor(integer)
          self.validate(tmp)
          self.check(index)
          self.data[index] = tmp

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of intlist class
#-------------------------------------------------------------------------------
class IntList2 :
  #The class IntList2 stores ID values as a list
  #It also stores functions which modify this list  

  #This is a copy of the class IntList, because I simply cannot 
  #be bothered to learn out how to make multiple class instances
  #in python without the list addresses overlapping.
  #Let me know if you know a good solution :)! /Nekos4Lyfe

      def validate(self, integer) :
        assert integer != None , "integer is NoneType!"
        assert isinstance(integer , int) , "Not a int!"

      def check(self, index) :
        assert not index == None , "Index is NoneType!"
        assert not (index > MAX_NUM_MIX or index < 0) , "Index out of bounds!"

      def __init__(self , default) :
        self.validate(default)
        IntList2.default = default
        IntList2.data = []
        for i in range (MAX_NUM_MIX):
          IntList2.data.append(default)

      def get(self,index) :
        self.check(index)
        return self.data[index]

      def place(self , integer , index) :
          tmp = math.floor(integer)
          self.validate(tmp)
          self.check(index)
          self.data[index] = tmp

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of IntList2 class
#-------------------------------------------------------------------------------
class IntList3 :
  #The class IntList3 stores ID values as a list
  #It also stores functions which modify this list  

  #This is a copy of the class IntList, because I simply cannot 
  #be bothered to learn out how to make multiple class instances
  #in python without the list addresses overlapping.
  #Let me know if you know a good solution :)! /Nekos4Lyfe

      def validate(self, integer) :
        assert integer != None , "integer is NoneType!"
        assert isinstance(integer , int) , "Not a int!"

      def check(self, index) :
        assert not index == None , "Index is NoneType!"
        assert not (index > MAX_NUM_MIX or index < 0) , "Index out of bounds!"

      def __init__(self , default) :
        self.validate(default)
        IntList3.default = default
        IntList3.data = []
        for i in range (MAX_NUM_MIX):
          IntList3.data.append(default)

      def get(self,index) :
        self.check(index)
        return self.data[index]

      def place(self , integer , index) :
          tmp = math.floor(integer)
          self.validate(tmp)
          self.check(index)
          self.data[index] = tmp

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of IntList3 class