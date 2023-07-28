
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

class BooList :
  #The class BooList stores booleans as a list
  #It also stores functions which modify this list  
      def validate(self, cond) :
        assert cond != None , "Null boolean!"
        assert isinstance(cond , bool) , "Not a boolean!"

      def check(self, index) :
        assert not index == None , "Index is NoneType!"
        assert not (index > MAX_NUM_MIX or index < 0) , "Index out of bounds!"

      def __init__(self , default) :
        self.validate(default)
        BooList.default = default
        BooList.data = []
        for i in range (MAX_NUM_MIX):
          BooList.data.append(default)

      def get(self,index) :
        self.check(index)
        self.validate(self.data[index])
        return self.data[index]

      def place(self , cond , index) :
          self.validate(cond)
          self.check(index)
          self.data[index] = cond

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of Boolist class
#-------------------------------------------------------------------------------

class BooList2 :
  #The class BooList2 stores booleans as a list
  #It also stores functions which modify this list 

  #This is a copy of the class Boolist, because I simply cannot 
  #be bothered to learn out how to make multiple class instances
  #in python without the list addresses overlapping.
  #Let me know if you know a good solution :)! /Nekos4Lyfe 
      def validate(self, cond) :
        assert cond != None , "Null boolean!"
        assert isinstance(cond , bool) , "Not a boolean!"

      def check(self, index) :
        assert not index == None , "Index is NoneType!"
        assert not (index > MAX_NUM_MIX or index < 0) , "Index out of bounds!"

      def __init__(self , default) :
        self.validate(default)
        BooList2.default = default
        BooList2.data = []
        for i in range (MAX_NUM_MIX):
          BooList2.data.append(default)

      def get(self,index) :
        self.check(index)
        self.validate(self.data[index])
        return self.data[index]

      def place(self , cond , index) :
          self.validate(cond)
          self.check(index)
          self.data[index] = cond

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of BooList2 class
#-------------------------------------------------------------------------------

class BooList3 :
  #The class BooList3 stores booleans as a list
  #It also stores functions which modify this list  

  #This is a copy of the class Boolist, because I simply cannot 
  #be bothered to learn out how to make multiple class instances
  #in python without the list addresses overlapping.
  #Let me know if you know a good solution :)! /Nekos4Lyfe
      def validate(self, cond) :
        assert cond != None , "Null boolean!"
        assert isinstance(cond , bool) , "Not a boolean!"

      def check(self, index) :
        assert not index == None , "Index is NoneType!"
        assert not (index > MAX_NUM_MIX or index < 0) , "Index out of bounds!"

      def __init__(self , default) :
        self.validate(default)
        BooList3.default = default
        BooList3.data = []
        for i in range (MAX_NUM_MIX):
          BooList3.data.append(default)

      def get(self,index) :
        self.check(index)
        self.validate(self.data[index])
        return self.data[index]

      def place(self , cond , index) :
          self.validate(cond)
          self.check(index)
          self.data[index] = cond

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of BooList3 class