
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

class StringList :
  #The class StringList stores token names as a list
  #It also stores functions which modify this list  
      def validate(self, string) :
        cond1 = isinstance(string , str)
        cond2 = (string == None)
        assert (cond1 or cond2) , "Not a String!"

      def check(self, index) :
        assert not index == None , "Index is NoneType!"
        assert not (index > MAX_NUM_MIX or index < 0) , "Index out of bounds!"

      def __init__(self) :
        StringList.default = None
        StringList.data = []
        for i in range (MAX_NUM_MIX):
          StringList.data.append(None)

      def get(self,index) :
        self.check(index)
        self.validate(self.data[index])
        return self.data[index]

      def place(self , string , index) :
          self.validate(string)
          self.check(index)
          if string == None: self.data[index] = None
          else: self.data[index] = copy.copy(string)

      def clear(self , index):
        assert self.default == None , "Default is not NoneType!"
        self.check(index)
        self.data[index] = None
#End of StringList class
#-------------------------------------------------------------------------------
class StringList2 :
  #The class StringList2 stores token names as a list
  #It also stores functions which modify this list  
      def validate(self, string) :
        cond1 = isinstance(string , str)
        cond2 = (string == None)
        assert (cond1 or cond2) , "Not a String!"

      def check(self, index) :
        assert not index == None , "Index is NoneType!"
        assert not (index > MAX_NUM_MIX or index < 0) , "Index out of bounds!"

      def __init__(self) :
        StringList2.default = None
        StringList2.data = []
        for i in range (MAX_NUM_MIX):
          StringList2.data.append(None)

      def get(self,index) :
        self.check(index)
        self.validate(self.data[index])
        return self.data[index]

      def place(self , string , index) :
          self.validate(string)
          self.check(index)
          if string == None: self.data[index] = None
          else: self.data[index] = copy.copy(string)

      def clear(self , index):
        assert self.default == None , "Default is not NoneType!"
        self.check(index)
        self.data[index] = None
#End of StringList2 class
#-------------------------------------------------------------------------------
class StringList3 :
  #The class StringList3 stores token names as a list
  #It also stores functions which modify this list  
      def validate(self, string) :
        cond1 = isinstance(string , str)
        cond2 = (string == None)
        assert (cond1 or cond2) , "Not a String!"

      def check(self, index) :
        assert not index == None , "Index is NoneType!"
        assert not (index > MAX_NUM_MIX or index < 0) , "Index out of bounds!"

      def __init__(self) :
        StringList3.default = None
        StringList3.data = []
        for i in range (MAX_NUM_MIX):
          StringList3.data.append(None)

      def get(self,index) :
        self.check(index)
        self.validate(self.data[index])
        return self.data[index]

      def place(self , string , index) :
          self.validate(string)
          self.check(index)
          if string == None: self.data[index] = None
          else: self.data[index] = copy.copy(string)

      def clear(self , index):
        assert self.default == None , "Default is not NoneType!"
        self.check(index)
        self.data[index] = None
#End of StringList3 class
#-------------------------------------------------------------------------------