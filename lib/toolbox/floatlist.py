
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
class FloatList :
  #The class FloatList stores float values as a list
  #It also stores functions which modify this list  
      def validate(self, dec) :
        assert dec != None , "Float is NoneType!"
        #assert isinstance(dec , float) , "Not a Float!"

      def check(self, index) :
        assert not index == None , "Index is NoneType!"
        assert not (index >= MAX_NUM_MIX or index < 0) , "Index out of bounds!"

      def __init__(self , default) :
        self.validate(default)
        FloatList.default = default
        FloatList.data = []
        for i in range (MAX_NUM_MIX):
          FloatList.data.append(default)

      def get(self,index) :
        self.check(index)
        return self.data[index]

      def place(self , dec , index) :
          self.validate(dec)
          self.check(index)
          self.data[index] = copy.copy(dec)

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of FloatList class
#-------------------------------------------------------------------------------
class FloatList2 :
  #The class FloatList stores float values as a list
  #It also stores functions which modify this list  
      def validate(self, dec) :
        assert dec != None , "Float is NoneType!"
        #assert isinstance(dec , float) , "Not a Float!"

      def check(self, index) :
        assert not index == None , "Index is NoneType!"
        assert not (index >= MAX_NUM_MIX or index < 0) , "Index out of bounds!"

      def __init__(self , default) :
        self.validate(default)
        FloatList2.default = default
        FloatList2.data = []
        for i in range (MAX_NUM_MIX):
          FloatList2.data.append(default)

      def get(self,index) :
        self.check(index)
        return self.data[index]

      def place(self , dec , index) :
          self.validate(dec)
          self.check(index)
          self.data[index] = copy.copy(dec)

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of FloatList2 class
#-------------------------------------------------------------------------------
class FloatList3 :
  #The class FloatList stores float values as a list
  #It also stores functions which modify this list  
      def validate(self, dec) :
        assert dec != None , "Float is NoneType!"
        #assert isinstance(dec , float) , "Not a Float!"

      def check(self, index) :
        assert not index == None , "Index is NoneType!"
        assert not (index >= MAX_NUM_MIX or index < 0) , "Index out of bounds!"

      def __init__(self , default) :
        self.validate(default)
        FloatList3.default = default
        FloatList3.data = []
        for i in range (MAX_NUM_MIX):
          FloatList3.data.append(default)

      def get(self,index) :
        self.check(index)
        return self.data[index]

      def place(self , dec , index) :
          self.validate(dec)
          self.check(index)
          self.data[index] = copy.copy(dec)

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of FloatList3 class