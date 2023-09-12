
import gradio as gr
import torch, os
import collections, math, random , numpy
import re #used to parse string to int
import copy
from library.toolbox.constants import MAX_NUM_MIX

from library.toolbox.constants import TENSOR_DEVICE_TYPE , TENSOR_DATA_TYPE
choosen_device = TENSOR_DEVICE_TYPE
datatype = TENSOR_DATA_TYPE

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
#-------------------------------------------------------------------------------
class FloatList4 :
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
        FloatList4.default = default
        FloatList4.data = []
        for i in range (MAX_NUM_MIX):
          FloatList4.data.append(default)

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
#End of FloatList4 class

class FloatList5 :
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
        self.default = default
        self.data = []
        for i in range (MAX_NUM_MIX):
          self.data.append(default)

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
#End of FloatList5 class


class FloatList6 :
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
        self.default = default
        self.data = []
        for i in range (MAX_NUM_MIX):
          self.data.append(default)

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
#End of FloatList6 class

class FloatList7 :
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
        self.default = default
        self.data = []
        for i in range (MAX_NUM_MIX):
          self.data.append(default)

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
#End of FloatList7 class

class FloatList8 :
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
        self.default = default
        self.data = []
        for i in range (MAX_NUM_MIX):
          self.data.append(default)

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
#End of FloatList8 class

class FloatList9 :
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
        self.default = default
        self.data = []
        for i in range (MAX_NUM_MIX):
          self.data.append(default)

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
#End of FloatList9 class




