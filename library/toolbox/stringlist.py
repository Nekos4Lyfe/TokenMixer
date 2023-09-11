import gradio as gr
import torch, os
import collections, math, random , numpy
import re #used to parse string to int
import copy
from library.toolbox.constants import MAX_NUM_MIX
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
class StringList4 :
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
        StringList4.default = None
        StringList4.data = []
        for i in range (MAX_NUM_MIX):
          StringList4.data.append(None)

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
#End of StringList4 class

class StringList5 :
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
        self.default = None
        self.data = []
        for i in range (MAX_NUM_MIX):
          self.data.append(None)

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
#End of StringList5 class

class StringList6 :
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
        self.default = None
        self.data = []
        for i in range (MAX_NUM_MIX):
          self.data.append(None)

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
#End of StringList6 class

class StringList7 :
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
        self.default = None
        self.data = []
        for i in range (MAX_NUM_MIX):
          self.data.append(None)

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
#End of StringList7 class

class StringList8 :
  #The class StringList8 stores token names as a list
  #It also stores functions which modify this list  
      def validate(self, string) :
        cond1 = isinstance(string , str)
        cond2 = (string == None)
        assert (cond1 or cond2) , "Not a String!"

      def check(self, index) :
        assert not index == None , "Index is NoneType!"
        assert not (index > MAX_NUM_MIX or index < 0) , "Index out of bounds!"

      def __init__(self) :
        self.default = None
        self.data = []
        for i in range (MAX_NUM_MIX):
          self.data.append(None)

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
#End of StringList8 class

class StringList9 :
  #The class StringList9 stores token names as a list
  #It also stores functions which modify this list  
      def validate(self, string) :
        cond1 = isinstance(string , str)
        cond2 = (string == None)
        assert (cond1 or cond2) , "Not a String!"

      def check(self, index) :
        assert not index == None , "Index is NoneType!"
        assert not (index > MAX_NUM_MIX or index < 0) , "Index out of bounds!"

      def __init__(self) :
        self.default = None
        self.data = []
        for i in range (MAX_NUM_MIX):
          self.data.append(None)

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
#End of StringList9 class