import gradio as gr
import torch, os
import collections, math, random , numpy
import re #used to parse string to int
import copy
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
          self.data[index] = copy.copy(tmp)

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
          self.data[index] = copy.copy(tmp)

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of IntList2 class
#-------------------------------------------------------------------------------
class IntList3 :
  #The class IntList3 stores ID values as a list
  #It also stores functions which modify this list  

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
          self.data[index] = copy.copy(tmp)

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of IntList3 class
#-------------------------------------------------------------------------------
class IntList4 :
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
        IntList4.default = default
        IntList4.data = []
        for i in range (MAX_NUM_MIX):
          IntList4.data.append(default)

      def get(self,index) :
        self.check(index)
        return self.data[index]

      def place(self , integer , index) :
          tmp = math.floor(integer)
          self.validate(tmp)
          self.check(index)
          self.data[index] = copy.copy(tmp)

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of IntList4 class

class IntList5 :
  #The class IntList5 stores ID values as a list
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
        IntList5.default = default
        IntList5.data = []
        for i in range (MAX_NUM_MIX):
          IntList5.data.append(default)

      def get(self,index) :
        self.check(index)
        return self.data[index]

      def place(self , integer , index) :
          tmp = math.floor(integer)
          self.validate(tmp)
          self.check(index)
          self.data[index] = copy.copy(tmp)

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of IntList5 class

class IntList6 :
  #The class IntList5 stores ID values as a list
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
        self.default = default
        self.data = []
        for i in range (MAX_NUM_MIX):
          self.data.append(default)

      def get(self,index) :
        self.check(index)
        return self.data[index]

      def place(self , integer , index) :
          tmp = math.floor(integer)
          self.validate(tmp)
          self.check(index)
          self.data[index] = copy.copy(tmp)

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of IntList6 class

class IntList7 :
  #The class IntList5 stores ID values as a list
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
        self.default = default
        self.data = []
        for i in range (MAX_NUM_MIX):
          self.data.append(default)

      def get(self,index) :
        self.check(index)
        return self.data[index]

      def place(self , integer , index) :
          tmp = math.floor(integer)
          self.validate(tmp)
          self.check(index)
          self.data[index] = copy.copy(tmp)

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of IntList7 class

class IntList8 :
  #The class IntList5 stores ID values as a list
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
        self.default = default
        self.data = []
        for i in range (MAX_NUM_MIX):
          self.data.append(default)

      def get(self,index) :
        self.check(index)
        return self.data[index]

      def place(self , integer , index) :
          tmp = math.floor(integer)
          self.validate(tmp)
          self.check(index)
          self.data[index] = copy.copy(tmp)

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of IntList8 class

class IntList9 :
  #The class IntList9 stores ID values as a list
  #It also stores functions which modify this list  

      def validate(self, integer) :
        assert integer != None , "integer is NoneType!"
        assert isinstance(integer , int) , "Not a int!"

      def check(self, index) :
        assert not index == None , "Index is NoneType!"
        assert not (index > MAX_NUM_MIX or index < 0) , "Index out of bounds!"

      def __init__(self , default) :
        self.validate(default)
        self.default = default
        self.data = []
        for i in range (MAX_NUM_MIX):
          self.data.append(default)

      def get(self,index) :
        self.check(index)
        return self.data[index]

      def place(self , integer , index) :
          tmp = math.floor(integer)
          self.validate(tmp)
          self.check(index)
          self.data[index] = copy.copy(tmp)

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of IntList9 class