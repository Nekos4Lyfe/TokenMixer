
import gradio as gr
import torch, os
import collections, math, random , numpy
import re #used to parse string to int
import copy
from library.toolbox.constants import MAX_NUM_MIX
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
          self.data[index] = copy.copy(cond)

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
          self.data[index] = copy.copy(cond)

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
          self.data[index] = copy.copy(cond)

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of BooList3 class
#-------------------------------------------------------------------------------
class BooList4 :
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
        BooList4.default = default
        BooList4.data = []
        for i in range (MAX_NUM_MIX):
          BooList4.data.append(default)

      def get(self,index) :
        self.check(index)
        self.validate(self.data[index])
        return self.data[index]

      def place(self , cond , index) :
          self.validate(cond)
          self.check(index)
          self.data[index] = copy.copy(cond)

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of BooList4 class

class BooList5 :
  #The class BooList5 stores booleans as a list
  #It also stores functions which modify this list  

      def validate(self, cond) :
        assert cond != None , "Null boolean!"
        assert isinstance(cond , bool) , "Not a boolean!"

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
        self.validate(self.data[index])
        return self.data[index]

      def place(self , cond , index) :
          self.validate(cond)
          self.check(index)
          self.data[index] = copy.copy(cond)

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of BooList5 class

class BooList6 :
  #The class BooList6 stores booleans as a list
  #It also stores functions which modify this list  

      def validate(self, cond) :
        assert cond != None , "Null boolean!"
        assert isinstance(cond , bool) , "Not a boolean!"

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
        self.validate(self.data[index])
        return self.data[index]

      def place(self , cond , index) :
          self.validate(cond)
          self.check(index)
          self.data[index] = copy.copy(cond)

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of BooList6 class

class BooList7 :
  #The class BooList7 stores booleans as a list
  #It also stores functions which modify this list  

      def validate(self, cond) :
        assert cond != None , "Null boolean!"
        assert isinstance(cond , bool) , "Not a boolean!"

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
        self.validate(self.data[index])
        return self.data[index]

      def place(self , cond , index) :
          self.validate(cond)
          self.check(index)
          self.data[index] = copy.copy(cond)

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of BooList7 class

class BooList8 :
  #The class BooList5 stores booleans as a list
  #It also stores functions which modify this list  

      def validate(self, cond) :
        assert cond != None , "Null boolean!"
        assert isinstance(cond , bool) , "Not a boolean!"

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
        self.validate(self.data[index])
        return self.data[index]

      def place(self , cond , index) :
          self.validate(cond)
          self.check(index)
          self.data[index] = copy.copy(cond)

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of BooList8 class

class BooList9 :
  #The class BooList5 stores booleans as a list
  #It also stores functions which modify this list  

      def validate(self, cond) :
        assert cond != None , "Null boolean!"
        assert isinstance(cond , bool) , "Not a boolean!"

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
        self.validate(self.data[index])
        return self.data[index]

      def place(self , cond , index) :
          self.validate(cond)
          self.check(index)
          self.data[index] = copy.copy(cond)

      def clear(self , index):
        self.validate(self.default)
        self.check(index)
        self.data[index] = self.default
#End of BooList9 class