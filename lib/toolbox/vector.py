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

from lib.toolbox.floatlist import FloatList
from lib.toolbox.intlist import IntList 
from lib.toolbox.boolist import BooList
from lib.toolbox.stringlist import StringList
from lib.toolbox.constants import MAX_NUM_MIX
#-------------------------------------------------------------------------------

class Vector :
#Vector is a class which stores torch tensors as a list
#It also stores functions which modify this tensor list
#The Vector class also contains StringList lists 
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

  def swap(self, target , index1 , index2) : 
    self.tmp = copy.copy(target.get(index1))
    target.place(target.get(index2) , index1)
    target.place(self.tmp , index2)

  def shuffle(self):
    randlist = []
    for i in range(MAX_NUM_MIX):
      randlist.append(i)
      if self.isEmpty.get(i) : break
    
    random.shuffle(randlist)

    index_list = []
    for i in range(MAX_NUM_MIX):
      if i < len(randlist): index_list.append(randlist[i])
      else: index_list.append(i)

    for index1 in range(MAX_NUM_MIX):
      for index2 in index_list : 
        if self.isEmpty.get(index1): continue
        if self.isEmpty.get(index2): continue

        self.swap(self.ID      , index1 , index2)
        self.swap(self.name    , index1 , index2)
        self.swap(self.isEmpty , index1 , index2)
        self.swap(self.weight  , index1 , index2)
        #####
        self.tmp = self.data[index1].cpu()
        self.data[index1] = self.data[index2]
        self.data[index2] = self.tmp
  ####### End of shuffle function

  def sample(self) :

    log = []
    log.append("Sample Mode:")

    distance = torch.nn.PairwiseDistance(p=2)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    r = self.randomization/100
    rand = None 
    prev = None
    current = None
    origin = self.origin 
    tokenCount = 0
    tmp = None
    for index in range(MAX_NUM_MIX):
      if self.isEmpty.get(index): continue
      rand = 2*torch.rand(self.size) - torch.ones(self.size)
      rdist = distance(rand , origin).numpy()[0]
      #########
      prev = self.data[index]
      prev_dist = distance(prev , origin).numpy()[0]
      current = prev*(1 - r)*(1/prev_dist) + rand * r * (1/rdist)
      #########
      curr_dist = distance(current , origin).numpy()[0]
      self.data[index] = (current * (prev_dist/curr_dist)).cpu()
      ########
      tmp = self.data[index]
      similarity = 100*cos(tmp, prev).numpy()[0]
      if similarity < 0 : similarity = -similarity
      dist = distance(tmp , origin).numpy()[0]
      tokenCount +=1
      similarity = round(similarity ,1)
      dist = round(dist ,2)
      log.append("Token '" + self.name.get(index) + "' was replaced by new vector with " + \
       "length " + str(dist)  + " and similarity " + str(similarity) + " %")
      log.append('New embedding has ' + str(tokenCount) + " tokens")
      log.append('-------------------------------------------')
    #########
    return '\n'.join(log)



  def __init__(self , size):
    Vector.size = size
    Vector.origin = torch.zeros(size).unsqueeze(0).cpu()
    Vector.costheta = 0
    Vector.radius = 0
    Vector.randomization = 0
    Vector.interpolation = 0
    Vector.itermax = 1000
    Vector.gain = 1
    Vector.allow_negative_gain = False
    Vector.data = []
    Vector.tmp = None

    for i in range (MAX_NUM_MIX):
      tmp = torch.zeros(size).unsqueeze(0).cpu()
      Vector.data.append(tmp)
      tmp=None

    Vector.ID = IntList(0)
    Vector.name = StringList()
    Vector.isEmpty = BooList(True)
    Vector.weight = FloatList(1)
#End of Vector class