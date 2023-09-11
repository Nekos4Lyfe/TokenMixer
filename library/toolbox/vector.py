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

from library.toolbox.floatlist import FloatList
from library.toolbox.intlist import IntList 
from library.toolbox.boolist import BooList
from library.toolbox.stringlist import StringList
from library.toolbox.constants import MAX_NUM_MIX
#-------------------------------------------------------------------------------

# Check that MPS is available (for MAC users)
choosen_device = torch.device("cpu")
#if torch.backends.mps.is_available(): 
#  choosen_device = torch.device("mps")
#else : choosen_device = torch.device("cpu")
######

class Vector :
#self is a class which stores torch tensors as a list
#It also stores functions which modify this tensor list
#The self class also contains StringList lists 
#and Boolist lists 

  def get(self,index) :
    output = self.data[index]
    assert not output == None , "Faulty get!"
    return output

  def validate(self, tensor) :
    assert tensor != None , "Null tensor!"
    assert tensor.shape[0] != None , "tensor.shape[0] is NoneType!"
    assert tensor.shape[0] == 1 , "Too many tensors!"
    assert tensor.shape[1] != None , "tensor.shape[1] is NoneType!"
    assert tensor.shape[1] == self.size , \
    "Wrong tensor dim! Size should be " + str(self.size) + " but input was " + \
    "size " + str(tensor.shape[1])  + " ."

  def place(self, tensor , index) :
    if (tensor != None) :
      self.validate(tensor)
      assert not (index > MAX_NUM_MIX or index < 0) , "Index out of bounds!"
      self.data[index] = tensor.to(device = choosen_device , dtype = torch.float32)
      self.isEmpty.place(False , index)
      assert not self.isEmpty.data[index] , "Faulty place!"

  def clear (self , index) :
    assert not index == None , "Index is NoneType!"
    assert not (index > MAX_NUM_MIX or index < 0) ,  "Index out of bounds!"
    self.data[index] = torch.zeros(self.size).unsqueeze(0)\
    .to(device = choosen_device , dtype = torch.float32)
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
        self.tmp = self.data[index1]\
        .to(device = choosen_device , dtype = torch.float32)
        self.data[index1] = self.data[index2]\
        .to(device = choosen_device , dtype = torch.float32)
        self.data[index2] = self.tmp\
        .to(device = choosen_device , dtype = torch.float32)
  ####### End of shuffle function

  def similarity (self , tensor1 , tensor2):
        distance = torch.nn.PairwiseDistance(p=2)\
        .to(device = choosen_device , dtype = torch.float32)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)\
        .to(device = choosen_device , dtype = torch.float32)
        origin = (self.origin)\
        .to(device = choosen_device , dtype = torch.float32)
        #######

        current = tensor1.to(device = choosen_device , dtype = torch.float32)
        dist1 = distance(current, origin).numpy()[0]
        current = current * (1/dist1)

        ####

        ref = tensor2.to(device = choosen_device , dtype = torch.float32)
        dist2 = distance(current, origin).numpy()[0]
        ref = ref * (1/dist2)

        ########
        sim = (100*cos(current , ref))\
        .to(device = choosen_device , dtype = torch.float32)
        if sim < 0 : sim = -sim
        output = str(round(sim.numpy()[0] , 2))
        return  output

  def distance (self, tensor1 , tensor2):
        distance = torch.nn.PairwiseDistance(p=2)\
        .to(device = choosen_device , dtype = torch.float32)
        #######
        current = tensor1.to(device = choosen_device , dtype = torch.float32)
        ref = tensor2.to(device = choosen_device , dtype = torch.float32)
        dist = distance(current, ref).numpy()[0]
        output = str(round(dist , 2))
        return  output

  def roll(self) :
    log = []
    log.append("Roll Mode:")
    distance = torch.nn.PairwiseDistance(p=2)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    r = self.randomization/100
    rand = None 
    prev = None
    current = None
    origin = self.origin.to(device = choosen_device , dtype = torch.float32)
    tokenCount = 0
    tmp = None
    for index in range(MAX_NUM_MIX):
      if self.isEmpty.get(index): continue
      if self.isFiltered(index) : continue
      #######
      r = self.rollcountrand/100
      rollCount = math.floor(self.rollcount * ((1 - r) + (r) * random.random()))
      #########
      prev = self.data[index]\
        .to(device = choosen_device , dtype = torch.float32)
      current = torch.roll(prev , rollCount)\
        .to(device = choosen_device , dtype = torch.float32)
      self.data[index] = current.to(device = choosen_device , dtype = torch.float32)
      ########
      similarity = self.similarity(current, prev)
      dist = self.distance(current, self.origin)
      tokenCount +=1
      log.append("Token '" + self.name.get(index) + "' was rolled " + str(rollCount) + \
      " times to create a new vector with length " + \
       str(dist)  + " and similarity " + str(similarity) + " %")
    log.append('New embedding has ' + str(tokenCount) + " tokens")
    log.append('-------------------------------------------')
    #########
    assert log != None , "log is None!"
    return '\n'.join(log)

  
  def random_quick(self):
    randvec = torch.rand(self.size)\
        .to(device = choosen_device , dtype = torch.float32)
    gain = None
    for k in range(self.size):
      gain = self.samplegain * (1 - (self.samplerand/100)*random.random())
      randvec[k] = (randvec[k] * gain)\
        .to(device = choosen_device , dtype = torch.float32)
    #####
    return randvec


  def random(self , internal_embs):

        output = torch.ones(self.size)\
        .to(device = choosen_device , dtype = torch.float32)

        #This list shares storage with 'output'
        output_data = output.numpy() 

        gain = None

        for k in range(self.size):
          randvec = \
          random.choice(internal_embs)\
          .to(device = choosen_device , dtype = torch.float32)
          gain = self.samplegain * (1 - (self.samplerand/100)*random.random())
          internal_emb_data = randvec.numpy() 
          randvec_data = internal_emb_data.copy()

          ####
          randval = random.choice(randvec_data)
          ####
          output_data[k] = (output_data[k] * randval * gain)
        #####
      
        return output

  def isFiltered(self , index):
    log = []
    if not self.filter_by_name : 
      #log.append("isFiltered : Not enabled")
      return False , '\n'.join(log)
    for unfiltered_index in self.unfiltered_indices:
      if str(unfiltered_index) == str(index): return False , '\n'.join(log)
      #log.append("isFiltered : " + str(unfiltered_index) + " and " + str(index))
    #########
    return True , '\n'.join(log)

  def sample(self , interal_embs) :
    log = []
    log.append("Sample Mode:")
    distance = torch.nn.PairwiseDistance(p=2)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    r = self.randomization/100
    rand = None 
    prev = None
    current = None
    origin = self.origin.to(device = choosen_device , dtype = torch.float32)
    tokenCount = 0
    tmp = None
    gain = None
    for index in range(MAX_NUM_MIX):
      if self.isEmpty.get(index): continue
      isFiltered , message = self.isFiltered(index)
      log.append(message)
      if isFiltered: continue

      rand = self.random(interal_embs)\
      .to(device = choosen_device , dtype = torch.float32)
      rdist = distance(rand , origin).numpy()[0]
      gain = self.vecsamplegain * (1 - (self.vecsamplerand/100)*random.random())
      #########
      prev = self.data[index]\
        .to(device = choosen_device , dtype = torch.float32)
      prev_dist = distance(prev , origin).numpy()[0]
      current = (prev*(1 - r)*(1/prev_dist) + rand * r * (1/rdist))\
      .to(device = choosen_device , dtype = torch.float32)
      #########
      curr_dist = distance(current , origin).numpy()[0]
      self.data[index] = \
      (current * (gain*prev_dist/curr_dist))\
      .to(device = choosen_device , dtype = torch.float32)
      ########
      tmp = self.data[index]\
        .to(device = choosen_device , dtype = torch.float32)
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
    self.size = size
    self.origin = (torch.zeros(size)\
    .to(device = choosen_device , dtype = torch.float32)).unsqueeze(0)
    self.randomization = 0
    self.interpolation = 0
    self.itermax = 1000
    self.gain = 1
    self.allow_negative_gain = False
    self.data = []
    self.tmp = None

    self.unfiltered_indices = []
    self.filter_by_name = False

    self.rollcount = 0
    self.rollcountrand = 0

    self.samplegain = 1
    self.samplerand = 0

    self.vecsamplegain = 1
    self.vecsamplerand = 0

    for i in range (MAX_NUM_MIX):
      tmp = torch.zeros(size).unsqueeze(0)\
        .to(device = choosen_device , dtype = torch.float32)
      self.data.append(tmp)
      tmp=None

    self.ID = IntList(0)
    self.name = StringList()
    self.isEmpty = BooList(True)
    self.weight = FloatList(1)
#End of self class

###### SDXL STUFF #####

#Vector768
from library.toolbox.floatlist import FloatList5
from library.toolbox.intlist import IntList5 
from library.toolbox.boolist import BooList5
from library.toolbox.stringlist import StringList5
####

class Vector1280 :
#self is a class which stores torch tensors as a list
#It also stores functions which modify this tensor list
#The self class also contains StringList lists 
#and Boolist lists 

  def get(self,index) :
    output = self.data[index]
    assert not output == None , "Faulty get!"
    return output

  def validate(self, tensor) :
    assert tensor != None , "Null tensor!"
    assert tensor.shape[0] != None , "tensor.shape[0] is NoneType!"
    assert tensor.shape[0] == 1 , "Too many tensors!"
    assert tensor.shape[1] != None , "tensor.shape[1] is NoneType!"
    assert tensor.shape[1] == self.size , \
    "Wrong tensor dim! Size should be " + str(self.size) + " but input was " + \
    "size " + str(tensor.shape[1])  + " ."

  def place(self, tensor , index) :
    if (tensor != None) :
      self.validate(tensor)
      assert not (index > MAX_NUM_MIX or index < 0) , "Index out of bounds!"
      self.data[index] = tensor.to(device = choosen_device , dtype = torch.float32)
      self.isEmpty.place(False , index)
      assert not self.isEmpty.data[index] , "Faulty place!"

  def clear (self , index) :
    assert not index == None , "Index is NoneType!"
    assert not (index > MAX_NUM_MIX or index < 0) ,  "Index out of bounds!"
    self.data[index] = torch.zeros(self.size).unsqueeze(0)\
        .to(device = choosen_device , dtype = torch.float32)
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
        self.tmp = self.data[index1]\
        .to(device = choosen_device , dtype = torch.float32)
        self.data[index1] = self.data[index2]\
        .to(device = choosen_device , dtype = torch.float32)
        self.data[index2] = self.tmp\
        .to(device = choosen_device , dtype = torch.float32)
  ####### End of shuffle function

  def similarity (self , tensor1 , tensor2):
        distance = torch.nn.PairwiseDistance(p=2)\
        .to(device = choosen_device , dtype = torch.float32)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)\
        .to(device = choosen_device , dtype = torch.float32)
        origin = (self.origin)\
        .to(device = choosen_device , dtype = torch.float32)
        #######

        current = tensor1.to(device = choosen_device , dtype = torch.float32)
        dist1 = distance(current, origin).numpy()[0]
        current = current * (1/dist1)

        ####

        ref = tensor2.to(device = choosen_device , dtype = torch.float32)
        dist2 = distance(current, origin).numpy()[0]
        ref = ref * (1/dist2)

        ########
        sim = (100*cos(current , ref))\
        .to(device = choosen_device , dtype = torch.float32)
        if sim < 0 : sim = -sim
        output = str(round(sim.numpy()[0] , 2))
        return  output

  def distance (self, tensor1 , tensor2):
        distance = torch.nn.PairwiseDistance(p=2)\
        .to(device = choosen_device , dtype = torch.float32)
        #######
        current = tensor1.to(device = choosen_device , dtype = torch.float32)
        ref = tensor2.to(device = choosen_device , dtype = torch.float32)
        dist = distance(current, ref).numpy()[0]
        output = str(round(dist , 2))
        return  output

  def roll(self) :
    log = []
    log.append("Roll Mode:")
    distance = torch.nn.PairwiseDistance(p=2)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    r = self.randomization/100
    rand = None 
    prev = None
    current = None
    origin = self.origin.to(device = choosen_device , dtype = torch.float32)
    tokenCount = 0
    tmp = None
    for index in range(MAX_NUM_MIX):
      if self.isEmpty.get(index): continue
      if self.isFiltered(index) : continue
      #######
      r = self.rollcountrand/100
      rollCount = math.floor(self.rollcount * ((1 - r) + (r) * random.random()))
      #########
      prev = self.data[index]\
        .to(device = choosen_device , dtype = torch.float32)
      current = torch.roll(prev , rollCount)\
        .to(device = choosen_device , dtype = torch.float32)
      self.data[index] = current.to(device = choosen_device , dtype = torch.float32)
      ########
      similarity = self.similarity(current, prev)
      dist = self.distance(current, self.origin)
      tokenCount +=1
      log.append("Token '" + self.name.get(index) + "' was rolled " + str(rollCount) + \
      " times to create a new vector with length " + \
       str(dist)  + " and similarity " + str(similarity) + " %")
    log.append('New embedding has ' + str(tokenCount) + " tokens")
    log.append('-------------------------------------------')
    #########
    assert log != None , "log is None!"
    return '\n'.join(log)

  
  def random_quick(self):
    randvec = torch.rand(self.size)\
        .to(device = choosen_device , dtype = torch.float32)
    gain = None
    for k in range(self.size):
      gain = self.samplegain * (1 - (self.samplerand/100)*random.random())
      randvec[k] = (randvec[k] * gain)\
        .to(device = choosen_device , dtype = torch.float32)
    #####
    return randvec


  def random(self , internal_embs):

        output = torch.ones(self.size)\
        .to(device = choosen_device , dtype = torch.float32)

        #This list shares storage with 'output'
        output_data = output.numpy() 

        gain = None

        for k in range(self.size):
          randvec = \
          random.choice(internal_embs)\
          .to(device = choosen_device , dtype = torch.float32)
          gain = self.samplegain * (1 - (self.samplerand/100)*random.random())
          internal_emb_data = randvec.numpy() 
          randvec_data = internal_emb_data.copy()

          ####
          randval = random.choice(randvec_data)
          ####
          output_data[k] = (output_data[k] * randval * gain)
        #####
      
        return output

  def isFiltered(self , index):
    log = []
    if not self.filter_by_name : 
      #log.append("isFiltered : Not enabled")
      return False , '\n'.join(log)
    for unfiltered_index in self.unfiltered_indices:
      if str(unfiltered_index) == str(index): return False , '\n'.join(log)
      #log.append("isFiltered : " + str(unfiltered_index) + " and " + str(index))
    #########
    return True , '\n'.join(log)

  def sample(self , interal_embs) :
    log = []
    log.append("Sample Mode:")
    distance = torch.nn.PairwiseDistance(p=2)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    r = self.randomization/100
    rand = None 
    prev = None
    current = None
    origin = self.origin.to(device = choosen_device , dtype = torch.float32)
    tokenCount = 0
    tmp = None
    gain = None
    for index in range(MAX_NUM_MIX):
      if self.isEmpty.get(index): continue
      isFiltered , message = self.isFiltered(index)
      log.append(message)
      if isFiltered: continue

      rand = self.random(interal_embs)\
      .to(device = choosen_device , dtype = torch.float32)
      rdist = distance(rand , origin).numpy()[0]
      gain = self.vecsamplegain * (1 - (self.vecsamplerand/100)*random.random())
      #########
      prev = self.data[index]\
        .to(device = choosen_device , dtype = torch.float32)
      prev_dist = distance(prev , origin).numpy()[0]
      current = (prev*(1 - r)*(1/prev_dist) + rand * r * (1/rdist))\
      .to(device = choosen_device , dtype = torch.float32)
      #########
      curr_dist = distance(current , origin).numpy()[0]
      self.data[index] = \
      (current * (gain*prev_dist/curr_dist))\
      .to(device = choosen_device , dtype = torch.float32)
      ########
      tmp = self.data[index]\
        .to(device = choosen_device , dtype = torch.float32)
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
    self.size = size
    self.origin = (torch.zeros(size)\
    .to(device = choosen_device , dtype = torch.float32)).unsqueeze(0)
    self.randomization = 0
    self.interpolation = 0
    self.itermax = 1000
    self.gain = 1
    self.allow_negative_gain = False
    self.data = []
    self.tmp = None

    self.unfiltered_indices = []
    self.filter_by_name = False

    self.rollcount = 0
    self.rollcountrand = 0

    self.samplegain = 1
    self.samplerand = 0

    self.vecsamplegain = 1
    self.vecsamplerand = 0

    for i in range (MAX_NUM_MIX):
      tmp = torch.zeros(size).unsqueeze(0)\
        .to(device = choosen_device , dtype = torch.float32)
      self.data.append(tmp)
      tmp=None

    self.ID = IntList5(0)
    self.name = StringList5()
    self.isEmpty = BooList5(True)
    self.weight = FloatList5(1)
#End of Vector768

