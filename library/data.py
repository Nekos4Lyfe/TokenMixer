import gradio as gr
import torch, os , math, random , numpy , copy
#####
from modules import  shared, sd_hijack
######
from library.toolbox.vector import Vector , Vector1280
from library.toolbox.negative import Negative , Negative1280
from library.toolbox.positive import Positive , Positive1280
from library.toolbox.temporary import Temporary , Temporary1280
from library.toolbox.tools import Tools
from library.toolbox.constants import MAX_NUM_MIX
from library.toolbox.operations import Operations
from library.toolbox.constants import TENSOR_DEVICE_TYPE , TENSOR_DATA_TYPE
choosen_device = TENSOR_DEVICE_TYPE
datatype = TENSOR_DATA_TYPE
#-------------------------------------------------------------------------------

#use_sdxl_dim
# Check that MPS is available (for MAC users)
#if torch.backends.mps.is_available(): 
#  choosen_device = torch.device("mps")
#else : choosen_device = torch.device("cpu")
######

class Data :
  #The Data class consists of all the shared resources between the modules
  #This includes the vector class and the tools class


  # Merge all tokens into single token
  def merge_all(self , similar_mode , use_1280_dim = False):
    log = []
    target = None
    if use_1280_dim : target = self.vector1280
    else : target = self.vector
    #####
    if similar_mode : 
      for index in range(MAX_NUM_MIX):
        current , message = self.replace_with_similar(index , use_1280_dim)
        target.place(current , index)
        log.append(message)
    #####
    output , message = \
    self.operations._merge_all (target , self.positive , self.negative)
    log.append(message)
    #######
    return output , '\n'.join(log)
  #### End of merge_all()


  # Concactinate all tokens into embedding 
  # (this is the default operation of TokenMixer)
  def concat_all(self , similar_mode , use_1280_dim = False):
    concat = self.operations.using_tensors._concat_all
    target = None
    if use_1280_dim : target = self.vector1280
    else : target = self.vector
    assert target != None , "Fetched vector is NoneType!"
    return concat(target , similar_mode, use_1280_dim)
  ##### End of concat_all()


  # Replace token at index with similar token
  # using randomization parameterns stored with the 'vector' class
  def replace_with_similar(self , index , use_1280_dim = False):

    similar_token = self.operations.using_tensors._replace_with_similar
    pursuit_value = self.pursuit_strength/100
    doping_value = self.doping_strength/100

    if use_1280_dim: 
      return similar_token(
        index , 
        self.vector1280, 
        self.positive1280 , 
        self.negative1280 , 
        pursuit_value , 
        doping_value)
    ########

    if not use_1280_dim:
      return similar_token(
        index , 
        self.vector, 
        self.positive , 
        self.negative , 
        pursuit_value , 
        doping_value)
  ###### End of replace_with_similar()

  #
  # Merge the tokens if they are similar enough
  # This mode is referred as "Interpolate Mode" in the TokenMixer
  def merge_if_similar(self , similar_mode , use_1280_dim = False):
    merge_similar = self.operations.using_tensors._merge_if_similar
    log = []
    target = None
    if use_1280_dim : target = self.vector1280
    else : target = self.vector
    #####
    if similar_mode : 
      for index in range(MAX_NUM_MIX):
        current , message = self.replace_with_similar(index , use_1280_dim)
        target.place(current , index)
        log.append(message)
    #######
    embedding , message = \
    merge_similar(target , self.positive , self.negative)
    log.append(message)
    #######
    return embedding , '\n'.join(log)
  ##### End of merge_if_similar()

  def text_to_emb_ids(self, text):
    is_sdxl , is_sd2 , is_sd1 = self.tools.get_flags()
    tokenizer = self.tools.tokenizer
    emb_ids = self.operations.using_text.\
    _text_to_emb_ids(text , tokenizer , is_sdxl , is_sd2 , is_sd1)
    assert emb_ids != None , "emb_ids are NoneType!"
    return emb_ids

  def emb_id_to_vec(self, emb_id , use_1280_dim = False) : 
    target = None
    index_max = None
    if use_1280_dim: 
      target = self.tools.internal_embs1280
      index_max = self.tools.no_of_internal_embs1280
    else : 
      target = self.tools.internal_embs
      index_max = self.tools.no_of_internal_embs
    ########
    # Do some checks
    assert isinstance(emb_id, int) , "Embedding ID is not int!"
    assert (emb_id < index_max), \
    "emb_id with value " + str(emb_id) + " is out of bounds. Must not exceed " + \
    "internal_embedding size of " + str(index_max)
    assert emb_id >= 0 ,  \
    "emb_id with value " + str(emb_id) + " is out of bounds. " + \
    "Index be greater then 0."
    #########
    return target[emb_id].to(device= choosen_device , dtype = datatype)
    
  def emb_id_to_name(self, text):
    emb_id = copy.copy(text)
    emb_name_utf8 = self.tools.tokenizer.decoder.get(emb_id)
    if emb_name_utf8 != None:
        byte_array_utf8 = bytearray([self.tools.tokenizer.byte_decoder[c] for c in emb_name_utf8])
        emb_name = byte_array_utf8.decode("utf-8", errors='backslashreplace')
    else: emb_name = '!Unknown ID!'
    if emb_name.find('</w>')>=0: 
      emb_name = emb_name.split('</w>')[0]
    return emb_name # return embedding name for embedding ID

  def get_embedding_info(self, string , use_1280_dim = False):
      emb_id = None
      text = copy.copy(string.lower())
      loaded_emb = self.tools.loaded_embs.get(text, None)
      if loaded_emb == None:
        for neg_index in self.tools.loaded_embs.keys():
            if text == neg_index.lower():
              loaded_emb = self.tools.loaded_embs.get(neg_index, None)
              break
              
      if loaded_emb != None: 
        emb_name = loaded_emb.name
        if use_1280_dim : 
          emb_id = 'unknown' #<<<< Will figure this out later
          emb_vec = loaded_emb.vec.get("clip_l")\
          .to(device = choosen_device , dtype = datatype)
        else: 
          emb_id = '['+ loaded_emb.checksum()+']' # emb_id is string for loaded embeddings
          emb_vec = loaded_emb.vec\
          .to(device = choosen_device , dtype = datatype)
        return emb_name, emb_id, emb_vec, loaded_emb #also return loaded_emb reference

      emb_ids = self.text_to_emb_ids(text)
      if emb_ids == None : return None, None, None, None
      
      emb_names = []
      emb_vecs = []
      emb_name = None
      emb_vec = None

      for emb_id in emb_ids:
        emb_name = self.emb_id_to_name(emb_id)
        emb_names.append(emb_name)

        if use_1280_dim : emb_vec = self.tools.internal_embs1280[emb_id].unsqueeze(0)
        else : emb_vec = self.tools.internal_embs[emb_id].unsqueeze(0)
        emb_vecs.append(emb_vec)
      ##########
    
      #Might have to fix later , idk
      if emb_name == None : return None, None, None, None
      emb_ids = emb_ids[0]
      emb_names = emb_names[0]
      emb_vecs = emb_vecs[0]
      ##########
      return emb_names, emb_ids, emb_vecs, None # return embedding name, ID, vector

  def shuffle(self , to_negative = None , to_mixer = None , \
  to_positive = None , to_temporary = None , use_1280_dim = False):

    if to_negative == None and to_mixer == None \
    and to_positive == None and to_temporary == None : 
      if use_1280_dim : self.vector1280.shuffle()
      self.vector.shuffle()

    else:
      if to_negative != None :
        if to_negative : pass # Not implemented
      #####
      if to_mixer != None : 
        if to_mixer : 
          if use_1280_dim : self.vector1280.shuffle()
          self.vector.shuffle()
      #####
      if to_temporary != None : 
        if to_temporary : pass # Not implemented

      if to_positive != None : 
        if to_positive : pass # Not implemented
  ######## End of shuffle function



  def roll (self , to_negative = None , to_mixer = None , \
  to_positive = None , to_temporary = None , use_1280_dim = False):
    message = ''
    log = []
    if to_negative == None and to_mixer == None \
    and to_positive == None and to_temporary == None : 
      if use_1280_dim :
        log.append("Performing roll on 1280 dimension vectors : ") 
        message = self.vector1280.roll()
        log.append(message)
      #######
      log.append("Performing roll on 768 dimension vectors : ") 
      message = self.vector.roll()
      log.append(message)
    else:
      if to_negative != None :
        if to_negative : pass # Not implemented
      #####
      if to_mixer != None : 
        if to_mixer : 
          if use_1280_dim : 
            log.append("Performing roll on 1280 dimension vectors : ")
            message = self.vector1280.roll()
            log.append(message)
          ######
          log.append("Performing roll on 768 dimension vectors : ") 
          message = self.vector.roll()
          log.append(message)
      #####
      if to_temporary != None : 
        if to_temporary : pass # Not implemented
      ######
      if to_positive != None : 
        if to_positive : pass # Not implemented
    #####
    return '\n'.join(log)
  ######## End of roll function

  def random(self , use_1280_dim = False):
    if use_1280_dim : return self.vector1280.random(self.tools.internal_embs1280)
    else : return self.vector.random(self.tools.internal_embs)
  ### End of random()

  def random_quick(self):
    return self.vector.random_quick()
  ## End of random_quick()

  def sample(self , to_negative = None , to_mixer = None , \
    to_positive = None , to_temporary = None , use_1280_dim = False):
    message = ''
    log = []
    if to_negative == None and to_mixer == None \
    and to_positive == None and to_temporary == None:
      if use_1280_dim : 
        log.append("Performing sample on 1280 dim vectors")
        message = self.vector1280.sample(self.tools.internal_embs1280)
        log.append(message)
      #######
      log.append("Performing sample on 768 dim vectors")
      message = self.vector.sample(self.tools.internal_embs)
      log.append(message)
    else:
      if to_negative != None :
        if to_negative  : pass #Not implemented
      #####
      if to_mixer != None : 
        if to_mixer : 
          if use_1280_dim : 
            log.append("Performing sample on 1280 dim vectors")
            message = self.vector1280.sample(self.tools.internal_embs1280)
            log.append(message)
          #######
          log.append("Performing sample on 768 dim vectors")
          message = self.vector.sample(self.tools.internal_embs)
          log.append(message)
      #########
      if to_temporary != None : 
        if to_temporary : pass #Not implemented
    #####
      if to_positive != None : 
        if to_positive : pass #Not implemented
    return '\n'.join(log)
  ### End of sample()


  # Helper function
  def _place_values_in(self, target , vector , ID, name, weight , index) :
      if vector != None: target.place(vector\
      .to(device = choosen_device , dtype = datatype),index)
      if ID != None: target.ID.place(ID, index)
      if name != None: target.name.place(name, index)
      if weight != None : target.weight.place(float(weight) , index)
  ##### End of _place_values_in()


  # Place the input somewhere in the Dataclass
  def place(self, index , vector = None , ID = None ,  name = None , \
    weight = None , to_negative = None , to_mixer = None ,  \
    to_positive = None ,  to_temporary = None , use_1280_dim = False):

    target = None
    cond = \
    (to_negative == None) and \
    (to_mixer == None) and \
    (to_temporary == None) and \
    (to_positive == None)

    # Default operation 
    if cond:
      target = self.vector
      self._place_values_in(target , vector , ID, name, weight , index)
    #######

    # Write to the 'negative' field
    if not cond and (to_negative != None) :
      if use_1280_dim : target = self.negative1280
      else: target = self.negative
      if to_negative : self._place_values_in\
      (target , vector , ID, name, weight , index)
    #######

    # Write to the 'vector' field
    if not cond and (to_mixer != None) : 
      if use_1280_dim : target = self.vector1280
      else: target = self.vector
      if to_mixer : self._place_values_in\
      (target , vector , ID, name, weight , index)
    #######

    # Write to the 'temporary' field
    if not cond and (to_temporary != None) :
      if use_1280_dim : target = self.temporary1280
      else: target = self.temporary
      if to_temporary : self._place_values_in\
      (target , vector , ID, name, weight , index)
    #######

    # Write to the 'positive' field
    if not cond and (to_positive != None) : 
      if use_1280_dim : target = self.positive1280
      else: target = self.positive
      if to_positive : self._place_values_in\
      (target , vector , ID, name, weight , index)
    #######
  #### End of place()


  # Overwrite the contents of temporary() with the contents
  # of vector() in the Data class
  def memorize(self) : 
    #######
    for index in range(MAX_NUM_MIX):
      if self.vector.isEmpty.get(index) : continue
      self.vec = self.vector.get(index)\
      .to(device = choosen_device , dtype = datatype)
      self.ID = copy.copy(self.vector.ID.get(index))
      self.name = copy.copy(self.vector.name.get(index))
      self.weight = copy.copy(self.vector.weight.get(index))
      ########
      self.temporary.clear(index)
      self.temporary.place(self.vec , index)
      self.temporary.ID.place(self.ID , index)
      self.temporary.name.place(self.name , index)
      self.temporary.weight.place(self.weight , index)
      ########
      if self.vector1280.isEmpty.get(index) : continue
      self.vec = self.vector1280.get(index)\
      .to(device = choosen_device , dtype = datatype)
      self.ID = copy.copy(self.vector1280.ID.get(index))
      self.name = copy.copy(self.vector1280.name.get(index))
      self.weight = copy.copy(self.vector1280.weight.get(index))
      ########
      self.temporary1280.clear(index)
      self.temporary1280.place(self.vec , index)
      self.temporary1280.ID.place(self.ID , index)
      self.temporary1280.name.place(self.name , index)
      self.temporary1280.weight.place(self.weight , index)
  ###### End of memorize()

  # Overwrite the contents of vector() with the contents
  # of temporary() in the Data class
  def recall(self) :
    for index in range(MAX_NUM_MIX):
      if self.temporary.isEmpty.get(index) : continue
      self.vec = self.temporary.get(index)\
      .to(device = choosen_device , dtype = datatype)
      self.ID = copy.copy(self.temporary.ID.get(index))
      self.name = copy.copy(self.temporary.name.get(index))
      self.weight = copy.copy(self.temporary.weight.get(index))
      #######
      self.vector.clear(index)
      self.vector.place(self.vec , index)
      self.vector.ID.place(self.ID , index)
      self.vector.name.place(self.name, index)
      self.vector.weight.place(self.weight , index)
      ########
      if self.temporary1280.isEmpty.get(index) : continue
      self.vec = self.temporary1280.get(index)\
      .to(device = choosen_device , dtype = datatype)
      self.ID = copy.copy(self.temporary1280.ID.get(index))
      self.name = copy.copy(self.temporary1280.name.get(index))
      self.weight = copy.copy(self.temporary1280.weight.get(index))
      #######
      self.vector1280.clear(index)
      self.vector1280.place(self.vec , index)
      self.vector1280.ID.place(self.ID , index)
      self.vector1280.name.place(self.name, index)
      self.vector1280.weight.place(self.weight , index)
  ###### End of recall()

  # Return a string of the distance from the 
  # endpoints of tensor1 to the endpoint of tensor2
  def distance (self, tensor1 , tensor2):
        distance = torch.nn.PairwiseDistance(p=2)\
        .to(device = choosen_device , dtype = datatype)
        #######
        current = tensor1.to(device = choosen_device , dtype = datatype)
        ref = tensor2.to(device = choosen_device , dtype = datatype)
        dist = distance(current, ref).numpy()[0]
        return  str(round(dist , 2))
  ###### End of distance()

  # Return a string of the similary % between
  # Tensor1 and Tensor2
  def similarity (self , tensor1 , tensor2):
        distance = torch.nn.PairwiseDistance(p=2)\
        .to(device = choosen_device , dtype = datatype)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)\
        .to(device = choosen_device , dtype = datatype)
        origin = (self.vector.origin)\
        .to(device = choosen_device , dtype = datatype)
        current = tensor1.to(device = choosen_device , dtype = datatype)
        dist1 = distance(current, origin).numpy()[0]
        current = current * (1/dist1)
        ref = tensor2.to(device = choosen_device , dtype = datatype)
        dist2 = distance(current, origin).numpy()[0]
        ref = ref * (1/dist2)
        sim = (100*cos(current , ref)).to(device = choosen_device , dtype = datatype)
        if sim < 0 : sim = -sim
        return  str(round(sim.numpy()[0] , 2))
  ###### End of similarity()

  # Get tensor from Data class at given index
  def get_vector(self , index , use_1280_dim = False):
    target = None
    if use_1280_dim: target = self.vector1280
    else:  target = self.vector
    return target.get(index).to(device = choosen_device , dtype = datatype)
  #######

  # Get ID from Data class at given index
  def get_ID(self , index , use_1280_dim = False):
    target = None
    if use_1280_dim: target = self.vector1280
    else:  target = self.vector
    return target.ID.get(index)
  ######

  # Get name from Data class at given index
  def get_name(self , index , use_1280_dim = False):
    target = None
    if use_1280_dim: target = self.vector1280
    else:  target = self.vector
    return target.name.get(index)
  #######

  #sample
  def update_loaded_embs(self):
    self.refresh()

  def refresh(self):
    log = []
    #Check if new embeddings have been added 
    try: 
      sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
      log.append('Reloading all embeddings')
    except:
      log.append("TokenMixer: Couldn't reload embedding database!") 
      sd_hijack.model_hijack.embedding_db.dir_mtime=0
      sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
    ##########
    assert self.tools.loaded , "Data : Checkpoint model was not loaded!"
    count = self.tools.count
    self.tools = Tools(count)
    return '\n'.join(log)
  ######### End of refresh()

  def clear (self, index , to_negative = None , to_mixer = None , \
  to_positive = None , to_temporary = None , use_1280_dim = False):

    if to_negative == None and to_mixer == None \
    and to_positive == None and to_temporary == None:

      self.vector.clear(index)
      self.negative.clear(index)
      self.positive.clear(index)
      self.temporary.clear(index)

      self.vector1280.clear(index)
      self.negative1280.clear(index)
      self.positive1280.clear(index)
      self.temporary1280.clear(index)

    else:

      if to_negative != None:
        if to_negative: 
          self.negative.clear(index)
          self.negative1280.clear(index)

      if to_mixer!= None:
        if to_mixer: 
          self.vector.clear(index)
          self.vector1280.clear(index)

      if to_temporary!= None:
        if to_temporary: 
          self.temporary.clear(index)
          self.temporary1280.clear(index)

      if to_positive!= None:
        if to_positive: 
          self.positive.clear(index)
          self.positive1280.clear(index)
  ########## End of clear()

  def set_unfiltered_indices(self, unfiltered_indices):
    self.vector.unfiltered_indices = copy.deepcopy(unfiltered_indices)
  ##### End of set_filtered_names()

  def set_filter_by_name(self, filter_by_name):
    self.vector.filter_by_name = copy.copy(filter_by_name)
  ###### End of set_filter_by_name()

  def __init__(self):

    #Check if new embeddings have been added 
    try: sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
    except: 
      sd_hijack.model_hijack.embedding_db.dir_mtime=0
      sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
      
    #Data parameteres
    Data.tools = Tools()

    # Check if a valid SD model is loaded (SD 1.5 , SD2 or SDXL)
    model_is_loaded = self.tools.model_is_loaded()
    is_sdxl , is_sd2 , is_sd1 = self.tools.get_flags()
    ######
    #Default values
    Data.vector = None
    Data.negative = None
    Data.positive = None
    Data.temporary = None
    Data.vector1280 = None
    Data.negative1280 = None
    Data.positive1280 = None
    Data.temporary1280 = None
    Data.pursuit_strength = 0
    Data.doping_strength = 0
    Data.operations = Operations()
    ########
    #Default initial values
    Data.vector = Vector(3)
    Data.negative = Negative(3)
    Data.positive = Positive(3)
    Data.temporary = Temporary(3)
    Data.vector1280 = Vector1280(3)
    Data.negative1280 = Negative1280(3)
    Data.positive1280 = Positive1280(3)
    Data.temporary1280 = Temporary1280(3)
    ########
    if model_is_loaded:
      emb_name, emb_id, emb_vec , loaded_emb = \
      self.get_embedding_info(',')
      size = emb_vec.shape[1]
      ####
      if is_sdxl:
        sdxl_emb_name, sdxl_emb_id, sdxl_emb_vec , sdxl_loaded_emb = \
        self.get_embedding_info(',', use_1280_dim = True)
        sdxl_size = sdxl_emb_vec.shape[1]
        self.vector1280 = Vector1280(sdxl_size)
        self.negative1280 = Negative1280(sdxl_size)
        self.positive1280 = Positive1280(sdxl_size)
        self.temporary1280 = Temporary1280(sdxl_size)
      ####
      self.vector = Vector(size)
      self.negative = Negative(size)
      self.positive = Positive(size)
      self.temporary = Temporary(size)
    #####
    
#End of Data class
dataStorage = Data() #Create data
