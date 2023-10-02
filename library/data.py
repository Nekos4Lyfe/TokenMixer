#DATA.PY
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

  # Randomize the order of the tensors in the 
  #given field(s) of the Data class
  def shuffle(self, index , vector = None , ID = None ,  name = None , \
    weight = None , to_negative = None , to_mixer = None ,  \
    to_positive = None ,  to_temporary = None , use_1280_dim = False):

    self.operations.using_tensors._shuffle(\
      self.vector , self.positive , self.negative , self.temporary ,  \
      self.vector1280 , self.positive1280 , self.negative1280 , \
      self.temporary1280 , \
      to_negative , to_mixer,  to_positive ,  to_temporary , use_1280_dim)
  ######### End of place()

  # Replace a fraction of each tensor with a random vector 
  # within given field(s) in Data class
  def roll (self , to_negative = None , to_mixer = None , \
    to_positive = None, to_temporary = None, use_1280_dim = False):
    log = []
    message = \
    self.operations.using_tensors._roll(\
      self.vector , self.positive , self.negative , self.temporary ,  \
      self.vector1280 , self.positive1280 , self.negative1280 , \
      self.temporary1280 , \
      to_negative , to_mixer , to_positive , to_temporary , use_1280_dim)
    log.append(message)
    return '\n'.join(log)
  ### End of sample()

  def random(self , use_1280_dim = False):
    if use_1280_dim : return self.vector1280.random(self.tools.internal_embs1280)
    else : return self.vector.random(self.tools.internal_embs)
  ### End of random()

  def random_quick(self):
    return self.vector.random_quick()
  ## End of random_quick()

  # Replace a fraction of each tensor with a random vector 
  # within given field(s) in Data class
  def sample (self , to_negative = None , to_mixer = None , \
    to_positive = None, to_temporary = None, use_1280_dim = False):
    log = []
    message = \
    self.operations.using_tensors._sample(\
      self.vector , self.positive , self.negative , self.temporary ,  \
      self.vector1280 , self.positive1280 , self.negative1280 , \
      self.temporary1280 , \
      self.internal_embs , self.internal_embs1280 , \
      to_negative , to_mixer , to_positive , to_temporary , use_1280_dim)
    log.append(message)
    return '\n'.join(log)
  ### End of sample()

  # Place the input at assigned position(s) in the Data class
  def place(self, index , vector = None , ID = None ,  name = None , \
    weight = None , to_negative = None , to_mixer = None ,  \
    to_positive = None ,  to_temporary = None , use_1280_dim = False):

    self.operations.using_tensors._place(\
      self.vector , self.positive , self.negative , self.temporary ,  \
      self.vector1280 , self.positive1280 , self.negative1280 , \
      self.temporary1280 , \
      index , vector , ID , name , weight , \
      to_negative , to_mixer,  to_positive ,  to_temporary , use_1280_dim)
  ######### End of place()

  def memorize(self):
    target = self.vector
    destination = self.temporary
    self.operations.using_tensors._move(\
    target , destination)
    ########
    if self.tools.is_sdxl:
      target = self.vector1280
      destination = self.temporary1280
      self.operations.using_tensors._move(\
      target , destination)
  #### End of memorize()

  def recall(self):
    target = self.temporary
    destination = self.vector
    self.operations.using_tensors._move(\
    target , destination)
    ######
    if self.tools.is_sdxl:
      target = self.temporary1280
      destination = self.vector1280
      self.operations.using_tensors._move(\
      target , destination)
  #### End of recall()

  def distance (self, tensor1 , tensor2):
    return self.operations.using_tensors._distance(\
    tensor1 , tensor2)
  ### End of distance()

  def similarity(self , tensor1 , tensor2):
    return self.operations.using_tensors._similarity(\
    tensor1 , tensor2 , self.vector)
  #### End of similarity()

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

  # Place the input at assigned position(s) in the Dataclass
  def clear(self, index , vector = None , ID = None ,  name = None , \
    weight = None , to_negative = None , to_mixer = None ,  \
    to_positive = None ,  to_temporary = None , use_1280_dim = False):

    self.operations.using_tensors._clear(\
      self.vector , self.positive , self.negative , self.temporary ,  \
      self.vector1280 , self.positive1280 , self.negative1280 , self.temporary1280 , \
      index ,\
      to_negative , to_mixer,  to_positive ,  to_temporary , use_1280_dim)    
  #########

  # If the approximate distance if equal
  # to the distance of either of the cutoff tokens
  # then return True
  def isCutoff(self , emb_vec):
    size = emb_vec.shape[1]
    size1280 =  self.vector1280.size
    use_1280_dim = (size ==  size1280) 
    #######
    origin = None
    if use_1280_dim: origin = self.data.vector1280.origin
    else : origin = self.data.vector.origin
    ########
    dist = self.data.distance(emb_vec , origin)
    #######
    sot_dist = None
    if use_1280_dim : sot_dist = self.start_of_text_dist1280
    else: sot_dist = self.start_of_text_dist768
    if dist == sot_dist : return True
    #######
    eot_dist = None
    if use_1280_dim : sot_dist = self.end_of_text_dist1280
    else: sot_dist = self.end_of_text_dist768
    if dist == eot_dist : return True
    #######
    return False
  #############

  
    return (dist == sot_dist or dist == eot_dist)

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
    ######
    Data.operations = Operations(
    self.tools.internal_embs , self.tools.internal_embs1280 , \
    self.tools.no_of_internal_embs , self.tools.no_of_internal_embs1280 , \
    self.tools.tokenizer , self.tools.loaded_embs , \
    is_sdxl ,is_sd2 ,  is_sd1)
    #####
    #These will be deleted 
    Data.text_to_emb_ids = self.operations.using_text.text_to_emb_ids
    Data.emb_id_to_vec = self.operations.using_text.emb_id_to_vec 
    Data.emb_id_to_name = self.operations.using_text.emb_id_to_name
    Data.get_embedding_info = self.operations.using_text.get_embedding_info
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
      emb_vec = self.tools.process(',' , to = 'tensors')
      size = emb_vec.shape[1]
      ####
      if is_sdxl:
        sdxl_emb_vec = self.tools.process(',' , to = 'tensors', use_1280_dim = True)
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

      from pprint import pprint
      pprint("1280 DIMENSION:")
      emb_ids1280 = self.tools.process("girl on beach" , to = 'ids' , use_1280_dim = True)
      emb_vecs1280 = self.tools.process("girl on beach" , use_1280_dim = True)
      pprint(emb_vecs1280.shape)
      pprint(emb_ids1280)
      pprint("768 DIMENSION:")
      emb_ids768 = self.tools.process("girl on beach" , to = 'ids')
      emb_vecs768 = self.tools.process("girl on beach")
      pprint(emb_vecs768.shape)
      pprint(emb_ids768)

#End of Data class
dataStorage = Data() #Create data

