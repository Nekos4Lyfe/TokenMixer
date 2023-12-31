import torch , math, random , numpy , copy
from library.toolbox.constants import MAX_NUM_MIX
from library.toolbox.constants import TENSOR_DEVICE_TYPE , TENSOR_DATA_TYPE
choosen_device = TENSOR_DEVICE_TYPE
datatype = TENSOR_DATA_TYPE


# The Operations class is used to store 
# helper functions to the Data.py class
class TextFunctions:

##### ALL OF THESE ARE TO BE DELETED ####

  def emb_id_to_vec(self, emb_id , use_1280_dim = False) : 
    target = None
    index_max = None
    if use_1280_dim: 
      target = self.internal_embs1280
      index_max = self.no_of_internal_embs1280
    else : 
      target = self.internal_embs
      index_max = self.no_of_internal_embs
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
    emb_name_utf8 = self.tokenizer.decoder.get(emb_id)
    if emb_name_utf8 != None:
        byte_array_utf8 = bytearray([self.tokenizer.byte_decoder[c] for c in emb_name_utf8])
        emb_name = byte_array_utf8.decode("utf-8", errors='backslashreplace')
    else: emb_name = '!Unknown ID!'
    if emb_name.find('</w>')>=0: 
      emb_name = emb_name.split('</w>')[0]
    return emb_name # return embedding name for embedding ID

  def get_embedding_info(self, string , use_1280_dim = False):
      emb_id = None
      text = copy.copy(string.lower())
      loaded_emb = self.loaded_embs.get(text, None)
      if loaded_emb == None:
        for neg_index in self.loaded_embs.keys():
            if text == neg_index.lower():
              loaded_emb = self.loaded_embs.get(neg_index, None)
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

        if use_1280_dim : emb_vec = self.internal_embs1280[emb_id].unsqueeze(0)
        else : emb_vec = self.internal_embs[emb_id].unsqueeze(0)
        emb_vecs.append(emb_vec)
      ##########
    
      #Might have to fix later , idk
      if emb_name == None : return None, None, None, None
      emb_ids = emb_ids[0]
      emb_names = emb_names[0]
      emb_vecs = emb_vecs[0]
      ##########
      return emb_names, emb_ids, emb_vecs, None # return embedding name, ID, vector

  def text_to_emb_ids(self, text) :
    text = copy.copy(text.lower())
    emb_ids = None
    if self.is_sdxl : #SDXL detected
        emb_ids = self.tokenizer.encode(text)
    elif self.is_sd1 : # SD1.x detected
        emb_ids = self.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    elif self.is_sd2 : # SD2.0 detected
        emb_ids =  self.tokenizer.encode(text)
    else: emb_ids = None
    return emb_ids # return list of embedding IDs for text
##################

  def __init__(self , \
  internal_embs , internal_embs1280 , \
  no_of_internal_embs , no_of_internal_embs1280 , \
  tokenizer , loaded_embs , \
  is_sdxl ,is_sd2 ,  is_sd1):

    self.internal_embs = internal_embs
    self.internal_embs1280 = internal_embs1280
    self.no_of_internal_embs = no_of_internal_embs
    self.no_of_internal_embs1280 = no_of_internal_embs1280
    self.tokenizer = tokenizer
    self.loaded_embs = loaded_embs
    self.is_sdxl = is_sdxl
    self.is_sd2 = is_sd2
    self.is_sd1 = is_sd1
