import torch , math, random , numpy , copy
from library.toolbox.constants import MAX_NUM_MIX
from library.toolbox.constants import TENSOR_DEVICE_TYPE , TENSOR_DATA_TYPE
choosen_device = TENSOR_DEVICE_TYPE
datatype = TENSOR_DATA_TYPE

from library.toolbox.constants import START_OF_TEXT_ID , END_OF_TEXT_ID
start_of_text_ID = START_OF_TEXT_ID
end_of_text_ID = END_OF_TEXT_ID

# The Unused class is used to store 
# stuff that is unused now but might
# be good to have later
class Unused:

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
    

 # If the approximate distance if equal
  # to the distance of either of the cutoff tokens
  # then return True
  def isCutoffVec(self , emb_vec):
    origin768 = self.data.vector.origin
    origin1280 = self.data.vector1280.origin
    start_of_text_vec768 = self.emb_id_to_vec(start_of_text_ID)
    end_of_text_vec768 = self.emb_id_to_vec(end_of_text_ID)
    start_of_text_vec1280 = self.emb_id_to_vec(start_of_text_ID , use_1280_dim = True)
    end_of_text_vec1280 = self.emb_id_to_vec(end_of_text_ID , use_1280_dim = True)

    #These are distance of the cutoff tokens 
    #rounded to 2 decimals convertet to a string
    self.start_of_text_dist768 = self.data.distance(start_of_text_vec768 , origin768)
    self.end_of_text_dist768 = self.data.distance(end_of_text_vec768 , origin768)  
    self.start_of_text_dist1280 = self.data.distance(start_of_text_vec1280 , origin1280)
    self.end_of_text_dist1280 = self.data.distance(end_of_text_vec1280 , origin1280) 
    
    #######
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

  def __init__(self):
    pass
