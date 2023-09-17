#UNUSED.PY
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

  def process (self , text_literal , k_value , offset_input , \
  sendtomix , send_to_positives , send_to_negatives , send_to_temporary):

      k = k_value
      offset = offset_input
      tokenbox = ''
      index = k + offset
      is_sdxl = self.data.tools.is_sdxl
      
      #DIMENSION 768 INIT
      found_IDs = self.data.tools.get_emb_ids_from(text_literal).numpy()
      found_vecs768 = self.data.tools.get_emb_vecs_from(text_literal)
      no_of_IDs = len(found_IDs)
      assert no_of_IDs - 2 == found_vecs768.shape[0] , \
      "Size mismatch between found vecs and found IDs! , " + \
      "found_vecs768.shape[0] = " + str(found_vecs768.shape[0]) + \
      " and len(found_IDs) minus 2 cutoff tokens = " + str(no_of_IDs - 2)
      tensor_index = 0
      placed_vecs768 = 0
      emb_vec768 = None

      #DIMENSION 768 LOOP
      for _ID in found_IDs:
            if no_of_IDs <= 2 : break
            emb_vec768 = found_vecs768[tensor_index].to(device = choosen_device) 
            tensor_index = tensor_index + 1
            if self.isCutoff(emb_vec768): continue
            emb_name = self.data.emb_id_to_name(_ID)
            emb_id = _ID
            #######
            self.data.place(index , 
            vector =  emb_vec768.unsqueeze(0) ,
            ID =  _ID ,
            name = emb_name ,
            to_negative = send_to_negatives ,
            to_mixer = sendtomix , 
            to_positive = send_to_positives , 
            to_temporary = send_to_temporary)
            #######
            offset = offset + 1
            index = k + offset
            placed_vecs768 = placed_vecs768 + 1
            ######
            name = self.data.vector.name.get(index)
            if (tokenbox != '') : tokenbox = tokenbox + ' , '
            tokenbox =  tokenbox + name + '_#' + str(_ID)
            #######
            neg_name = self.data.negative.name.get(index)
            if neg_name != None :
              if (negbox != '') : negbox = negbox + ' , ' 
              negbox = negbox + neg_name
            #####
            pos_name = self.data.positive.name.get(index)
            if pos_name != None :
              if (posbox != '') : posbox = posbox + ' , ' 
              posbox = posbox + pos_name
          ####### End of Loop
        ###### End 768 Dimension stuff

      if is_sdxl :
          #SDXL DIMENSION 1280 INIT
          found_vecs1280 = None
          found_IDs1280 = None
          no_of_IDs1280 = None
          found_IDs1280 = self.data.tools.\
          get_emb_ids_from(text_literal , use_1280_dim = True).numpy()
          found_vecs1280 = self.data.tools\
          .get_emb_vecs_from(text_literal , use_1280_dim = True) 
          no_of_IDs1280 = len(found_IDs1280)
          assert no_of_IDs1280 -2 == found_vecs1280.shape[0] , \
          "Size mismatch between found vecs1280 and found IDs! , " + \
          "found_vecs1280.shape[0] = " + str(found_vecs1280.shape[0]) + \
          " and len(found_IDs1280) minus 2 cutoff tokens = " + str(no_of_IDs1280 - 2)
          tensor_index = 0
          placed_vecs1280 = 0
          emb_vec1280 = None

          #SDXL DIMENSION 1280 LOOP
          from pprint import pprint
          for _ID in found_IDs1280:
            if no_of_IDs1280 <= 2 : break
            if placed_vecs1280 == placed_vecs768 : break
            emb_vec1280 = found_vecs1280[tensor_index].to(device = choosen_device) 
            tensor_index = tensor_index + 1
            if self.isCutoff(emb_vec1280): continue
            emb_name = self.data.emb_id_to_name(_ID)
            ########
            self.data.place(index , 
            vector =  emb_vec1280.unsqueeze(0) ,
            ID =  _ID ,
            name = emb_name ,
            to_negative = send_to_negatives ,
            to_mixer = sendtomix , 
            to_positive = send_to_positives , 
            to_temporary = send_to_temporary , 
            use_1280_dim = True)
            #######
            placed_vecs1280 = placed_vecs1280 + 1
            offset = offset + 1
            index = k + offset
          ##### End of 1280 Dimension Loop
      ######## End of SDXL Stuff
      return offset , tokenbox
  ##### End of process()
  

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

