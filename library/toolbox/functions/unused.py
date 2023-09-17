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

  def leftovers (self):

    send_to_temporary = False
    tokenmixer_vectors = ''
    if not sendtomix : tokenmixer_vectors= mix_input

    # Vector length stuff
    distance = torch.nn.PairwiseDistance(p=2)
    origin = self.data.vector.origin\
    .to(device = choosen_device , dtype = datatype)
    origin1280 = self.data.vector1280.origin\
    .to(device = choosen_device , dtype = datatype)
    #######

    #Parameters
    section = None
    tokenbox = ''
    splitbox = ''
    negbox = ''
    posbox = ''
    emb_name = None
    found_IDs = None
    reading_word = False
    ID_index = None
    ##########
    start = 0
    end = MAX_NUM_MIX
    tmp = None
    numbers = None
    no_of_tokens = 0
    token_num = 0
    no_of_IDs = None
    emb_vecs = None
    emb_vec = None
    sdxl_emb_vec = None
    trailing_end_of_text = False

    ## SDXL stuff
    is_sdxl = self.data.tools.is_sdxl
    ####
    
    #Append start_of_text_ID to output before loop
    first_index = 0
    self.place (first_index,\
    send_to_negatives , sendtomix , send_to_positives , send_to_temporary , \
    start_of_text_ID) 
    ######

    placed = False
    offset = 1
    text_literal = ''
    ########## Start loop : 
    for k in range(MAX_NUM_MIX):
      index = k + offset
      if not index < MAX_NUM_MIX : break
      ######
      if placed : #Store the values from the previous iteration
        neg_name = self.data.negative.name.get(index-1)
        if neg_name != None :
          if (negbox != '') : negbox = negbox + ' , ' 
          negbox = negbox + neg_name
        #####
        pos_name = self.data.positive.name.get(index-1)
        if pos_name != None :
          if (posbox != '') : posbox = posbox + ' , ' 
          posbox = posbox + pos_name
        ######
        name = self.data.vector.name.get(index-1)
        if name != None and sendtomix:
          if tokenmixer_vectors != '': tokenmixer_vectors = tokenmixer_vectors + ' , '
          tokenmixer_vectors = tokenmixer_vectors + name
        ######
        placed = False
      #########

      if sendtomix and not self.data.vector.isEmpty.get(index): continue
      #Go word-for-word through the list of words
      if not word_index>0: break
      word = sentence[no_of_words-word_index]
      if word == "," :  
        word_index -=1 
        continue  #Ignore comma inputs
      ########

      # If word is '_' represent it as a random token
      if word == "_":
        emb_vec = torch.rand(self.data.vector.size).to(device = choosen_device , dtype = datatype)
        dist = distance(emb_vec , origin).numpy()[0]
        tmp = random_token_length * \
        (1 - random_token_length_randomization*random.random())
        emb_vec = (tmp/dist)*emb_vec
        #####
        if is_sdxl: 
          sdxl_emb_vec = torch.rand(self.data.vector1280.size)\
          .to(device = choosen_device , dtype = datatype)
          dist = distance(sdxl_emb_vec  , origin1280).numpy()[0]
          tmp = random_token_length * \
          (1 - random_token_length_randomization*random.random())
          sdxl_emb_vec  = (tmp/dist)*sdxl_emb_vec 
        #######
        emb_id = 0
        emb_name = "random_" + str(index)
      ###########
        placed = True
        self.data.place(index , 
            vector =  emb_vec.unsqueeze(0) ,
            ID =  emb_id ,
            name = emb_name , 
            to_negative = send_to_negatives ,
            to_mixer = sendtomix , 
            to_positive = send_to_positives , 
            to_temporary = send_to_temporary)

        if is_sdxl:
            placed = True
            self.data.place(index , 
              vector = sdxl_emb_vec.unsqueeze(0) ,
              ID = 0 ,
              name = emb_name + '_' + str(token_num) , 
              to_negative = send_to_negatives , 
              to_mixer = sendtomix , 
              to_positive = send_to_positives ,
              to_temporary = send_to_temporary , 
              use_1280_dim =True)
        ##########
        if (tokenbox != '') : tokenbox = tokenbox + ' , '
        tokenbox =  tokenbox + emb_name + '_#' + str(emb_id)
        word_index -=1 
        continue
      ########## End of the '_' random token stuff

      #Place a start-of-text token if running SDXL
      if word == "<" :
        if word_index < no_of_words - 1 :
          self.place (index  , \
            send_to_negatives , sendtomix , send_to_positives , send_to_temporary , \
            start_of_text_ID) 
          #####
          emb_name = "<|startoftext|>_#49406"
          if (tokenbox != '') : tokenbox = tokenbox + ' , '
          tokenbox =  tokenbox + emb_name
        ########
        word_index -=1 
        continue
      ####### End of the '<' start-of-text stuff

      #Place a end-of-text token if running SDXL
      if word == ">" :
        self.place (index  , \
          send_to_negatives , sendtomix , send_to_positives , send_to_temporary , \
          end_of_text_ID) 
        #####
        emb_name = "<|endoftext|>_#49407"
        if (tokenbox != '') : tokenbox = tokenbox + ' , '
        tokenbox =  tokenbox + emb_name
        word_index -=1 
        trailing_end_of_text = True
        continue
      else :  trailing_end_of_text = False
      #######End of the '>' end-of-text stuff

      #Extract emb_vec from emb_id if id_mode is selected
      if (id_mode and word.isdigit()):
        emb_id = int(word)
        if emb_id >= no_of_internal_embs: continue
        emb_vec = self.data.tools.internal_embs[emb_id]\
        .to(device = choosen_device , dtype = datatype)
        if is_sdxl: sdxl_emb_vec = self.data.tools.internal_embs1280[emb_id]\
        .to(device = choosen_device , dtype = datatype)
        emb_name = self.data.emb_id_to_name(emb_id)
        ######
        assert emb_vec != None , "emb_vec is NoneType"
        assert not placed , "Overwrite error!"
        placed = True
        self.data.place(index , 
            vector =  emb_vec.unsqueeze(0) ,
            ID =  emb_id  ,
            name = emb_name , 
            to_negative = send_to_negatives , 
            to_mixer = sendtomix , 
            to_positive = send_to_positives , 
            to_temporary = send_to_temporary)
        ######
        if is_sdxl:
            placed = True
            self.data.place(index , 
              vector = sdxl_emb_vec.unsqueeze(0) ,
              ID = 0 ,
              name = emb_name + '_' + str(token_num) , 
              to_negative = send_to_negatives , 
              to_mixer = sendtomix , 
              to_positive = send_to_positives ,
              to_temporary = send_to_temporary , 
              use_1280_dim =True)
        ###########
        if (tokenbox != '') : tokenbox = tokenbox + ' , '
        tokenbox =  tokenbox + emb_name + '_#' + str(emb_id)
        word_index -=1 
        continue
      ######### End of id_mode stuff

      #Find which section of embedding vectors to 
      #add to the output if the user has written [n:m] in
      #the mini_input
      ##########
      tmp = word.strip().lower()
      if (word.find('[')>=0 and word.find(']')>=0 and (word.find('[]')<0)):
          tmp =  word.split('[')[0]
          tmp = tmp.strip().lower()
          section = word.split('[')[1]
          section = section.split(']')[0]
          numbers = [int(num) for num in re.findall(r'\d+', section)]
          if (len(numbers)>1):
            start = numbers[0]
            end = numbers[1]
      ##########
      #self.data.get
      #####
      emb_name, emb_ids, emb_vecs , loaded_emb  = self.data.get_embedding_info(tmp)
      ###
      sdxl_emb_name = None
      sdxl_emb_ids = None
      sdxl_emb_vecs = None 
      sdxl_loaded_emb = None
      if is_sdxl:
        sdxl_emb_name, sdxl_emb_ids, sdxl_emb_vecs , sdxl_loaded_emb  = \
        self.data.get_embedding_info(tmp , use_1280_dim =True)
      ######## End of the [n:m] in mini_input stuff

      no_of_tokens = emb_vecs.shape[0]
      if no_of_tokens > MAX_NUM_MIX : no_of_tokens = MAX_NUM_MIX
      if tmp == None : end = no_of_tokens

      #If we read an embedding in 'literal mode'
      #we discard the input and interpret the embedding 
      #name as a CLIP token
      if literal_mode :
        no_of_tokens = 0 
        emb_vecs = []
        for emb_id in emb_ids:
          emb_vec = self.data.emb_id_to_vec(emb_id)\
          .to(device = choosen_device , dtype = datatype)
          if is_sdxl: sdxl_emb_vec = \
          self.data.emb_id_to_vec(emb_id , use_1280_dim =True)\
          .to(device = choosen_device , dtype = datatype)
          no_of_tokens +=1
          break
      ########## End of 'literal mode' stuff

      # 'Normal operation'
      if no_of_tokens > 1 :

        # If we looped through single tokens then process them
        if text_literal != '': 
          offset , tokenbox = self.process (\
          text_literal , k , offset , \
          sendtomix , send_to_positives , send_to_negatives , send_to_temporary)
          index = k + offset
          text_literal = ''
        ########



        #If embedding contains multiple tokens
          if (token_num+1>min(end, no_of_tokens)) or (start>end) :
            token_num = 0  #Reset token_num
            word_index -=1 #Go to next word
            continue
          if (token_num<start):
            token_num += 1 #Skip until token_num==start
            continue
          #Fetch the vector
          emb_vec = emb_vecs[token_num].to(device = choosen_device , dtype = datatype)
          assert emb_vec != None , "emb_vec is NoneType"
          ####
          if is_sdxl: 
            sdxl_emb_vec = sdxl_emb_vecs[token_num]\
            .to(device = choosen_device , dtype = datatype)
            assert sdxl_emb_vec != None , "sdxl_emb_vec is NoneType"
          ######
          assert not placed , "Overwrite error!"
          placed = True
          self.data.place(index , 
              vector = emb_vec.unsqueeze(0) ,
              ID = 0 ,
              name = emb_name + '_' + str(token_num) , 
              to_negative = send_to_negatives , 
              to_mixer = sendtomix , 
              to_positive = send_to_positives ,
              to_temporary = send_to_temporary)
          ######
          if is_sdxl:
            placed = True
            self.data.place(index , 
              vector = sdxl_emb_vec.unsqueeze(0) ,
              ID = 0 ,
              name = emb_name + '_' + str(token_num) , 
              to_negative = send_to_negatives , 
              to_mixer = sendtomix , 
              to_positive = send_to_positives ,
              to_temporary = send_to_temporary , 
              use_1280_dim =True)
          #######
          if (splitbox != '') : splitbox = splitbox + ' , '    
          splitbox =  splitbox + emb_name + '_' + str(token_num)
          token_num += 1
      ###########
      else:
        #If embedding is single token
        if text_literal != '': text_literal = text_literal + ' '
        text_literal = text_literal + word
        word_index -=1 
        continue
      #### End of 'Normal operation'
    ####### End main loop

    # If we looped through single tokens then process them
    if text_literal != '': 
          offset , tokenbox = self.process (\
          text_literal , k , offset , \
          sendtomix , send_to_positives , send_to_negatives , send_to_temporary)
          index = k + offset
          text_literal = ''
    ############

    
    # Find the first empty slot in the list
    # after the loop (the last index to fill)
    last_index = None
    for index in range(MAX_NUM_MIX):
      if trailing_end_of_text : break
      if sendtomix and not self.data.vector.isEmpty.get(index): continue
      last_index = copy.copy(index)
    ########

    # Append end_of_text_ID token to output after loop
    # if none were placed at the end
    #if not trailing_end_of_text:
      #self.place (last_index  , \
      #send_to_negatives , sendtomix , send_to_positives , send_to_temporary , \
      #end_of_text_ID) 
      if (tokenbox != '') : tokenbox = tokenbox + ' , '
      tokenbox =  tokenbox + "<|endoftext|>_#49407"
    ###### End of append end_of_text_ID to output

    #Filter stuff
    name_list = []
    for index in range(MAX_NUM_MIX):
        if self.data.vector.isEmpty.get(index): continue
        name_list.append(self.data.vector.name.get(index))
    ######### End of filter stuff






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

