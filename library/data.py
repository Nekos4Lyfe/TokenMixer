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
from library.toolbox.operations import TensorOperations

from library.toolbox.constants import TENSOR_DEVICE_TYPE , TENSOR_DATA_TYPE
choosen_device = TENSOR_DEVICE_TYPE
datatype = TENSOR_DATA_TYPE
#-------------------------------------------------------------------------------

# Check that MPS is available (for MAC users)
#if torch.backends.mps.is_available(): 
#  choosen_device = torch.device("mps")
#else : choosen_device = torch.device("cpu")
######

class Data :
  #The Data class consists of all the shared resources between the modules
  #This includes the vector class and the tools class

  def merge_all (self, similar_mode):
    log = []
    log.append('Merge mode :')
    current = None
    output = None
    vsum = None
    dsum = 0
    no_of_tokens = 0

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    distance = torch.nn.PairwiseDistance(p=2)
    #shuffle
    #Count the negatives
    no_of_negatives = 0
    for neg_index in range(MAX_NUM_MIX):
      if self.negative.isEmpty.get(neg_index): continue
      no_of_negatives += 1
    #####

    dist = None
    candidate_vec = None
    tmp = None
    origin = self.vector.origin

    for index in range (MAX_NUM_MIX):
      if (self.vector.isEmpty.get(index)): continue
      if (self.vector.isFiltered(index)): continue 
      no_of_tokens += 1

      if similar_mode : 
        current , message = self.replace_with_similar(index)
        log.append(message)
      else : current = self.vector.get(index).to(device = choosen_device , dtype = torch.float32)

      current_weight = 1
      #self.vector.weight.get(index)
      current = current*current_weight

      dist = distance(current, origin).numpy()[0]
      dsum = dsum + dist

      if vsum == None : vsum = current
      else : vsum = vsum + current

    if vsum == None : 
      log.append('No vectors to merge!')
      return output , '\n'.join(log)

    tensor = vsum/no_of_tokens #Expected average of vector
    dist_of_mean = distance(tensor, origin).numpy()[0]
    norm_expected = tensor * (1/dist_of_mean) #Normalized expected average of vectors
    dist_expected = dsum/no_of_tokens # average length on inputs

    radialGain = 0.001 #deprecated
    randomGain = self.vector.randomization/100  
    size = self.vector.size
    gain = self.vector.gain
    N = self.vector.itermax
    T = 1/N  

    radialRandom = 1

    similarity_score = 0 #similarity index from 0 to 1, where 0 is no similarity at all
    similarity_minimum = None #similarity between output and the least similar input token
    similarity_maximum = None #similarity between output and the most similar input token
    mintoken = None #Token with least similarity
    maxtoken = None #Token with most similarity

    for step in range (N):
      rand_vec  = ((2*torch.rand(size) - torch.ones(size))\
      .to(device = choosen_device , dtype = torch.float32)).unsqueeze(0)
      rdist = distance(rand_vec, origin).numpy()[0]
      rando = (1/rdist) * rand_vec #Create random unit vector

      candidate_vec = norm_expected * (1 - randomGain)  + rando*randomGain

      minimum = 20000 #set initial minimum at impossible 200% similarity 
      maximum = -10000 #Set initial maximum at impossible -100% similarity
      similarity_sum = 0
      for index in range (MAX_NUM_MIX):
        if (self.vector.isEmpty.get(index)): continue
        if (self.vector.isFiltered(index)): continue

        #Get current token
        tmp = self.vector.get(index).to(device = choosen_device , dtype = torch.float32)
        dist = distance(tmp, origin).numpy()[0]
        current = (1/dist) * tmp 

        #Compute similarity of current token to 
        #candidate_vec
        similarity = 100*cos(current, candidate_vec).numpy()[0]
        worst_nsim = None
        worst_neg_index = None
        nsim = None
        neg_vec = None
        strength = self.negative.strength/100
        negative_similarity = None
        #######
        #Check negatives
        if no_of_negatives > 0: 
          tmp = None
          for neg_index in range(MAX_NUM_MIX) :
            if self.negative.isEmpty.get(neg_index): continue
            neg_vec = self.negative.get(neg_index)
            tmp = math.floor((100*100*cos(neg_vec, candidate_vec)).numpy()[0])
            if tmp<0 : tmp = -tmp
            nsim = tmp/100
            #######
            if worst_nsim == None : 
              worst_nsim = nsim
              worst_neg_index = neg_index
            #######
            if nsim > worst_nsim : 
              worst_nsim = nsim
              worst_neg_index = neg_index
          ########
          negative_similarity = 100 - worst_nsim
        ########### #Done checking negatives
        

        if similarity == None : similarity = 0
        if negative_similarity != None : 
          similarity_sum += similarity * (1 - strength) + negative_similarity*strength
        else: similarity_sum += similarity
        
        if similarity == min(minimum , similarity) :
          minimum = similarity
          mintoken = index

        if similarity == max(maximum , similarity) :
          maximum = similarity
          maxtoken = index
   
      #Check if this candidate vector is better then
      #the ones before it
      if similarity_sum>similarity_score : 
        similarity_score = similarity_sum
        similarity_minimum = minimum
        similarity_maximum = maximum
        output = candidate_vec

    #length of candidate vector
    cdist = distance(candidate_vec , origin).numpy()[0]

    #length of output vector
    output_length = gain * dist_expected * radialRandom 

    #Cap the length of the output vector according to radialGain slider setting 
    if output_length>dist_expected: output_length = min(output_length , dist_expected/radialGain)
    else: output_length = max(output_length  , dist_expected*radialGain)

    #Normalize candidate vector and multiply with output vector length
    output = candidate_vec *(output_length/cdist)
    output_length = round(output_length, 2)

    #round the values before printing them
    similarity_minimum = round(similarity_minimum,1)
    similarity_maximum = round(similarity_maximum,1)
    dist_expected = round(dist_expected, 2)
    dist_of_mean = round(dist_of_mean , 2)

    negsim = None
    if negative_similarity != None :
      negsim = round(100-negative_similarity , 3) #Invert the value
    
    log.append ('Merged ' + str(no_of_tokens) + ' tokens into single token')
    log.append('Lowest similarity : ' + str(similarity_minimum)+' % to token #' + str(mintoken) + \
      '( ' + self.vector.name.get(mintoken) + ')')

    log.append('Highest similarity : ' + str(similarity_maximum)+' % to token #' + str(maxtoken) + \
      '( ' + self.vector.name.get(maxtoken) + ')')

    if negsim != None :
      name = self.negative.name.get(worst_neg_index)
      log.append('Max similarity to negative tokens : ' + str(negsim) + ' %' + \
      "to token '" + name + "'")

    log.append('Average length of the input tokens : ' + str(dist_expected) )
    log.append('Length of the token average : ' + str(dist_of_mean) )
    log.append('Length of generated merged token : ' + str (output_length))
    log.append('New embedding has 1 token')
    log.append('-------------------------------------------')
  
    return output , '\n'.join(log)
  #End of merge-all function

  def concat_all(self , similar_mode , is_sdxl = False):
    target = None
    if is_sdxl : target = self.vector1280
    else : target = self.vector
    assert target != None , "Fetched vector is NoneType!"
    return self.operations._concat_all(target , similar_mode, is_sdxl)
  ######

  def replace_with_similar(self , index , is_sdxl = False):

    pursuit_value = self.pursuit_strength/100
    doping_value = self.doping_strength/100

    if is_sdxl: 
      return self.operations._replace_with_similar(
        index , 
        self.vector1280, 
        self.positive1280 , 
        self.negative1280 , 
        pursuit_value , 
        doping_value)
    ########

    if not is_sdxl:
      return self._replace_with_similar(
        index , 
        self.vector, 
        self.positive , 
        self.negative , 
        pursuit_value , 
        doping_value)
  ###### End of replace_with_similar

  

  def merge_if_similar(self, similar_mode):
    log = []
    log.append('Interpolate Mode :')
    no_of_tokens = 0

    for index in range (MAX_NUM_MIX):
      if (self.vector.isEmpty.get(index)): continue
      if (self.vector.isFiltered(index)): continue
      no_of_tokens +=1
      continue

    if no_of_tokens <= 0: 
      log.append('No inputs in mixer')
      return None , '\n'.join(log) 

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    distance = torch.nn.PairwiseDistance(p=2)

    radialGain = 0.001 #deprecated
    randomGain = self.vector.randomization/100 
    lowerBound = self.vector.interpolation 
    origin = self.vector.origin.to(device="cpu" , dtype = torch.float32)
    size = self.vector.size
    gain = self.vector.gain
    N = self.vector.itermax
    current_dist = None
    current_norm = None
    similarity = None
    merge_norm = None
    prev_dist = None
    prev_norm = None
    current = None
    avg_dist=None
    output = None
    prev = None
    tmp = None
    T = 1/N  
    iters = 0
    found = False 
    tensor = None
    message = None
    output_length = None
    current_weight = None

    for index in range (MAX_NUM_MIX):
      if (self.vector.isEmpty.get(index)): continue
      if (self.vector.isFiltered(index)): continue

      if similar_mode : 
        current , message = self.replace_with_similar(index)
        log.append(message)
      else : current = self.vector.get(index).to(device = choosen_device , dtype = torch.float32)

      current_weight = 1
      #self.vector.weight.get(index)
      current = current*current_weight

      if prev == None : 
        prev = current
        continue

      assert not current == None , "current tensor is None!"
      assert not prev == None , "prev tensor is None!"

      tmp = ((current + prev) * 0.5).to(device = choosen_device , dtype = torch.float32) #Expected merge vector
      avg_dist = distance(tmp, origin).numpy()[0]
      merge_norm = ((1/avg_dist)*tmp).to(device = choosen_device , dtype = torch.float32)

      current_dist = distance(current, origin).numpy()[0]
      current_norm = ((1/current_dist)*current).to(device = choosen_device , dtype = torch.float32) 

      prev_dist = distance(prev, origin).numpy()[0]
      prev_norm = ((1/prev_dist)*prev).to(device = choosen_device , dtype = torch.float32)

      assert not ( 
        merge_norm == None or
        current_norm==None or
        prev_norm == None ) , "norm tensor is None!"

      #Randomization parameters: 
      radialRandom = (1 - randomGain) + randomGain*(2*random.random() - 1)

      if not (self.vector.allow_negative_gain): 
        if radialRandom<0: radialRandom = -radialRandom

      found = False 
      for step in range (N):
        iters+=1

        tmp  = ((torch.rand(size) - 0.5* torch.ones(size))\
        .to(device = choosen_device , dtype = torch.float32)).unsqueeze(0)
        rdist = distance(tmp, origin).numpy()[0]
        rand_norm = (1/rdist) * tmp  #Create random unit vector

        cand = (step * merge_norm + (N - step)* rand_norm) * T 
        cdist = distance(cand, origin).numpy()[0]
        candidate_norm = (1/cdist)*cand #Calculate candidate vector

        sim1 =  (cos(current_norm, candidate_norm)).numpy()[0]
        sim2 =  (cos(prev, candidate_norm)).numpy()[0]
        dist_expected = sim1*current_dist + sim2*prev_dist

        similarity = min(sim1, sim2) #Get lowest similarity
        if (100*similarity > lowerBound) : 
          found = True 
          prev = None
          break
   
      #round the values before printing them
      similarity = round(100*similarity, 1)
      lowerBound = round(lowerBound, 1)

      if(found):

        #length of output vector
        output_length = gain * dist_expected * radialRandom 

        #Cap the length of the output vector according to radialGain slider setting 
        if output_length>dist_expected: output_length = min(output_length , dist_expected/radialGain)
        else: output_length = max(output_length  , dist_expected*radialGain)

        #Set the length of found similar vector
        similar_token = output_length*candidate_norm

        #round the values before printing them
        output_length = round(output_length, 2)
        dist_expected = round(dist_expected , 2)

        log.append('Token ' + str(index-1) + ' and '+str(index)+ \
        ' with expected length '+ str(dist_expected) + ' merged into 1 token with ' + \
         str(similarity)+ '% similarity and ' + str(output_length) + ' length after ' + \
        str(iters) + ' steps)')
        no_of_tokens = no_of_tokens - 1

        if output == None : output = similar_token
        else :  output = torch.cat([output,similar_token], dim=0)\
        .to(device = choosen_device , dtype = torch.float32)
        log.append('Placed token with length '+ str(output_length)  +' in new embedding ')

      else:
        log.append('Skipping merge between token ' + str(index-1)+ \
        ' and '+str(index))
        log.append('Token similarity ' + str(similarity) + \
        '% is less then req. similarity' + str(lowerBound) + '%')

    log.append('New embedding now has ' + str(no_of_tokens) + ' tokens')
    log.append('-------------------------------------------')
    return output , '\n'.join(log)

  def text_to_emb_ids(self, text):
    text = copy.copy(text.lower())
    emb_ids = None
    if self.tools.is_sdxl : #SDXL detected
        emb_ids = self.tools.tokenizer.encode(text)
    elif self.tools.tokenizer.__class__.__name__== 'CLIPTokenizer': # SD1.x detected
        emb_ids = self.tools.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    elif self.tools.tokenizer.__class__.__name__== 'SimpleTokenizer': # SD2.0 detected
        emb_ids =  self.tools.tokenizer.encode(text)
    else: emb_ids = None

    return emb_ids # return list of embedding IDs for text

  def emb_id_to_vec(self, emb_id , is_sdxl = False):
    if not isinstance(emb_id, int):
      return None
    if (emb_id > self.tools.no_of_internal_embs):
      return None
    if emb_id < 0 :
      return None
    if is_sdxl: return self.tools.internal_embs1280[emb_id].to(device="cpu" , dtype = torch.float32)
    else : return self.tools.internal_embs[emb_id].to(device="cpu" , dtype = torch.float32)

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

  def get_embedding_info(self, string , is_sdxl = False):

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
        if is_sdxl : 
          emb_id = 'unknown' #<<<< Will figure this out later
          emb_vec = loaded_emb.vec.get("clip_l")\
          .to(device = choosen_device , dtype = torch.float32)
        else: 
          emb_id = '['+ loaded_emb.checksum()+']' # emb_id is string for loaded embeddings
          emb_vec = loaded_emb.vec\
          .to(device = choosen_device , dtype = torch.float32)
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

        if is_sdxl : emb_vec = self.tools.internal_embs1280[emb_id].unsqueeze(0)
        else : emb_vec = self.tools.internal_embs[emb_id].unsqueeze(0)
        
        emb_vecs.append(emb_vec)
    
      #Might have to fix later , idk
      if emb_name == None : return None, None, None, None
      emb_ids = emb_ids[0]
      emb_names = emb_names[0]
      emb_vecs = emb_vecs[0]
      ##########
      
      return emb_names, emb_ids, emb_vecs, None # return embedding name, ID, vector

  def shuffle(self , to_negative = None , to_mixer = None , \
  to_positive = None , to_temporary = None , is_sdxl = False):

    if to_negative == None and to_mixer == None \
    and to_positive == None and to_temporary == None : 

      if is_sdxl : self.vector1280.shuffle()
      self.vector.shuffle()

    else:
      if to_negative != None :
        if to_negative : pass # Not implemented
      #####
      if to_mixer != None : 
        if to_mixer : 
          if is_sdxl : self.vector1280.shuffle()
          self.vector.shuffle()
      #####
      if to_temporary != None : 
        if to_temporary : pass # Not implemented

      if to_positive != None : 
        if to_positive : pass # Not implemented
  ######## End of shuffle function

  def roll (self , to_negative = None , to_mixer = None , \
  to_positive = None , to_temporary = None , is_sdxl = False):
    message = ''
    log = []
    if to_negative == None and to_mixer == None \
    and to_positive == None and to_temporary == None : 
      if is_sdxl :
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
          if is_sdxl : 
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

  def random(self , is_sdxl = False):
    if is_sdxl : return self.vector1280.random(self.tools.internal_embs1280)
    else : return self.vector.random(self.tools.internal_embs)
  ### End of random()

  def random_quick(self):
    return self.vector.random_quick()
  ## End of random_quick()

  def sample(self , to_negative = None , to_mixer = None , \
    to_positive = None , to_temporary = None , is_sdxl = False):
    message = ''
    log = []
    if to_negative == None and to_mixer == None \
    and to_positive == None and to_temporary == None:
      if is_sdxl : 
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
          if is_sdxl : 
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

  def place(self, index , vector = None , ID = None ,  name = None , \
    weight = None , to_negative = None , to_mixer = None ,  \
    to_positive = None ,  to_temporary = None , is_sdxl = False):

    #Helper function
    def _place_values_in(target) :
      if vector != None: target.place(vector\
      .to(device = choosen_device , dtype = torch.float32),index)
      if ID != None: target.ID.place(ID, index)
      if name != None: target.name.place(name, index)
      if weight != None : target.weight.place(float(weight) , index)
    #### End of helper function

    if to_negative == None and to_mixer == None \
    and to_temporary == None and to_positive == None:
      _place_values_in(self.vector)
    else:
    
      if to_negative != None :
        if to_negative : 
          if is_sdxl : _place_values_in(self.negative1280)
          else: _place_values_in(self.negative)

      if to_mixer != None : 
        if to_mixer : 
          if is_sdxl : _place_values_in(self.vector1280)
          else: _place_values_in(self.vector)

      if to_temporary != None : 
        if to_temporary : 
          if is_sdxl : _place_values_in(self.temporary1280)
          else: _place_values_in(self.temporary)

      if to_positive != None : 
        if to_positive : 
          if is_sdxl : _place_values_in(self.positive1280)
          else: _place_values_in(self.positive)
  #### End of place()

  def memorize(self) : 
    #######
    for index in range(MAX_NUM_MIX):
      if self.vector.isEmpty.get(index) : continue
      self.vec = self.vector.get(index)\
      .to(device = choosen_device , dtype = torch.float32)
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
      .to(device = choosen_device , dtype = torch.float32)
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

  def recall(self) :
    for index in range(MAX_NUM_MIX):
      if self.temporary.isEmpty.get(index) : continue
      self.vec = self.temporary.get(index)\
      .to(device = choosen_device , dtype = torch.float32)
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
      .to(device = choosen_device , dtype = torch.float32)
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

  def norm (self, tensor , origin_input , distance_fcn):
        current = tensor.to(device = choosen_device , dtype = torch.float32)
        origin = origin_input.to(device = choosen_device , dtype = torch.float32)
        return current

  def distance (self, tensor1 , tensor2):
        distance = torch.nn.PairwiseDistance(p=2)\
        .to(device = choosen_device , dtype = torch.float32)
        #######
        current = tensor1.to(device = choosen_device , dtype = torch.float32)
        ref = tensor2.to(device = choosen_device , dtype = torch.float32)
        dist = distance(current, ref).numpy()[0]
        return  str(round(dist , 2))

  def similarity (self , tensor1 , tensor2):
        distance = torch.nn.PairwiseDistance(p=2)\
        .to(device = choosen_device , dtype = torch.float32)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)\
        .to(device = choosen_device , dtype = torch.float32)
        origin = (self.vector.origin)\
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
        sim = (100*cos(current , ref)).to(device = choosen_device , dtype = torch.float32)
        if sim < 0 : sim = -sim
        return  str(round(sim.numpy()[0] , 2))

  def get(self , index , is_sdxl = False):

    vector = None
    ID = None
    name = None

    if is_sdxl:
      vector = self.vector1280.get(index).to(device = choosen_device , dtype = torch.float32)
      ID = self.vector1280.ID.get(index)
      name = self.vector1280.name.get(index)
    else:
      vector = self.vector.get(index).to(device = choosen_device , dtype = torch.float32)
      ID = self.vector.ID.get(index)
      name = self.vector.name.get(index)
    return vector , ID ,  name
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
  to_positive = None , to_temporary = None , is_sdxl = False):

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
    ###
    Data.vector = None
    Data.negative = None
    Data.positive = None
    Data.temporary = None
    ###
    Data.vector1280 = None
    Data.negative1280 = None
    Data.positive1280 = None
    Data.temporary1280 = None
    ###
    Data.pursuit_strength = 0
    Data.doping_strength = 0
    ######

    ## There params should be removed
    Data.vec= None
    Data.ID = None
    Data.name = None
    Data.weight = None 
    Data.emb_name = None
    Data.emb_id = None
    Data.emb_vec = None
    Data.loaded_emb = None
    ########
    #Default initial values
    self.vector = Vector(3)
    self.negative = Negative(3)
    self.positive = Positive(3)
    self.temporary = Temporary(3)
    ####
    self.vector1280 = Vector1280(3)
    self.negative1280 = Negative1280(3)
    self.positive1280 = Positive1280(3)
    self.temporary1280 = Temporary1280(3)
    ########
    if self.tools.loaded:
      emb_name, emb_id, emb_vec , loaded_emb = \
      self.get_embedding_info(',')
      size = emb_vec.shape[1]
      ####
      if self.tools.is_sdxl:
        sdxl_emb_name, sdxl_emb_id, sdxl_emb_vec , sdxl_loaded_emb = \
        self.get_embedding_info(',', is_sdxl = True)
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
      self.operations = TensorOperations()
    #####


#End of Data class
dataStorage = Data() #Create data
