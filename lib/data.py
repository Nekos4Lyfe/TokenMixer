
import gradio as gr
from modules import script_callbacks, shared, sd_hijack
from modules.shared import cmd_opts
from pandas import Index
from pandas.core.groupby.groupby import OutputFrameOrSeries
from pydantic.env_settings import BaseSettings
import torch, os
from modules.textual_inversion.textual_inversion import Embedding
import collections, math, random , numpy
import re #used to parse string to int
import copy
from torch.nn.modules import ConstantPad1d, container

from lib.toolbox.vector import Vector
from lib.toolbox.negative import Negative
from lib.toolbox.temporary import Temporary
from lib.toolbox.tools import Tools
from lib.toolbox.constants import MAX_NUM_MIX
#-------------------------------------------------------------------------------

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

    for i in range (MAX_NUM_MIX):
      if (self.vector.isEmpty.get(i)): continue 
      no_of_tokens += 1

      if similar_mode : 
        current , message = self.replace_with_similar(i)
        log.append(message)
      else : current = self.vector.get(i).cpu()

      current_weight = 1
      #self.vector.weight.get(i)
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

    radialGain = max(0.001 , self.vector.radius/100) 
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
      rand_vec  = (2*torch.rand(size) - torch.ones(size)).unsqueeze(0)
      rdist = distance(rand_vec, origin).numpy()[0]
      rando = (1/rdist) * rand_vec #Create random unit vector

      candidate_vec = norm_expected * (1 - randomGain)  + rando*randomGain

      minimum = 20000 #set initial minimum at impossible 200% similarity 
      maximum = -10000 #Set initial maximum at impossible -100% similarity
      similarity_sum = 0
      for i in range (MAX_NUM_MIX):
        if (self.vector.isEmpty.get(i)): continue

        #Get current token
        tmp = self.vector.get(i).cpu()
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
          mintoken = i

        if similarity == max(maximum , similarity) :
          maximum = similarity
          maxtoken = i
   
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


  def concat_all(self , similar_mode) :
    log = []
    log.append('Concat Mode :')

    output = None 
    current = None
    dist = None
    tmp = None
    origin = self.vector.origin.cpu()

    no_of_tokens = 0
    distance = torch.nn.PairwiseDistance(p=2)

    for i in range (MAX_NUM_MIX):
      if (self.vector.isEmpty.get(i)): continue 
      no_of_tokens+=1
      
      if similar_mode : 
        current , message = self.replace_with_similar(i)
        log.append(message)
      else : current = self.vector.get(i).cpu()

      current_weight = 1 #Will fix later

      current = current*current_weight

      assert not current == None , "current vector is NoneType!"
      dist = round(distance(current , origin).numpy()[0],2)
      if output == None : 
          output = current
          log.append('Placed token with length '+ str(dist)  +' in new embedding ')
          continue
      output = torch.cat([output,current], dim=0)
      log.append('Placed token with length '+ str(dist)  +' in new embedding ')
    
    log.append('New embedding has '+ str(no_of_tokens) + ' tokens')
    log.append('-------------------------------------------')
    return output , '\n'.join(log)
  
  def replace_with_similar(self , i):
      assert not (self.vector.isEmpty.get(i)) , "Empty token!"
      log = []
      dist = None
      current = None
      cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
      distance = torch.nn.PairwiseDistance(p=2)
      radialGain = max(0.001 , self.vector.radius/100) 
      randomGain = self.vector.randomization/100  
      costheta = self.vector.costheta
      origin = self.vector.origin
      size = self.vector.size
      gain = self.vector.gain
      N = self.vector.itermax
      T = 1/N  


      similarity = None
      radialRandom = None
      tmp = None
      no_of_tokens = 0
      iters = 0


      tensor = self.vector.get(i).cpu()
      dist_expected = distance(tensor, origin).numpy()[0] # Tensor length
      current = (1/dist_expected) * tensor  #Tensor as unit vector

      #Count the negatives
      no_of_negatives = 0
      for neg_index in range(MAX_NUM_MIX):
        if self.negative.isEmpty.get(neg_index): continue
        no_of_negatives += 1
      #####

      good_vecs = []
      good_sims = []

      candidate_vec = None
      rando = None
      rdist = None
      rand_vec = None

      #Randomization parameters: 
      #radialRandom = (1 - randomGain) + randomGain*(2*random.random() - 1)
      #if radialRandom<0: radialRandom = -radialRandom
      radialRandom = 1
      lowerBound  = costheta  + (100 - costheta) * (randomGain * random.random())

      #Get vectors with similarity > lowerBound
      combined_similarity_score = None
      worst_neg_index = None 
      negative_similarity = None
      worst_nsim = None
      tmp = None
      strength = self.negative.strength/100
      #######
      best_similarity_score = None
      best_negative_similarity = None
      best_similarity = None
      best_worst_neg_index = None
      best = None
      #######
      for step in range (N):
        iters+=1
        #Check similarity to original token
        rand_vec  = (torch.rand(size) - 0.5* torch.ones(size)).unsqueeze(0)
        rdist = distance(rand_vec , origin)
        rando = (1/rdist) * rand_vec 
        candidate_vec = rando * randomGain +  current * (1 - randomGain)
        similarity = (100*cos(current, candidate_vec)).numpy()[0]
        if similarity<0 : similarity = -similarity
        #######
        #Check negatives
        if no_of_negatives > 0: 
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
            #######
            negative_similarity = 100 - worst_nsim
        ###########
        if negative_similarity == None: combined_similarity_score = similarity
        else: combined_similarity_score = similarity *(1 - strength) + negative_similarity*strength
        ###########
        assert candidate_vec != None , "candidate_vec is NoneType!"
        if best == None: 
            best_similarity_score = combined_similarity_score
            best_negative_similarity = negative_similarity
            best_similarity = similarity
            best_worst_neg_index = worst_neg_index
            best = candidate_vec
        ##########
        if combined_similarity_score > best_similarity_score:
            best_similarity_score = combined_similarity_score
            best_negative_similarity = negative_similarity
            best_similarity = similarity
            best_worst_neg_index = worst_neg_index
            best = candidate_vec
      #############
      
      #If negative similarity condition was not fulfilled
      #get the best candidate vector
      if best != None : 
          combined_similarity_score = best_similarity_score 
          negative_similarity = best_negative_similarity
          similarity = best_similarity
          worst_neg_index = best_worst_neg_index 
          candidate_vec = best
                
      #length of candidate vector
      cdist = distance(candidate_vec , origin).numpy()[0]

      #length of output vector
      output_length = gain * dist_expected

      #Cap the length of the output vector according to radialGain slider setting 
      if output_length>dist_expected: output_length = min(output_length , dist_expected/radialGain)
      else: output_length = max(output_length  , dist_expected*radialGain)

      #Set the length of found similar vector
      similar_token = gain* (output_length/cdist) * candidate_vec * radialRandom

      #round the values before printing them
      output_length = round(output_length, 2)
      similarity = round(similarity, 1)
      randomization = round(self.vector.randomization , 2)
      req_similarity = round (self.vector.costheta , 1)
      dist_expected = round(dist_expected, 2)
      lowerBound = round(lowerBound , 1)
      #######
      if combined_similarity_score != None : 
        combined_similarity_score = round(combined_similarity_score , 1)
      negsim = None
      if negative_similarity != None :
        negsim = round(100-negative_similarity , 3) #Invert the value

      #print the result
      log.append('Similar Mode : Token #' + str(i) + ' with length ' + \
       str(dist_expected) + ' was replaced by new token ' + \
      'with ' + str(similarity) + '% similarity and ' + str(output_length) + \
      ' length')
      ######
      if negsim != None :
        name = self.negative.name.get(worst_neg_index)
        log.append('Max similarity to negative tokens : ' + str(negsim) + ' %' + \
        "to token '" + name + "'")
      ######
      log.append('Search took ' + str(iters) + ' iterations')

      return similar_token , '\n'.join(log)

  def merge_if_similar(self, similar_mode):
    log = []
    log.append('Interpolate Mode :')
    no_of_tokens = 0

    for i in range (MAX_NUM_MIX):
      if (self.vector.isEmpty.get(i)): continue
      no_of_tokens +=1
      continue

    if no_of_tokens <= 0: 
      log.append('No inputs in mixer')
      return None , '\n'.join(log) 

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    distance = torch.nn.PairwiseDistance(p=2)

    radialGain = max(0.001 , self.vector.radius/100) 
    randomGain = self.vector.randomization/100 
    lowerBound = self.vector.interpolation 
    costheta = self.vector.costheta
    origin = self.vector.origin.cpu()
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

    for i in range (MAX_NUM_MIX):
      if (self.vector.isEmpty.get(i)): continue

      if similar_mode : 
        current , message = self.replace_with_similar(i)
        log.append(message)
      else : current = self.vector.get(i).cpu()

      current_weight = 1
      #self.vector.weight.get(i)
      current = current*current_weight

      if prev == None : 
        prev = current
        continue

      assert not current == None , "current tensor is None!"
      assert not prev == None , "prev tensor is None!"

      tmp = ((current + prev) * 0.5).cpu() #Expected merge vector
      avg_dist = distance(tmp, origin).numpy()[0]
      merge_norm = ((1/avg_dist)*tmp).cpu()

      current_dist = distance(current, origin).numpy()[0]
      current_norm = ((1/current_dist)*current).cpu() 

      prev_dist = distance(prev, origin).numpy()[0]
      prev_norm = ((1/prev_dist)*prev).cpu()

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

        tmp  = (torch.rand(size) - 0.5* torch.ones(size)).unsqueeze(0)
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

        log.append('Token ' + str(i-1) + ' and '+str(i)+ \
        ' with expected length '+ str(dist_expected) + ' merged into 1 token with ' + \
         str(similarity)+ '% similarity and ' + str(output_length) + ' length after ' + \
        str(iters) + ' steps)')
        no_of_tokens = no_of_tokens - 1

        if output == None : output = similar_token
        else :  output = torch.cat([output,similar_token], dim=0)
        log.append('Placed token with length '+ str(output_length)  +' in new embedding ')

      else:
        log.append('Skipping merge between token ' + str(i-1)+ \
        ' and '+str(i))
        log.append('Token similarity ' + str(similarity) + \
        '% is less then req. similarity' + str(lowerBound) + '%')

    log.append('New embedding now has ' + str(no_of_tokens) + ' tokens')
    log.append('-------------------------------------------')
    return output , '\n'.join(log)

  def text_to_emb_ids(self, text):
    text = copy.copy(text.lower())
    if self.tools.tokenizer.__class__.__name__== 'CLIPTokenizer': # SD1.x detected
        emb_ids = self.tools.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    elif self.tools.tokenizer.__class__.__name__== 'SimpleTokenizer': # SD2.0 detected
        emb_ids =  self.tools.tokenizer.encode(text)
    else: emb_ids = None
    return emb_ids # return list of embedding IDs for text

  def emb_id_to_vec(self, emb_id):
    if not isinstance(emb_id, int):
      return None
    if (emb_id > self.tools.no_of_internal_embs):
      return None
    if emb_id < 0 :
      return None
    return self.tools.internal_embs[emb_id]

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

  def get_embedding_info(self, string):
      text = copy.copy(string.lower())
      loaded_emb = self.tools.loaded_embs.get(text, None)

      if loaded_emb == None:
        for neg_index in self.tools.loaded_embs.keys():
            if text == neg_index.lower():
              loaded_emb = self.tools.loaded_embs.get(neg_index, None)
              break

      if loaded_emb!=None:
        emb_name = loaded_emb.name
        emb_id = '['+ loaded_emb.checksum()+']' # emb_id is string for loaded embeddings
        emb_vec = loaded_emb.vec.cpu()
        return emb_name, emb_id, emb_vec, loaded_emb #also return loaded_emb reference

      # support for #nnnnn format
      val = None
      if text.startswith('#'):
        try:
            val = int(text[1:])
            if (val<0) or (val>=self.tools.internal_embs.shape[0]): val = None
        except: val = None

      # obtain internal embedding ID
      if val!=None: emb_id = val
      else:
        emb_ids = self.text_to_emb_ids(text)
        if emb_ids == None : return None, None, None, None
        emb_id = emb_ids[0] # emb_id is int for internal embeddings

      emb_name = self.emb_id_to_name(emb_id)
      emb_vec = self.tools.internal_embs[emb_id].unsqueeze(0)
      return emb_name, emb_id, emb_vec, None # return embedding name, ID, vector

  def shuffle(self , to_negative = None , to_mixer = None , to_temporary = None):
    if to_negative == None and to_mixer == None and to_temporary == None:
      self.data.vector.shuffle()
    else:
      if to_negative != None :
        if to_negative : self.data.negative.shuffle()
      #####
      if to_mixer != None : 
        if to_mixer : self.data.vector.shuffle()
      #####
      if to_temporary != None : 
        if to_temporary : self.data.temporary.shuffle()
  ######## End of shuffle function

  def sample(self , to_negative = None , to_mixer = None , to_temporary = None):

    message = ''

    if to_negative == None and to_mixer == None and to_temporary == None:
      message = self.vector.sample()
    else:
      if to_negative != None :
        if to_negative  : pass #Not implemented
      #####
      if to_mixer != None : 
        if to_mixer : message = self.vector.sample()
      #####
      if to_temporary != None : 
        if to_temporary : pass #Not implemented
    #####

    return message



    
  
  
  def place(self, index , vector = None , ID = None ,  name = None , \
    weight = None , to_negative = None , to_mixer = None , to_temporary = None):

    def _place_values_in(target) :
      if vector != None: target.place(vector.cpu(),index)
      if ID != None: target.ID.place(ID, index)
      if name != None: target.name.place(name, index)
      if weight != None : target.weight.place(float(weight) , index)
    #### End of helper function

    if to_negative == None and to_mixer == None and to_temporary == None:
      _place_values_in(self.vector)
    else:
    
      if to_negative != None :
        if to_negative : _place_values_in(self.negative)

      if to_mixer != None : 
        if to_mixer : _place_values_in(self.vector)

      if to_temporary != None : 
        if to_temporary : _place_values_in(self.temporary)
  #### End of place function


  def get(self , index):
    vector = self.vector.get(index)
    ID = self.vector.ID.get(index)
    name = self.vector.name.get(index)
    return vector , ID ,  name

  def update_loaded_embs(self):
      self.tools.update_loaded_embs()

  def clear (self, index , to_negative = None , to_mixer = None , to_temporary = None):

    if to_negative == None and to_mixer == None and to_temporary == None:
      self.vector.clear(index)
      self.negative.clear(index)
      self.temporary.clear(index)
    else:

      if to_negative != None:
        if to_negative: self.negative.clear(index)

      if to_mixer!= None:
        if to_mixer: self.vector.clear(index)

      if to_temporary!= None:
        if to_temporary: self.temporary.clear(index)

  def __init__(self):
    #Check if new embeddings have been added 
    try: sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
    except: 
      sd_hijack.model_hijack.embedding_db.dir_mtime=0
      sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
      
    Data.tools = Tools()

    if self.tools.loaded:
      Data.emb_name, Data.emb_id, Data.emb_vec , Data.loaded_emb = self.get_embedding_info('test')
      Data.vector = Vector(self.emb_vec.shape[1])
      Data.negative = Negative(self.emb_vec.shape[1])
      Data.temporary = Temporary(self.emb_vec.shape[1])
    else: 
      Data.vector = Vector(3)
      Data.negative = Negative(3)
      Data.temporary = Temporary(3)

#End of Data class
dataStorage = Data() #Create data
