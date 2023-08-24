import gradio as gr
import torch, os
import collections, math, random , numpy
import re #used to parse string to int
import copy
from modules import  shared, sd_hijack

from lib.toolbox.vector import Vector
from lib.toolbox.negative import Negative
from lib.toolbox.positive import Positive
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
      .to(device = "cpu" , dtype = torch.float32)).unsqueeze(0)
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
      output = torch.cat([output,current], dim=0)\
      .to(device = "cpu" , dtype = torch.float32)
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
      radialGain = 0.001 #deprecated
      randomGain = self.vector.randomization/100  
      pursuit_strength = self.pursuit_strength/100
      origin = self.vector.origin
      size = self.vector.size
      gain = self.vector.gain
      N = self.vector.itermax
      T = 1/N  

      similarity = None
      radialRandom = 1
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

      #Count the positives
      no_of_positives = 0
      for pos_index in range(MAX_NUM_MIX):
        if self.positive.isEmpty.get(pos_index): continue
        no_of_positives += 1
      #####

      good_vecs = []
      good_sims = []

      candidate_vec = None
      rando = None
      rdist = None
      bdist = None
      rand_vec = None

      #Get vectors with similarity > lowerBound
      combined_similarity_score = None
      worst_neg_index = None 
      worst_pos_index = None 
      negative_similarity = None
      positive_similarity = None
      worst_nsim = None
      worst_psim = None
      tmp = None
      neg_strength = self.negative.strength/100
      pos_strength = self.positive.strength/100
      #######
      best_similarity_score = None
      best_negative_similarity = None
      best_positive_similarity = None
      best_similarity = None
      best_worst_neg_index = None
      best_worst_pos_index = None
      best = None
      #######

      randomReduce = pursuit_strength*randomGain/N

      for step in range (N):
        iters+=1
        #Check similarity to original token
        rand_vec = torch.rand(self.vector.size)\
        .to(device = "cpu" , dtype = torch.float32)
        rand_vec  = (2*rand_vec - torch.ones(self.vector.size))\
        .to(device = "cpu" , dtype = torch.float32)

        rdist = distance(rand_vec , origin).to(device = "cpu" , dtype = torch.float32)
        rando = ((1/rdist) * rand_vec).to(device = "cpu" , dtype = torch.float32)

        #Calculate candidate vector
        if best == None : 
          candidate_vec = (rando * randomGain +  current * (1 - randomGain))\
          .to(device = "cpu" , dtype = torch.float32)
        else:
          #Reduce randomness a bit for every step
          randomGain = copy.copy(randomGain - randomReduce)
          bdist = distance(best, origin).to(device = "cpu" , dtype = torch.float32)
          if randomGain<0 : randomGain = 0
          #####
          candidate_vec = (rando * randomGain +  best * (1/bdist)*(1 - randomGain))\
          .to(device = "cpu" , dtype = torch.float32)
        #####

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
            nsim = copy.copy(tmp/100)
            #######
            if worst_nsim == None or nsim > worst_nsim: 
              worst_nsim = nsim
              negative_similarity = copy.copy(100 - nsim)
              worst_neg_index = neg_index
              continue
            #######

        ###########
        #Check positives
        if no_of_positives > 0: 
          for pos_index in range(MAX_NUM_MIX) :
            if self.positive.isEmpty.get(pos_index): continue
            pos_vec = self.positive.get(pos_index)
            tmp = math.floor((100*100*cos(pos_vec, candidate_vec)).numpy()[0])
            if tmp<0 : tmp = -tmp
            psim = copy.copy(tmp/100)
            #######
            if worst_psim == None or psim < worst_psim: 
              worst_psim = psim
              positive_similarity = copy.copy(psim)
              worst_pos_index = pos_index
              continue
            #######
        ###########
        #Calculate the similarity score 
        if positive_similarity == None : 
          if negative_similarity == None: combined_similarity_score = copy.copy(similarity)
          else: combined_similarity_score = copy.copy(\
          similarity *(1 - neg_strength) + negative_similarity*neg_strength)
        else:
          if negative_similarity == None: 
            combined_similarity_score = copy.copy(\
            (similarity * (1 - pos_strength) + positive_similarity*pos_strength))
          else: combined_similarity_score = copy.copy(\
          (similarity * (1 - pos_strength) + positive_similarity*pos_strength) \
          *(1 - neg_strength) + negative_similarity*neg_strength)
        ###########
        assert candidate_vec != None , "candidate_vec is NoneType!"

        #Update the best vector
        if best == None or combined_similarity_score > best_similarity_score: 
            best_similarity_score = combined_similarity_score
            best_negative_similarity = negative_similarity
            best_positive_similarity = positive_similarity
            best_similarity = similarity
            best_worst_neg_index = copy.copy(worst_neg_index)
            best_worst_pos_index = copy.copy(worst_pos_index)
            best = candidate_vec.to(device= "cpu" , dtype = torch.float32)
            continue
      #############

      #Copy the best vector to output if found
      if best != None : 
          combined_similarity_score = best_similarity_score 
          negative_similarity = best_negative_similarity
          positive_similarity = best_positive_similarity
          worst_neg_index = best_worst_neg_index
          worst_pos_index = best_worst_pos_index
          similarity = best_similarity
          candidate_vec = best.to(device= "cpu" , dtype = torch.float32)
                
      #length of candidate vector
      cdist = distance(candidate_vec , origin).numpy()[0]

      #length of output vector
      output_length = gain * dist_expected

      #Cap the length of the output vector according to radialGain slider setting 
      if output_length>dist_expected: output_length = min(output_length , dist_expected/radialGain)
      else: output_length = max(output_length  , dist_expected*radialGain)

      #Set the length of found similar vector
      similar_token = (gain* (output_length/cdist) * candidate_vec * radialRandom)\
      .to(device = "cpu" , dtype = torch.float32)

      #round the values before printing them
      output_length = round(output_length, 2)
      similarity = round(similarity, 1)
      randomization = round(self.vector.randomization , 2)
      dist_expected = round(dist_expected, 2)
      #######
      if combined_similarity_score != None : 
        combined_similarity_score = round(combined_similarity_score , 1)

      negsim = None
      if negative_similarity != None :
        negsim = round(100-negative_similarity , 3) #Invert the value
      
      possim = None
      if positive_similarity != None :
        possim = round(positive_similarity , 3)

      #print the result
      log.append('Similar Mode : Token #' + str(i) + ' with length ' + \
       str(dist_expected) + ' was replaced by new token ' + \
      'with ' + str(similarity) + '% similarity and ' + str(output_length) + \
      ' length')
      ######
      if negsim != None and worst_neg_index != None:
        log.append('Highest similarity to negative tokens : ' + str(negsim) + ' %' + \
        "to token '" + self.negative.name.get(worst_neg_index) + "'")
      if possim != None and worst_pos_index != None:
        log.append('Lowest similarity to positive tokens : ' + str(possim) + ' %' + \
        "to token '" + self.positive.name.get(worst_pos_index) + "'")
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

    radialGain = 0.001 #deprecated
    randomGain = self.vector.randomization/100 
    lowerBound = self.vector.interpolation 
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

        tmp  = ((torch.rand(size) - 0.5* torch.ones(size))\
        .to(device = "cpu" , dtype = torch.float32)).unsqueeze(0)
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
        else :  output = torch.cat([output,similar_token], dim=0)\
        .to(device = "cpu" , dtype = torch.float32)
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
        emb_id = '['+ loaded_emb.checksum()+']' # emb_id is string for loaded embeddings
        emb_vec = loaded_emb.vec.cpu()
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
        emb_vec = self.tools.internal_embs[emb_id].unsqueeze(0)
        emb_vecs.append(emb_vec)
    
      #Emergency fix
      if emb_name == None : return None, None, None, None
      emb_ids = emb_ids[0]
      emb_names = emb_names[0]
      emb_vecs = emb_vecs[0]
      ##########
      
      return emb_names, emb_ids, emb_vecs, None # return embedding name, ID, vector

  def shuffle(self , to_negative = None , to_mixer = None , \
  to_positive = None , to_temporary = None):

    if to_negative == None and to_mixer == None \
    and to_positive == None and to_temporary == None : 
      self.vector.shuffle()
    else:
      if to_negative != None :
        if to_negative : pass # Not implemented
      #####
      if to_mixer != None : 
        if to_mixer : self.vector.shuffle()
      #####
      if to_temporary != None : 
        if to_temporary : pass # Not implemented

      if to_positive != None : 
        if to_positive : pass # Not implemented
  ######## End of shuffle function

  def roll (self , to_negative = None , to_mixer = None , \
  to_positive = None , to_temporary = None):
    message = ''
    if to_negative == None and to_mixer == None \
    and to_positive == None and to_temporary == None : 
      message = self.vector.roll()
    else:
      if to_negative != None :
        if to_negative : pass # Not implemented
      #####
      if to_mixer != None : 
        if to_mixer : message = self.vector.roll()
      #####
      if to_temporary != None : 
        if to_temporary : pass # Not implemented

      if to_positive != None : 
        if to_positive : pass # Not implemented
    return message
  ######## End of roll function

  def random(self):
    return self.vector.random(self.tools.internal_embs)
  ### End of random()

  def random_quick(self):
    return self.vector.random_quick()
  ## End of random_quick()

  def sample(self , to_negative = None , to_mixer = None , \
    to_positive = None , to_temporary = None):
    message = ''
    if to_negative == None and to_mixer == None \
    and to_positive == None and to_temporary == None:
      message = self.vector.sample(self.tools.internal_embs)
    else:
      if to_negative != None :
        if to_negative  : pass #Not implemented
      #####
      if to_mixer != None : 
        if to_mixer : message = self.vector.sample(self.tools.internal_embs)
      #####
      if to_temporary != None : 
        if to_temporary : pass #Not implemented
    #####
      if to_positive != None : 
        if to_positive : pass #Not implemented
    return message
  ### End of sample()

  def place(self, index , vector = None , ID = None ,  name = None , \
    weight = None , to_negative = None , to_mixer = None ,  \
    to_positive = None ,  to_temporary = None):

    #Helper function
    def _place_values_in(target) :
      if vector != None: target.place(vector\
      .to(device = "cpu" , dtype = torch.float32),index)
      if ID != None: target.ID.place(ID, index)
      if name != None: target.name.place(name, index)
      if weight != None : target.weight.place(float(weight) , index)
    #### End of helper function

    if to_negative == None and to_mixer == None \
    and to_temporary == None and to_positive == None:
      _place_values_in(self.vector)
    else:
    
      if to_negative != None :
        if to_negative : _place_values_in(self.negative)

      if to_mixer != None : 
        if to_mixer : _place_values_in(self.vector)

      if to_temporary != None : 
        if to_temporary : _place_values_in(self.temporary)

      if to_positive != None : 
        if to_positive : _place_values_in(self.positive)
  #### End of place()

  def memorize(self) : 
    #######
    for index in range(MAX_NUM_MIX):
      if self.vector.isEmpty.get(index) : continue
      self.vec = self.vector.get(index)\
      .to(device = "cpu" , dtype = torch.float32)
      self.ID = copy.copy(self.vector.ID.get(index))
      self.name = copy.copy(self.vector.name.get(index))
      self.weight = copy.copy(self.vector.weight.get(index))
      ########
      self.temporary.clear(index)
      self.temporary.place(self.vec , index)
      self.temporary.ID.place(self.ID , index)
      self.temporary.name.place(self.name , index)
      self.temporary.weight.place(self.weight , index)
  ###### End of memorize()

  def recall(self) :
    for index in range(MAX_NUM_MIX):
      if self.temporary.isEmpty.get(index) : continue
      self.vec = self.temporary.get(index)\
      .to(device = "cpu" , dtype = torch.float32)
      self.ID = copy.copy(self.temporary.ID.get(index))
      self.name = copy.copy(self.temporary.name.get(index))
      self.weight = copy.copy(self.temporary.weight.get(index))
      #######
      self.vector.clear(index)
      self.vector.place(self.vec , index)
      self.vector.ID.place(self.ID , index)
      self.vector.name.place(self.name, index)
      self.vector.weight.place(self.weight , index)
  ###### End of recall()


  def norm (self, tensor , origin_input , distance_fcn):
        current = tensor.to(device = "cpu" , dtype = torch.float32)
        origin = origin_input.to(device = "cpu" , dtype = torch.float32)
        return current

  def distance (self, tensor1 , tensor2):
        distance = torch.nn.PairwiseDistance(p=2)\
        .to(device = "cpu" , dtype = torch.float32)
        #######
        current = tensor1.to(device = "cpu" , dtype = torch.float32)
        ref = tensor2.to(device = "cpu" , dtype = torch.float32)
        dist = distance(current, ref).numpy()[0]
        return  str(round(dist , 2))

  def similarity (self , tensor1 , tensor2):
        distance = torch.nn.PairwiseDistance(p=2)\
        .to(device = "cpu" , dtype = torch.float32)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)\
        .to(device = "cpu" , dtype = torch.float32)
        origin = (self.vector.origin)\
        .to(device = "cpu" , dtype = torch.float32)
        #######

        current = tensor1.to(device = "cpu" , dtype = torch.float32)
        dist1 = distance(current, origin).numpy()[0]
        current = current * (1/dist1)

        ####

        ref = tensor2.to(device = "cpu" , dtype = torch.float32)
        dist2 = distance(current, origin).numpy()[0]
        ref = ref * (1/dist2)

        ########
        sim = (100*cos(current , ref)).to(device = "cpu" , dtype = torch.float32)
        if sim < 0 : sim = -sim
        return  str(round(sim.numpy()[0] , 2))

  def get(self , index):
    vector = self.vector.get(index).to(device = "cpu" , dtype = torch.float32)
    ID = self.vector.ID.get(index)
    name = self.vector.name.get(index)
    return vector , ID ,  name

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
  to_positive = None , to_temporary = None):

    if to_negative == None and to_mixer == None \
    and to_positive == None and to_temporary == None:

      self.vector.clear(index)
      self.negative.clear(index)
      self.positive.clear(index)
      self.temporary.clear(index)

    else:

      if to_negative != None:
        if to_negative: self.negative.clear(index)

      if to_mixer!= None:
        if to_mixer: self.vector.clear(index)

      if to_temporary!= None:
        if to_temporary: self.temporary.clear(index)

      if to_positive!= None:
        if to_positive: self.positive.clear(index)

  def __init__(self):
    #Check if new embeddings have been added 
    try: sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
    except: 
      sd_hijack.model_hijack.embedding_db.dir_mtime=0
      sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
      
    Data.tools = Tools()

    Data.vec= None
    Data.ID = None
    Data.name = None
    Data.weight = None 
    Data.emb_name = None
    Data.emb_id = None
    Data.emb_vec = None
    Data.loaded_emb = None
    Data.vector = None
    Data.negative = None
    Data.temporary = None

    if self.tools.loaded:
      self.emb_name, Data.emb_id, Data.emb_vec , Data.loaded_emb = self.get_embedding_info('test')
      self.vector = Vector(self.emb_vec.shape[1])
      self.negative = Negative(self.emb_vec.shape[1])
      self.positive = Positive(self.emb_vec.shape[1])
      self.temporary = Temporary(self.emb_vec.shape[1])
    else: 
      #Just set size to 3 if no model can be loaded
      self.vector = Vector(3)
      self.negative = Negative(3)
      self.positive = Positive(3)
      self.temporary = Temporary(3)


#End of Data class
dataStorage = Data() #Create data
