import torch , math, random , numpy , copy
from library.toolbox.constants import MAX_NUM_MIX
from library.toolbox.constants import TENSOR_DEVICE_TYPE , TENSOR_DATA_TYPE
choosen_device = TENSOR_DEVICE_TYPE
datatype = TENSOR_DATA_TYPE


# The Operations class is used to store 
# helper functions to the Data.py class
class TensorOperations:


  @staticmethod
  def _concat_all(vector , similar_mode , is_sdxl) :

    log = []
    log.append('Concat Mode :')

    output = None 
    current = None
    dist = None
    tmp = None
    origin = vector.origin.to(device = choosen_device , dtype = torch.float32)

    no_of_tokens = 0
    distance = torch.nn.PairwiseDistance(p=2)

    for index in range (MAX_NUM_MIX):
      if (vector.isEmpty.get(index)): continue 
      no_of_tokens+=1
      
      if similar_mode : 
        current , message = self.replace_with_similar(index , is_sdxl)
        log.append(message)
      else : current = vector.get(index).to(device = choosen_device , dtype = torch.float32)

      current_weight = 1 #Will fix later

      current = current*current_weight

      assert not current == None , "current vector is NoneType!"
      dist = round(distance(current , origin).numpy()[0],2)
      if output == None : 
          output = current
          log.append('Placed token with length '+ str(dist)  +' in new embedding ')
          continue
      output = torch.cat([output,current], dim=0)\
      .to(device = choosen_device , dtype = torch.float32)
      log.append('Placed token with length '+ str(dist)  +' in new embedding ')
    
    log.append('New embedding has '+ str(no_of_tokens) + ' tokens')
    log.append('-------------------------------------------')
    return output , '\n'.join(log)
  #############
  
  @staticmethod
  def _replace_with_similar(index , \
   vector , positive , negative , \
   pursuit_value , doping_value):

      assert not (vector.isEmpty.get(index)) , "Empty token!"
      log = []
      dist = None
      current = None
      cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
      distance = torch.nn.PairwiseDistance(p=2)
      radialGain = 0.001 #deprecated
      randomGain = vector.randomization/100  
      origin = vector.origin.to(device=choosen_device , dtype = datatype)
      size = vector.size
      gain = vector.gain
      N = vector.itermax
      T = 1/N  

      similarity = None
      radialRandom = 1
      tmp = None
      no_of_tokens = 0
      iters = 0

      tensor = vector.get(index).to(device=choosen_device , dtype = datatype)
      dist_expected = distance(tensor, origin).to(device=choosen_device , dtype = datatype)
      current = ((1/dist_expected) * tensor)\
      .to(device=choosen_device , dtype = datatype)  #Tensor as unit vector

      #Count the negatives
      no_of_negatives = 0
      for neg_index in range(MAX_NUM_MIX):
        if negative.isEmpty.get(neg_index): continue
        no_of_negatives += 1
      #####

      #Count the positives
      #and get a local list of positives to use for doping
      positives = []
      no_of_positives = 0
      for pos_index in range(MAX_NUM_MIX):
        if positive.isEmpty.get(pos_index): continue
        no_of_positives += 1
        positives.append(pos_index)
      #####

      good_vecs = []
      good_sims = []

      candidate_vec = None
      rando = None
      rdist = None
      bdist = None
      ddist = None
      pdist = None
      cdist = None
      ndist = None
      rand_vec = None

      #Get vectors with similarity > lowerBound
      combined_similarity_score = None
      worst_neg_index = None 
      worst_pos_index = None 
      best_pos_index = None
      negative_similarity = None
      positive_similarity = None
      special_similarity = None
      worst_nsim = None
      worst_psim = None
      best_psim = None
      tmp = None
      neg_strength = negative.strength/100
      pos_strength = positive.strength/100
      #######
      best_similarity_score = None
      best_negative_similarity = None
      best_positive_similarity = None
      best_similarity = None
      best_worst_neg_index = None
      best_worst_pos_index = None
      best = None
      doping = torch.zeros(vector.size).to(device=choosen_device , dtype = datatype)
      #######

      randomReduce = pursuit_value*randomGain/N

      lenpos = len(positives)
      for step in range (N):
        iters+=1
        ##### Calculate doping vector
        doping = None
        if lenpos>0 and doping_value>0:

          doping = positive.get(random.choice(positives))\
          .to(device = choosen_device , dtype = datatype)

          ddist = distance(doping , origin)\
          .to(device = choosen_device , dtype = datatype)
          doping = (doping * (1/ddist))\
          .to(device = choosen_device , dtype = datatype)
        else:
          doping = origin\
          .to(device = choosen_device , dtype = datatype) 

        ##### Calculate random vector
        rand_vec = torch.rand(vector.size)\
        .to(device = choosen_device , dtype = datatype)
        ones = torch.ones(vector.size)\
        .to(device = choosen_device , dtype = datatype)
        rand_vec  = (2*rand_vec - ones)\
        .to(device = choosen_device , dtype = datatype)
        rdist = distance(rand_vec , origin)\
        .to(device = choosen_device , dtype = datatype)
        rand_vec = ((1/rdist) * rand_vec)\
        .to(device = choosen_device , dtype = datatype)
        ##### Perform doping on the random vector
        rand_vec = (doping*doping_value  + rand_vec*(1 - doping_value))\
        .to(device = choosen_device , dtype = datatype)
        #### Calculate the doped random vector
        rdist = distance(rand_vec , origin)\
        .to(device = choosen_device , dtype = datatype)
        rando = (rand_vec * (1/rdist))\
        .to(device = choosen_device , dtype = datatype)

        #Calculate candidate vector
        if best == None : 
          candidate_vec = (rando * randomGain +  current * (1 - randomGain))\
          .to(device = choosen_device , dtype = datatype)
        else:
          #Reduce randomness a bit for every step
          randomGain = copy.copy(randomGain - randomReduce)
          bdist = distance(best, origin).to(device = choosen_device , dtype = datatype)
          if randomGain<0 : randomGain = 0
          #####
          candidate_vec = (rand_vec*(1/rdist)*randomGain +  best*(1/bdist)*(1 - randomGain))\
          .to(device = choosen_device , dtype = datatype)
        #####
        cdist = distance(candidate_vec, origin).to(device = choosen_device , dtype = datatype)
        candidate_vec = (candidate_vec*(1/cdist)).to(device = choosen_device , dtype = datatype)

        similarity = (100*cos(current, candidate_vec)\
        .to(device = choosen_device , dtype = datatype)).numpy()[0]
        if similarity<0 : similarity = -similarity
        #######
        #Check negatives
        if no_of_negatives > 0: 
          for neg_index in range(MAX_NUM_MIX) :
            if negative.isEmpty.get(neg_index): continue
            neg_vec = negative.get(neg_index).to(device=choosen_device , dtype = datatype)
            ndist = distance(neg_vec, origin).to(device = choosen_device , dtype = datatype)
            neg_vec = (neg_vec *(1/ndist)).to(device = choosen_device , dtype = datatype)
            tmp = math.floor((100*100*cos(neg_vec, candidate_vec)).numpy()[0])
            if tmp<0 : tmp = -tmp
            nsim = copy.copy(tmp/100)
            #######
            if worst_nsim == None or nsim > worst_nsim: 
              worst_nsim = copy.copy(nsim)
              negative_similarity = copy.copy(100 - nsim)
              worst_neg_index = copy.copy(neg_index)
            continue
            #######

        ###########

        #Check positives
        doped = False
        pos_vec = None
        if no_of_positives > 0: 
          for pos_index in range(MAX_NUM_MIX) :
            if positive.isEmpty.get(pos_index): continue
            pos_vec = positive.get(pos_index)\
            .to(device=choosen_device , dtype = datatype)

            pdist = distance(pos_vec, origin)\
            .to(device = choosen_device , dtype = datatype)

            pos_vec = (pos_vec*(1/pdist))\
            .to(device = choosen_device , dtype = datatype)

            tmp = math.floor((100*100*cos(pos_vec,candidate_vec))\
            .to(device = choosen_device , dtype = datatype).numpy()[0])
            
            if tmp<0 : tmp = -tmp
            psim = copy.copy(tmp/100)
            #######
            if worst_psim == None or psim < worst_psim: 
              worst_psim = copy.copy(psim)
              positive_similarity = copy.copy(psim)
              worst_pos_index = copy.copy(pos_index)
            if best_psim == None or psim > best_psim:
              best_psim = copy.copy(psim)
              best_pos_index = copy.copy(pos_index)
              special_similarity = copy.copy(psim)
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
            best_similarity_score = copy.copy(combined_similarity_score)
            best_negative_similarity = copy.copy(negative_similarity)
            best_positive_similarity = copy.copy(positive_similarity)
            best_similarity = copy.copy(similarity)
            best_worst_neg_index = copy.copy(worst_neg_index)
            best_worst_pos_index = copy.copy(worst_pos_index)
            best = candidate_vec.to(device= choosen_device , dtype = datatype)
            continue
      #############

      #Copy the best vector to output if found
      if best != None : 
          combined_similarity_score = copy.copy(best_similarity_score)
          negative_similarity = copy.copy(best_negative_similarity)
          positive_similarity = copy.copy(best_positive_similarity)
          worst_neg_index = copy.copy(best_worst_neg_index)
          worst_pos_index = copy.copy(best_worst_pos_index)
          similarity = copy.copy(best_similarity)
          candidate_vec = best.to(device = choosen_device , dtype = datatype)
                
      #length of candidate vector
      cdist = distance(candidate_vec , origin)\
      .to(device = choosen_device , dtype = datatype)

      #length of output vector
      output_length = copy.copy(gain * dist_expected)

      #Set the length of found similar vector
      similar_token = ((output_length/cdist) * candidate_vec)\
      .to(device = choosen_device , dtype = datatype)

      #round the values before printing them
      output_length = output_length.numpy()[0]
      dist_expected = dist_expected.numpy()[0]
      output_length = round(output_length, 2)
      similarity = round(similarity, 1)
      randomization = round(vector.randomization , 2)
      dist_expected = round(dist_expected, 2)
      #######
      if combined_similarity_score != None : 
        combined_similarity_score = round(combined_similarity_score , 1)

      negsim = None
      if negative_similarity != None :
        negsim = round(100-negative_similarity , 3) #Invert the value to get
                                                    #Highest neg. similarity
      possim = None
      if positive_similarity != None :
        possim = round(positive_similarity , 3) #<--Lowest pos. similiarity

      specsim = None
      if special_similarity != None:
        specsim = round(special_similarity , 3)#<--Highest pos. similiarity

      #print the result
      log.append('Similar Mode : Token #' + str(index) + ' with length ' + \
       str(dist_expected) + ' was replaced by new token ' + \
      'with ' + str(similarity) + '% similarity and ' + str(output_length) + \
      ' length')
      ######
      if negsim != None and worst_neg_index != None:
        log.append('Highest similarity to negative tokens : ' + str(negsim) + ' %' + \
        "to token '" + negative.name.get(worst_neg_index) + "'")
      if possim != None and worst_pos_index != None:
        log.append('Lowest similarity to positive tokens : ' + str(possim) + ' %' + \
        "to token '" + positive.name.get(worst_pos_index) + "'")
      if specsim != None and best_pos_index != None:
        log.append('Highes similarity to positive tokens : ' + str(specsim) + ' %' + \
        "to token '" + positive.name.get(best_pos_index) + "'")
      ######
      log.append('Search took ' + str(iters) + ' iterations')
      return similar_token , '\n'.join(log)
  ######## End of _replace_with_similar

  def __init__(self):
    pass
