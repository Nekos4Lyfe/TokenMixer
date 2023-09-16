import torch , math, random , numpy , copy
from library.toolbox.constants import MAX_NUM_MIX
from library.toolbox.constants import TENSOR_DEVICE_TYPE , TENSOR_DATA_TYPE
choosen_device = TENSOR_DEVICE_TYPE
datatype = TENSOR_DATA_TYPE


# The Operations class is used to store 
# helper functions to the Data.py class
class TensorFunctions:

  # Randomize the order of the tensors in the 
  #given field(s) of the Data class
  @staticmethod
  def _shuffle(\
  vector , positive , negative , temporary , \
  vector1280 , positive1280 , negative1280 , temporary1280 , \
  to_negative , to_mixer,to_positive, to_temporary , use_1280_dim):

    if to_negative == None and to_mixer == None \
    and to_positive == None and to_temporary == None : 
      if use_1280_dim : vector1280.shuffle()
      vector.shuffle()
    else:
      if to_negative != None :
        if to_negative : pass # Not implemented
      #####
      if to_mixer != None : 
        if to_mixer : 
          if use_1280_dim : vector1280.shuffle()
          vector.shuffle()
      #####
      if to_temporary != None : 
        if to_temporary : pass # Not implemented

      if to_positive != None : 
        if to_positive : pass # Not implemented
  ######## End of shuffle function


  # Perform torch.roll of each tensor
  # within given field(s) in Data class
  @staticmethod
  def _roll (\
  vector , positive , negative , temporary , \
  vector1280 , positive1280 , negative1280 , temporary1280 , \
  to_negative , to_mixer, to_positive , to_temporary , use_1280_dim):
    message = ''
    log = []
    if to_negative == None and to_mixer == None \
    and to_positive == None and to_temporary == None : 
      if use_1280_dim :
        log.append("Performing roll on 1280 dimension vectors : ") 
        message = vector1280.roll()
        log.append(message)
      #######
      log.append("Performing roll on 768 dimension vectors : ") 
      message = vector.roll()
      log.append(message)
    else:
      if to_negative != None :
        if to_negative : pass # Not implemented
      #####
      if to_mixer != None : 
        if to_mixer : 
          if use_1280_dim : 
            log.append("Performing roll on 1280 dimension vectors : ")
            message = vector1280.roll()
            log.append(message)
          ######
          log.append("Performing roll on 768 dimension vectors : ") 
          message = vector.roll()
          log.append(message)
      #####
      if to_temporary != None : 
        if to_temporary : pass # Not implemented
      ######
      if to_positive != None : 
        if to_positive : pass # Not implemented
    #####
    return '\n'.join(log)
  ######## End of _roll function


  #_concat
  # Replace a fraction of each tensor with a random vector 
  # within given field(s) in Data class
  @staticmethod
  def _sample(\
  vector , positive , negative , temporary ,\
  vector1280 , positive1280 , negative1280 , temporary1280, \
  internal_embs , internal_embs1280 ,
  to_negative , to_mixer , to_positive , to_temporary , use_1280_dim):
    message = ''
    log = []
    if to_negative == None and to_mixer == None \
    and to_positive == None and to_temporary == None:
      if use_1280_dim : 
        log.append("Performing sample on 1280 dim vectors")
        message = vector1280.sample(internal_embs1280)
        log.append(message)
      #######
      log.append("Performing sample on 768 dim vectors")
      message = vector.sample(internal_embs)
      log.append(message)
    else:
      if to_negative != None :
        if to_negative  : pass #Not implemented
      #####
      if to_mixer != None : 
        if to_mixer : 
          if use_1280_dim : 
            log.append("Performing sample on 1280 dim vectors")
            message = vector1280.sample(internal_embs1280)
            log.append(message)
          #######
          log.append("Performing sample on 768 dim vectors")
          message = vector.sample(internal_embs)
          log.append(message)
      #########
      if to_temporary != None : 
        if to_temporary : pass #Not implemented
    #####
      if to_positive != None : 
        if to_positive : pass #Not implemented
    return '\n'.join(log)
  ### End of sample()

  @staticmethod
  def _clear (\
  vector , positive , negative , temporary , 
  vector1280 , positive1280 , negative1280 , temporary1280 , 
  index , \
  to_negative , to_mixer, to_positive, to_temporary , use_1280_dim) :

    if to_negative == None and to_mixer == None \
    and to_positive == None and to_temporary == None:

      vector.clear(index)
      negative.clear(index)
      positive.clear(index)
      temporary.clear(index)
      vector1280.clear(index)
      negative1280.clear(index)
      positive1280.clear(index)
      temporary1280.clear(index)

    else:

      if to_negative != None:
        if to_negative: 
          negative.clear(index)
          negative1280.clear(index)

      if to_mixer!= None:
        if to_mixer: 
          vector.clear(index)
          vector1280.clear(index)

      if to_temporary!= None:
        if to_temporary: 
          temporary.clear(index)
          temporary1280.clear(index)

      if to_positive!= None:
        if to_positive: 
          positive.clear(index)
          positive1280.clear(index)
  ########## End of clear()


  # Place the input at assigned position(s) in the Dataclass
  @staticmethod
  def _place(\
    vector , positive , negative , temporary ,  \
    vector1280 , positive1280 , negative1280 , temporary1280 , \
    index , tensor , ID ,  name , weight , \
    to_negative , to_mixer,  to_positive ,  to_temporary , use_1280_dim ) :

    # Helper function
    def _place_values_in(target , tensor  , ID, name, weight , index) :
      if tensor != None: target.place(tensor\
      .to(device = choosen_device , dtype = datatype),index)
      if ID != None: target.ID.place(ID, index)
      if name != None: target.name.place(name, index)
      if weight != None : target.weight.place(float(weight) , index)
    ##### End of _place_values_in()

    target = None
    cond = \
    (to_negative == None) and \
    (to_mixer == None) and \
    (to_temporary == None) and \
    (to_positive == None)

    # Default operation 
    if cond:
      target = vector
      _place_values_in(target , tensor , ID, name, weight , index)
    #######

    # Write to the 'negative' field
    if not cond and (to_negative != None) :
      if use_1280_dim : target = negative1280
      else: target = negative
      if to_negative : _place_values_in\
      (target , tensor , ID, name, weight , index)
    #######

    # Write to the 'vector' field
    if not cond and (to_mixer != None) : 
      if use_1280_dim : target = vector1280
      else: target = vector
      if to_mixer : _place_values_in\
      (target , tensor , ID, name, weight , index)
    #######

    # Write to the 'temporary' field
    if not cond and (to_temporary != None) :
      if use_1280_dim : target = temporary1280
      else: target = temporary
      if to_temporary : _place_values_in\
      (target , tensor , ID, name, weight , index)
    #######

    # Write to the 'positive' field
    if not cond and (to_positive != None) : 
      if use_1280_dim : target = positive1280
      else: target = positive
      if to_positive : _place_values_in\
      (target , tensor , ID, name, weight , index)
    #######
  #### End of place()


  # Move values from one field to another
  # withing the Data class
  @staticmethod
  def _move(target , destination) : 
    for index in range(MAX_NUM_MIX):
      if target.isEmpty.get(index) : continue
      tensor = target.get(index)\
      .to(device = choosen_device , dtype = datatype)
      ID = copy.copy(target.ID.get(index))
      name = copy.copy(target.name.get(index))
      weight = copy.copy(target.weight.get(index))
      ########
      destination.clear(index)
      destination.place(tensor , index)
      destination.ID.place(ID , index)
      destination.name.place(name , index)
      destination.weight.place(weight , index)
      ########
  ###### End of _move()


  # Return a string of the distance from the 
  # endpoints of tensor1 to the endpoint of tensor2
  @staticmethod
  def _distance (tensor1 , tensor2):
        distance = torch.nn.PairwiseDistance(p=2)\
        .to(device = choosen_device , dtype = datatype)
        #######
        current = tensor1.to(device = choosen_device , dtype = datatype)
        ref = tensor2.to(device = choosen_device , dtype = datatype)
        dist = distance(current, ref).numpy()[0]
        return  str(round(dist , 4))
  ###### End of distance()

  # Return a string of the similary % between
  # Tensor1 and Tensor2
  @staticmethod
  def _similarity (tensor1 , tensor2 , target):
        distance = torch.nn.PairwiseDistance(p=2)\
        .to(device = choosen_device , dtype = datatype)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)\
        .to(device = choosen_device , dtype = datatype)
        origin = (target.origin)\
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
      current_weight = 1 #Will fix later
      current = vector.get(index)\
      .to(device = choosen_device , dtype = datatype)
      current = current*current_weight

      assert not current == None , "current vector is NoneType!"
      dist = round(distance(current , origin).numpy()[0],2)
      if output == None : 
          output = current
          log.append('Placed token with length '+ str(dist)  +' in new embedding ')
          continue
      output = torch.cat([output,current], dim=0)\
      .to(device = choosen_device , dtype = datatype)
      log.append('Placed token with length '+ str(dist)  +' in new embedding ')
    
    log.append('New embedding has '+ str(no_of_tokens) + ' tokens')
    log.append('-------------------------------------------')
    return output , '\n'.join(log)
  #### End of _concat_all()


  @staticmethod
  def _merge_all (vector , positive , negative):
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
      if negative.isEmpty.get(neg_index): continue
      no_of_negatives += 1
    #####

    dist = None
    candidate_vec = None
    tmp = None
    origin = vector.origin.to(device = choosen_device , dtype = datatype)

    for index in range (MAX_NUM_MIX):
      if (vector.isEmpty.get(index)): continue
      if (vector.isFiltered(index)): continue 
      no_of_tokens += 1

      current_weight = 1
      current = vector.get(index)\
      .to(device = choosen_device , dtype = datatype)
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
    randomGain = vector.randomization/100  
    size = vector.size
    gain = vector.gain
    N = vector.itermax
    T = 1/N  

    radialRandom = 1

    similarity_score = 0 #similarity index from 0 to 1, where 0 is no similarity at all
    similarity_minimum = None #similarity between output and the least similar input token
    similarity_maximum = None #similarity between output and the most similar input token
    mintoken = None #Token with least similarity
    maxtoken = None #Token with most similarity

    for step in range (N):
      rand_vec  = ((2*torch.rand(size) - torch.ones(size))\
      .to(device = choosen_device , dtype = datatype)).unsqueeze(0)
      rdist = distance(rand_vec, origin).numpy()[0]
      rando = (1/rdist) * rand_vec #Create random unit vector

      candidate_vec = norm_expected * (1 - randomGain)  + rando*randomGain

      minimum = 20000 #set initial minimum at impossible 200% similarity 
      maximum = -10000 #Set initial maximum at impossible -100% similarity
      similarity_sum = 0
      for index in range (MAX_NUM_MIX):
        if (vector.isEmpty.get(index)): continue
        if (vector.isFiltered(index)): continue

        #Get current token
        tmp = vector.get(index).to(device = choosen_device , dtype = datatype)
        dist = distance(tmp, origin).numpy()[0]
        current = (1/dist) * tmp 

        #Compute similarity of current token to 
        #candidate_vec
        similarity = 100*cos(current, candidate_vec).numpy()[0]
        worst_nsim = None
        worst_neg_index = None
        nsim = None
        neg_vec = None
        strength = negative.strength/100
        negative_similarity = None
        #######
        #Check negatives
        if no_of_negatives > 0: 
          tmp = None
          for neg_index in range(MAX_NUM_MIX) :
            if negative.isEmpty.get(neg_index): continue
            neg_vec = negative.get(neg_index)
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
      '( ' + vector.name.get(mintoken) + ')')

    log.append('Highest similarity : ' + str(similarity_maximum)+' % to token #' + str(maxtoken) + \
      '( ' + vector.name.get(maxtoken) + ')')

    if negsim != None :
      name = negative.name.get(worst_neg_index)
      log.append('Max similarity to negative tokens : ' + str(negsim) + ' %' + \
      "to token '" + name + "'")

    log.append('Average length of the input tokens : ' + str(dist_expected) )
    log.append('Length of the token average : ' + str(dist_of_mean) )
    log.append('Length of generated merged token : ' + str (output_length))
    log.append('New embedding has 1 token')
    log.append('-------------------------------------------')
  
    return output , '\n'.join(log)
  #####End of _merge_all()


  @staticmethod
  def _merge_if_similar(vector , positive , negative):
    log = []
    log.append('Interpolate Mode :')
    no_of_tokens = 0

    for index in range (MAX_NUM_MIX):
      if (vector.isEmpty.get(index)): continue
      if (vector.isFiltered(index)): continue
      no_of_tokens +=1
      continue

    if no_of_tokens <= 0: 
      log.append('No inputs in mixer')
      return None , '\n'.join(log) 

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    distance = torch.nn.PairwiseDistance(p=2)

    radialGain = 0.001 #deprecated
    randomGain = vector.randomization/100 
    lowerBound = vector.interpolation 
    origin = vector.origin.to(device= choosen_device , dtype = datatype)
    size = vector.size
    gain = vector.gain
    N = vector.itermax
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
      if (vector.isEmpty.get(index)): continue
      if (vector.isFiltered(index)): continue

      current_weight = 1
      current = vector.get(index)\
      .to(device = choosen_device , dtype = datatype)
      current = current*current_weight

      if prev == None : 
        prev = current
        continue

      assert not current == None , "current tensor is None!"
      assert not prev == None , "prev tensor is None!"

      tmp = ((current + prev) * 0.5).to(device = choosen_device , dtype = datatype) #Expected merge vector
      avg_dist = distance(tmp, origin).numpy()[0]
      merge_norm = ((1/avg_dist)*tmp).to(device = choosen_device , dtype = datatype)

      current_dist = distance(current, origin).numpy()[0]
      current_norm = ((1/current_dist)*current).to(device = choosen_device , dtype = datatype) 

      prev_dist = distance(prev, origin).numpy()[0]
      prev_norm = ((1/prev_dist)*prev).to(device = choosen_device , dtype = datatype)

      assert not ( 
        merge_norm == None or
        current_norm==None or
        prev_norm == None ) , "norm tensor is None!"

      #Randomization parameters: 
      radialRandom = (1 - randomGain) + randomGain*(2*random.random() - 1)

      if not (vector.allow_negative_gain): 
        if radialRandom<0: radialRandom = -radialRandom

      found = False 
      for step in range (N):
        iters+=1

        tmp  = ((torch.rand(size) - 0.5* torch.ones(size))\
        .to(device = choosen_device , dtype = datatype)).unsqueeze(0)
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
        .to(device = choosen_device , dtype = datatype)
        log.append('Placed token with length '+ str(output_length)  +' in new embedding ')

      else:
        log.append('Skipping merge between token ' + str(index-1)+ \
        ' and '+str(index))
        log.append('Token similarity ' + str(similarity) + \
        '% is less then req. similarity' + str(lowerBound) + '%')

    log.append('New embedding now has ' + str(no_of_tokens) + ' tokens')
    log.append('-------------------------------------------')
    return output , '\n'.join(log)
  ### End of _merge_if_similar()
  
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
