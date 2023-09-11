import gradio as gr
from modules import sd_hijack
import torch, os
import collections, math, random , numpy
import re #used to parse word to int
import copy
from library.toolbox.constants import MAX_NUM_MIX

from library.data import dataStorage

# Check that MPS is available (for MAC users)
choosen_device = torch.device("cpu")
#if torch.backends.mps.is_available(): 
#  choosen_device = torch.device("mps")
#else : choosen_device = torch.device("cpu")
#######

class TokenComparator :

  def reset (self , mini_input , tokenbox) : 
    return '' , ''

  def compare (self, *args):

    first_text  = args[0]
    second_text = args[1]
    show_similarity = args[2]
    stack_mode = args[3]
    mix_input = args[4]
    show_distance = args[5]
    negbox_input = args[6]
    id_mode = args[7]
    literal_mode = args[8]
    #######
    tokenmixer_vectors = mix_input 
    negbox = negbox_input
    tokenbox= []
    ########

    #clear temporary data
    for index in range(MAX_NUM_MIX):
      self.data.temporary.clear(index)
    ####
  
    first_compounds = []
    first_compound = [None , None]
    first_args = self._minitokenizer_input(first_text , \
    id_mode , mix_input , literal_mode)
    ###
    self.tokenize(*first_args) #Run the copy-pasted Tokenize() function
    ###
    for index in range(MAX_NUM_MIX):
      if self.data.temporary.isEmpty.get(index): break
      first_compound[0] = self.data.temporary.name.get(index) #vector name (string)
      first_compound[1] = self.data.temporary.get(index) #vector (torch.tensor)
      first_compounds.append(list(first_compound))
    ######

    #clear temporary data
    for index in range(MAX_NUM_MIX):
      self.data.temporary.clear(index)
    ####

    second_compounds = []
    second_compound = [None , None]
    second_args = self._minitokenizer_input(second_text , \
    id_mode , mix_input , literal_mode)
    ###
    self.tokenize(*second_args) #Run the copy-pasted Tokenize() function
    ###
    for index in range(MAX_NUM_MIX):
      if self.data.temporary.isEmpty.get(index): break
      second_compound[0] = self.data.temporary.name.get(index) #vector name (string)
      second_compound[1] = self.data.temporary.get(index) #vector (torch.tensor)
      second_compounds.append(list(second_compound))
    #######

    #Compute the similarities
    compounds = set()
    #####
    for first_compound in first_compounds :
      for second_compound in second_compounds :
        compound = []
        compound.append(first_compound[0]) #first vector name
        compound.append(second_compound[0]) #second vector name
        #####
        if show_similarity : 
          compound.append(self.data.similarity(\
            first_compound[1] , \
            second_compound[1]))
        else: compound.append('')
        
        if show_distance :
          compound.append(self.data.distance(\
            first_compound[1] , \
            second_compound[1]))
        else: compound.append('')
        ####
        compounds.add(tuple(compound))
      ########
    #########

    #Print the results
    tokenbox.append("Metrics for vector pairs : ")

    for compound in list(compounds):

      tokenbox.append("............." + \
      "................................." + \
      "................................." + \
      ".....................................")

      tokenbox.append(\
      "'" + compound[0] + "'" + \
       "  <-->  " +  \
       "'" + compound[1] + "'")

      if show_similarity: 
        tokenbox.append("Similarity  : " + compound[2] + "  %" )
      
      if show_distance :
        tokenbox.append("Distance  : " + compound[3])
      
      tokenbox.append(" ")

    #########

    return tokenmixer_vectors , negbox , '\n'.join(tokenbox)


  def _minitokenizer_input(self, mini_input , \
  id_mode , mix_input , literal_mode) :

    #Set data to the "temporary folder"
    sendtomix = False
    send_to_negatives = False
    send_to_temporary = True
    #####

    #Set fixed values
    random_token_length = 0.35
    random_token_length_randomization = 0
    ########

    #Don't use stack mode
    stack_mode = False
    ########

    args = []
    args.append(mini_input)
    args.append(sendtomix) 
    args.append(id_mode)
    args.append(send_to_negatives)
    args.append(random_token_length)
    args.append(send_to_temporary)
    args.append(mix_input)
    args.append(stack_mode)
    args.append(literal_mode)
    args.append(random_token_length_randomization)

    return args

  #COPY PASTE FROM MINITOKENIZER FUNCTION
  def tokenize (self  , *args) :

    mini_input = args[0]
    sendtomix = args[1]
    id_mode = args[2]
    send_to_negatives = args[3]
    random_token_length = args[4]
    send_to_temporary = args[5]
    mix_input = args[6]
    stack_mode = args[7]
    literal_mode = args[8]
    random_token_length_randomization = (1/100) * args[9]

    if mini_input == None : 
      return tokenmixer_vectors , '' , '' , ''

    assert not (
          sendtomix == None or
          id_mode == None or
          send_to_negatives == None or
          random_token_length == None or
          literal_mode == None or
          send_to_temporary == None
          ) , "NoneType in Tokenizer input!"

    tokenmixer_vectors = mix_input
    if sendtomix : tokenmixer_vectors= ''

    distance = torch.nn.PairwiseDistance(p=2)
    origin = self.data.vector.origin\
    .to(device = choosen_device , dtype = torch.float32)

    #Check if new embeddings have been added 
    try: sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
    except: 
      sd_hijack.model_hijack.embedding_db.dir_mtime=0
      sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
    self.data.update_loaded_embs() 

    #clear all slots to give room for new data
    for index in range(MAX_NUM_MIX):
      self.data.clear(index , 
          to_negative = send_to_negatives and not stack_mode , 
          to_mixer = sendtomix and not stack_mode , 
          to_temporary = send_to_temporary and not stack_mode)
    ######

    sentence = mini_input.strip().split()
    word_index = 0
    for splits in sentence :
      word_index += 1
    no_of_words = word_index 
    no_of_internal_embs = self.data.tools.no_of_internal_embs

    #Parameters
    section = None
    tokenbox = ''
    splitbox = ''
    negbox = ''
    emb_name = None
    found_IDs = None
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

    ########## Start loop : 
    for index in range(MAX_NUM_MIX):
      if sendtomix and not self.data.vector.isEmpty.get(index): continue
      ######
      if index>0: #Store the values from the previous iteration
        neg_name = self.data.negative.name.get(index - 1)
        if neg_name != None :
          if (negbox != '') : negbox = negbox + ' , ' 
          negbox = negbox + neg_name
        ######
        name = self.data.vector.name.get(index - 1)
        if name != None and sendtomix:
          if tokenmixer_vectors != '': tokenmixer_vectors = tokenmixer_vectors + ' , '
          tokenmixer_vectors = tokenmixer_vectors + name
      ######

      if not word_index>0: continue

      word = sentence[no_of_words-word_index]


      if word == "," :  
        word_index -=1 
        continue  #Ignore comma inputs

      if word == "_":
        emb_vec = torch.rand(self.data.vector.size)
        dist = distance(emb_vec , origin).numpy()[0]
        tmp = random_token_length * \
        (1 - random_token_length_randomization*random.random())
        emb_vec = (tmp/dist)*emb_vec
        emb_id = 0
        emb_name = "random_" + str(index)

        self.data.place(index , 
            vector =  emb_vec.unsqueeze(0) ,
            ID =  emb_id ,
            name = emb_name , 
            to_negative = send_to_negatives ,
            to_mixer = sendtomix , 
            to_temporary = send_to_temporary
            )

        if (tokenbox != '') : tokenbox = tokenbox + ' , '
        tokenbox =  tokenbox + emb_name + '_#' + str(emb_id)
        word_index -=1 
        continue
      ##########

      #Extract emb_vec from emb_id if id_mode is selected
      if (id_mode and word.isdigit()):
        emb_id = int(word)
        if emb_id >= no_of_internal_embs: continue
        emb_vec = self.data.tools.internal_embs[emb_id]
        emb_name = self.data.emb_id_to_name(emb_id)
       
        assert emb_vec != None , "emb_vec is NoneType"
        self.data.place(index , 
            vector =  emb_vec.unsqueeze(0) ,
            ID =  emb_id  ,
            name = emb_name , 
            to_negative = send_to_negatives , 
            to_mixer = sendtomix , 
            to_temporary = send_to_temporary)

        if (tokenbox != '') : tokenbox = tokenbox + ' , '
        tokenbox =  tokenbox + emb_name + '_#' + str(emb_id)
        word_index -=1 
        continue
      #########
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
      emb_name, emb_ids, emb_vecs , loaded_emb  = self.data.get_embedding_info(tmp)
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
          emb_vec = self.data.emb_id_to_vec(emb_id)
          no_of_tokens +=1
          break
      #########

      if no_of_tokens > 1 :
          if (token_num+1>min(end, no_of_tokens)) or (start>end) :
            token_num = 0  #reset token_num
            word_index -=1 #Go to next word
            continue

          if (token_num<start):
            token_num += 1 #Skip until token_num==start
            continue

          emb_vec = emb_vecs[token_num]

          assert emb_vec != None , "emb_vec is NoneType"
          self.data.place(index , 
              vector = emb_vec.unsqueeze(0) ,
              ID = 0 ,
              name = emb_name + '_' + str(token_num) , 
              to_negative = send_to_negatives , 
              to_mixer = sendtomix , 
              to_temporary = send_to_temporary)
          

          if (splitbox != '') : splitbox = splitbox + ' , '    
          splitbox =  splitbox + emb_name + '_' + str(token_num)
          token_num += 1
      ###########
      else:
        if found_IDs == None:
          found_IDs = self.data.text_to_emb_ids(word)
          no_of_IDs = len(found_IDs)
          ID_index = 0
        

        _ID = found_IDs[ID_index] 
        
        emb_name = self.data.emb_id_to_name(_ID)
        emb_vec = self.data.emb_id_to_vec(_ID)

        assert emb_vec != None , "emb_vec is NoneType"

        if not _ID ==318:
          self.data.place(index , 
            vector =  emb_vec.unsqueeze(0) ,
            ID =  _ID ,
            name = emb_name ,
            to_negative = send_to_negatives ,
            to_mixer = sendtomix , 
            to_temporary = send_to_temporary)

        ID_index+=1 
        if ID_index+1> no_of_IDs : 
            found_IDs = None
            word_index -=1 
     
        if (tokenbox != '') and not _ID ==318 : tokenbox = tokenbox + ' , '
        tokenbox =  tokenbox + emb_name + '_#' + str(_ID)
    ####### End loop
        
    return tokenmixer_vectors , tokenbox , splitbox , negbox
       
       
    
  def setupIO_with (self, module):

      cond = self.data.tools.loaded
      input_list = []
      output_list = []

      if (module.ID == "Comparator") and cond:
        return 1
      
      if (module.ID == "TokenMixer") and cond:
        input_list.append(self.inputs.first)             #0
        input_list.append(self.inputs.second)            #1
        input_list.append(self.inputs.show_similarity)   #2
        input_list.append(self.inputs.stack_mode)        #3
        input_list.append(module.inputs.mix_input)       #4
        input_list.append(self.inputs.show_distance)     #5
        input_list.append(module.inputs.negbox)          #6
        input_list.append(self.inputs.id_mode)           #7
        input_list.append(self.inputs.literal_mode)      #8
        #######
        output_list.append(module.inputs.mix_input)      #0
        output_list.append(module.inputs.negbox)         #1
        output_list.append(self.outputs.tokenbox)        #3
        #######
        self.buttons.compare.click(fn=self.compare , inputs = input_list , outputs = output_list)


  def __init__(self , label , vis = False ) :

      self.data = dataStorage
      self.ID = "Comparator"

      class Outputs :
        def __init__(self):
          Outputs.tokenbox = []

      class Inputs :
        def __init__(self):
          Inputs.input_image = []
          Inputs.sendtomix = []
          Inputs.stack_mode = []
          Inputs.send_to_negative = []
          Inputs.length = []
          Inputs.show_similarity = []
          Inputs.show_distance = []

      class Buttons :
        def __init__(self):
          Buttons.compare = []
          Buttons.reset = []

      self.outputs= Outputs()
      self.inputs = Inputs()
      self.buttons= Buttons()
      self.refresh_symbol = "\U0001f504"  # 🔄


      with gr.Accordion(label , open=True , visible = vis) as show :
        gr.Markdown("Compare tokens or embeddings")
  
        with gr.Row() :  
          self.buttons.compare = gr.Button(value="Compare", variant="primary", size="sm")
          self.buttons.reset = gr.Button(value="Reset", variant = "secondary", size="sm") 
        with gr.Row() :  
          self.inputs.first = gr.Textbox(label='', lines=2, \
            placeholder="Enter name of token or embedding" , interactive = True)

          self.inputs.second = gr.Textbox(label='', lines=2, \
            placeholder="Enter name of token or embedding" , interactive = True)

        with gr.Row() : 
          self.outputs.tokenbox = gr.Textbox(label = '' , lines=2, interactive = False)
        with gr.Row() :   
            self.inputs.show_similarity = gr.Checkbox(value=True, label="Show similarity %", interactive = True)
            self.inputs.show_distance = gr.Checkbox(value=False, label="Show distance", interactive = True)
            
            self.inputs.stack_mode = gr.Checkbox(\
            value=False, label="Stack Mode", interactive = True , visible = False)
            self.inputs.send_to_negative = gr.Checkbox(\
            value=False, label="Send to negatives", interactive = True , visible =False)
            
            self.inputs.id_mode = gr.Checkbox(value=False, label="ID input mode", interactive = True)
            self.inputs.literal_mode = gr.Checkbox(value=False, label="String Literal Mode", interactive = True)



        with gr.Accordion('Tutorial : What is this?',open=False , visible= False) as tutorial_0 :
          gr.Markdown("This tutorial has not been written yet.")

      self.tutorials = [tutorial_0]
      self.show = [show]
      if self.data.tools.loaded : self.setupIO_with(self)


      # Need to check the input_image for API compatibility
      #if isinstance(raw_input_image['input_image'], str):
      #  from library.pfd.utils import decode_base64_to_image
      #  input_image = HWC3(np.asarray(decode_base64_to_image(raw_input_image['input_image'])))
      #else:
      #  input_image = HWC3(raw_input_image['input_image'])

        
