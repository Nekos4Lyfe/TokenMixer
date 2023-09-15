import gradio as gr
from modules import sd_hijack
import torch, os
from modules.textual_inversion.textual_inversion import Embedding
import collections, math, random , numpy
import re #used to parse word to int
import copy
from torch.nn.modules import ConstantPad1d, container
from library.toolbox.constants import MAX_NUM_MIX
from library.data import dataStorage

from library.toolbox.constants import TENSOR_DEVICE_TYPE , TENSOR_DATA_TYPE
choosen_device = TENSOR_DEVICE_TYPE
datatype = TENSOR_DATA_TYPE

from library.toolbox.constants import START_OF_TEXT_ID , END_OF_TEXT_ID
start_of_text_ID = START_OF_TEXT_ID
end_of_text_ID = END_OF_TEXT_ID

# Check that MPS is available (for MAC users)
#if torch.backends.mps.is_available(): 
#  choosen_device = torch.device("mps")
#else : choosen_device = torch.device("cpu")
#######

class MiniTokenizer:

  def Reset (self , mini_input , tokenbox) : 
    return '' , ''

  def isCutoff(self, ID):
    return ((ID == start_of_text_ID) or (ID == end_of_text_ID))

  def place (self , index ,\
    send_to_negatives , sendtomix , send_to_positives , send_to_temporary , \
    emb_id):

    is_sdxl = self.data.tools.is_sdxl
    #Do some checks
    valid_ID = (emb_id < self.data.tools.no_of_internal_embs) and emb_id>=0
    assert isinstance(emb_id , int) , \
    "Error: emb id is not an integer , it is a " + str(type(emb_id)) + " !" 
    if is_sdxl : valid_ID = valid_ID and \
    (emb_id < self.data.tools.no_of_internal_embs1280)
    assert valid_ID , "Error: ID with value " + str(emb_id) + \
    " is outside the range of permissable values from 0 to " + \
    str(self.data.tools.no_of_internal_embs)
    ####

    #### Append start-of-text token (if model is SDXL)
    if valid_ID :
        emb_vec = self.data.tools.internal_embs[emb_id]\
        .to(device = choosen_device , dtype = datatype)
        emb_name = self.data.emb_id_to_name(emb_id)
        assert emb_vec != None ,"emb_vec is NoneType!"

        #Add to 768 dimension vectors
        self.data.place(index ,\
            vector =  emb_vec.unsqueeze(0) ,
            ID =  emb_id ,
            name = emb_name , 
            to_negative = send_to_negatives ,
            to_mixer = sendtomix , 
            to_positive = send_to_positives , 
            to_temporary = send_to_temporary)

        if is_sdxl:
          sdxl_emb_vec = self.data.tools.internal_embs1280[emb_id]\
          .to(device = choosen_device , dtype = datatype)
          assert sdxl_emb_vec != None , "sdxl_emb_vec is NoneType!"

          #Add to 1280 dimension vectors
          self.data.place(index , 
              vector = sdxl_emb_vec.unsqueeze(0) ,
              ID = emb_id ,
              name = emb_name , 
              to_negative = send_to_negatives , 
              to_mixer = sendtomix , 
              to_positive = send_to_positives ,
              to_temporary = send_to_temporary , 
              use_1280_dim =True)
    ###############
    return valid_ID

  def Tokenize (self  , *args) :

    #Get the inputs
    mini_input = args[0]
    sendtomix = args[1]
    id_mode = args[2]
    send_to_negatives = args[3]
    random_token_length = args[4]
    send_to_positives = args[5]
    mix_input = args[6]
    stack_mode = args[7]
    literal_mode = args[8]
    random_token_length_randomization = (1/100) * args[9]
    name_list = []
    
    # Do some checks
    if mini_input == None : 
      for index in range(MAX_NUM_MIX):
        if self.data.vector.isEmpty.get(index): continue
        name_list.append(self.data.vector.name.get(index))
      return tokenmixer_vectors , '' , '' , '' , '' , gr.Dropdown.update(choices = name_list)
    assert not (
          sendtomix == None or
          id_mode == None or
          send_to_negatives == None or
          random_token_length == None or
          literal_mode == None
          ) , "NoneType in Tokenizer input!"
    ###########

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

    #Check if new embeddings have been added 
    try: sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
    except: 
      sd_hijack.model_hijack.embedding_db.dir_mtime=0
      sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
    self.data.update_loaded_embs() 
    ########

    #clear all slots to give room for new data
    for index in range(MAX_NUM_MIX):
      self.data.clear(index , 
          to_negative = send_to_negatives , 
          to_mixer = sendtomix and not stack_mode , 
          to_positive = send_to_positives ,
          to_temporary = send_to_temporary)
    ######


    #Convert text input to list of 'words'
    sentence = mini_input.strip().split()
    word_index = 0
    for splits in sentence :
      word_index += 1
    no_of_words = word_index 
    no_of_internal_embs = self.data.tools.no_of_internal_embs
    ########

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
    index = 0
    ########## Start loop : 
    for index in range(MAX_NUM_MIX):
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
        #index = index + 1
        placed = False
      ######
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
        if not reading_word:
          reading_word = True
          found_IDs = self.data.tools.get_emb_ids_from(word).numpy()
          found_IDs1280 = self.data.tools.get_emb_ids_from(word , use_1280_dim = True).numpy()

          found_vecs768 = None
          found_vecs1280 = None
          no_of_vecs768 = 0
          no_of_vecs1280 = 0

          if is_sdxl : 
            found_vecs768 = \
            self.data.tools.get_emb_vecs_from(word)

            ######
            found_vecs1280 = \
            self.data.tools.get_emb_vecs_from(word , use_1280_dim = True) 
            #######

          ##########
          if len(found_IDs)<=0:
            reading_word = False
            word_index -=1 
            continue
          ###########
          no_of_IDs = len(found_IDs)
          if is_sdxl: 
            assert no_of_IDs == found_vecs768.shape[0] , \
            "Size mismatch between found vecs and found IDs! , " + \
            "found_vecs768.shape[0] = " + str(found_vecs768.shape[0]) + \
            " and len(found_IDs) = " + str(no_of_IDs)
          #######
          ID_index = 0
        #######
        _ID = int(found_IDs[ID_index])
        #######

        if is_sdxl : 
          emb_vec = found_vecs768[ID_index]\
          .to(device = choosen_device) #Good method (SDXL)
          sdxl_emb_vec = found_vecs1280[ID_index]\
          .to(device = choosen_device)  # Good method(SDXL)
          assert sdxl_emb_vec != None , "sdxl_emb_vec is NoneType!"
        else : 
          emb_vec = self.data.emb_id_to_vec(_ID)\
          .to(device = choosen_device , dtype = datatype) # Bad method (SD1.5)
        #########
        emb_name = self.data.emb_id_to_name(_ID)
        assert _ID != None , "_ID is NoneType"
        assert emb_vec != None , "emb_vec is NoneType"
        ######
        #if is_sdxl: 
          #sdxl_emb_vec = self.data.emb_id_to_vec(_ID , use_1280_dim = True)\
          #.to(device= choosen_device , dtype = datatype) # Bad method (SDXL)
        #######
        if not self.isCutoff(_ID):
          assert not placed , "Overwrite error!"
          placed = True 
          self.data.place(index , 
            vector =  emb_vec.unsqueeze(0) ,
            ID =  _ID ,
            name = emb_name ,
            to_negative = send_to_negatives ,
            to_mixer = sendtomix , 
            to_positive = send_to_positives , 
            to_temporary = send_to_temporary)
          ########
          if is_sdxl:
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
        tokenbox =  tokenbox + emb_name + '_#' + str(_ID)
        ID_index+=1
        if ID_index+1> no_of_IDs : 
            reading_word = False
            word_index -=1 
            continue
        ##########
      #### End of 'Normal operation'
    ####### End loop

    #Try to append end_of_text_ID to output after loop
    last_index = None
    for index in range(MAX_NUM_MIX):
      if trailing_end_of_text : break
      if sendtomix and not self.data.vector.isEmpty.get(index): continue
      last_index = copy.copy(index)
    ########

    if not trailing_end_of_text:
      self.place (last_index  , \
      send_to_negatives , sendtomix , send_to_positives , send_to_temporary , \
      end_of_text_ID) 
      if (tokenbox != '') : tokenbox = tokenbox + ' , '
      tokenbox =  tokenbox + "<|endoftext|>_#49407"
    ###### End of append end_of_text_ID to output

    #Filter stuff
    name_list = []
    for index in range(MAX_NUM_MIX):
        if self.data.vector.isEmpty.get(index): continue
        name_list.append(self.data.vector.name.get(index))
    ######### End of filter stuff

    return tokenmixer_vectors , tokenbox , splitbox , negbox , posbox , gr.Dropdown.update(choices = name_list)
    
    
  def setupIO_with (self, module):
    #Tell the buttons in this class what to do when 
    #pressed by the user

    cond = self.data.tools.loaded
    input_list = []
    output_list = []

    if (module.ID == "MiniTokenizer") and cond:
      output_list.append(self.inputs.mini_input)
      output_list.append(self.outputs.tokenbox)
      self.buttons.reset.click(fn = self.Reset, inputs = input_list , outputs = output_list)


    if (module.ID == "TokenMixer") and cond:
      input_list.append(self.inputs.mini_input)   #0
      input_list.append(self.inputs.sendtomix)    #1
      input_list.append(self.inputs.id_mode)      #2
      input_list.append(self.inputs.negatives)    #3
      input_list.append(self.inputs.randlen)      #4
      input_list.append(self.inputs.positives)    #5
      input_list.append(module.inputs.mix_input)  #6
      input_list.append(self.inputs.stack_mode)   #7
      input_list.append(self.inputs.literal_mode) #8
      input_list.append(self.inputs.randlenrand)  #9
      #######
      output_list.append(module.inputs.mix_input) #0
      output_list.append(self.outputs.tokenbox)   #1
      output_list.append(self.outputs.splitbox)   #2
      output_list.append(module.inputs.negbox)    #3
      output_list.append(module.inputs.posbox)    #4
      output_list.append(module.inputs.unfiltered_names)  #5
      #######
      self.buttons.tokenize.click(fn=self.Tokenize , inputs = input_list , outputs = output_list)

  #End of setupIO_with()


  def __init__(self, label , vis = False):

    #Pass reference to global object "dataStorage" to class
    self.data = dataStorage 

    class Outputs :
      def __init__(self):
        Outputs.tokenbox = []

    class Inputs :
      def __init__(self):
        Inputs.mini_input = []
        Inputs.sendtomix = []
        Inputs.id_mode = []
        Inputs.randlen = []
        Inputs.positives = []
        Inputs.stack_mode = []
        Inputs.literal_mode =[]
        Inputs.randlenrand = []

    class Buttons :
      def __init__(self):
        Buttons.tokenize = []
        Buttons.reset = []

    self.outputs= Outputs()
    self.inputs = Inputs()
    self.buttons= Buttons()
    self.ID = "MiniTokenizer"

    #test = gr.Label('Prompt MiniTokenizer' , color = "red")
    #create UI
    #
    with gr.Accordion(label , open=True , visible = vis) as show :
      with gr.Row() :  
          self.buttons.tokenize = gr.Button(value="Tokenize", variant="primary", size="sm")
          self.buttons.reset = gr.Button(value="Reset", variant = "secondary", size="sm")
      with gr.Row() : 
        self.inputs.mini_input = gr.Textbox(label='', lines=2, \
        placeholder="Enter a short prompt or name of embedding" , interactive = True)
      with gr.Row() : 
        self.outputs.tokenbox = gr.Textbox(label="CLIP tokens", lines=2, interactive = False)
        self.outputs.splitbox = gr.Textbox(label="Embedded vectors", lines=2, interactive = False)
      with gr.Row():
          self.inputs.sendtomix = gr.Checkbox(value=False, label="Send to input", interactive = True)
          self.inputs.id_mode = gr.Checkbox(value=False, label="ID input mode", interactive = True)
          self.inputs.negatives = gr.Checkbox(value=False, label="Send to negatives", interactive = True , visible = True) 
          self.inputs.positives = gr.Checkbox(value=False, label="Send to positives", interactive = True , visible = True)
          self.inputs.stack_mode = gr.Checkbox(value=False, label="Stack Mode", interactive = True)
          self.inputs.literal_mode = gr.Checkbox(value=False, label="String Literal Mode", interactive = True)
      with gr.Accordion("Random ' _ ' token settings" ,open=False , visible = True) as randset : 
        self.inputs.randlen = gr.Slider(minimum=0, maximum=10, step=0.01, label="Randomized ' _ ' token max length", default=0.35 , interactive = True)
        self.inputs.randlenrand = \
          gr.Slider(value = 50 , minimum=0, maximum=100, step=0.1, \
          label="Randomized ' _ ' token length randomization %", default=50 , interactive = True)
      
      with gr.Accordion('Tutorial : What is this?',open=False , visible= False) as tutorial_0 :
          gr.Markdown("The Minitokenizer is a tool which allows you to get CLIP tokens " + \
          "from a text prompt. The MiniTokenizer can also split embedddings into individual tokens. \n \n " +  \
          "You can write the name of an embedding to get its single vector components. " + \
          "By writing example[3:5] you can extract vector 3 to 5 in an embedding. \n \n " + \
          "By writing ' _ ' you can generate random vector. \n \n To input token IDs " + \
          " intead of token names , check the ID input mode checkbox and " + \
          "write a ID number between 0 and" + str(self.data.tools.no_of_internal_embs) + \
          ",  like for example '6228 20198' which gives the " + \
          "tokens 'example' and 'prompt' . \n \n " + \
          "when using SDXL , you can write '<' to place a 'start-of-text' token  " + \
          " and '>' to place a 'end-of-text' token in the output , e.g " + \
          " < red banana > < blue car > ." + \
          "Observe : When you write example[3:5] and send the vectors to the mixer, " + \
          "the vectors will not show up at the top. You will need to scroll down to see them. \n \n " + \
          "The code for this module is built upon the extension provided by tkalayci71: \n " + \
          "https://github.com/tkalayci71/embedding-inspector")
    #end of UI

    self.tutorials = [tutorial_0]
    self.show = [show]
    self.randset = [randset]

    self.inputs.randlen.value = 0.35

    if self.data.tools.loaded : self.setupIO_with(self)
## End of class MiniTokenizer--------------------------------------------------#