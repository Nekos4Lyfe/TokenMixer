import gradio as gr
from modules import sd_hijack
import torch, os
from modules.textual_inversion.textual_inversion import Embedding
import collections, math, random , numpy
import re #used to parse word to int
import copy
from torch.nn.modules import ConstantPad1d, container
from lib.toolbox.constants import MAX_NUM_MIX
from lib.data import dataStorage

class MiniTokenizer:

  def Reset (self , mini_input , tokenbox) : 
    return '' , ''


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
    
    send_to_temporary = False

    if mini_input == None : 
      for index in range(MAX_NUM_MIX):
        if self.data.vector.isEmpty.get(index): continue
        name_list.append(self.data.vector.name.get(index))
      #####
      return tokenmixer_vectors , '' , '' , '' , '' , gr.Dropdown.update(choices = name_list)

    assert not (
          sendtomix == None or
          id_mode == None or
          send_to_negatives == None or
          random_token_length == None or
          literal_mode == None or
          send_to_temporary == None
          ) , "NoneType in Tokenizer input!"

    tokenmixer_vectors = ''
    if not sendtomix : tokenmixer_vectors= mix_input

    distance = torch.nn.PairwiseDistance(p=2)
    origin = self.data.vector.origin.cpu()
    #get_embedding_info
    #Check if new embeddings have been added 
    try: sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
    except: 
      sd_hijack.model_hijack.embedding_db.dir_mtime=0
      sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
    self.data.update_loaded_embs() 

    #clear all slots to give room for new data
    for index in range(MAX_NUM_MIX):
      self.data.clear(index , 
          to_negative = send_to_negatives , 
          to_mixer = sendtomix and not stack_mode , 
          to_positive = send_to_positives ,
          to_temporary = send_to_temporary)
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
    posbox = ''
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
        #####
        pos_name = self.data.positive.name.get(index - 1)
        if pos_name != None :
          if (posbox != '') : posbox = posbox + ' , ' 
          posbox = posbox + pos_name
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
        emb_vec = torch.rand(self.data.vector.size).to(device = "cpu" , dtype = torch.float32)
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
            to_positive = send_to_positives , 
            to_temporary = send_to_temporary)

        if (tokenbox != '') : tokenbox = tokenbox + ' , '
        tokenbox =  tokenbox + emb_name + '_#' + str(emb_id)
        word_index -=1 
        continue
      ##########

      #Extract emb_vec from emb_id if id_mode is selected
      if (id_mode and word.isdigit()):
        emb_id = int(word)
        if emb_id >= no_of_internal_embs: continue
        emb_vec = self.data.tools.internal_embs[emb_id]\
        .to(device = "cpu" , dtype = torch.float32)
        emb_name = self.data.emb_id_to_name(emb_id)
       
        assert emb_vec != None , "emb_vec is NoneType"
        self.data.place(index , 
            vector =  emb_vec.unsqueeze(0) ,
            ID =  emb_id  ,
            name = emb_name , 
            to_negative = send_to_negatives , 
            to_mixer = sendtomix , 
            to_positive = send_to_positives , 
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
      ###
      #SDXL Stuff
      sdxl_emb_name = None
      sdxl_emb_ids = None
      sdxl_emb_vecs = None 
      sdxl_loaded_emb = None
      if self.tools.is_sdxl:
        sdxl_emb_name, sdxl_emb_ids, sdxl_emb_vecs , sdxl_loaded_emb  = \
        self.data.get_embedding_info(tmp , is_sdxl = True)
      ########

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
          .to(device = "cpu" , dtype = torch.float32)
          no_of_tokens +=1
          break
      #########

      if no_of_tokens > 1 :
          if (token_num+1>min(end, no_of_tokens)) or (start>end) :
            token_num = 0  #Reset token_num
            word_index -=1 #Go to next word
            continue

          if (token_num<start):
            token_num += 1 #Skip until token_num==start
            continue

          #Fetch the vector
          emb_vec = emb_vecs[token_num].to(device = "cpu" , dtype = torch.float32)
          assert emb_vec != None , "emb_vec is NoneType"
          ####
          sdxl_emb_vec = None
          if self.tools.is_sdxl: 
            sdxl_emb_vec = sdxl_emb_vecs[token_num]\
            .to(device = "cpu" , dtype = torch.float32)
            assert sdxl_emb_vec != None , "sdxl_emb_vec is NoneType"
          ######

          self.data.place(index , 
              vector = emb_vec.unsqueeze(0) ,
              ID = 0 ,
              name = emb_name + '_' + str(token_num) , 
              to_negative = send_to_negatives , 
              to_mixer = sendtomix , 
              to_positive = send_to_positives ,
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
        emb_vec = self.data.emb_id_to_vec(_ID).to(device = "cpu" , dtype = torch.float32)

        assert emb_vec != None , "emb_vec is NoneType"

        if not _ID ==318:
          self.data.place(index , 
            vector =  emb_vec.unsqueeze(0) ,
            ID =  _ID ,
            name = emb_name ,
            to_negative = send_to_negatives ,
            to_mixer = sendtomix , 
            to_positive = send_to_positives , 
            to_temporary = send_to_temporary)

        ID_index+=1 
        if ID_index+1> no_of_IDs : 
            found_IDs = None
            word_index -=1 
     
        if (tokenbox != '') and not _ID ==318 : tokenbox = tokenbox + ' , '
        tokenbox =  tokenbox + emb_name + '_#' + str(_ID)
    ####### End loop

    name_list = []
    for index in range(MAX_NUM_MIX):
        if self.data.vector.isEmpty.get(index): continue
        name_list.append(self.data.vector.name.get(index))
      #####

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