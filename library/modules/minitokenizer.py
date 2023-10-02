#MINITOKENIZER.PY
import gradio as gr
from modules import sd_hijack
import torch, os
from modules.textual_inversion.textual_inversion import Embedding
import random , numpy
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

class MiniTokenizer:
    
    def Reset (self , mini_input , tokenbox) : 
        return '' , ''
    
    @staticmethod
    def isCutoff(_ID):
      return _ID == start_of_text_ID or _ID == end_of_text_ID

    def get_emb_vec_from(self , _ID , use_1280_dim = False):

      #Helper function which fetches an embedding vector
      #from an ID with flag exceptions included
      # Unlike the function get_emb_vecs_from in Tools.py , 
      # this helper function can only process a single ID
      # and return a single embedding vector

      # Get flags
      use_random , use_start_of_text , use_end_of_text = \
      self.data.tools.get_token_type_from(_ID)

      # Get functions
      get_emb_vecs_from = self.data.tools.get_emb_vecs_from
      random = self.data.tools.random
      ######

      #Fetch the vector from ID
      emb_vec = get_emb_vecs_from(_ID , use_1280_dim)
      #######

      # Overwrite with different embedding vector 
      # for certain types of ID:s
      if use_random : emb_vec = random(use_1280_dim)
      if use_start_of_text : emb_vec = \
        get_emb_vecs_from(start_of_text_ID , use_1280_dim)
      if use_end_of_text : emb_vec = \
        get_emb_vecs_from(end_of_text_ID , use_1280_dim)
      ######

      return emb_vec.to(device = choosen_device , dtype = datatype)
    ### End of get_emb_vec_from()

 
    def place(self , input , \
      send_to_vector = False , send_to_positives = False  , \
      send_to_negatives = False):

      #Places token with given _ID at index
      # in the given fields of the dataclass 
      #(vector , positive , negative , temporary)
        
        # Fetch _IDs from input
        assert input != None , "_IDs is NoneType!"
        _IDs = None
        if type(_IDs) == torch.Tensor : 
          _IDs = input.to(choosen_device).numpy()
        else: 
          assert type(input) == list , "_IDs is neither a tensor nor a list. " + \
          "_IDs is the type " + str(type(_IDs)) + " !" 
          _IDs = input
          assert len(_IDs)>0 , "_IDs is an empty list!"
        ###########

        # Fetch flags
        model_is_sdxl , is_sd2 , is_sd1 = self.data.tools.get_flags()
        ######

        #Fetch functions
        get_emb_vec_from = self.get_emb_vec_from
        get_name_from = self.data.tools.get_name_from
        isEmpty_at = self.data.vector.isEmpty.get
        isCutoff = self.isCutoff
        token = self.data.tools.token
        #######

        # Initialize params
        emb_vec768 = None
        emb_vec1280 = None
        rand_index = 0
        count = 0
        no_of_IDs = len(_IDs)
        #######

        for _ID in _IDs:
            if isCutoff(_ID): continue #Skip ahead of all padding tokens
            count = count + 1

            # Fetch the name of the vector
            emb_name = get_name_from(_ID , using_index = count)
            assert emb_name != None ,"emb_name is NoneType!"
            ##########

            # Fetch 768 Dimension vectors
            if True: 
              emb_vec768 = get_emb_vec_from(_ID)
              assert emb_vec768 != None ,"emb_vec768 is NoneType!"
            ###########

            # Fetch 1280 Dimension vectors
            if model_is_sdxl:
              emb_vec1280 = get_emb_vec_from(_ID)
              assert emb_vec1280 != None , "emb_vec1280 is NoneType!"
            ###########
        
            #Add to 768 dimension vector
            for index in range (MAX_NUM_MIX):
              if not isEmpty_at(index): continue
              self.data.place(index ,\
                  vector =  emb_vec768.unsqueeze(0) ,
                  ID =  _ID ,
                  name = emb_name , 
                  to_negative = send_to_negatives ,
                  to_mixer = send_to_vector , 
                  to_positive = send_to_positives)
            ######### End of of for-loop
        
            #Add to 1280 dimension vectors 
            for index in range (MAX_NUM_MIX):
              if not model_is_sdxl: break
              if not isEmpty_at(index): continue
              self.data.place(index , 
                  vector = emb_vec1280.unsqueeze(0) ,
                  ID = _ID ,
                  name = emb_name , 
                  to_negative = send_to_negatives , 
                  to_mixer = send_to_vector , 
                  to_positive = send_to_positives , 
                  use_1280_dim =True)
            ######## End of for-loop  
      ######## End of place()

    def Process (self  , *args) :

      #Get the inputs
      mini_input = args[0]
      send_to_vector = args[1]
      id_mode = args[2]
      send_to_negatives = args[3]
      random_token_length = args[4]
      send_to_positives = args[5]
      mix_input = args[6]
      stack_mode = args[7]
      literal_mode = args[8]
      random_token_length_randomization = (1/100) * args[9]

      # Fetch functions
      tokenize = self.data.tools.tokenize
      place = self.place 
      concat = self.concat
      get_emb_vecs_from = self.data.tools.get_emb_vecs_from
      name_at = self.data.vector.name.get
      ID_at = self.data.vector.ID.get
      isEmpty_at = self.data.vector.isEmpty.get
      ########

     # Clear all slots to give room for new data
      for index in range(MAX_NUM_MIX):
        self.data.clear(index , 
          to_negative = send_to_negatives , 
          to_mixer = send_to_vector and not stack_mode , 
          to_positive = send_to_positives)
      ######

      # Set default value for output params
      name_list = []
      tokenbox = ''
      embgenbox = ''
      splitbox = ''
      negbox = ''
      posbox = ''
      #######

      # If mini_input is empty , return nothing
      if mini_input == None : 
        return '' , '' , '' , '' , '' , gr.Dropdown.update(choices = name_list)
      ##########
      
      # Do some checks
      assert not (
          send_to_vector == None or
          id_mode == None or
          send_to_negatives == None or
          random_token_length == None or
          literal_mode == None
          ) , "NoneType in Tokenizer input!"
      ###########

      # Update Tools.py with randomizaton settings
      self.data.tools.random_token_length = random_token_length
      self.data.tools.random_token_length_randomization = \
      random_token_length_randomization
      #########

      #Check if new embeddings have been added 
      try: sd_hijack.model_hijack.embedding_db.\
      load_textual_inversion_embeddings(force_reload=True)
      except: 
        sd_hijack.model_hijack.embedding_db.dir_mtime=0
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
      self.data.update_loaded_embs() 
      #######
    
      # Get flags
      is_sdxl , is_sd2 , is_sd1 = self.data.tools.get_flags()
      #####

      # START PROCESSING THE VECTORS

      # Place the start-of-text cutoff token
      place(start_of_text_ID , \
        send_to_vector , send_to_positives  , send_to_negatives)
      #########

      # Place the tokenized vectors (excluding any cutoff tokens)
      _IDs = tokenize(mini_input , max_length = True)
      emb_vecs = get_emb_vecs_from(mini_input)
      place(_IDs , send_to_vector , send_to_positives  , send_to_negatives)
      ######### 

      # Place the end-of-text cutoff token
      place(end_of_text_ID , \
        send_to_vector , send_to_positives  , send_to_negatives)
      ########

      # END PROCESSING THE VECTORS

       # Process output params
      if not send_to_vector :  embgenbox = mix_input
      for index in range(MAX_NUM_MIX):
        if isEmpty_at(index) : continue
        name = name_at(index)
        #emb_id = ID_at(index)
        if tokenbox != '': tokenbox = tokenbox + " , "
        if embgenbox != '': embgenbox = embgenbox + " , "
        name_list.append(name)
        tokenbox = tokenbox + name
        embgenbox = embgenbox + name
      #########

      return embgenbox , tokenbox , splitbox , negbox , posbox , gr.Dropdown.update(choices = name_list)
    

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
        self.buttons.tokenize.click(fn=self.Process , inputs = input_list , outputs = output_list)
    #End of setupIO_with()


    def __init__(self, label , vis = False):

      #Pass reference to global object "dataStorage" to class
      self.data = dataStorage 

      #Random vector generation from "_" symbol
      self.random_token_length = 0
      self.random_token_length_randomization = 0
      ########

      class Outputs :
        def __init__(self):
          Outputs.tokenbox = []
      ### End of Outputs() class

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
      ### End of Inputs() class

      class Buttons :
        def __init__(self):
          Buttons.tokenize = []
          Buttons.reset = []
      ### End of Buttons() class

      self.outputs= Outputs()
      self.inputs = Inputs()
      self.buttons= Buttons()
      self.ID = "MiniTokenizer"
    
      with gr.Accordion(label , open=True , visible = vis) as show :
        with gr.Row() :  
          self.buttons.tokenize = gr.Button(value="Process", variant="primary", size="sm")
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
          self.inputs.randlen = gr.Slider(value = 0.35 , minimum=0, maximum=10, step=0.01, label="Randomized ' _ ' token max length", default=0.35 , interactive = True)
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

      # Add some stuff for setting up the UI
      self.tutorials = [tutorial_0]
      self.show = [show]
      self.randset = [randset]
      model_is_loaded = self.data.tools.loaded
      if model_is_loaded : self.setupIO_with(self)
      #######

## End of class MiniTokenizer--------------------------------------------------#




