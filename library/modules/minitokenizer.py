#MINITOKENIZER.PY
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


class MiniTokenizer:

  def Reset (self , mini_input , tokenbox) : 
    return '' , ''

  # Checks if an ID is valid
  def assert_ID(self, emb_id):
    assert isinstance(emb_id , int) , \
    "Error: emb id is not an integer , it is a " + str(type(emb_id)) + " !" 
    if self.data.tools.is_sdxl : 
      assert self.data.tools.no_of_internal_embs1280 == \
      self.data.tools.no_of_internal_embs , \
      "Mismatch between no. of IDs in no_of_internal_embs1280 " + \
      "and no_of_internal_embs for loaded SDXL model!"
    ########
    assert (emb_id < self.data.tools.no_of_internal_embs) and emb_id>=0 , \
    "Error: ID with value " + str(emb_id)  + " is outside the range of " + \
    "permissable values from 0 to "  + str(self.data.tools.no_of_internal_embs)
  ###### End of assert_ID()

  #Places token with given _ID at index
  # in the given fields of the dataclass 
  #(vector , positive , negative , temporary)
  def place(self , _ID , index ,\
    send_to_vector , send_to_positives  , send_to_negatives , send_to_temporary):

    self.assert_ID(_ID)
    model_is_sdxl , is_sd2 , is_sd1 = self.data.tools.get_flags()

    emb_vec768 = self.data.tools.internal_embs768[_ID]\
    .to(device = choosen_device , dtype = datatype)
    assert emb_vec768 != None ,"emb_vec768 is NoneType!"
    emb_name = self.data.emb_id_to_name(_ID)
  
    #Add to 768 dimension vectors
    self.data.place(index ,\
      vector =  emb_vec768.unsqueeze(0) ,
      ID =  _ID ,
      name = emb_name , 
      to_negative = send_to_negatives ,
      to_mixer = send_to_vector , 
      to_positive = send_to_positives)

    #Add to 1280 dimension vectors (if model is SDXL)
    if model_is_sdxl:
      sdxl_emb_vec = self.data.tools.internal_embs1280[_ID]\
      .to(device = choosen_device , dtype = datatype)
      assert sdxl_emb_vec != None , "sdxl_emb_vec is NoneType!"
      ######
      self.data.place(index , 
        vector = sdxl_emb_vec.unsqueeze(0) ,
        ID = _ID ,
        name = emb_name , 
        to_negative = send_to_negatives , 
        to_mixer = send_to_vector , 
        to_positive = send_to_positives , 
        use_1280_dim =True)
  ######## End of place()

  def random(self , use_1280_dim = False):
    target = None
    if use_1280_dim : target = self.data.vector1280
    else : target = self.data.vector768
    distance = torch.nn.PairwiseDistance(p=2)
    size = target.size 
    origin = target.origin
    random_token_length = self.random_token_length
    random_token_length_randomization = self.random_token_length_randomization
    #########
    emb_vec = torch.rand(size).to(device = choosen_device , dtype = datatype)
    dist = distance(emb_vec , origin).numpy()[0]
    rdist = random_token_length * \
    (1 - random_token_length_randomization*random.random())
    emb_vec = (rdist/dist)*emb_vec
    #######
    return emb_vec.to(device = choosen_device , dtype = datatype)

  @staticmethod
  def concat(tensor , target):
    if target == None: 
      return tensor.to(device = choosen_device , dtype = datatype)
    else : return torch.cat([tensor , target] , dim = 0)\
    .to(device = choosen_device , dtype = datatype)


  def get_embedding_vecs(name):
    return [] , 0

  # Check which words have special symbols in them
  def filter_symbols(self, text , id_mode = False) : 

    # Initialize
    is_sdxl , is_sd2 , is_sd1 = self.data.tools.get_flags()
    symbols = ("_" , "<" , ">" , "[" , "]" , "," , "#")
    words = text.strip().lower().split()
    word_index = 0 
    for word in words :
      word_index += 1
    no_of_words = word_index 
    replacements768 = [None]*no_of_words
    replacements1280 = [None]*no_of_words
    ########

    processed = None
    for iteration in range(10):
      processed = ''
      for index in range(no_of_words):

        for symbol in symbols:
          location = word.find(symbol) # equals -1 if None found
          if location>=0 : 
            #Remove the symbol from the word
            word = copy.copy(words[index])
            words[index] = ''
            for fragment in word.split(symbol): 
              words[index] = words[index] + fragment
            ######

            if symbol == "_":
              self.concat(replacements768[index] , self.random())
              self.concat(replacements1280[index] , self.random(use_1280_dim = is_sdxl))
          
            elif symbol == "<":
              self.concat(\
              replacements768[index] ,
              self.data.tools.internal_embs768[start_of_text_ID])

              self.concat(\
              replacements1280[index] ,
              self.data.tools.internal_embs1280[start_of_text_ID])
          
            elif symbol == ">":

              self.concat(\
              replacements768[index] ,
              self.random(end_of_text_ID))

              self.concat(\
              replacements1280[index] ,
              self.random(end_of_text_ID , use_1280_dim = is_sdxl))
      
            elif symbol == "[": 
              emb_vecs = self.get_embedding_vecs()
              numbers = [int(num) for num in re.findall(r'\d+', words[index])]
              start = 0
              end = len(emb_vecs)
              if (len(numbers)>1):
                start = min(numbers[0] , 0)
                end = max(numbers[1] , len(emb_vecs))
              ######

              k=None
              for emb_vec in emb_vecs:
                if k==None : k = 0
                else: k = k+1
                if k<start: continue
                self.concat(replacements1280[index] , emb_vec)
                if k==end : break
              #########

            elif id_mode and symbol == "#" and words[index].isdigit():
        
              _ID = int(words[index])
              words[index] = ''

              self.concat(\
              self.concat(replacements768[index] ,
              self.data.tools.internal_embs768[_ID]))
              self.concat(\
              self.concat(replacements768[index] ,
              self.data.tools.internal_embs768[_ID]))
            ##########
            if processed != '' : processed = processed + ' '
            processed = processed  + words[index]
    
    return processed , replacements768 , replacements1280


  def Process (self  , *args) :

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

    self.random_token_length = random_token_length
    self.random_token_length_randomization = random_token_length_randomization
    is_sdxl , is_sd2 , is_sd1 = self.data.tools.get_flags()

    tokenmixer_vectors = ''
    if not sendtomix : tokenmixer_vectors= mix_input

    # Do some checks prior to running
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
          to_positive = send_to_positives)
    ######

    #Convert text input to list of 'words'

    text = mini_input.strip().lower().split()
    processed , replacements768 , replacements1280 = \
    self.filter_symbols(text)

    from pprint import pprint
    pprint(processed)

    tokenbox = ''
    splitbox = ''
    negbox = ''
    posbox = ''
    
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
      self.buttons.tokenize.click(fn=self.Process , inputs = input_list , outputs = output_list)

  #End of setupIO_with()


  def __init__(self, label , vis = False):

    #Pass reference to global object "dataStorage" to class
    self.data = dataStorage 

    # Random vector generation from "_" symbol
    self.random_token_length = 0
    self.random_token_length_randomization = 0
    ########

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



