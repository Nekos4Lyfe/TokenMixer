import gradio as gr
from modules import script_callbacks, shared, sd_hijack
from modules.shared import cmd_opts
from pandas import Index
from pandas.core.groupby.groupby import OutputFrameOrSeries
import torch, os
from modules.textual_inversion.textual_inversion import Embedding
import collections, math, random , numpy
import re #used to parse string to int
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
    send_to_temporary = args[5]
    mix_input = args[6]
    stack_mode = args[7]

    assert not (
          sendtomix == None or
          id_mode == None or
          send_to_negatives == None or
          random_token_length == None or
          send_to_temporary == None
          ) , "NoneType in Tokenizer input!"

    tokenmixer_vectors = mix_input
    if sendtomix and not stack_mode : tokenmixer_vectors= ''

    distance = torch.nn.PairwiseDistance(p=2)
    origin = self.data.vector.origin.cpu()

    start = None
    comma = None
    end = None
    section = None
    numbers = None
    start = 0
    end = MAX_NUM_MIX
    tokenbox = ''
    splitbox = ''
    negbox = ''
    emb_name = None
    emb_vecs = None
    found_IDs = None
    ID_index = None

    if mini_input == None : 
      return tokenmixer_vectors , '' , '' , ''

    #Check if new embeddings have been added 
    try: sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
    except: 
      sd_hijack.model_hijack.embedding_db.dir_mtime=0
      sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
    self.data.update_loaded_embs() 

    #clear all slots to give room for new data
    for i in range(MAX_NUM_MIX):
      self.data.clear(i , 
          to_negative = send_to_negatives , 
          to_mixer = sendtomix and not stack_mode , 
          to_temporary = send_to_temporary)
    ######

    no_of_tokens = 0
    token_num = 0
    no_of_IDs = None

    splitList = mini_input.strip().split()
    split_index = 0
    for splits in splitList :
      split_index += 1
    split_length = split_index 

    no_of_internal_embs = self.data.tools.no_of_internal_embs

    for i in range(MAX_NUM_MIX):
      if sendtomix and not self.data.vector.isEmpty.get(i): continue

      ######
      if i>0: #Store the values from the previous iteration
        neg_name = self.data.negative.name.get(i - 1)
        if neg_name != None :
          if (negbox != '') : negbox = negbox + ' , ' 
          negbox = negbox + neg_name
        ######
        name = self.data.vector.name.get(i - 1)
        if name != None and sendtomix:
          if tokenmixer_vectors != '': tokenmixer_vectors = tokenmixer_vectors + ' , '
          tokenmixer_vectors = tokenmixer_vectors + name
      ######

      if not split_index>0: continue
      string = splitList[split_length-split_index]
      if string == "," :  continue  #Ignore comma inputs

      if string == "_":
        emb_vec = torch.rand(self.data.vector.size)
        dist = distance(emb_vec , origin).numpy()[0]
        emb_vec = (random_token_length/dist)*emb_vec
        emb_id = 0
        emb_name = "random_" + str(i)

        self.data.place(i , 
            vector =  emb_vec.unsqueeze(0) ,
            ID =  emb_id ,
            name = emb_name , 
            to_negative = send_to_negatives ,
            to_mixer = sendtomix , 
            to_temporary = send_to_temporary
            )

        if (tokenbox != '') : tokenbox = tokenbox + ' , '
        tokenbox =  tokenbox + emb_name + '_#' + str(emb_id)
        split_index -=1 
        continue
      ##########

      #Extract emb_vec from emb_id if id_mode is selected
      if (id_mode and string.isdigit()):
        emb_id = int(string)
        if emb_id >= no_of_internal_embs: continue
        emb_vec = self.data.tools.internal_embs[emb_id]
        emb_name = self.data.emb_id_to_name(emb_id)
       
        self.data.place(i , 
            vector =  emb_vec.unsqueeze(0) ,
            ID =  emb_id  ,
            name = emb_name , 
            to_negative = send_to_negatives , 
            to_mixer = sendtomix , 
            to_temporary = send_to_temporary)

        if (tokenbox != '') : tokenbox = tokenbox + ' , '
        tokenbox =  tokenbox + emb_name + '_#' + str(emb_id)
        split_index -=1 
        continue
      #########

        
      tmp = None
      if (string.find('[')>=0 and string.find(']')>=0):
          tmp =  string.split('[')[0]
          tmp = tmp.strip().lower()
          section = string.split('[')[1]
          section = section.split(']')[0]
          numbers = [int(num) for num in re.findall(r'\d+', section)]
          if (len(numbers)>1):
            start = numbers[0]
            end = numbers[1]

      if tmp == None : tmp = string.strip().lower()
      emb_name, emb_ids, emb_vecs , loaded_emb  = self.data.get_embedding_info(tmp)
      no_of_tokens = min(emb_vecs.shape[0] , MAX_NUM_MIX)


###########################
      if no_of_tokens > 1:           
          if (token_num>end): continue
          if (token_num+1>no_of_tokens) :
            no_of_tokens = 0
            token_num = 0
            split_index -=1 #Go to next word
            continue
          if (token_num<start):
            token_num += 1 #Skip until token_num==start
            continue

          tmp = emb_vecs[token_num]
          tmp = copy.deepcopy(tmp)

          self.data.place(i , 
              vector = (emb_vecs[token_num]).unsqueeze(0) ,
              ID = 0 ,
              name = emb_name + '_' + str(token_num) , 
              to_negative = send_to_negatives , 
              to_mixer = sendtomix , 
              to_temporary = send_to_temporary)

          if (splitbox != '') : splitbox = splitbox + ' , '    
          splitbox =  splitbox + emb_name + '_' + str(token_num)
          token_num += 1

#########################
      else:
        if found_IDs == None:
          found_IDs = self.data.text_to_emb_ids(string)
          no_of_IDs = len(found_IDs)
          ID_index = 0
        
        _ID = found_IDs[ID_index]
        emb_name = self.data.emb_id_to_name(_ID)
        emb_vec = self.data.emb_id_to_vec(_ID)
        assert emb_vec != None , "emb_vec is None!"


        self.data.place(i , 
            vector =  emb_vec.unsqueeze(0) ,
            ID =  _ID ,
            name = emb_name ,
            to_negative = send_to_negatives ,
            to_mixer = sendtomix , 
            to_temporary = send_to_temporary)

        ID_index+=1 
        if ID_index+1> no_of_IDs : 
            found_IDs = None
            split_index -=1 
     
        if (tokenbox != '') : tokenbox = tokenbox + ' , '
        tokenbox =  tokenbox + emb_name + '_#' + str(_ID)

####################################
        
    return tokenmixer_vectors , tokenbox , splitbox , negbox
       
    
  def setupIO_with (self, module):
    #Tell the buttons in this class what to do when 
    #pressed by the user

    input_list = []
    output_list = []

    if (module.ID == "MiniTokenizer") :
      output_list.append(self.inputs.mini_input)
      output_list.append(self.outputs.tokenbox)
      self.buttons.reset.click(fn = self.Reset, inputs = input_list , outputs = output_list)


    if (module.ID == "TokenMixer") :
      input_list.append(self.inputs.mini_input)   #0
      input_list.append(self.inputs.sendtomix)    #1
      input_list.append(self.inputs.id_mode)      #2
      input_list.append(self.inputs.negatives)    #3
      input_list.append(self.inputs.randlen)      #4
      input_list.append(self.inputs.positives)    #5
      input_list.append(module.inputs.mix_input)  #6
      input_list.append(self.inputs.stack_mode)   #7
      #######
      output_list.append(module.inputs.mix_input) #0
      output_list.append(self.outputs.tokenbox)   #1
      output_list.append(self.outputs.splitbox)   #2
      output_list.append(module.inputs.negbox)    #3
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
    with gr.Accordion(label , open=False , visible = vis) as show :
      gr.Markdown("Get CLIP tokens and embedding vectors")
      with gr.Row() :  
          self.buttons.tokenize = gr.Button(value="Tokenize", variant="primary")
          self.buttons.reset = gr.Button(value="Reset", variant = "secondary")
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
          self.inputs.positives = gr.Checkbox(value=False, label="Send to positives", interactive = True , visible = False) #Experimental
          self.inputs.stack_mode = gr.Checkbox(value=False, label="Stack Mode", interactive = True)
      with gr.Accordion("Random ' _ ' token settings" ,open=False , visible = False) as randset : 
        self.inputs.randlen = gr.Slider(minimum=0, maximum=10, step=0.01, label="Randomized ' _ ' token length", default=0.35 , interactive = True)
      
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
    self.buttons.tokenize.style(size="sm")
    self.buttons.reset.style(size="sm")

    self.setupIO_with(self)
## End of class MiniTokenizer--------------------------------------------------#