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

from lib.toolbox.constants import \
MAX_NUM_MIX , SHOW_NUM_MIX , MAX_SIMILAR_EMBS , \
VEC_SHOW_TRESHOLD , VEC_SHOW_PROFILE , SEP_STR , \
SHOW_SIMILARITY_SCORE , ENABLE_GRAPH , GRAPH_VECTOR_LIMIT , \
ENABLE_SHOW_CHECKSUM , REMOVE_ZEROED_VECTORS , EMB_SAVE_EXT 

from lib.data import dataStorage

class TokenCalculator:

  def Reset (self , calc_input , sumbox) : 
    return '' , ''

  def addition (self, tensor1 , tensor2):
    if (tensor1 == None) or (tensor2 == None) : return None
    tmp1 = tensor1.cpu()
    tmp2 = tensor2.cpu()
    return tmp1 + tmp2
  
  def subtraction (self, tensor1 , tensor2):
    if (tensor1 == None) or (tensor2 == None) : return None
    tmp1 = tensor1.cpu()
    tmp2 = tensor2.cpu()
    return tmp1 - tmp2

  def get_emb_vec (self, string , id_mode , no_of_internal_embs) :
      log = []
      dist = None

      #Set random vector if string matches "_"
      if string == "_":

        size = self.size
        rand = 2*torch.rand(size)-torch.ones(size)
        dist , rand = self.get_length(rand)
        rand = (self.random_vector_length/dist)*rand
        dist , rand = self.get_length(rand)

        log.append("random vector with length " + str(dist))
        return rand , '\n'.join(log)
      #####

      if (id_mode and string.isdigit()):
        emb_id = int(string)
        if emb_id < no_of_internal_embs: 
          emb_vec = self.data.tools.internal_embs[emb_id]
          dist , emb_vec = self.get_length(emb_vec)
          emb_name = self.data.emb_id_to_name(emb_id)
        ######
      else:
        emb_name, emb_id, tmp , loaded_emb = \
        self.data.get_embedding_info(string)

        if emb_name != None:
          if (emb_name.find('</w>')>=0):
            emb_name = emb_name.split('</w>')[0]
        else : return None , '\n'.join(log)

        if tmp != None:
          dist , emb_vec = self.get_length(tmp[0])
          emb_vec = tmp[0]
        else: return None , '\n'.join(log)
      ######
    
      log.append("token '" + emb_name + "' , ID #" + str(emb_id) + \
      " , length :" +  str(dist))
      return emb_vec , '\n'.join(log)
    ######

  def Calculate (self, *args) :
    
    calc_input = args[0]
    sendtomix = args[1]
    id_mode = args[2]
    output_length = args[3]
    self.random_vector_length = copy.copy(args[4])
    mix_inputs = args[5:MAX_NUM_MIX+5]
    vectors = args[MAX_NUM_MIX+6:2*MAX_NUM_MIX+6]

    tokenizer = self.data.tools.tokenizer
    internal_embs = self.data.tools.internal_embs

    no_of_internal_embs = 0
    if id_mode : no_of_internal_embs = \
    len(self.data.tools.internal_embs)

    assert not (
          sendtomix == None or
          id_mode == None or
          tokenizer == None or
          internal_embs == None or
          output_length  == None
          ) , "NoneType in Tokenizer input!"

    func = self.addition
    log = []

    #Check if new embeddings have been added 
    try: sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
    except: 
      sd_hijack.model_hijack.embedding_db.dir_mtime=0
      sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
    self.data.update_loaded_embs() 

    #clear all slots
    for i in range(MAX_NUM_MIX):
      self.data.clear(i)
    ######

    if calc_input == None :
      print ("calc_input is NoneType")
      return *emptyList , '', '\n'.join(log)

    splitList = calc_input.strip().split(' ')
    split_index = 0
    for splits in splitList :
      split_index += 1
    split_length = split_index 

    prev = None
    current = None
    operation = None

    size = copy.copy(self.data.vector.size)
    calc_sum = None
    calc = None
    output = None

    emptyList = [None]*MAX_NUM_MIX
    tokenmixer_vectors= [None]*MAX_NUM_MIX

    sumbox = ''
    string = None
    emb_name = None
    emb_vec = None

    for i in range(MAX_NUM_MIX):
      if split_index<=0: continue
      string = splitList[split_length-split_index]

      if (string.find('+')>=0) :
        func = self.addition
        log.append("plus")
        split_index -=1
        continue
      
      if (string.find('-')>=0) :
        func = self.subtraction
        log.append("minus")
        split_index -=1
        continue

      emb_vec , message = \
      self.get_emb_vec(string, id_mode , no_of_internal_embs)
      log.append(message)

      if calc_sum == None : calc_sum = emb_vec
      else: calc_sum = func(calc_sum , emb_vec)

      split_index -=1
      continue
    ######

    if calc_sum == None :
      log.append("Empty sum")
      return *emptyList, '' , '\n'.join(log)

    distance = torch.nn.PairwiseDistance(p=2)
    output = calc_sum.unsqueeze(0).cpu()
    dist = distance(output , 0*output).numpy()[0]

    output = (output_length /dist)*output
    dist = round(distance(output , 0*output).numpy()[0] , 2)

    log.append("------------------------")
    log.append("=  vector with length " + str(dist))

    #Find an empty slot to save the calc_sum
    for i in range(MAX_NUM_MIX):
      if (not self.data.vector.isEmpty.get(i)) : continue

      self.data.place(i , 
          vector = output ,
          ID = 0,
          name = "emb_sum")

      sumbox = "emb_sum"
      if sendtomix : 
        tokenmixer_vectors[i] = self.data.vector.name.get(i)
      break
    #End of loop

    return *tokenmixer_vectors , sumbox , '\n'.join(log)
       
    
  def setupIO_with (self, module):
    #Tell the buttons in this class what to do when 
    #pressed by the user

    input_list = []
    output_list = []

    if (module.ID == "TokenCalculator") :
      output_list.append(self.inputs.calc_input)
      output_list.append(self.outputs.sumbox)
      self.buttons.reset.click(fn = self.Reset, inputs = input_list , outputs = output_list)
      return 1


    if (module.ID == "TokenMixer") :
      input_list.append(self.inputs.calc_input) #0
      input_list.append(self.inputs.sendtomix)  #1
      input_list.append(self.inputs.id_mode)    #2
      input_list.append(self.inputs.length)     #3
      input_list.append(self.inputs.randlen)    #4
      for i in range(MAX_NUM_MIX):
        input_list.append(module.inputs.mix_inputs[i])
        output_list.append(module.inputs.mix_inputs[i])
      output_list.append(self.outputs.sumbox)
      output_list.append(self.outputs.log)
      self.buttons.calculate.click(fn=self.Calculate , inputs = input_list , outputs = output_list)
  #End of setupIO_with()

  def get_length(self, emb_vec):
    self.emb_vec = emb_vec.cpu()
    assert self.emb_vec != None, "emb_vec is None!"
    dist = self.distance(self.emb_vec, self.origin).numpy()[0]
    dist = round(dist , 2)
    return dist , emb_vec

  def __init__(self , label , vis = False):
    #Pass reference to global object "dataStorage" to class
    self.data = dataStorage 
    self.emb_vec = None
    self.size = copy.copy(self.data.vector.size)
    self.origin = torch.zeros(self.size).unsqueeze(0).cpu()
    self.distance = torch.nn.PairwiseDistance(p=2)
    self.random_vector_length = 0


    class Outputs :
      def __init__(self):
        Outputs.sumbox = []
        Outputs.log = []

    class Inputs :
      def __init__(self):
        Inputs.calc_input = []
        Inputs.sendtomix = []
        Inputs.id_mode = []
        Inputs.length = []
        Inputs.randlen = []

    class Buttons :
      def __init__(self):
        Buttons.calculate = []
        Buttons.reset = []

    self.outputs= Outputs()
    self.inputs = Inputs()
    self.buttons= Buttons()
    self.ID = "TokenCalculator"

    #create UI
    with gr.Accordion(label ,open=False , visible = vis) as show: 
      with gr.Row() :  
          self.inputs.length = gr.Slider(label="Desired vector length",value=0.35, \
                                      minimum=0, maximum=2, step=0.01 , interactive = True)
      with gr.Row() :  
          self.buttons.calculate = gr.Button(value="Calculate", variant="primary")
          self.buttons.reset = gr.Button(value="Reset", variant = "secondary")
      with gr.Row() : 
        self.inputs.calc_input = gr.Textbox(label="Input", lines=2, \
        placeholder="Enter a short prompt (loaded embeddings or modifiers are not supported)" , interactive = True)
      with gr.Row() : 
        self.outputs.sumbox = gr.Textbox(label="CLIP tokens", lines=2, interactive = False)
      with gr.Row():
          self.inputs.sendtomix = gr.Checkbox(value=False, label="Send to TokenMixer", interactive = True)
          self.inputs.id_mode = gr.Checkbox(value=False, label="ID input mode", interactive = True)
      with gr.Accordion("Random ' _ ' token settings" ,open=False , visible = False) as randset: 
        self.inputs.randlen = gr.Slider(minimum=0, maximum=10, step=0.01, \
        label="Randomized ' _ ' token length", default=0.35 , interactive = True)     
      with gr.Accordion('Output Log',open=False) as logs : 
        self.outputs.log = gr.Textbox(label="Log", lines=4, interactive = False)
      with gr.Accordion('Tutorial : What is this?',open=False , visible= False) as tutorial_0 : 
        gr.Markdown("The Token Calculator is a tool which allows you to create " + \
        "custom tokens by adding or subtracting CLIP tokens." + \
          "\n \n Example : write 'king - man + woman' " + \
          " in the input will generate a 'queen' token. \n \n " + \
          "The Token Calculator  can add a random token by writing " + \
          "_ in the input. Example : ' car + _' \n \n " + \
          "The _ token can host both positive and negative values , meaning the + " + \
          "and - operators acts equivalently for the random token. \n \n "
          "You can set the desired length of the output token using the length slider \n \n" + \
          "\n \n How does length affect the output? : " + \
          "Each of the vectors 768 elements correspond to the amplitude of a sine function when " + \
          "processed by the layers in Stable diffusion. \n \n "  + \
          "Given this relationship , it is expected that tokens with " + \
          "long length can overpower tokens with short length when placed " + \
          "together in a prompt. An analogy to this can be music played at high volume " + \
          "that makes it difficult to hear sounds at low volume.  \n \n " + \
          "Note that Vector length does NOT correspond to vector weight " + \
          ", i.e the (''example prompt'' : 1.4) statement you sometimes see in prompts. ")
    #end of UI

    self.tutorials = [tutorial_0]
    self.show = [show]
    self.randset = [randset]
    self.logs = [logs]
    
    
    self.inputs.randlen.value = 0.35
    self.buttons.calculate.style(size="sm")
    self.buttons.reset.style(size="sm")
    self.setupIO_with(self)
## End of class TokenCalculator--------------------------------------------------#