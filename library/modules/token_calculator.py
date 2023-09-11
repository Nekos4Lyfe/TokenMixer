import gradio as gr
from modules import  sd_hijack
import torch, os
import collections, math, random , numpy
import re #used to parse string to int
import copy

from library.toolbox.constants import MAX_NUM_MIX

from library.data import dataStorage

# Check that MPS is available (for MAC users)
choosen_device = torch.device("cpu")
#if torch.backends.mps.is_available(): 
#  choosen_device = torch.device("mps")
#else : choosen_device = torch.device("cpu")
#######

class TokenCalculator:

  def Reset (self , calc_input , sumbox) : 
    return '' , ''

  def addition (self, tensor1 , tensor2):
    if (tensor1 == None) or (tensor2 == None) : return None
    tmp1 = tensor1.to(device = choosen_device , dtype = torch.float32)
    tmp2 = tensor2.to(device = choosen_device , dtype = torch.float32)
    return tmp1 + tmp2
  
  def subtraction (self, tensor1 , tensor2):
    if (tensor1 == None) or (tensor2 == None) : return None
    tmp1 = tensor1.to(device = choosen_device , dtype = torch.float32)
    tmp2 = tensor2.to(device = choosen_device , dtype = torch.float32)
    return tmp1 - tmp2

  def get_emb_vec (self, string , id_mode , no_of_internal_embs) :
      log = []
      dist = None

      #Set random vector if string matches "_"
      if string == "_":

        size = self.size
        rand = 2*torch.rand(size)-torch.ones(size)
        dist , rand = self.get_length(rand)
        tmp = self.random_token_length * \
        (1 - self.random_token_length_randomization*random.random())
        rand = (tmp/dist)*rand
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
    self.random_token_length = copy.copy(args[4])
    mix_input = args[5]
    stack_mode = args[6]
    send_to_negatives = args[7]
    neg_input = args[8]
    self.random_token_length_randomization = copy.copy((1/100) * args[9])
    output_length_rand = (1/100) * args[10]

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

    negbox = neg_input
    tokenmixer_vectors = mix_input
    if sendtomix and not stack_mode: tokenmixer_vectors= ''
    if send_to_negatives and not stack_mode: negbox= ''

    func = self.addition
    log = []

    #Check if new embeddings have been added 
    try: sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
    except: 
      sd_hijack.model_hijack.embedding_db.dir_mtime=0
      sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
    self.data.update_loaded_embs() 

    #clear all slots
    for index in range(MAX_NUM_MIX):
      self.data.clear(
        index , 
        to_mixer = sendtomix and (not stack_mode) , 
        to_temporary = True , 
        to_negative = send_to_negatives and (not stack_mode))
    ######

    if calc_input == None :
      print ("calc_input is NoneType")
      return tokenmixer_vectors , '', '\n'.join(log) , negbox

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
      return tokenmixer_vectors , '' , '\n'.join(log) , negbox

    distance = torch.nn.PairwiseDistance(p=2)
    output = calc_sum.unsqueeze(0).to(device = choosen_device , dtype = torch.float32)
    dist = distance(output , 0*output).numpy()[0]

    tmp = output_length * \
    (1 - output_length_rand*random.random())    
    output = (tmp /dist)*output
    dist = round(distance(output , 0*output).numpy()[0] , 2)

    log.append("------------------------")
    log.append("=  vector with length " + str(dist))

    #Find an empty slot to save the calc_sum
    for i in range(MAX_NUM_MIX):
      if (not self.data.vector.isEmpty.get(i)) : continue
      
      sumbox = "emb_sum"
      self.data.place(i , 
          vector = output ,
          ID = 0,
          name = sumbox , 
          to_mixer = sendtomix , 
          to_temporary = True ,
          to_negative = send_to_negatives)

      
      if sendtomix : 
        name = self.data.vector.name.get(i)
        if name != None and sendtomix:
          if tokenmixer_vectors != '': tokenmixer_vectors = tokenmixer_vectors + ' , '
          tokenmixer_vectors = tokenmixer_vectors  + name
      ##########
      if send_to_negatives: 
        if self.data.negative.isEmpty.get(i): continue
        name = self.data.negative.name.get(i)
        if name != None and send_to_negatives :
          if negbox != '': negbox = negbox + ' , '
          negbox = negbox  + name

      break
    #End of loop

    return tokenmixer_vectors , sumbox , '\n'.join(log) , negbox
       
    
  def setupIO_with (self, module):
    #Tell the buttons in this class what to do when 
    #pressed by the user

    cond = self.data.tools.loaded
    input_list = []
    output_list = []

    if (module.ID == "TokenCalculator") and cond:
      output_list.append(self.inputs.calc_input)
      output_list.append(self.outputs.sumbox)
      self.buttons.reset.click(fn = self.Reset, inputs = input_list , outputs = output_list)
      return 1


    if (module.ID == "TokenMixer") and cond:
      input_list.append(self.inputs.calc_input)  #0
      input_list.append(self.inputs.sendtomix)   #1
      input_list.append(self.inputs.id_mode)     #2
      input_list.append(self.inputs.length)      #3
      input_list.append(self.inputs.randlen)     #4
      input_list.append(module.inputs.mix_input) #5
      input_list.append(self.inputs.stack_mode)  #6
      input_list.append(self.inputs.negatives)   #7
      input_list.append(module.inputs.negbox)    #8
      input_list.append(self.inputs.randlenrand) #9
      input_list.append(self.inputs.lenrand) #10


      output_list.append(module.inputs.mix_input) #0
      output_list.append(self.outputs.sumbox)     #1
      output_list.append(self.outputs.log)        #2
      output_list.append(module.inputs.negbox)    #3

      self.buttons.calculate.click(fn=self.Calculate , inputs = input_list , outputs = output_list)
  #End of setupIO_with()

  def get_length(self, emb_vec):
    self.emb_vec = emb_vec\
    .to(device = choosen_device , dtype = torch.float32)
    assert self.emb_vec != None, "emb_vec is None!"
    dist = self.distance(self.emb_vec, self.origin).numpy()[0]
    dist = round(dist , 2)
    return dist , emb_vec

  def __init__(self , label , vis = False):
    #Pass reference to global object "dataStorage" to class
    self.data = dataStorage 
    self.emb_vec = None
    self.size = copy.copy(self.data.vector.size)
    self.origin = torch.zeros(self.size).unsqueeze(0)\
    .to(device = choosen_device , dtype = torch.float32)
    self.distance = torch.nn.PairwiseDistance(p=2)
    self.random_token_length = 0
    self.random_token_length_randomization = 0


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
        Inputs.stack_mode = []
        Inputs.randlenrand = []
        Inputs.lenrand = []

    class Buttons :
      def __init__(self):
        Buttons.calculate = []
        Buttons.reset = []

    self.outputs= Outputs()
    self.inputs = Inputs()
    self.buttons= Buttons()
    self.ID = "TokenCalculator"

    #create UI
    with gr.Accordion(label ,open=True , visible = vis) as show: 
      gr.Markdown("Add or subtract tokens into a new vector")
      with gr.Row() :  
          self.inputs.length = gr.Slider(label="Desired vector length",value=0.35, \
                                      minimum=0, maximum=2, step=0.01 , interactive = True)
      with gr.Row() :  
          self.buttons.calculate = gr.Button(value="Calculate", variant="primary", size="sm")
          self.buttons.reset = gr.Button(value="Reset", variant = "secondary", size="sm")
      with gr.Row() : 
        self.inputs.calc_input = gr.Textbox(label='', lines=2, \
        placeholder="Enter a short prompt (loaded embeddings or modifiers are not supported)" , interactive = True)
      with gr.Row() : 
        self.outputs.sumbox = gr.Textbox(label="CLIP tokens", lines=2, interactive = False)
      with gr.Row():
          self.inputs.sendtomix = gr.Checkbox(value=False, label="Send to input", interactive = True)
          self.inputs.id_mode = gr.Checkbox(value=False, label="ID input mode", interactive = True)
          self.inputs.stack_mode = gr.Checkbox(value=False, label="Stack Mode", interactive = True)
          self.inputs.negatives = gr.Checkbox(value=False, label="Send to negatives", interactive = True , visible = True) 
      with gr.Accordion("Random ' _ ' vector settings" ,open=False , visible = True) as randset: 
        self.inputs.randlen = gr.Slider(value = 0.35 , minimum=0, maximum=10, step=0.01, \
          label="Randomized ' _ ' vector length", default=0.35 , interactive = True) 
        self.inputs.randlenrand = \
          gr.Slider(value = 50 , minimum=0, maximum=100, step=0.1, \
          label="Randomized ' _ ' vector length randomization %", default=50 , interactive = True)
        self.inputs.lenrand = \
          gr.Slider(value = 0 , minimum=0, maximum=100, step=0.1, \
          label="Desired vector length randomization %", default=0 , interactive = True)

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

    if self.data.tools.loaded : self.setupIO_with(self)
## End of class TokenCalculator--------------------------------------------------#