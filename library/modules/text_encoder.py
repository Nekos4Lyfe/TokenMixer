import gradio as gr
from PIL import Image
from modules import shared
from modules.shared import interrogator
from modules import devices
import torch, os
import collections, math, random , numpy
import copy
from torch.nn.modules import ConstantPad1d, container
from library.toolbox.constants import MAX_NUM_MIX
from library.data import dataStorage

from library.toolbox.constants import TENSOR_DEVICE_TYPE , TENSOR_DATA_TYPE
choosen_device = TENSOR_DEVICE_TYPE
datatype = TENSOR_DATA_TYPE

class TextEncoder:

  def Reset (self , mini_input , tokenbox) : 
    return '' , ''


  def get_text_features (self, input_text_tokens, length):

    interrogator.load()
    devices.torch_gc()


    text_features = None 
    
    text_tokens = input_text_tokens.to\
    (device= devices.device_interrogate , dtype = torch.long)

    with torch.no_grad(), devices.autocast():
      text_features = interrogator.clip_model.encode_text\
      (text_tokens).type(interrogator.dtype)
      text_features /= text_features.norm(dim=-1, keepdim=True)
      text_features = text_features * length

    text_features = text_features.to(device = "cpu")

    return text_features


  def create_embedding (self  , *args) :

    #Get the inputs
    input_text = args[0]
    sendtomix = args[1]
    stack_mode = args[2]
    mix_input = args[3]
    send_to_negative = args[4]
    length = args[5]
    neg_input = args[6]

    tokenbox = ''
    ####
    negbox = ''
    if stack_mode : negbox = neg_input
    ####
    mixbox = ''
    if stack_mode : mixbox = mix_input

    #Clear everything
    for index in range(MAX_NUM_MIX):
      self.data.clear(index , 
      to_mixer = sendtomix and not stack_mode , 
      to_negative = send_to_negative and not stack_mode)
    #####

    if not hasattr(input_text , "split") : 
      return None , None , ''

    text_tokens = torch.zeros(self.data.vector.size , 77)
    text_features = self.get_text_features(text_tokens, length)
    #######

    if not hasattr(text_features , "shape") : 
      return None , None , ''

    no_of_vectors = text_features.shape[0]
    emb_vec = None
    emb_name = None
    #######
    for k in range(no_of_vectors):
      emb_name = "clip_" + str(k)
      emb_vec = text_features[k]
      for index in range(MAX_NUM_MIX):
        if not self.data.vector.isEmpty.get(index): continue
        self.data.place(index, 
          vector = emb_vec.unsqueeze(0) , 
          name = emb_name , 
          to_mixer = sendtomix , 
          to_negative = send_to_negative)
        break
      ########
      if tokenbox != '' : tokenbox += " , "
      tokenbox += emb_name

      if negbox != '' : negbox += " , "
      negbox += emb_name

      if mixbox != '' : mixbox += " , "
      mixbox += emb_name
    ########

    if not sendtomix : mixbox = mix_input
    if not send_to_negative : negbox = neg_input

    return mixbox , negbox , tokenbox
       
    
  def setupIO_with (self, module):

      cond = self.data.tools.loaded
      input_list = []
      output_list = []

      if (module.ID == "Interrogator") and cond:
        return 1
      
      if (module.ID == "TokenMixer") and cond:
        input_list.append(self.inputs.input_text)        #0
        input_list.append(self.inputs.sendtomix)    #1
        input_list.append(self.inputs.stack_mode)   #2
        input_list.append(module.inputs.mix_input)  #3
        input_list.append(self.inputs.send_to_negative)  #4
        input_list.append(self.inputs.length) #5
        input_list.append(module.inputs.negbox) #6
        #######
        output_list.append(module.inputs.mix_input) #0
        output_list.append(module.inputs.negbox) #0
        output_list.append(self.outputs.tokenbox)   #1
        #######
        self.buttons.create.click(fn=self.create_embedding , inputs = input_list , outputs = output_list)


  def __init__(self , label , vis = False) -> None:

      self.data = dataStorage
      self.ID = "Interrogator"

      class Outputs :
        def __init__(self):
          Outputs.tokenbox = []

      class Inputs :
        def __init__(self):
          Inputs.input_text = []
          Inputs.sendtomix = []
          Inputs.stack_mode = []
          Inputs.send_to_negative = []
          Inputs.length = []

      class Buttons :
        def __init__(self):
          Buttons.create = []
          Buttons.reset = []

      self.outputs= Outputs()
      self.inputs = Inputs()
      self.buttons= Buttons()
      self.refresh_symbol = "\U0001f504"  # 🔄


      with gr.Accordion(label , open=False , visible = vis) as show :
        gr.Markdown("Get CLIP tokens and embedding vectors")
  
        with gr.Row() :  
          self.buttons.create = gr.Button(value="Create Embedding", variant="primary", size="sm")
          self.buttons.reset = gr.Button(value="Reset", variant = "secondary" , size="sm") 
        with gr.Row() :  
          self.inputs.length = gr.Slider(label="Desired vector length",value=0.35, \
                                      minimum=0, maximum=2, step=0.01 , interactive = True)

        with gr.Row() :  
          self.inputs.input_text = \
          gr.Textbox(placeholder = "Input text prompt" , lines = 2, interactive = True)

        with gr.Row() : 
          self.outputs.tokenbox = gr.Textbox(label="Encoding", lines=2, interactive = False)
        with gr.Row() :   
          self.inputs.sendtomix = gr.Checkbox(value=False, label="Send to input", interactive = True)
          self.inputs.stack_mode = gr.Checkbox(value=False, label="Stack Mode", interactive = True)
          self.inputs.send_to_negative = gr.Checkbox(value=False, label="Send to negatives", interactive = True)

        with gr.Accordion('Tutorial : What is this?',open=False , visible= False) as tutorial_0 :
          gr.Markdown("This tutorial has not been written yet")

      self.tutorials = [tutorial_0]
      self.show = [show]
      if self.data.tools.loaded : self.setupIO_with(self)


      # Need to check the input_image for API compatibility
      #if isinstance(raw_input_image['input_image'], str):
      #  from library.pfd.utils import decode_base64_to_image
      #  input_image = HWC3(np.asarray(decode_base64_to_image(raw_input_image['input_image'])))
      #else:
      #  input_image = HWC3(raw_input_image['input_image'])

        
