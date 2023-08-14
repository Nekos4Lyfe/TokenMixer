import gradio as gr
from modules import scripts ,script_callbacks, shared, sd_hijack , devices
import modules.shared as shared
import torch, os
import collections, math, random
import numpy as np
import copy
from torch.nn.modules import ConstantPad1d, container
#####
from PIL import Image
from torch import nn, einsum
from einops import rearrange
from ldm.modules.attention import CrossAttention
######
from lib.toolbox.constants import MAX_NUM_MIX
from lib.data import dataStorage

class CrossAttentionMixer:

  def Reset (self , mini_input , tokenbox) : 
    return '' , ''

  def update_layer_names(self, model):
    self.hidden_layers = {}
    for n, m in model.named_modules():
        if(isinstance(m, CrossAttention)):
            self.hidden_layers[n] = m
    self.hidden_layer_names = list(filter(lambda s : "attn2" in s, self.hidden_layers.keys())) 
    if self.hidden_layer_select != None:
        self.hidden_layer_select.update(value=self.default_hidden_layer_name, choice=self.hidden_layer_names)

  def get_attn(self, emb, ret):
    def hook(self, sin, sout):
        h = self.heads
        q = self.to_q(sin[0])
        context = emb
        k = self.to_k(context)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        ret["out"] = attn
    return hook

  def generate_vxa(self, image, prompt, idx, time, layer_name, output_mode):
    if(not isinstance(image, np.ndarray)):
        return image
    output = image.copy()
    image = image.astype(np.float32) / 255.0
    image = np.moveaxis(image, 2, 0)
    image = torch.from_numpy(image).unsqueeze(0)

    model = shared.sd_model
    layer = self.hidden_layers[layer_name]
    cond_model = model.cond_stage_model
    with torch.no_grad(), devices.autocast():
        image = image.to(devices.device)
        latent = model.get_first_stage_encoding(model.encode_first_stage(image))
        try:
            t = torch.tensor([float(time)]).to(devices.device)
        except:
            return output
        emb = cond_model([prompt])

        attn_out = {}
        handle = layer.register_forward_hook(self.get_attn(emb, attn_out))
        try:
            model.apply_model(latent, t, emb)
        finally:
            handle.remove()

    if (idx == ""):
        img = attn_out["out"][:,:,1:].sum(-1).sum(0)
    else:
        try:
            idxs = list(map(int, filter(lambda x : x != '', idx.strip().split(','))))
            img = attn_out["out"][:,:,idxs].sum(-1).sum(0)
        except:
            return output

    scale = round(math.sqrt((image.shape[2] * image.shape[3]) / img.shape[0]))
    h = image.shape[2] // scale
    w = image.shape[3] // scale
    img = img.reshape(h, w) / img.max()
    img = img.to("cpu").numpy()
    output = output.astype(np.float64)
    if output_mode == "masked":
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i][j] *= img[i // scale][j // scale]
    elif output_mode == "grey":
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i][j] = [img[i // scale][j // scale] * 255.0] * 3
    output = output.astype(np.uint8)
    return output
    
  def setupIO_with (self, module):
    #Tell the buttons in this class what to do when 
    #pressed by the user

    cond = self.data.tools.loaded

    input_list = []
    output_list = []

    if (module.ID == "MiniTokenizer") and cond:
      return 1

    if (module.ID == "CrossAttentionMixer") and cond :
      self.vxa_generate.click(
            fn=self.generate_vxa,
            inputs=[self.input_image, self.vxa_prompt, \
            self.vxa_token_indices, self.vxa_time_embedding, \
            self.hidden_layer_select, self.vxa_output_mode],
            outputs=[self.vxa_output])

  #End of setupIO_with()


  def __init__(self, label , vis = False):

    #Pass reference to global object "dataStorage" to class
    self.data = dataStorage 

    self.hidden_layers = {}
    self.hidden_layer_names = []
    self.default_hidden_layer_name = "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2"
    self.hidden_layer_select = None

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
    self.ID = "CrossAttentionMixer"

    #test = gr.Label('Prompt MiniTokenizer' , color = "red")
    #create UI
    with gr.Accordion(label , open=False , visible = vis) as show :
      gr.Markdown("Visualize cross attention.")
      with gr.Row():
            with gr.Column():
                self.input_image = gr.Image(elem_id="vxa_input_image")
                self.vxa_prompt = gr.Textbox(label="Prompt", lines=2, placeholder="Prompt to be visualized")
                self.vxa_token_indices = gr.Textbox(value="", label="Indices of tokens to be visualized", lines=2, placeholder="Example: 1, 3 means the sum of the first and the third tokens. 1 is suggected for a single token. Leave blank to visualize all tokens.")
                self.vxa_time_embedding = gr.Textbox(value="1.0", label="Time embedding")
                ##########
                if self.data.tools.loaded : 
                  for n, m in shared.sd_model.named_modules():
                    if(isinstance(m, CrossAttention)):
                        self.hidden_layers[n] = m
                  self.hidden_layer_names = list(filter(lambda s : "attn2" in s, self.hidden_layers.keys())) 
                  self.hidden_layer_select = gr.Dropdown(value=self.default_hidden_layer_name, label="Cross-attention layer", choices=self.hidden_layer_names)
                ##########
                self.vxa_output_mode = gr.Dropdown(value="masked", label="Output mode", choices=["masked", "grey"])
                self.vxa_generate = gr.Button(value="Visualize Cross-Attention", elem_id="vxa_gen_btn")
            with gr.Column():
                self.vxa_output = gr.Image(elem_id = "self.vxa_output", interactive=False)
            #########
      gr.Markdown("Code from : https://github.com/benkyoujouzu/stable-diffusion-webui-visualize-cross-attention-extension")
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

    if self.data.tools.loaded : self.setupIO_with(self)
## End of class MiniTokenizer--------------------------------------------------#