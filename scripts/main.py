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

#---------------------------------

from lib.modules.token_mixer import TokenMixer     
from lib.modules.moduleTab import Modules
from lib.modules.cross_attention_mixer import CrossAttentionMixer

def add_tab():

  left_column = gr.Column(scale=1)
  right_column = gr.Column(scale=2)

  settings_update_button = gr.Button(value="Update columns", variant="secondary")
  settings_update_button.style(size="sm")

  #Create UI
  with gr.Blocks() as ui :
    with gr.Tabs():
        with gr.Row(): 
            gr.HTML("<header><h1>TokenMixer </h1></header>") #Top Bar
        with gr.Row():
            gr.Markdown("Create, merge , randomize or interpolate tokens to create new embeddings ")
        with gr.Row():    
            header = gr.Textbox(label="", lines=2, placeholder="Rearrange embeddings here", interactive = True)
        with gr.Row(): 
            with left_column.render() : #Left Column
              modules = Modules("all" , True , True) 
              modules2  = Modules("minit" , True)
              modules3 = Modules("embin" , True)
              modules4 = Modules("tocal" , True)
              modules5 = Modules("tokex" , True)
              modules6 = Modules("synom" , True)
            #####
                      
            with right_column.render() : #Right Column
              tokm = TokenMixer("Embedding generator", True , True)
              tokm2 = TokenMixer("Embedding generator #2") 
              tokm3 = TokenMixer("Embedding generator #3")
              tokm4 = TokenMixer("Embedding generator #4")
              tokm5 = TokenMixer("Embedding generator #5")
              #####
              catm = CrossAttentionMixer("Cross Attention Visualizer" , True)
              
        with gr.Row() : 
          with gr.Accordion('Hide Columns',open=False):
              show_left_column = gr.Checkbox(value=True, label="Show left column", interactive = True)
              show_right_column = gr.Checkbox(value=True, label="Show right column", interactive = True)
              settings_update_button.render() 

    if not tokm.data.tools.loaded : 
      header.value = "NOTE : No viable model was detected upon starting the UI. " + \
      "The TokenMixer is currently disabled."

    
    def hide_columns (show_left_column , show_right_column) :
      return  { 
        left_column  : gr.Column.update(visible = show_left_column) , 
        right_column  : gr.Column.update(visible = show_right_column) }

    settings_update_button.click(fn = hide_columns , \
    inputs = [show_left_column , show_right_column] , \
    outputs = [left_column , right_column , modules.tab])
    
    modules.setupIO_with(tokm , tokm2 , tokm3 , tokm4 , tokm5)
    modules2.setupIO_with(tokm , tokm2 , tokm3 , tokm4 , tokm5)
    modules3.setupIO_with(tokm , tokm2 , tokm3 , tokm4 , tokm5)
    modules4.setupIO_with(tokm , tokm2 , tokm3 , tokm4 , tokm5)
    modules5.setupIO_with(tokm , tokm2 , tokm3 , tokm4 , tokm5)

  ########## End of UI
  
  return [(ui, "TokenMixer", "TokenMixer")]
########

script_callbacks.on_ui_tabs(add_tab)
        

