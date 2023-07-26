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

from lib.toolbox.constants import \
MAX_NUM_MIX , SHOW_NUM_MIX , MAX_SIMILAR_EMBS , \
VEC_SHOW_TRESHOLD , VEC_SHOW_PROFILE , SEP_STR , \
SHOW_SIMILARITY_SCORE , ENABLE_GRAPH , GRAPH_VECTOR_LIMIT , \
ENABLE_SHOW_CHECKSUM , REMOVE_ZEROED_VECTORS , EMB_SAVE_EXT 

from lib.modules.minitokenizer import MiniTokenizer
from lib.modules.embedding_inspector import EmbeddingInspector
from lib.modules.token_mixer import TokenMixer      
from lib.modules.token_calculator import TokenCalculator 
from lib.modules.token_extrapolator import TokenExtrapolator 


def add_tab():

  tutorial_string = "The TokenMixer is a tool that takes" + \
  "tokens and/or embedding vectors as inputs and processes these into new " + \
  "tokens and/or embeddings. \n \n Above this tab ↑ you see a number of modules . \n \n " + \
  "These modules will take a token and/or embedding vector as an input ,  " + \
  "split them into individual vectors which are then passed " + \
  "on to the TokenMixer on the right → . \n \n " + \
  "The TokenMixer then proceses these vectors embeddings which you can use in the prompt. \n \n" + \
  "If you wish to see an in-depth tutorial for each module, enable the 'Show tutorial' " + \
  "checkbox and press the 'Update' button. \n \n This will append a 'Tutorial : What is this?' " + \
  "tab to relevant section where a description of the given module is provided."

  no_of_minit = gr.Slider(value = 1, minimum=0, maximum=5, step=1, label="Number of MiniTokenizers", default=1 , interactive = True)
  no_of_embin = gr.Slider(value = 1 , minimum=0, maximum=5, step=1, label="Number of Embedding Inspectors", default=1 , interactive = True)
  no_of_tocal = gr.Slider(value = 1 , minimum=0, maximum=5, step=1, label="Number of Token Calculators", default=1 , interactive = True)
  no_of_tokex = gr.Slider(value = 1 , minimum=0, maximum=5, step=1, label="Number of Token Extrapolators", default=1 , interactive = True)
  no_of_tokm = gr.Slider(value = 1 , minimum=0, maximum=5, step=1, label="Number of Token Mixers", default=1 , interactive = True)

  left_column = gr.Column(scale=1)
  right_column = gr.Column(scale=2)

  settings_update_button = gr.Button(value="Update columns", variant="secondary")
  settings_update_button.style(size="sm")

  module_update_button = gr.Button(value="Update modules", variant="primary")
  module_update_button.style(size="sm")

  #Create UI
  with gr.Blocks() as ui :
    with gr.Tabs():
        with gr.Row(): 
            gr.HTML("<header><h1>TokenMixer </h1></header>") #Top Bar
        with gr.Row():    
            gr.Textbox(label="", lines=2, placeholder="Rearrange embeddings here", interactive = True)
        with gr.Row(): 
            with left_column.render() : #Left Column 
  
                    #####
                    minit = MiniTokenizer("MiniTokenizer" , True)
                    minit2 = MiniTokenizer("MiniTokenizer #2")
                    minit3 = MiniTokenizer("MiniTokenizer #3")
                    minit4 = MiniTokenizer("MiniTokenizer #4")
                    minit5 = MiniTokenizer("MiniTokenizer #5")

                    embin = EmbeddingInspector("Embedding Inspector" , True) 
                    embin2 = EmbeddingInspector("Embedding Inspector#2") 
                    embin3 = EmbeddingInspector("Embedding Inspector#3") 
                    embin4 = EmbeddingInspector("Embedding Inspector#4") 
                    embin5 = EmbeddingInspector("Embedding Inspector#5") 

                    tocal = TokenCalculator("Token Calculator" , True)
                    tocal2 = TokenCalculator("Token Calculator#2")
                    tocal3 = TokenCalculator("Token Calculator#3")
                    tocal4 = TokenCalculator("Token Calculator#4")
                    tocal5 = TokenCalculator("Token Calculator#5")

                    tokex = TokenExtrapolator("Token Extrapolator" , True)
                    tokex2 = TokenExtrapolator("Token Extrapolator#2")
                    tokex3 = TokenExtrapolator("Token Extrapolator#3")
                    tokex4 = TokenExtrapolator("Token Extrapolator#4")
                    tokex5 = TokenExtrapolator("Token Extrapolator#5")


                    with gr.Accordion('Remove/add modules',open=False):
                      no_of_minit.render()
                      no_of_embin.render()
                      no_of_tocal.render()
                      no_of_tokex.render()
                      no_of_tokm.render()
                      with gr.Row():
                        show_tutorial = gr.Checkbox(value=False, label="Show tutorials", interactive = True)
                        show_output_logs = gr.Checkbox(value=True, label="Show output logs", interactive = True)
                        show_rand_settings = gr.Checkbox(value=True, label="Show random '_' token settings ", interactive = True)
                      module_update_button.render()  
                      with gr.Accordion('How do I use this extension?',open=False):
                        gr.Markdown(tutorial_string)  
                        

            with right_column.render() : #Right Column
              tokm = TokenMixer("TokenMixer", True , True)
              tokm2 = TokenMixer("TokenMixer#2") 
              tokm3 = TokenMixer("TokenMixer#3")
              tokm4 = TokenMixer("TokenMixer#4")
              tokm5 = TokenMixer("TokenMixer#5")
        with gr.Row() : 
          with gr.Accordion('Hide Columns',open=False):
              show_left_column = gr.Checkbox(value=True, label="Show left column", interactive = True)
              show_right_column = gr.Checkbox(value=True, label="Show right column", interactive = True)
              settings_update_button.render() 

    ########## End of UI

    def Hide_column (show_left_column , show_right_column) :
      return  { 
        left_column  : gr.Column.update(visible = show_left_column) , 
        right_column  : gr.Column.update(visible = show_right_column)}

    settings_update_button.click(fn = Hide_column , 
    inputs = [show_left_column , show_right_column] , outputs = [left_column , right_column])

    #######
    minit_list = minit.show + minit2.show + minit3.show + minit4.show + minit5.show
    embin_list = embin.show + embin2.show + embin3.show + embin4.show + embin5.show
    tocal_list = tocal.show + tocal2.show + tocal3.show + tocal4.show + tocal5.show
    tokex_list = tokex.show + tokex2.show + tokex3.show + tokex4.show + tokex5.show
    tokm_list = tokm.show + tokm2.show + tokm3.show + tokm4.show + tokm5.show
    module_output_list = minit_list + embin_list + tocal_list + tokex_list + tokm_list
    module_input_list = [no_of_minit , no_of_embin , no_of_tocal , no_of_tokex, no_of_tokm]
    #######
    tutorial_output_list = \
    minit.tutorials + minit2.tutorials + minit3.tutorials + minit4.tutorials + minit5.tutorials + \
    embin.tutorials + embin2.tutorials + embin3.tutorials + embin4.tutorials + embin5.tutorials + \
    tocal.tutorials + tocal2.tutorials + tocal3.tutorials + tocal4.tutorials + tocal5.tutorials + \
    tokex.tutorials + tokex2.tutorials + tokex3.tutorials + tokex4.tutorials + tokex5.tutorials + \
    tokm.tutorials + tokm2.tutorials + tokm3.tutorials + tokm4.tutorials + tokm5.tutorials
    tutorial_input_list = [show_tutorial]
    #######
    minit_randset_list = \
    minit.randset + minit2.randset + minit3.randset + minit4.randset + minit4.randset + minit5.randset
    tocal_randset_list = \
    tocal.randset + tocal2.randset + tocal3.randset + tocal4.randset + tocal4.randset + tocal5.randset
    randset_output_list = minit_randset_list + tocal_randset_list
    randset_input_list = [show_rand_settings]
    ######
    embin_logs = embin.logs + embin2.logs + embin3.logs + embin4.logs + embin5.logs
    tocal_logs = tocal.logs + tocal2.logs + tocal3.logs + tocal4.logs + tocal5.logs
    tokex_logs = tokex.logs + tokex2.logs + tokex3.logs + tokex4.logs + tokex5.logs
    tokm_logs = tokm.logs + tokm2.logs + tokm3.logs + tokm4.logs + tokm5.logs
    logs_output_list = embin_logs + tocal_logs + tokex_logs + tokm_logs
    logs_input_list = [show_output_logs]
    ######

    def Show (no_of_minit , no_of_embin , no_of_tocal , no_of_tokex , no_of_tokm , 
    show_tutorial , show_rand_settings , show_output_logs) : 
      return  {
              minit_list[0]  : gr.Accordion.update(visible=no_of_minit>0) ,
              minit_list[1]  : gr.Accordion.update(visible=no_of_minit>1) ,
              minit_list[2]  : gr.Accordion.update(visible=no_of_minit>2) ,
              minit_list[3]  : gr.Accordion.update(visible=no_of_minit>3) ,
              minit_list[4]  : gr.Accordion.update(visible=no_of_minit>4) ,

              embin_list[0]  : gr.Accordion.update(visible=no_of_embin>0) ,
              embin_list[1]  : gr.Accordion.update(visible=no_of_embin>1) ,
              embin_list[2]  : gr.Accordion.update(visible=no_of_embin>2) ,
              embin_list[3]  : gr.Accordion.update(visible=no_of_embin>3) ,
              embin_list[4]  : gr.Accordion.update(visible=no_of_embin>4) ,

              tocal_list[0]  : gr.Accordion.update(visible=no_of_tocal>0) ,
              tocal_list[1]  : gr.Accordion.update(visible=no_of_tocal>1) ,
              tocal_list[2]  : gr.Accordion.update(visible=no_of_tocal>2) ,
              tocal_list[3]  : gr.Accordion.update(visible=no_of_tocal>3) ,
              tocal_list[4]  : gr.Accordion.update(visible=no_of_tocal>4) ,

              tokex_list[0]  : gr.Accordion.update(visible=no_of_tokex>0) ,
              tokex_list[1]  : gr.Accordion.update(visible=no_of_tokex>1) ,
              tokex_list[2]  : gr.Accordion.update(visible=no_of_tokex>2) ,
              tokex_list[3]  : gr.Accordion.update(visible=no_of_tokex>3) ,
              tokex_list[4]  : gr.Accordion.update(visible=no_of_tokex>4) ,

              tokm_list[0]  : gr.Accordion.update(visible=no_of_tokm>0) ,
              tokm_list[1]  : gr.Accordion.update(visible=no_of_tokm>1) ,
              tokm_list[2]  : gr.Accordion.update(visible=no_of_tokm>2) ,
              tokm_list[3]  : gr.Accordion.update(visible=no_of_tokm>3) ,
              tokm_list[4]  : gr.Accordion.update(visible=no_of_tokm>4) ,

              tutorial_output_list[0]   : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[1]   : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[2]   : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[3]   : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[4]   : gr.Accordion.update(visible=show_tutorial) ,       
              tutorial_output_list[5]   : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[6]   : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[7]   : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[8]   : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[9]   : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[10]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[11]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[12]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[13]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[14]  : gr.Accordion.update(visible=show_tutorial) ,
              
              tutorial_output_list[15]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[16]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[17]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[18]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[19]  : gr.Accordion.update(visible=show_tutorial) ,       
              tutorial_output_list[20]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[21]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[22]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[23]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[24]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[25]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[26]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[27]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[28]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[29]  : gr.Accordion.update(visible=show_tutorial) ,

              tutorial_output_list[30]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[31]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[32]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[33]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[34]  : gr.Accordion.update(visible=show_tutorial) ,       
              tutorial_output_list[35]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[36]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[37]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[38]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[39]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[40]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[41]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[42]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[43]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[44]  : gr.Accordion.update(visible=show_tutorial) ,

              tutorial_output_list[45]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[46]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[47]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[48]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[49]  : gr.Accordion.update(visible=show_tutorial) ,       
              tutorial_output_list[50]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[51]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[52]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[53]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[54]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[55]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[56]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[57]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[58]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[59]  : gr.Accordion.update(visible=show_tutorial) ,

              tutorial_output_list[60]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[61]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[62]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[63]  : gr.Accordion.update(visible=show_tutorial) ,
              tutorial_output_list[64]  : gr.Accordion.update(visible=show_tutorial) ,

              minit_randset_list[0]  : gr.Accordion.update(visible=show_rand_settings) ,
              minit_randset_list[1]  : gr.Accordion.update(visible=show_rand_settings) ,
              minit_randset_list[2]  : gr.Accordion.update(visible=show_rand_settings) ,
              minit_randset_list[3]  : gr.Accordion.update(visible=show_rand_settings) ,
              minit_randset_list[4]  : gr.Accordion.update(visible=show_rand_settings) ,

              tocal_randset_list[0]  : gr.Accordion.update(visible=show_rand_settings) ,
              tocal_randset_list[1]  : gr.Accordion.update(visible=show_rand_settings) ,
              tocal_randset_list[2]  : gr.Accordion.update(visible=show_rand_settings) ,
              tocal_randset_list[3]  : gr.Accordion.update(visible=show_rand_settings) ,
              tocal_randset_list[4]  : gr.Accordion.update(visible=show_rand_settings) ,

              embin_logs[0]  : gr.Accordion.update(visible=show_output_logs) , 
              embin_logs[1]  : gr.Accordion.update(visible=show_output_logs) , 
              embin_logs[2]  : gr.Accordion.update(visible=show_output_logs) , 
              embin_logs[3]  : gr.Accordion.update(visible=show_output_logs) , 
              embin_logs[4]  : gr.Accordion.update(visible=show_output_logs) , 

              tocal_logs[0]  : gr.Accordion.update(visible=show_output_logs) , 
              tocal_logs[1]  : gr.Accordion.update(visible=show_output_logs) , 
              tocal_logs[2]  : gr.Accordion.update(visible=show_output_logs) , 
              tocal_logs[3]  : gr.Accordion.update(visible=show_output_logs) , 
              tocal_logs[4]  : gr.Accordion.update(visible=show_output_logs) , 

              tokex_logs[0]  : gr.Accordion.update(visible=show_output_logs) , 
              tokex_logs[1]  : gr.Accordion.update(visible=show_output_logs) , 
              tokex_logs[2]  : gr.Accordion.update(visible=show_output_logs) , 
              tokex_logs[3]  : gr.Accordion.update(visible=show_output_logs) , 
              tokex_logs[4]  : gr.Accordion.update(visible=show_output_logs) , 

              tokm_logs[0]  : gr.Accordion.update(visible=show_output_logs) , 
              tokm_logs[1]  : gr.Accordion.update(visible=show_output_logs) , 
              tokm_logs[2]  : gr.Accordion.update(visible=show_output_logs) , 
              tokm_logs[3]  : gr.Accordion.update(visible=show_output_logs) , 
              tokm_logs[4]  : gr.Accordion.update(visible=show_output_logs) 

              } 

    output_list = \
    module_output_list + tutorial_output_list + randset_output_list + logs_output_list

    input_list =  \
    module_input_list + tutorial_input_list + randset_input_list + logs_input_list

    module_update_button.click(fn = Show , inputs = input_list , outputs =  output_list)

    minits = [minit , minit2 , minit3 , minit4 , minit5]
    embins = [embin , embin2 , embin3 , embin4 , embin5]
    tocals = [tocal , tocal2 , tocal3 , tocal4 , tocal5]
    tokexs = [tokex , tokex2 , tokex3 , tokex4 , tokex5]
    tokm_modules = minits + embins + tocals + tokexs
    #tokms =  [tokm , tokm2 , tokm3 , tokm4 , tokm5] 
    #(Ignore writing to extra tokm modules as this will just
    # reduce performance for no real benefit. 
    #Extra tokm modules can still be used, 
    #since they share the data class , but their input fields
    #will appear blank /Nekos)

    #Assign functionality to buttons
    for _module in tokm_modules : 
      _module.setupIO_with(tokm)

  #End of UI
  return [(ui, "TokenMixer", "TokenMixer")]

script_callbacks.on_ui_tabs(add_tab)
        

