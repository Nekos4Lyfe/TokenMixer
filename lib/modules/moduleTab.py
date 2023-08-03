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

from lib.modules.minitokenizer import MiniTokenizer
from lib.modules.embedding_inspector import EmbeddingInspector
from lib.modules.token_mixer import TokenMixer      
from lib.modules.token_calculator import TokenCalculator 
from lib.modules.token_extrapolator import TokenExtrapolator 
from lib.modules.cross_attention_mixer import CrossAttentionMixer
from lib.modules.synonymizer import Synonymizer

from lib.data import dataStorage

class Modules : 

  def __init__(self, label , vis = False , first = False):

    #Pass reference to global object "dataStorage" to class
    self.data = dataStorage 
    self.ID = "Modules"

    tutorial_string = "The TokenMixer is a tool that takes" + \
    "tokens and/or embedding vectors as inputs and processes these into new " + \
    "tokens and/or embeddings. \n \n Above this tab ↑ you see a number of modules . \n \n " + \
    "These modules will take a token and/or embedding vector as an input ,  " + \
    "split them into individual vectors which are then passed " + \
    "on to the Embedding generator on the right → . \n \n " + \
    "The Embedding generator then proceses these vectors into embeddings which you can use in your prompts. \n \n" + \
    "If you wish to see an in-depth tutorial for each module, enable the 'Show tutorial' " + \
    "checkbox inside the 'Remove/add modules' tab ↑ and press the 'Update modules' button. \n \n This will append a 'Tutorial : What is this?' " + \
    "tab to the relevant section, where information about the given module is provided. \n \n" + \
    "The tabs have different names to make organization easier. All tabs works the same. \n \n" + \
    "You can also refer to the examples provided at the TokenMixer github page: \n" + \
    "https://github.com/Nekos4Lyfe/TokenMixer"

    #test = gr.Label('Prompt MiniTokenizer' , color = "red")
    #create UI
    with gr.Tab(label , visible = vis) as tab :
      gr.Markdown("") 
      #######
      self.minit = MiniTokenizer("MiniTokenizer" , vis = first)
      ######
      self.minit2 = MiniTokenizer("MiniTokenizer #2")
      self.minit3 = MiniTokenizer("MiniTokenizer #3")
      self.minit4 = MiniTokenizer("MiniTokenizer #4")
      self.minit5 = MiniTokenizer("MiniTokenizer #5")
      ########
      self.embin = EmbeddingInspector("Embedding Inspector" , vis = first) 
      self.embin2 = EmbeddingInspector("Embedding Inspector#2") 
      self.embin3 = EmbeddingInspector("Embedding Inspector#3") 
      self.embin4 = EmbeddingInspector("Embedding Inspector#4") 
      self.embin5 = EmbeddingInspector("Embedding Inspector#5") 
      ########
      self.tocal = TokenCalculator("Token Calculator" , vis = first)
      self.tocal2 = TokenCalculator("Token Calculator#2")
      self.tocal3 = TokenCalculator("Token Calculator#3")
      self.tocal4 = TokenCalculator("Token Calculator#4")
      self.tocal5 = TokenCalculator("Token Calculator#5")
      ########
      self.tokex = TokenExtrapolator("Token Extrapolator" , vis = first)
      self.tokex2 = TokenExtrapolator("Token Extrapolator#2")
      self.tokex3 = TokenExtrapolator("Token Extrapolator#3")
      self.tokex4 = TokenExtrapolator("Token Extrapolator#4")
      self.tokex5 = TokenExtrapolator("Token Extrapolator#5")
      ########
      self.synz = Synonymizer("Token Synonymizer" , vis = first)
      self.synz2 = Synonymizer("Token Synonymizer #2")
      self.synz3 = Synonymizer("Token Synonymizer #3")
      self.synz4 = Synonymizer("Token Synonymizer #4")
      self.synz5 = Synonymizer("Token Synonymizer #5")
      gr.Markdown(" ")
      ########
      with gr.Accordion('Remove/add modules',open=False):
        if first :
          self.no_of_minit = gr.Slider(value = 1, minimum=0, maximum=5, \
          step=1, label="Number of MiniTokenizers", \
          default=1 , interactive = True)
          self.no_of_embin = gr.Slider(value = 1 , minimum=0, maximum=5, \
          step=1, label="Number of Embedding Inspectors", \
          default=1 , interactive = True)
          self.no_of_tocal = gr.Slider(value = 1 , minimum=0, maximum=5, \
          step=1, label="Number of Token Calculators", \
          default=1 , interactive = True)
          self.no_of_tokex = gr.Slider(value = 1 , minimum=0, maximum=5, \
          step=1, label="Number of Token Extrapolators", \
          default=1 , interactive = True)
          self.no_of_tokm = gr.Slider(value = 1 , minimum=0, maximum=5, \
          step=1, label="Number of Token Mixers", \
          default=1 , interactive = True)
          self.no_of_synz = gr.Slider(value = 1 , minimum=0, maximum=5, \
          step=1, label="Number of Synonymizers", \
          default=1 , interactive = True)           
        else : 
          self.no_of_minit = gr.Slider(value = 0, minimum=0, maximum=5, \
          step=1, label="Number of MiniTokenizers", \
          default=0 , interactive = True)
          self.no_of_embin = gr.Slider(value = 0 , minimum=0, maximum=5, \
          step=1, label="Number of Embedding Inspectors", \
          default=0 , interactive = True)
          self.no_of_tocal = gr.Slider(value = 0 , minimum=0, maximum=5, \
          step=1, label="Number of Token Calculators", \
          default=0 , interactive = True)
          self.no_of_tokex = gr.Slider(value = 0 , minimum=0, maximum=5, \
          step=1, label="Number of Token Extrapolators", \
          default=0 , interactive = True)
          self.no_of_tokm = gr.Slider(value = 1 , minimum=0, maximum=5, \
          step=1, label="Number of Token Mixers", \
          default=1 , interactive = True)
          self.no_of_synz = gr.Slider(value = 0 , minimum=0, maximum=5, \
          step=1, label="Number of Synonymizers", \
          default=0 , interactive = True)
        ######
        with gr.Row():
          self.show_tutorial = gr.Checkbox(value=False, label="Show tutorials", interactive = True)
          self.show_output_logs = gr.Checkbox(value=True, label="Show output logs", interactive = True)
          self.show_rand_settings = gr.Checkbox(value=True, label="Show random '_' token settings ", interactive = True)
        with gr.Row():
          self.module_update_button = gr.Button(value="Update modules", variant="primary")
          self.module_update_button.style(size="sm")

        with gr.Accordion('How do I use this extension?',open=False):
          gr.Markdown(tutorial_string) 
        gr.Markdown("")
      ######### 
      self.tab = tab
   # settings_update_button.click(fn = Hide_column , 
   # inputs = [show_left_column , show_right_column] , outputs = [left_column , right_column])

  def Show (self , no_of_minit , no_of_embin , \
    no_of_tocal , no_of_tokex, no_of_tokm , no_of_synz , \
    show_tutorial , show_rand_settings , show_output_logs) :

      #Assign functionality to buttons in the rest of
      # the TokenMixer (if revealed by the user)
    if no_of_tokm>1:
        for _module in self.tokm_modules :
          for k in range(len(self.tokms)):
            if k> no_of_tokm : continue
            _module.setupIO_with(self.tokms[k])


    return  {
              self.minit_list[0]  : gr.Accordion.update(visible=no_of_minit>0) ,
              self.minit_list[1]  : gr.Accordion.update(visible=no_of_minit>1) ,
              self.minit_list[2]  : gr.Accordion.update(visible=no_of_minit>2) ,
              self.minit_list[3]  : gr.Accordion.update(visible=no_of_minit>3) ,
              self.minit_list[4]  : gr.Accordion.update(visible=no_of_minit>4) ,

              self.embin_list[0]  : gr.Accordion.update(visible=no_of_embin>0) ,
              self.embin_list[1]  : gr.Accordion.update(visible=no_of_embin>1) ,
              self.embin_list[2]  : gr.Accordion.update(visible=no_of_embin>2) ,
              self.embin_list[3]  : gr.Accordion.update(visible=no_of_embin>3) ,
              self.embin_list[4]  : gr.Accordion.update(visible=no_of_embin>4) ,

              self.tocal_list[0]  : gr.Accordion.update(visible=no_of_tocal>0) ,
              self.tocal_list[1]  : gr.Accordion.update(visible=no_of_tocal>1) ,
              self.tocal_list[2]  : gr.Accordion.update(visible=no_of_tocal>2) ,
              self.tocal_list[3]  : gr.Accordion.update(visible=no_of_tocal>3) ,
              self.tocal_list[4]  : gr.Accordion.update(visible=no_of_tocal>4) ,

              self.tokex_list[0]  : gr.Accordion.update(visible=no_of_tokex>0) ,
              self.tokex_list[1]  : gr.Accordion.update(visible=no_of_tokex>1) ,
              self.tokex_list[2]  : gr.Accordion.update(visible=no_of_tokex>2) ,
              self.tokex_list[3]  : gr.Accordion.update(visible=no_of_tokex>3) ,
              self.tokex_list[4]  : gr.Accordion.update(visible=no_of_tokex>4) ,

              self.tokm_list[0]  : gr.Accordion.update(visible=no_of_tokm>0) ,
              self.tokm_list[1]  : gr.Accordion.update(visible=no_of_tokm>1) ,
              self.tokm_list[2]  : gr.Accordion.update(visible=no_of_tokm>2) ,
              self.tokm_list[3]  : gr.Accordion.update(visible=no_of_tokm>3) ,
              self.tokm_list[4]  : gr.Accordion.update(visible=no_of_tokm>4) ,

              self.synz_list[0]  : gr.Accordion.update(visible=no_of_synz>0) ,
              self.synz_list[1]  : gr.Accordion.update(visible=no_of_synz>1) ,
              self.synz_list[2]  : gr.Accordion.update(visible=no_of_synz>2) ,
              self.synz_list[3]  : gr.Accordion.update(visible=no_of_synz>3) ,
              self.synz_list[4]  : gr.Accordion.update(visible=no_of_synz>4) ,

              self.tutorial_output_list[0]   : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[1]   : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[2]   : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[3]   : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[4]   : gr.Accordion.update(visible=show_tutorial) ,       
              self.tutorial_output_list[5]   : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[6]   : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[7]   : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[8]   : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[9]   : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[10]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[11]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[12]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[13]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[14]  : gr.Accordion.update(visible=show_tutorial) ,
              
              self.tutorial_output_list[15]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[16]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[17]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[18]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[19]  : gr.Accordion.update(visible=show_tutorial) ,       
              self.tutorial_output_list[20]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[21]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[22]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[23]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[24]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[25]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[26]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[27]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[28]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[29]  : gr.Accordion.update(visible=show_tutorial) ,

              self.tutorial_output_list[30]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[31]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[32]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[33]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[34]  : gr.Accordion.update(visible=show_tutorial) ,       
              self.tutorial_output_list[35]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[36]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[37]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[38]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[39]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[40]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[41]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[42]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[43]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[44]  : gr.Accordion.update(visible=show_tutorial) ,

              self.tutorial_output_list[45]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[46]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[47]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[48]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[49]  : gr.Accordion.update(visible=show_tutorial) ,       
              self.tutorial_output_list[50]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[51]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[52]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[53]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[54]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[55]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[56]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[57]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[58]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[59]  : gr.Accordion.update(visible=show_tutorial) ,

              self.tutorial_output_list[60]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[61]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[62]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[63]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[64]  : gr.Accordion.update(visible=show_tutorial) ,

              self.tutorial_output_list[65]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[66]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[67]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[68]  : gr.Accordion.update(visible=show_tutorial) ,
              self.tutorial_output_list[69]  : gr.Accordion.update(visible=show_tutorial) ,

              self.minit_randset_list[0]  : gr.Accordion.update(visible=show_rand_settings) ,
              self.minit_randset_list[1]  : gr.Accordion.update(visible=show_rand_settings) ,
              self.minit_randset_list[2]  : gr.Accordion.update(visible=show_rand_settings) ,
              self.minit_randset_list[3]  : gr.Accordion.update(visible=show_rand_settings) ,
              self.minit_randset_list[4]  : gr.Accordion.update(visible=show_rand_settings) ,

              self.tocal_randset_list[0]  : gr.Accordion.update(visible=show_rand_settings) ,
              self.tocal_randset_list[1]  : gr.Accordion.update(visible=show_rand_settings) ,
              self.tocal_randset_list[2]  : gr.Accordion.update(visible=show_rand_settings) ,
              self.tocal_randset_list[3]  : gr.Accordion.update(visible=show_rand_settings) ,
              self.tocal_randset_list[4]  : gr.Accordion.update(visible=show_rand_settings) ,

              self.embin_logs[0]  : gr.Accordion.update(visible=show_output_logs) , 
              self.embin_logs[1]  : gr.Accordion.update(visible=show_output_logs) , 
              self.embin_logs[2]  : gr.Accordion.update(visible=show_output_logs) , 
              self.embin_logs[3]  : gr.Accordion.update(visible=show_output_logs) , 
              self.embin_logs[4]  : gr.Accordion.update(visible=show_output_logs) , 

              self.tocal_logs[0]  : gr.Accordion.update(visible=show_output_logs) , 
              self.tocal_logs[1]  : gr.Accordion.update(visible=show_output_logs) , 
              self.tocal_logs[2]  : gr.Accordion.update(visible=show_output_logs) , 
              self.tocal_logs[3]  : gr.Accordion.update(visible=show_output_logs) , 
              self.tocal_logs[4]  : gr.Accordion.update(visible=show_output_logs) , 

              self.tokex_logs[0]  : gr.Accordion.update(visible=show_output_logs) , 
              self.tokex_logs[1]  : gr.Accordion.update(visible=show_output_logs) , 
              self.tokex_logs[2]  : gr.Accordion.update(visible=show_output_logs) , 
              self.tokex_logs[3]  : gr.Accordion.update(visible=show_output_logs) , 
              self.tokex_logs[4]  : gr.Accordion.update(visible=show_output_logs) , 

              self.tokm_logs[0]  : gr.Accordion.update(visible=show_output_logs) , 
              self.tokm_logs[1]  : gr.Accordion.update(visible=show_output_logs) , 
              self.tokm_logs[2]  : gr.Accordion.update(visible=show_output_logs) , 
              self.tokm_logs[3]  : gr.Accordion.update(visible=show_output_logs) , 
              self.tokm_logs[4]  : gr.Accordion.update(visible=show_output_logs) ,
              } 
    ################# End of show() functions 

  def setupIO_with (self, 
  tokm , tokm2 , tokm3 , tokm4  , tokm5):

    if not self.data.tools.loaded: return None
    if tokm== None : return None

    #Copy the adresses to the tokms
    minits = [self.minit , self.minit2 , self.minit3 , self.minit4 , self.minit5] # 5x Minitokenizer modules
    embins = [self.embin , self.embin2 , self.embin3 , self.embin4 , self.embin5] # 5x Embedding Inspector modules
    tocals = [self.tocal , self.tocal2 , self.tocal3 , self.tocal4 , self.tocal5] # 5x Token Calculator modules
    tokexs = [self.tokex , self.tokex2 , self.tokex3 , self.tokex4 , self.tokex5] # 5x Token Extrapolator modules
    synzs =  [self.synz  , self.synz2  , self.synz3  , self.synz4  , self.synz5]
    #########
    self.tokm_modules = minits + embins + tocals + tokexs + synzs
    self.tokms =  [tokm , tokm2 , tokm3 , tokm4 , tokm5] #5x TokenMixers


    ######
    input_list = []
    output_list = []

    #Assign functionality to buttons in the first TokenMixer
    #and the rest of the modules
    for _module in self.tokm_modules : 
      _module.setupIO_with(tokm)

    self.minit_list = self.minit.show + self.minit2.show + self.minit3.show + self.minit4.show + self.minit5.show
    self.embin_list = self.embin.show + self.embin2.show + self.embin3.show + self.embin4.show + self.embin5.show
    self.tocal_list = self.tocal.show + self.tocal2.show + self.tocal3.show + self.tocal4.show + self.tocal5.show
    self.tokex_list = self.tokex.show + self.tokex2.show + self.tokex3.show + self.tokex4.show + self.tokex5.show
    self.synz_list = self.synz.show   + self.synz2.show  + self.synz3.show  + self.synz4.show  + self.synz5.show
    self.tokm_list = tokm.show + tokm2.show + tokm3.show + tokm4.show + tokm5.show
    self.module_output_list = self.minit_list + self.embin_list + self.tocal_list + self.tokex_list + self.tokm_list + self.synz_list
    self.module_input_list = \
    [self.no_of_minit , self.no_of_embin , \
    self.no_of_tocal , self.no_of_tokex, self.no_of_tokm ,self.no_of_synz]

    #######
    self.tutorial_output_list = \
    self.minit.tutorials + self.minit2.tutorials + self.minit3.tutorials + self.minit4.tutorials + self.minit5.tutorials + \
    self.embin.tutorials + self.embin2.tutorials + self.embin3.tutorials + self.embin4.tutorials + self.embin5.tutorials + \
    self.tocal.tutorials + self.tocal2.tutorials + self.tocal3.tutorials + self.tocal4.tutorials + self.tocal5.tutorials + \
    self.tokex.tutorials + self.tokex2.tutorials + self.tokex3.tutorials + self.tokex4.tutorials + self.tokex5.tutorials + \
    tokm.tutorials + tokm2.tutorials + tokm3.tutorials + tokm4.tutorials + tokm5.tutorials + \
    self.synz.tutorials  + self.synz2.tutorials  + self.synz3.tutorials  + self.synz4.tutorials  + self.synz5.tutorials

    self.tutorial_input_list = [self.show_tutorial]
    #######
    self.minit_randset_list = \
    self.minit.randset + self.minit2.randset + self.minit3.randset + self.minit4.randset + self.minit4.randset + self.minit5.randset
    self.tocal_randset_list = \
    self.tocal.randset + self.tocal2.randset + self.tocal3.randset + self.tocal4.randset + self.tocal4.randset + self.tocal5.randset
    self.randset_output_list = self.minit_randset_list + self.tocal_randset_list
    self.randset_input_list = [self.show_rand_settings]
    ######
    self.embin_logs = self.embin.logs + self.embin2.logs + self.embin3.logs + self.embin4.logs + self.embin5.logs
    self.tocal_logs = self.tocal.logs + self.tocal2.logs + self.tocal3.logs + self.tocal4.logs + self.tocal5.logs
    self.tokex_logs = self.tokex.logs + self.tokex2.logs + self.tokex3.logs + self.tokex4.logs + self.tokex5.logs
    self.tokm_logs = tokm.logs + tokm2.logs + tokm3.logs + tokm4.logs + tokm5.logs
    self.logs_output_list = self.embin_logs + self.tocal_logs + self.tokex_logs + self.tokm_logs
    self.logs_input_list = [self.show_output_logs]
    ######
    ######
    output_list = \
    self.module_output_list + self.tutorial_output_list + \
    self.randset_output_list + self.logs_output_list

    input_list =  \
    self.module_input_list + self.tutorial_input_list + \
    self.randset_input_list + self.logs_input_list

    self.module_update_button.click(fn = self.Show , inputs = input_list , outputs =  output_list)
    ######
## End of class MiniTokenizer--------------------------------------------------#

    ########## End of UI


        

