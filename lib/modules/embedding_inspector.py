import gradio as gr
from modules import script_callbacks, shared, sd_hijack
from modules.shared import cmd_opts
from pandas import Index
from pandas.core.groupby.groupby import OutputFrameOrSeries
import torch, os
from modules.textual_inversion.textual_inversion import Embedding
import collections, math, random , numpy
import re #used to parse text to int
import copy
from torch.nn.modules import ConstantPad1d, container

from lib.toolbox.constants import MAX_NUM_MIX , SEP_STR

from lib.data import dataStorage

class EmbeddingInspector :

  def Inspect (self, *args) :

    mini_input = args[0]
    sendtomix = args[1]
    separate = args[2]  
    similarity = args[3]
    percentage = args[4]
    max_similar_embs = args[5]
    id_mode = args[6]
    mix_input = args[7]
    stack_mode = args[8]
    send_to_negatives = args[9]
    neg_input = args[10]

    #Check if new embeddings have been added 
    try: sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
    except: 
      sd_hijack.model_hijack.embedding_db.dir_mtime=0
      sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
    self.data.update_loaded_embs() 

    negbox = neg_input
    if neg_input == None : negbox = ''
    if send_to_negatives and not stack_mode : negbox = ''

    tokenmixer_vectors = mix_input
    if tokenmixer_vectors == None : tokenmixer_vectors = ''
    if sendtomix and not stack_mode : tokenmixer_vectors = ''

    #Clear all inputs from tokenmixer
    for index in range(MAX_NUM_MIX):
      self.data.clear(
        index , 
        to_mixer = sendtomix and (not stack_mode) , 
        to_temporary = True , 
        to_negative = send_to_negatives and (not stack_mode))
    #####

    results = []
    log = []

    if send_to_negatives :
      log.append("send_to_negatives : True")
    else:
      log.append("send_to_negatives : False")   

    if sendtomix :
      log.append("sendtomix : True")
    else: 
      log.append("sendtomix : False")

    log.append("----")

    emb_vec = None
    emb_id = None
    emb_name = None
    loaded_emb = None
    text = None

    if mini_input == None or mini_input == '' : 
      log.append("Mini-input is empty!")
      return '' , '' , '\n'.join(results) , '\n'.join(log)
    
    assert  \
    sendtomix != None or \
    send_to_negatives != None or \
    percentage != None or \
    separate != None or \
    id_mode != None
    max_similar_embs !=None
    similarity != None , "Nonetype input on Embedding Inspector!"

    #Process the input string to get all the vectors
    if id_mode : 
      for word in mini_input.strip().lower().split() :
        results.append("ID mode")
        text = copy.copy(word)
        break #Only take the first ID in the input
    else: text = copy.copy(mini_input.strip().lower())


    loaded_emb = self.data.tools.loaded_embs.get(text, None)
    emb_id = None

    if id_mode and  text.isdigit(): 
      emb_id = copy.copy(int(text))
      assert isinstance(emb_id , int) , "emb_id is not int!"
    else: emb_id = self.data.tools.no_of_internal_embs


    #ID input mode
    if (emb_id < self.data.tools.no_of_internal_embs) and (emb_id > 0): 
      emb_vec = self.data.tools.internal_embs[emb_id].unsqueeze(0)
      emb_name = self.data.emb_id_to_name(emb_id)
      results.append("Id mode : Found vector '" + \
      emb_name + "' with ID #" + str(emb_id))
    else:
      emb_name, emb_id, emb_vec, loaded_emb = self.data.get_embedding_info(text)
      results.append("Tokenize : Found vector '" + \
      emb_name + "' with ID #" + str(emb_id))
    ######

    if emb_vec == None : 
      results.append('emb_vec is Nonetype!') 
      return '' , '' , '\n'.join(results) , '\n'.join(log) 

    # add embedding info to results
    results.append('Embedding name: "'+emb_name+'"')
    results.append('Vector count: '+str(emb_vec.shape[0]))
    results.append('Vector size: '+str(emb_vec.shape[1]))
    
    if loaded_emb!=None: 
      results.append('Step: '+str(loaded_emb.step))
      results.append('SD checkpoint: '+str(loaded_emb.sd_checkpoint))
      results.append('SD checkpoint name: '+str(loaded_emb.sd_checkpoint_name))
      if hasattr(loaded_emb, 'filename'):
            results.append('Filename: '+str(loaded_emb.filename))

    if type(emb_id)==int:
        results.append('Embedding ID: '+str(emb_id)+' (internal)')
    else:
        results.append('Embedding ID: '+str(emb_id)+' (loaded)')

    results.append(SEP_STR)
    results.append("get_emb_info....")

    dist = None
    best_ids = None
    sorted_scores = None
    tmp = None
   
    if (emb_vec.shape[1] == self.data.vector.size) :
      
      distance = torch.nn.PairwiseDistance(p=2)
      origin = self.data.vector.origin.cpu()
      tmp = emb_vec.cpu()
      dist = distance(tmp , origin).numpy()[0]

      results.append('Vector length: '+str(dist))
      results.append('')
      results.append("Similar tokens:")

      best_ids_list = []
      sorted_scores_list = []

      if type(emb_id)==int:
        _best_ids , _sorted_scores = \
        self.data.tools.get_best_ids(emb_id , similarity , max_similar_embs , None)
        best_ids_list.append(_best_ids)
        sorted_scores_list.append(_sorted_scores)

      else:
        for k in range (emb_vec.shape[0]):
          _best_ids , _sorted_scores = \
          self.data.tools.get_best_ids(emb_id , similarity , max_similar_embs , emb_vec[k])
          best_ids_list.append(_best_ids)
          sorted_scores_list.append(_sorted_scores)

    else: 
      results.append('Vector size is not compatible with current SD model')
      results.append("emb vec size : " + str(emb_vec.shape[0]) + \
      ' x '  + str(emb_vec.shape[1]))
      results.append("required size :  1 x " + str(self.data.vector.size))
      return '' , '' , '\n'.join(results) , '\n'.join(log) 

    #Start loop

    for k in range(len(best_ids_list)):
      best_ids = best_ids_list[k]
      sorted_scores = sorted_scores_list[k]
      names = []
      ids = []
      vecs = []
      full = []
      sim = []
      emb_name = None
      for i in range(max_similar_embs):
        emb_id = best_ids[i].item()
        emb_name = self.data.emb_id_to_name(emb_id)
        emb_vec = self.data.emb_id_to_vec(emb_id)
        vecs.append(emb_vec)
        names.append(emb_name)
        ids.append(str(emb_id))
        full.append(emb_name+'('+str(emb_id)+')')
        sim.append(str(round((100*sorted_scores).numpy()[i] , 1))+'% ')


        if k==0 and send_to_negatives :
          for index in range(MAX_NUM_MIX):
            if (not self.data.negative.isEmpty.get(index)): continue
            self.data.place(index , \
                vector =   emb_vec.unsqueeze(0).cpu(),
                ID =  emb_id ,
                name = copy.copy(emb_name) , 
                to_mixer = False , 
                to_temporary = True , 
                to_negative = send_to_negatives)
            break
        #########

        if k==0 and sendtomix :
          for index in range(MAX_NUM_MIX):
            if (not self.data.vector.isEmpty.get(index)): continue
            self.data.place(index , \
                vector =   emb_vec.unsqueeze(0).cpu(),
                ID =  emb_id ,
                name = copy.copy(emb_name) , 
                to_mixer = sendtomix , 
                to_temporary = True , 
                to_negative = False)
            break
        #########
      ##########           

      results.append('Vector #' + str(k) + ' :')

      if separate :
        results.append('Names:')
        results.append('   '.join(names))
        results.append('IDs:')
        results.append('   '.join(ids))
      else: 
        results.append('   '.join(full))

      if percentage: 
        results.append('Score:')
        results.append('   '.join(sim))

      results.append(SEP_STR)

      for index in range(MAX_NUM_MIX):
        name = self.data.vector.name.get(index)
        if name != None :
          if sendtomix : 
            if tokenmixer_vectors != '': tokenmixer_vectors = tokenmixer_vectors + ' , '
            tokenmixer_vectors = tokenmixer_vectors  + name
          
      for index in range(MAX_NUM_MIX):
        name = self.data.negative.name.get(index)
        if name != None :        
          if send_to_negatives:
            if negbox != '': negbox = negbox + ' , '
            negbox = negbox  + name


    return  tokenmixer_vectors , negbox , '\n'.join(results) , '\n'.join(log)
       
    
  def setupIO_with (self, module):
    #Tell the buttons in this class what to do when 
    #pressed by the user

    cond = self.data.tools.loaded

    input_list = []
    output_list = []

    if (module.ID == "MiniTokenizer") and cond :
      return 1

    if (module.ID == "EmbeddingInspector") and cond:
      return 1

    if (module.ID == "TokenMixer") and cond :
      input_list.append(self.inputs.mini_input)       #0
      input_list.append(self.inputs.sendtomix)        #1
      input_list.append(self.inputs.separate)         #2
      input_list.append(self.inputs.similarity)       #3
      input_list.append(self.inputs.percentage)       #4
      input_list.append(self.inputs.max_similar_embs) #5
      input_list.append(self.inputs.id_mode)          #6
      input_list.append(module.inputs.mix_input)      #7
      input_list.append(self.inputs.stack_mode)       #8
      input_list.append(self.inputs.negatives)        #9
      input_list.append(module.inputs.negbox)         #10
      #######

      output_list.append(module.inputs.mix_input) #0
      output_list.append(module.inputs.negbox)    #1
      output_list.append(self.outputs.results)    #2
      output_list.append(self.outputs.log)        #3
      self.buttons.inspect.click(fn=self.Inspect , inputs = input_list , outputs = output_list)
  #End of setupIO_with()


  def __init__(self , label , vis = False):

    self.data = dataStorage 

    class Outputs :
      def __init__(self):
        Outputs.results = []
        Outputs.log = []

    class Inputs :
      def __init__(self):
        Inputs.mini_input = []
        Inputs.sendtomix = []
        Inputs.similarity = []
        Inputs.separate = []
        Inputs.percentage = []
        Inputs.max_similar_embs = []
        Inputs.id_mode = []
        Inputs.stack_mode = []
        Inputs.negatives = []

    class Buttons :
      def __init__(self):
        Buttons.inspect = []
        Buttons.reset = []

    self.outputs= Outputs()
    self.inputs = Inputs()
    self.buttons= Buttons()
    self.ID = "EmbeddingInspector"

    #create UI
    with gr.Accordion( label ,open=False , visible = vis) as show:
      gr.Markdown("Find CLIP tokens similar to input token")
      with gr.Row() :  
          self.inputs.similarity = gr.Slider(label="Similarity upper bound %",value=100, \
                                      minimum=-100, maximum=100, step=0.1 , interactive = True)
      with gr.Row() :  
          self.inputs.max_similar_embs = gr.Slider(label="Number of tokens to display",value=30, \
                                      minimum=0, maximum=300, step=1 , interactive = True)
      with gr.Row() : 
          self.buttons.inspect = gr.Button(value="Get similar tokens", variant="primary")
          self.buttons.reset = gr.Button(value="Reset", variant = "secondary") 
      with gr.Row() : 
        self.inputs.mini_input = gr.Textbox(label='', lines=2, \
        placeholder="Enter a single vector token or embedding" , interactive = True)
      with gr.Row() : 
        self.outputs.results = gr.Textbox(label="Output", lines=10, interactive = False)
      with gr.Row():
          self.inputs.sendtomix = gr.Checkbox(value=False, label="Send to input", interactive = True)
          self.inputs.id_mode = gr.Checkbox(value=False, label="ID Mode", interactive = True)
          self.inputs.separate = gr.Checkbox(value=False, label="Separate name and ID", interactive = True)
          self.inputs.percentage = gr.Checkbox(value=False, label="Show similarity %", interactive = True)
          self.inputs.stack_mode = gr.Checkbox(value=False, label="Stack Mode", interactive = True)
          self.inputs.negatives = gr.Checkbox(value=False, label="Send to negatives", interactive = True) 
      with gr.Accordion("Output Log" , open = False) as logs :
        self.outputs.log = gr.Textbox(label="Output log", lines=5, interactive = False)   
      
      
      with gr.Accordion('Tutorial : What is this?',open=False , visible= False) as tutorial_0 : 
          gr.Markdown("The embedding inspector allows you to find similar " + \
          "CLIP tokens to a given input token. \n \n" + \
          "By setting the 'Similarity upper bound %' slider you can set how " + \
          "similar the generated vectors should be at most to the input token. \n \n" + \
          "You can send the generated vectors to the TokenMixer by selecting the " + \
          "'Send vectors similar to first vector to TokenMixer ' checkbox. \n \n" + \
          "You can input the ID of the input token (an integer between 0 and " + \
          str(self.data.tools.no_of_internal_embs) + ") instead of its name by " + \
          "selecting the 'ID input mode' checkbox.\n \n"
          "If you wish to copy and paste the similar tokens, you can select the " + \
          "'Separate name and ID' to keep the token names and IDs apart " + \
          "from each other. \n \n" + \
          "If you wish to include the similarity score in the output, you can select " + \
          "the 'Show similarity %' checkbox . \n \n" + \
          "The code for this module is built upon the extension provided by tkalayci71: \n " + \
          "https://github.com/tkalayci71/embedding-inspector" )

          with gr.Accordion('How does this work?',open=False): 
            gr.Markdown("Token similarity is measured using cosine similarity , where a value of 1 ( or 100%) " + \
            "means the token vectors are parallell to each other and a value of 0 (or 0%) " + \
            "means that they are perpendicular to each other. \n \n " 
            "Negative cosine similarity values are possible, but in this tool they are treated " + \
            "the same as positive cosine similarity in order to make it easy to understand the " + \
            "similarity metric. \n \n This means that a token vector with " +\
            " 100% similarity (or some other value) to the input token vector can sometimes be " + \
            "oriented in the opposite direction of the input token vector. \n \n")
    #end of UI

    self.tutorials = [tutorial_0]
    self.show = [show]
    self.logs = [logs]
    
    self.buttons.inspect.style(size="sm")
    self.buttons.reset.style(size="sm")
    if self.data.tools.loaded : self.setupIO_with(self)

    #self.buttons.split.click(fn=self.Split , inputs = None , outputs = None )
## end of class  EmbeddingInspector
#------------------------------------------------------------------------------#
