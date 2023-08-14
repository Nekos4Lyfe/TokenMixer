import gradio as gr
from modules import sd_hijack
import torch, os
from modules.textual_inversion.textual_inversion import Embedding
import collections, math, random , numpy
import re #used to parse name_string to int
import copy
from lib.toolbox.constants import MAX_NUM_MIX 
from lib.data import dataStorage

class TokenExtrapolator:

  def Reset (self , first , tokenbox) : 
    return '' , ''

  def get_random_id_below(self , value) :
      self.emb_id = copy.copy(value)
      self.emb_id = math.floor(self.emb_id*random.random())
      return self.emb_id

  def get_random_id_above(self , value) :
      self.emb_id = copy.copy(value)
      self.emb_id = math.floor(self.emb_id + \
      (self.no_of_internal_embs - self.emb_id)*random.random())
      return self.emb_id

  def get_id(self, string):
      emb_name, self.emb_id , emb_vec, loaded_emb = \
      self.data.get_embedding_info(string)
      return self.emb_id

  def get_embedding_data(self, string):
      self.emb_name, self.emb_id , self.emb_vec, loaded_emb = \
      self.data.get_embedding_info(string)
      return self.emb_name, self.emb_id , self.emb_vec
  
  def get_name(self, emb_id):
    self.emb_name = self.data.emb_id_to_name(emb_id)
    if self.emb_name.find('</w>')>=0:
      self.emb_name = self.emb_name.split('</w>')[0]
    return self.emb_name

  def isText(self , string):
    if string == None: return False
    if not isinstance(string, str): return False
    if string == '': return False
    tmp = string.strip()
    if tmp == None: return False 
    if tmp == '': return False
    return True

  def travel (self , start , ending , suggestion , desired_no_of_tokens , deactivate , \
  sendtomix , randomization , region , skip_length , linger , steplinger) :  

    log = []
    names = []
    ids = []
    emb_id = copy.copy(start)
    tokenCount = desired_no_of_tokens 
    emb_name = None
    found = False

    for i in range (desired_no_of_tokens):
      if tokenCount<=0: continue

      #Place token
      emb_name = self.get_name(emb_id)

      if deactivate < 1 and emb_id == start:
        emb_name = '[ ' + emb_name + ' : : ' + str(deactivate) + ' ]'

      if tokenCount <= 1 or found:
        emb_id = copy.copy(ending)
        emb_name = self.get_name(ending)
        if deactivate < 1:
          emb_name = '[ ' + emb_name + ' : : ' + str(deactivate) + ' ]'

      names.append(emb_name)
      ids.append(emb_id)
      log.append("Placing token '" + emb_name + "' at i #" +str(i))
          
      if tokenCount <= 1 or found: break
      else : tokenCount = tokenCount - 1
              
      #Calculate how many ID tokenCount we should skip ahead of 
      rando = (1 - randomization) + (randomization *random.random())
      step_ahead = math.floor(region*skip_length*rando)

      if (step_ahead <= 0) or random.random()<linger : 
        step_ahead = math.floor(steplinger*rando)
        if step_ahead <=0 : 
          step_ahead = 1
        
      current = emb_id + step_ahead
      log.append("moving ahead " + str(step_ahead) + " ID steps")

      #Perform modulo operation on step_ahead
      #(make the index loop back around if it exceeds the given range)
      emb_id = current
      if current >= ending : found = True

    return names, ids , log 
#######


  def Set (self, names , ids , index , name_string , ID_string , \
  comma_probability , sendtomix , suggestion , send_to_negatives):

    emb_id = copy.copy(ids[index])
    emb_name = copy.copy(names[index])
    output_name_string = copy.copy(name_string)
    output_ID_string = copy.copy(ID_string)

    output_name_string = output_name_string + emb_name + ' '
    if comma_probability>random.random(): output_name_string = name_string + ' , '
    output_ID_string = output_ID_string + str(emb_id) + ' '

    emb_vec = self.data.tools.internal_embs[emb_id].unsqueeze(0)

    for k in range(MAX_NUM_MIX):
      if (not sendtomix) or suggestion > 0: break
      if not self.data.vector.isEmpty.get(k) : continue
      self.data.place(k , 
        vector =   emb_vec.cpu(),
        ID =  emb_id ,
        name = emb_name , 
        to_mixer = sendtomix , 
        to_temporary = True , 
        to_negative = False)
      break
    ##########
    for k in range(MAX_NUM_MIX):
      if (not send_to_negatives) or suggestion > 0: break
      if not self.data.negative.isEmpty.get(k) : continue
      self.data.place(k , 
        vector =   emb_vec.cpu(),
        ID =  emb_id ,
        name = emb_name , 
        to_mixer = False , 
        to_temporary = True , 
        to_negative = send_to_negatives)
      break
    
    return output_name_string , output_ID_string



  def Extrapolate (self, *args) :

    #Get the inputs
    first = args[0]
    first_padding = args[1]
    last = args[2]
    skip_length = args[3]/100
    randomization = args[4]/100
    desired_no_of_tokens = args[5]
    sendtomix = args[6]
    id_mode = args[7]
    comma_probability = args[8]/100
    no_of_suggestions = args[9]
    show_ids = args[10]
    linger = args[11]/100
    steplinger = args[12]
    last_padding = args[13]
    deactivate = args[14]/100
    padding_skip_length = args[15]/100
    mix_input = args[16]
    stack_mode = args[17]
    send_to_negatives = args[18]
    neg_input = args[19]

    tokenizer = self.data.tools.tokenizer
    internal_embs = self.data.tools.internal_embs

    assert not (
          skip_length == None or
          randomization == None or
          desired_no_of_tokens == None or
          sendtomix == None or
          id_mode == None or
          tokenizer == None or
          internal_embs == None or
          comma_probability == None or
          no_of_suggestions == None
          ) , "NoneType in Tokenizer input!"

    #set the outputs
    log = []
    results = []
    tokenbox = ''
    
    negbox = neg_input
    tokenmixer_vectors = mix_input

    if neg_input == None : neg_input = ''
    if tokenmixer_vectors == None : tokenmixer_vectors = ''

    if not stack_mode:
      if sendtomix : tokenmixer_vectors= ''
      if send_to_negatives : negbox= ''

    tmp = None

    #set function params
    current = None

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
    #####

    #Empty input
    if (first == None) and (last == None) :
      log.append("No inputs")
      return tokenmixer_vectors , '' , '\n'.join(log) , negbox
    ######

    if desired_no_of_tokens <=0 :
      log.append("Zero number of tokens to generate")
      return tokenmixer_vectors , '' , '\n'.join(log) , negbox


    log.append("Starting Token Extrapolator.....")
    log.append('-------------------------------------------')

    # FETCH START TOKEN
    start = None
    #Find token ID starting point
    if (self.isText(first)): 
      if id_mode and first.isdigit(): start = int(first)
      else: start = self.get_id(first)
    else: start = self.get_random_id_below(self.no_of_internal_embs)
    assert start != None ,"start is NoneType!"

    #If the input is an embedding
    if not isinstance(start , int):
      emb_name, start , emb_vec =  self.get_embedding_data(first)
      best_ids = []
      sorted_scores = []
      vec = None
      found = False
      if emb_vec.shape[0]>1: vec = emb_vec[0]
      else : vec = emb_vec
      #Find most similar CLIP token to first embedding vector
      for accuracy in range(1000):
        similarity = 100 - accuracy/10
        best_ids , sorted_scores = self.data.tools.get_best_ids (None , similarity , 3 , vec)
        if len(best_ids)>0 : 
          found = True
          break
      #######

      if found :
        vec = vec.unsqueeze(0)
        start = best_ids[0]
        name = self.get_name(start)
        score = str(round((100*sorted_scores).numpy()[0] , 1))
        #score = round(score , 1)
        log.append("Embedding '"  + str(emb_name) + "' found ( " + \
        str(vec.shape[0]) + " x " + str(vec.shape[1]) + " size) : " \
        "Placing token most similar to first embedding vector as initial token : #" + str(start) + \
        " ( '" + name + "' ) with " + str(score) + " % similarity")
        emb_name = name
      else: 
        start = self.get_random_id_below(self.no_of_internal_embs)
        emb_name = self.get_name(start)
        log.append("Could not find similar token. " + \
        "Placing random roken at intial token : #" + str(start) + \
        " ( '" + emb_name + "' )")

    #Place random token
    elif start == 318 :
      start = self.get_random_id_below(self.no_of_internal_embs)
      emb_name = self.get_name(start)
      log.append("Detected ' _ ' symbol on initial token. Placing random ID at initial token : #" + str(start) + \
      " ( '" + emb_name + "' )")

    #Normal operation
    else : 
      emb_name = self.get_name(start)
    ##### FETCH OF START TOKEN COMPLETE


    log.append("extrapolation first ID : #" + str(start) + \
    " ( '" + emb_name + "' )")

  
    #FETCH ENDING TOKEN
    ending = None
    if (self.isText(last)): 
      if id_mode and last.isdigit(): ending = int(last)
      else: ending = self.get_id(last)
    else: ending = self.get_random_id_above(start)
    assert ending != None ,"ending is NoneType!"

    #If the input is an embedding
    if not isinstance(ending , int):
      emb_name, ending , emb_vec =  self.get_embedding_data(last)
      best_ids = []
      sorted_scores = []
      vec = None
      found = False
      if emb_vec.shape[0]>1: vec = emb_vec[0]
      else : vec = emb_vec
      #Find most similar CLIP token to first embedding vector
      for accuracy in range(1000):
        similarity = 100 - accuracy/10
        best_ids , sorted_scores = self.data.tools.get_best_ids (None , similarity , 3 , vec)
        if len(best_ids)>0 : 
          found = True
          break
      #######

      if found :
        vec = vec.unsqueeze(0)
        ending = best_ids[0]
        name = self.get_name(ending)
        score = str(round((100*sorted_scores).numpy()[0] , 1))
        #score = round(score , 1)

        log.append("Embedding '"  + str(emb_name) + "' found ( " + \
        str(vec.shape[0]) + " x " + str(vec.shape[1]) + " size) : " \
        "Placing token most similar to first embedding vector as ending token : #" + str(ending) + \
        " ( '" + name + "' ) with " + str(score) + " % similarity")
        emb_name = name
      else: 
        ending = self.get_random_id_below(self.no_of_internal_embs)
        emb_name = self.get_name(ending)
        log.append("Could not find similar token. " + \
        "Placing random roken at intial token : #" + str(ending) + \
        " ( '" + emb_name + "' )")

    #Place random token
    elif ending == 318 :
      ending = self.get_random_id_below(self.no_of_internal_embs)
      emb_name = self.get_name(ending)
      log.append("Detected ' _ ' symbol on ending token. Placing random ID at ending token : #" + str(ending) + \
      " ( '" + emb_name + "' )")

    #Normal operation
    else : 
      emb_name = self.get_name(ending)
    ##### FETCH OF ENDING TOKEN COMPLETE

    emb_name = self.get_name(ending)
    log.append("extrapolation ending ID : #" + str(ending) + \
    " ( '" + emb_name + "' )")

    #Set region of ID tokens to navigate during 
    #extrapolation. Switch places on start and 
    #ending if start>ending
    emb_id = None
    reverse = False
    region = 0
    if start>ending:
        reverse = True
        log.append("switching places on start and ending")
        emb_id = copy.copy(ending)
        ending = copy.copy(start)
        start = emb_id
    region = max(ending - start , 1)

    log.append("region to navigate :" + str(region) + \
      " tokens available")

    pre_start = self.get_random_id_below(start)

    if pre_start > start  - steplinger : 
      if start  - steplinger <= 0 :
        pre_start = 0
        start = steplinger
      else : pre_start = start  - steplinger

    assert not pre_start > start , "pre_start > start!"
    pre_region = max(start - pre_start , 1)

    post_ending = self.get_random_id_above(ending)

    if post_ending < ending  + steplinger : 
      post_ending = ending  + steplinger

    assert not post_ending < ending , "post_ending < ending !"
    post_region = max(post_ending - ending , 1)

    results.append('Extrapolation results : ')
    results.append('----------------------------------')

    #Start loop
    name_string  = None
    ID_string = None

    pre_names = None
    pre_ids = None

    names = None
    ids = None

    post_names = None
    post_ids = None
    found = False

    tmp = None

    for suggestion in range (no_of_suggestions):

        name_string = ''
        ID_string = ''

        pre_names , pre_ids , pre_messages = self.travel(pre_start , \
        start - math.floor(steplinger*random.random() + 1) , suggestion ,  \
        first_padding  , 1 , sendtomix , randomization , \
        pre_region , padding_skip_length , linger , steplinger)

        names , ids , messages  = self.travel(start , ending, suggestion ,  \
        desired_no_of_tokens , deactivate , sendtomix , randomization , \
        region , skip_length , linger , steplinger)

        post_names , post_ids , post_messages = \
        self.travel(ending + math.floor(steplinger*random.random() + 1) , \
        post_ending, suggestion , last_padding  , 1 , \
        sendtomix , randomization , post_region , \
        padding_skip_length , linger , steplinger)

        if reverse : 
          for i in range (len(post_names)):
            index = len(post_names) - i -1
            name_string , ID_string  = \
            self.Set(post_names , post_ids , \
            index , name_string , ID_string , \
            comma_probability , sendtomix , suggestion , \
            send_to_negatives)
            log.append(post_messages[index])
        else:
          for index in range (len(pre_names)):
            name_string , ID_string  = \
            self.Set(pre_names , pre_ids , \
            index , name_string , ID_string ,\
            comma_probability , sendtomix , suggestion , \
            send_to_negatives)
            log.append(pre_messages[index])

        for i in range (len(names)):
          if reverse: index = len(names) - i -1
          else : index = i
          name_string , ID_string  = \
          self.Set(names , ids , \
          index , name_string , ID_string ,\
          comma_probability , sendtomix , suggestion , \
          send_to_negatives)
          log.append(messages[index])

        if reverse : 
          for i in range (len(pre_names)):
            index = len(pre_names) - i -1
            name_string , ID_string  = \
            self.Set(pre_names , pre_ids , \
            index , name_string , ID_string ,\
            comma_probability , sendtomix , suggestion , \
            send_to_negatives)
            log.append(pre_messages[index])
        else:
          for index in range (len(post_names)):
            name_string , ID_string  = \
            self.Set(post_names , post_ids , \
            index , name_string , ID_string ,\
            comma_probability , sendtomix , suggestion , \
            send_to_negatives)
            log.append(post_messages[index])

        results.append(name_string)
        if show_ids : 
          results.append(' ')
          results.append('IDs : ' + ID_string)
          results.append(' ')
        results.append('----------------------------------')
    ######


    for i in range(MAX_NUM_MIX):
        if self.data.vector.isEmpty.get(i): continue
        name = self.data.vector.name.get(i)
        if name != None and sendtomix :
          if tokenmixer_vectors != '': tokenmixer_vectors = tokenmixer_vectors + ' , '
          tokenmixer_vectors = tokenmixer_vectors  + name
        #######
    for i in range(MAX_NUM_MIX):
        if self.data.negative.isEmpty.get(i): continue
        name = self.data.negative.name.get(i)
        if name != None and send_to_negatives :
          if negbox != '': negbox = negbox + ' , '
          negbox = negbox  + name


    return tokenmixer_vectors , '\n'.join(results) , '\n'.join(log) , negbox
  ## End of Extrapolate function
        
  def setupIO_with (self, module):
    #Tell the buttons in this class what to do when 
    #pressed by the user

    cond = self.data.tools.loaded
    input_list = []
    output_list = []

    if (module.ID == "Extrapolator") and cond:
      #input_list.append(self.inputs.first)
      #input_list.append(self.outputs.tokenbox)
      #output_list.append(self.inputs.first)
      #output_list.append(self.outputs.tokenbox)
      #self.buttons.reset.click(fn = self.Reset, inputs = input_list , outputs = output_list)
      return 1


    if (module.ID == "TokenMixer") and cond:
      input_list.append(self.inputs.first)                #0
      input_list.append(self.inputs.first_padding)        #1
      input_list.append(self.inputs.last)                 #2
      input_list.append(self.inputs.skip_length)          #3
      input_list.append(self.inputs.randomization)        #4
      input_list.append(self.inputs.desired_no_of_tokens) #5
      input_list.append(self.inputs.sendtomix)            #6
      input_list.append(self.inputs.id_mode)              #7
      input_list.append(self.inputs.comma)                #8
      input_list.append(self.inputs.suggestions)          #9
      input_list.append(self.inputs.show_id)              #10
      input_list.append(self.inputs.linger)               #11
      input_list.append(self.inputs.steplinger)           #12
      input_list.append(self.inputs.last_padding)         #13
      input_list.append(self.inputs.deactivate)           #14
      input_list.append(self.inputs.padding_skip_length)  #15
      input_list.append(module.inputs.mix_input)          #16
      input_list.append(self.inputs.stack_mode)           #17
      input_list.append(self.inputs.negatives)            #18
      input_list.append(module.inputs.negbox)             #19

      ######
      output_list.append(module.inputs.mix_input)         #0
      output_list.append(self.outputs.tokenbox)           #1
      output_list.append(self.outputs.log)                #2
      output_list.append(module.inputs.negbox)            #3
      ######
      self.buttons.extrapolate.click(fn=self.Extrapolate , inputs = input_list , outputs = output_list)

  #End of setupIO_with()


  def __init__(self , label , vis = False):

    #Pass reference to global object "dataStorage" to class
    self.data = dataStorage 
    self.emb_id = None
    self.emb_name = None
    self.emb_vec = None


    self.no_of_internal_embs = self.data.tools.no_of_internal_embs

    class Outputs :
      def __init__(self):
        Outputs.tokenbox = []
        Outputs.log = []

    class Inputs :
      def __init__(self):
        Inputs.first = []
        Inputs.last = []
        Inputs.sendtomix = []
        Inputs.id_mode = []
        Inputs.skip_length = []
        Inputs.randomization = []
        Inputs.desired_no_of_tokens = []
        Inputs.comma = []
        Inputs.suggestions = []
        Inputs.show_id = []
        Inputs.linger = []
        Inputs.steplinger = []
        Inputs.deactivate = []
        Inputs.first_padding = []
        Inputs.last_padding = []
        Inputs.padding_skip_length = []
        Inputs.stack_mode = []
        Inputs.negatives = []

    class Buttons :
      def __init__(self):
        Buttons.extrapolate = []
        Buttons.reset = []

    self.outputs= Outputs()
    self.inputs = Inputs()
    self.buttons= Buttons()
    self.ID = "Extrapolator"

    with gr.Accordion(label , open=True , visible = vis) as show :
      gr.Markdown("Extrapolate tokens using the ID list in CLIP")
      with gr.Row() :
        self.inputs.skip_length = gr.Slider(label="Pursuit strength to end token %",value=20, \
                                minimum=0, maximum=100, step=0.1 , interactive = True)
      with gr.Row() :
        self.inputs.randomization = gr.Slider(label="Step length randomization %",value=20, \
                                      minimum=0, maximum=100, step=1 , interactive = True)                                                       
      with gr.Row() :
        self.inputs.desired_no_of_tokens = gr.Slider(label="Number of tokens",value=5, \
                                      minimum=0, maximum=100, step=1 , interactive = True)
      with gr.Row() :  
          self.buttons.extrapolate = gr.Button(value="Extrapolate", variant="primary")
          self.buttons.reset = gr.Button(value="Reset", variant = "secondary")
      with gr.Row() : 
        self.inputs.first = gr.Textbox(label="Initial token", lines=1, \
        placeholder="Enter a single token" , interactive = True)
        self.inputs.last = gr.Textbox(label="End token", lines=1, \
        placeholder="Enter a single token" , interactive = True)
      with gr.Row() : 
        self.outputs.tokenbox = gr.Textbox(label="Extrapolation results", lines=10, interactive = False)
      with gr.Row():
          self.inputs.sendtomix = gr.Checkbox(value=False, label="Send to input", interactive = True)
          self.inputs.id_mode = gr.Checkbox(value=False, label="ID input mode", interactive = True)
          self.inputs.show_id = gr.Checkbox(value=False, label="Include IDs in output", interactive = True)
          self.inputs.stack_mode = gr.Checkbox(value=False, label="Stack Mode", interactive = True)
          self.inputs.negatives = gr.Checkbox(value=False, label="Send to negatives", interactive = True , visible = True) 

      with gr.Accordion("Advanced options" , open = False):
        with gr.Row() : 
          self.inputs.suggestions = gr.Slider(label="Number of suggestions",value=5, \
                                      minimum=1, maximum=100, step=1 , interactive = True) 
        with gr.Accordion("Add comma in prompt" , open = False):                          
          self.inputs.comma = gr.Slider(label="Comma placement probability %",value=0, \
                                      minimum=0, maximum=100, step=0.1 , interactive = True)

          with gr.Accordion('Tutorial : What is this?',open=False , visible= False) as tutorial_0 :
            gr.Markdown("Commas are interpreted as any other token and will " + \
            "normally have minimal impact on your prompt. \n \n " +  \
            "However, with the  CutOff extension installed , " + \
            "(https://github.com/hnmr293/sd-webui-cutoff)" + \
            " the comma placement will affect the output. \n \n " + \
            "Using the 'Comma placement probability %' slider, you can randomly " + \
            "add commas after the extrapolated tokens.") 

        with gr.Accordion("ID Slowdown Mode settings" , open = False):
          with gr.Row() :
            self.inputs.linger = gr.Slider(label="Slowdown Mode probability %",value=0, \
                                      minimum=0, maximum=100, step=1 , interactive = True)
          with gr.Row() :    
            self.inputs.steplinger = gr.Slider(label="Max ID step during slowdown",value=5, \
                                      minimum=0, maximum=1000, step=1 , interactive = True)
          with gr.Accordion('Tutorial : What is this?',open=False , visible= False) as tutorial_1 :
            gr.Markdown("During Slowdown Mode the step length in the ID list will " + \
            "slow down to a few incremental steps at a time. \n \n " +  \
            "By taking slow steps as you advance through the ID list at random points " + \
            "you can make the extrapolated prompt more coherent. \n \n " + \
            "You can set the probability for Slowdown Mode to be enabled " + \
            "using the 'Slowdown Mode probability %' slider . \n \n" + \
            "You can set the maximum incremental step length using " + \
            "the 'Max ID step during slowdown' slider. The incremental step length " + \
            "during Slowdown Mode will be a random integer below the " + \
            "'Max ID step during slowdown' slider value . ") 

        with gr.Accordion("Token padding settings" , open = False):    
          self.inputs.first_padding = gr.Slider(label="No of padding tokens (random point to first token)",value=0, \
                                      minimum=0, maximum=100, step=1 , interactive = True)     
          with gr.Row() :    
            self.inputs.last_padding = gr.Slider(label="No of padding tokens (last token to random point)",value=0, \
                                      minimum=0, maximum=100, step=1 , interactive = True)
          with gr.Row() :    
            self.inputs.padding_skip_length = gr.Slider(label="Pursuit strength of padding tokens %",value=20, \
                                      minimum=0, maximum=100, step=0.1 , interactive = True)
          with gr.Accordion('Tutorial : What is this?',open=False , visible= False) as tutorial_2 :
            gr.Markdown("You can pad the input tokens with tokens . \n \n " +  \
            "generated from a random point in the ID list " + \
            "This can either be done from a random ID below the initial token index, \n \n " + \
            "or from the last token ID to a random ID above the last token ID. \n \n " + \
            "You can set the number of random tokens to add on either endpoint using the " + \
            "two 'No of padding tokens' sliders above. \n \n The paddings have their separate " + \
            "pursuit strength index (how fast the extrapolator will travel down the ID list) " + \
            "which can be set with the  'Pursuit strength of padding tokens %'. \n \n" + \
            "Note that the pursuit strength of the padding is still affected by " + \
            "randomization slider and the Slowdown mode sliders. ")  

        with gr.Accordion("Token deactivation settings" , open = False):
          self.inputs.deactivate = gr.Slider(label="Deactivate threshold",value=100, \
                                      minimum=0, maximum=100, step=1 , interactive = True)

          with gr.Accordion('Tutorial : What is this?',open=False , visible= False) as tutorial_3 : 
            gr.Markdown(" You can deactivate the input tokens in the extrapolated" +  \
            " prompt after a given number of steps. " + \
            "This will given the extrapolated tokens more influence on the output  . \n \n ")                                
      
      with gr.Accordion("Output Log" , open = False) as logs :
        self.outputs.log = gr.Textbox(label="Output log", lines=5, interactive = False)  
      with gr.Accordion('Tutorial : What is this?',open=False , visible= False) as tutorial_4 :
          gr.Markdown("The ID Extrapolation tool uses the properties of the token ID list " + \
          "in CLIP to generate a prompt from a pair of input tokens . \n \n " +  \
          "Leaving the 'Initial token' or 'End token' fields empty will generate " + \
          "a random token from the ID list in its place. \n \n " + \
          "By enabling 'ID Input mode' you can set the Initial token or End token by " + \
          "their ID (an integer from 0 to " + str(self.no_of_internal_embs) + ")" + \
          " instead of their name. \n \n " + \
          "'Pursuit strength to end token %' sets the speed which the extrapolator " + \
          "travels down the ID list. Note that high pursuit strength can " + \
          "result in fewer extrapolated tokens from the desired amount. \n \n " + \
          "'Step length randomization %' sets a percentage of the step length which is randomized. \n \n " + \
          " By selecting 'Include IDs in output' you can see the IDs " + \
          " of the generated tokens . \n \n  " + \
          " You can pass the first extrapolated prompt suggestion to the TokenMixer " + \
          " by selecting the 'Send first suggestion to TokenMixer' box. \n \n  " + \
          " This tool also features a list of 'Advanced options'. These can " + \
          " Be found at the bottom of this module, along with further details on " + \
          " how to use them.")

          with gr.Accordion('How does this work?',open=False): 
            gr.Markdown("Tokens in the ID list are ordered so that " + \
            " the tokens which most commonly appear next to each other " + \
            "are grouped together (i.e have similiar IDs). \n \n" + \
            "Note that similar token ID does not mean the tokens themselves are similar " + \
            "(meaning they will NOT generate the same thing when placed in the prompt). Rather " + \
            "a similar ID means that these tokens frequently appeared next to each other in the training data. \n \n " + \
            "If you wish to generate similar tokens, use the 'Similar Mode' feature " + \
            "in the TokenMixer . \n \n " + \
            "Note that the order of ID:s and how they usually appear in a squence is a generalization. \n \n " + \
            "IDs that are a 500 or more steps away " + \
            "from a given ID can appear just as frequently in a sequence in the CLIP training data as the " + \
            " ID located a single step away from the input token. \n \n But you can confidently say that using a token that is " + \
            "a 500 steps away or less will give you more consistent results then using a token " + \
            " that is 10 000 steps away or more in a given prompt sequence. " + \
            "This property in the ID list is the general idea behind this extrapolator. \n \n " + \
            "In addition :  Tokens which frequently appeared in the CLIP training data " + \
            "have low number IDs and vice versa for uncommon tokens.  \n \n " + \
            "Using the initial token ID as a starting point you " + \
            "can step down the ID list until you reach the End token ID. \n \n" + \
            "If initial token ID > end token ID then they will switch places with each other. \n \n" + \
            "By taking short steps in the ID you will generate ''typical'' token sequences" + \
            ", but you might not reach the End token ID within the desired number of tokens. \n \n " + \
            "By taking longer steps, you are more likely to reach the End token ID " + \
            "but the token sequence will be less ''normal'' then when taking shorter steps. \n \n " + \
            "Note that for large steps the End token ID might be reached pre-maturely " + \
            "before all the desired tokens will be placed. \n \n TLDR: High step count can give " + \
            " less tokens then desired. ")
    #end of UI

    self.tutorials = []
    self.tutorials.append(tutorial_0)
    self.tutorials.append(tutorial_1)
    self.tutorials.append(tutorial_2)
    self.tutorials.append(tutorial_3)
    self.tutorials.append(tutorial_4)

    self.show = [show]
    self.logs = [logs]

    self.buttons.extrapolate.style(size="sm")
    self.buttons.reset.style(size="sm")

    if self.data.tools.loaded : self.setupIO_with(self)
## End of class Extrapolator----------------------