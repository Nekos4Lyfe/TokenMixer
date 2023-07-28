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

class TokenMixer :

  def Reset (self):

    for index in range(MAX_NUM_MIX):
      self.data.clear(
        index, 
        to_negative = True ,
        to_mixer = True ,
        to_temporary = True)

    return '' , '' , '' , ''

  def Run(self, save_name) :
    log = []
    save_filename = None
    anything_saved =  False

    tmp = 1
    if (self.local.order_randomize_mode):
      log.append(str(tmp) + '. ' + 'Randomize token order selected: Order of the input tokens ' + \
      'will be randomized prior to embedding generation')
      tmp+=1
      log.append('-------------------------------------------')

    if(self.local.similar_mode):
      log.append(str(tmp) + '. ' + 'Similar mode selected: Input tokens will be replaced ' + \
      'by similar tokens prior to embedding generation')
      tmp+=1
      log.append('-------------------------------------------')

    if(self.local.interpolate_mode):
      log.append(str(tmp) + '. ' + 'Interpolate mode selected: Input token pairs with similarity will ' + \
      'be merged into single token prior to embedding generation')
      tmp+=1
      log.append('-------------------------------------------')

    elif(self.local.merge_mode):
      log.append(str(tmp) + '. ' + 'Merge mode: ' + \
      'Will interpolate input tokens into single token embedding with greatest possible similarity to each negative_weight input token')
      log.append('-------------------------------------------')
    else: 
      log.append(str(tmp) + '. ' + 'Concat mode (default): Concatinate input tokens into multi-token embedding... ')
      log.append('-------------------------------------------')
    
#BEGIN PROCESSING THE VECTORS
    if (self.local.order_randomize_mode): self.data.vector.shuffle()
    tot_vec = None
    if(self.local.interpolate_mode):
      tot_vec , message = self.data.merge_if_similar(self.local.similar_mode)
    elif (self.local.merge_mode): 
      tot_vec , message = self.data.merge_all(self.local.similar_mode)
    else : 
      tot_vec , message = self.data.concat_all(self.local.similar_mode)
    log.append(message)
#END PROCESSING THE VECTORS

      # save the mixed embedding
    if (tot_vec == None):
        log.append('No tokens in the TokenMixer.')
        log.append('You can add tokens into the TokenMixer from the modules on the left side, ' + \
        'or write them into the TokenMixer inputs directly. \n \n Note that when writing a text or embedding into the TokenMixer ' +\
        'directly only the first token will be processed. \n \n To process embeddings or prompts greater then one token, ' +\
        'use the modules on the left')
        return '\n'.join(log), None
    else:
            if False: # Remove zeroed vectors
                old_count = tot_vec.shape[0]
                tot_vec = tot_vec[torch.count_nonzero(tot_vec,dim=1)>0]
                new_count = tot_vec.shape[0]
                if (old_count!=new_count): log.append('Removed '+str(old_count-new_count)+' zeroed vectors, remaining vectors: '+str(new_count))

            if tot_vec.shape[0] > 0:
                log.append('Final embedding size: '+str(tot_vec.shape[0])+' x '+str(tot_vec.shape[1]))
                if tot_vec.shape[0]>75: 
                  log.append('⚠️WARNING: vector count>75, it may not work 🛑')

            save_filename = os.path.join(self.data.tools.emb_savepath , f"{save_name}.pt")

            if (os.path.exists(save_filename)):
                if not(self.local.enable_overwrite):
                  return('File already exists ('+save_filename+') overwrite not enabled, aborting.', None)
                else:  log.append('File already exists, overwrite is enabled')
            
            new_emb = Embedding(tot_vec, save_name) #Create the embedding

            try:
                new_emb.save(save_filename)
                log.append('Saved "'+save_filename+'"')
                anything_saved = True

            except: log.append('🛑 Error saving "'+save_filename+'" (filename might be invalid)')

    #Update the embedding database and the data class if we have saved an embedding
    if anything_saved==True:
        log.append('Reloading all embeddings')
        try: sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
        except: 
            sd_hijack.model_hijack.embedding_db.dir_mtime=0
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
        self.data.update_loaded_embs() 
    
    return '\n'.join(log) , save_filename


  def Save (self, *args):

    #Fetch parameters from the args input
    #assert not (None in args[0:16]), "Warning: Null input among settings in Save()"
    save_name = args[0]
    enable_overwrite = args[1]
    merge_mode = args[2]
    interpolate_mode = args[3]
    similar_mode = args[4]
    five_sets_mode = args[5]
    angularSlider = args[6]
    radialSlider = args[7]
    randSlider = args[8]
    interpolateSlider = args[9]
    iterationSlider = args[10]
    gainSlider = args[11]
    override_box = args[12]
    sub_name = args[13]
    allow_negative_gain = args[14]
    negative_strength = args[15]
    autoselect = args[16]
    positive_strength = args[17]
    mix_input = args[18]
    order_randomize_mode = args[19]
    neg_input = args[20]

    assert not self.data == None , "Warning: data class is null"

    #Store mixer multipliers
    for i in range(MAX_NUM_MIX):
      self.data.vector.weight.place(1  , i) # will fix later
    
    #Store relevant variables in our own class 
    #so they can be fetched by the Run() function
    self.local.enable_overwrite = copy.copy(enable_overwrite)
    self.local.merge_mode = copy.copy(merge_mode)
    self.local.interpolate_mode = copy.copy(interpolate_mode)
    self.local.similar_mode = copy.copy(similar_mode)
    self.local.five_sets_mode = copy.copy(five_sets_mode)
    self.local.order_randomize_mode = copy.copy(order_randomize_mode)

    #Set the strength of the token negatives from input
    self.data.negative.strength = copy.copy(negative_strength)

    log = [] 
    emptyList = [None]*MAX_NUM_MIX 

    #Helper function
    def user_wrote_something_in(string):
      cond1 = (isinstance(string , str))
      cond2 = (string != None) and (string != '')
      return (cond1 and cond2)

    #Check if user has written a filename for the embedding 
    if user_wrote_something_in(save_name):
      log.append('Starting TokenMixer...')
      log.append(' ')
      log.append('Order or operations to be performed : ')
      log.append('-------------------------------------------')
    else :
      log.append('Please enter a name for the embedding')
      return '\n'.join(log), None , None

    #Check if five-sets-mode is enabled and if the user has
    #assigned a sub_name for the embedding
    autosub = None
    if autoselect : autosub = self.data.tools.get_subname()

    if self.local.five_sets_mode : 
      if user_wrote_something_in(sub_name):
        log.append('Five sets mode : will repeat process 5 times ' + \
        'and create a new embedding for each using sub embedding name : '+ str(sub_name)+ \
        'N , where N is an integer from 1 to 5')
      elif autoselect :
        log.append('Five sets mode : will repeat process 5 times ' + \
        'and create a new embedding for each using automatic sub-embedding name : '+ str(autosub)+ \
        'N , where N is an integer from 1 to 5')
      else: 
        log.append('Five sets mode : will repeat process 5 times ' + \
        'and create a new embedding for each using default name system : '+ str(save_name) + '_(N)' + \
        ' , where N is an integer from 1 to 5')
      log.append('-------------------------------------------')

    #Update the data class with given slider values
    self.data.vector.costheta = angularSlider
    self.data.vector.radius = radialSlider
    self.data.vector.randomization = randSlider
    self.data.vector.interpolation = interpolateSlider
    self.data.vector.itermax = iterationSlider
    self.data.vector.gain = gainSlider
    self.data.vector.allow_negative_gain = allow_negative_gain

    #Check if user has pasted something into the override box
    if (user_wrote_something_in(override_box)):
      try:
        costh , rad ,rand , interp , iterm , gain  = \
        [int(s) for s in re.findall(r'\b\d+\b', override_box)]
        log.append('Setting override params : '+ str(costh)+','+ \
        str(rad)+','+str(rand)+','+str(interp)+','+str(iterm))
        if costh != None : self.data.vector.costheta = costh
        if rad != None : self.data.vector.radius = rad
        if rand != None : self.data.vector.randomization = rand
        if interp != None : self.data.vector.interpolation = interp
        if iterm != None : self.data.vector.itermax = iterm
        if gain != None : self.data.vector.gain = gain
      except: log.append('Warning: Could not read override string. Using slider values instead')

    iterations = None
    if (five_sets_mode): iterations = 5
    else : iterations = 1

    embox_output = '{'

    for i in range (iterations+1):
      if i == 0: continue

      #change filename if five_sets_mode is enabled
      if (five_sets_mode): 
        if (sub_name == None) or (sub_name == '') or (i==1):
          if autoselect and i>1:
            save_name = autosub + str(i)
          else: save_name = save_name + '.'
        else: save_name = sub_name + str(i)
        embox_output = embox_output + save_name
        
        if (i<iterations): embox_output = embox_output + '|'
        else: embox_output = embox_output + '}'

      #Run the TokenMixer
      message , save_filename = self.Run(save_name) 
      log.append(message)
      if save_filename == None : 
        return '\n'.join(log), None , None

    if (five_sets_mode) : save_name = embox_output

    #Load names from the data class and send them to the mixer inputs
    mix_name_output = ''
    for index in range (MAX_NUM_MIX):
      name = self.data.vector.name.get(index)
      if name != None :
        if mix_name_output != '' : mix_name_output = mix_name_output + ' , '
        mix_name_output = mix_name_output + self.data.vector.name.get(index)
    if (mix_name_output == ''): return 'No embeddings stored. Nothing to save' , None , mix_name_output

    return '\n'.join(log), save_name , mix_name_output
#End of the TokenMixer Save() Function


  def setupIO_with (self, module):
    #Tell the buttons in this class what to do when 
    #pressed by the user

    input_list = []
    output_list = []

    if (module.ID == "MiniTokenizer") :
      module.setupIO_with(self)

    if (module.ID == "TokenMixer") :
      input_list.append(self.inputs.save_name)                      #0
      input_list.append(self.inputs.settings.enable_overwrite)      #1
      input_list.append(self.inputs.settings.merge_mode)            #2
      input_list.append(self.inputs.settings.interpolate_mode)      #3
      input_list.append(self.inputs.settings.similar_mode)          #4
      input_list.append(self.inputs.settings.five_sets_mode)        #5
      input_list.append(self.inputs.sliders.angle)                  #6
      input_list.append(self.inputs.sliders.length)                 #7
      input_list.append(self.inputs.sliders.randomize)              #8
      input_list.append(self.inputs.sliders.interpolate)            #9
      input_list.append(self.inputs.sliders.iterations)             #10
      input_list.append(self.inputs.sliders.gain)                   #11
      input_list.append(self.inputs.override_box)                   #12
      input_list.append(self.inputs.sub_name)                       #13
      input_list.append(self.inputs.settings.allow_negative_gain)   #14
      input_list.append(self.inputs.sliders.negative_weight)        #15
      input_list.append(self.inputs.settings.autoselect)            #16
      input_list.append(self.inputs.sliders.positive_weight)        #17
      input_list.append(self.inputs.mix_input)                      #18
      input_list.append(self.inputs.settings.order_randomize_mode)  #19
      input_list.append(self.inputs.negbox)                         #20
      ########

      output_list.append(self.outputs.log)            #1
      output_list.append(self.outputs.embedding_box)  #2
      output_list.append(self.inputs.mix_input)       #3
      
      self.buttons.save.click(fn=self.Save , inputs = input_list , outputs = output_list)
      

      reset_output_list = []
      self.buttons.reset.click(
        fn=self.Reset , 
        inputs = None , 
        outputs = \
        [
          self.outputs.log , 
          self.outputs.embedding_box , 
          self.inputs.mix_input ,
          self.inputs.negbox])

    
    if (module.ID == "EmbeddingInspector") :
       module.setupIO_with(self)    

    if (module.ID == "EmbeddingAnalyzer") :
       module.setupIO_with(self)


  def __init__(self , label , vis = False , op = False):

          #Pass reference to global object "dataStorage" to class
          self.data = dataStorage 

          class Settings :
              def __init__(self):
                Settings.merge_mode = []
                Settings.similar_mode = []
                Settings.interpolate_mode = []
                Settings.enable_overwrite = []
                Settings.five_sets_mode = []
                Settings.section_extrapolation = []
                Settings.random_walk_extrapolation = []
                Settings.expansion_extrapolation = []
                Settings.angle_randomize_mode = []
                Settings.preset_names = []
                Settings.presets_dropdown = []
                Settings.allow_negative_gain = []
                Settings.autoselect = []
                Settings.order_randomize_mode = []

          class Local :
            #Class to store local variables
            def __init__(self):
              Local.interpolate_mode = False
              Local.similar_mode = False
              Local.merge_mode = False
              Local.enable_overwrite = False
              Local.five_sets_mode = False
              Local.weight_randomize_mode = False
              Local.save_name = ''
              Local.sub_name = ''
              Local.order_randomize_mode = False


          class Sliders :
              def __init__(self):
                Sliders.angle = []
                Sliders.length = []
                Sliders.randomize = []
                Sliders.interpolate = []
                Sliders.iterations = []
                Sliders.gain = []
                Sliders.negative_weight = []
                Sliders.positive_weight = []

          class Inputs :
              def __init__(self):
                Inputs.settings = Settings()
                Inputs.sliders = Sliders()
                Inputs.save_name = []
                #Inputs.eval_box = [0]
                Inputs.save_vector_name = []
                Inputs.mix_input =  []
                Inputs.mix_sliders =[]
                Inputs.override_box = []
                Inputs.sub_name = []
                Inputs.negbox = []
                Inputs.posbox = []

          class Outputs :
              def __init__(self):
                #Outputs.preset_names = [0]
                #Outputs.presets_dropdown = [0]
                Outputs.embedding_box = []
                Outputs.log = []
                Outputs.tolist = []

          class Buttons :
              def __init__(self):
                Buttons.reset = []
                Buttons.save_vector = []
                Buttons.save  = []

          self.inputs = Inputs()
          self.outputs = Outputs()
          self.buttons = Buttons()
          self.local = Local()
          self.ID = "TokenMixer"

          with gr.Accordion(label ,open=op , visible = vis) as show:
                        gr.Markdown("Create embeddings from module output tokens")
                        with gr.Row():  
                          self.inputs.mix_input = gr.Textbox(label='', lines=3, interactive = False)                                 
                        with gr.Row(): 
                          self.inputs.negbox = gr.Textbox(label= 'Negatives' , lines=3 , interactive = False)                                   
                        with gr.Row():                        
                          with gr.Column():
                                self.buttons.save = gr.Button(value="Create embedding", variant="primary")
                                self.inputs.save_name = gr.Textbox(label="New embedding name",lines=1,placeholder='Enter file name to save', interactive = True)
                                self.inputs.settings.enable_overwrite = gr.Checkbox(value=False,label="Enable overwrite", interactive = True)
        
                                with gr.Accordion('TokenMixer modes of operation',open=True):
                                  self.inputs.settings.order_randomize_mode = gr.Checkbox(value=False,label="Randomize token order ", interactive = True)
                                  self.inputs.settings.similar_mode = gr.Checkbox(value=False,label="Similar Mode  : Replace input tokens with similar tokens", interactive = True)
                                  self.inputs.settings.merge_mode = gr.Checkbox(value=False,label="Merge Mode : Find single token with greatest similarity to all input tokens ", interactive = True)
                                  self.inputs.settings.interpolate_mode = gr.Checkbox(value=False,label="Interpolate Mode  : Merge tokens with similarity (Work in progress)", interactive = True)       
                                  with gr.Accordion('Tutorial : What is this?',open=False , visible= False) as tutorial_1 :
                                      gr.Markdown("The TokenMixer has four modes of operation: Concat Mode (default) , " + \
                                      "Merge Mode , Similar Mode and Interpolate Mode. \n \n " + \
                                      "These modes can be used individually , or in combination with each other. \n \n " + \
                                      "You can open up the 'Output Log' → and run the TokenMixer while playing around with the above settings " +\
                                      "to get a feeling for how they work. \n \n  " + \
                                      "When NONE of the check-boxes are enabled , the TokenMixer defaults to 'Concat mode' . \n \n " + \
                                      "In 'Concat Mode' , all the N vectors in the TokenMixer input are placed into a N x " + str(self.data.vector.size) + " " + \
                                      "embedding (A single token has the dimensions 1 x " + str(self.data.vector.size)  + " for the model you are currently using) \n \n " + \
                                      "When 'Similar Mode' is enabled , all the input vectors in the TokenMixer will " + \
                                      "first be replaced by numerically similar tokens (vectors) prior to being placed in an embedding. \n \n " + \
                                      "You can set the similarity percentage % using the sliders found under the 'Similar Mode' settings → . \n \n" + \
                                      "A value of 100% means full similarity (parallell vectors) , and a value of 0 % means no similarity whatsoever " + \
                                      "(perpendicular vectors) . \n \n When 'Merge Mode' is selected , all the input tokens will be merged into " + \
                                      "a single token. This token will have the highest combined similarity to all the input tokens among a number of semi-random samples. \n \n" + \
                                      "You can set the randomness of the sampler and the number of samples performed using the sliders found under 'General Settings'  →  . \n \n" + \
                                      "The sliders found under 'General Settings' will affect the output of 'Similar Mode' , 'Merge Mode' and 'Interpolate Mode' " + \
                                      ", hence the name . \n \n When 'Interpolate Mode' is selected , tokens with a certain level of similarity among the token " + \
                                      "inputs will be merged into a single token. \n \n The level of similarity required to merge the tokens during 'Interpolate Mode' " + \
                                      "can be set by the 'Req. similarity to merge %' slider found under 'Interpolate Mode Settings'  → . \n \n " + \
                                      " Note that during 'Interpolate Mode' , only similar tokens that are 'adjecent' by to each other will be merged. \n \n" + \
                                      " 'Adjecent' means in this case that their TokenMixer input index number are adjecent , so it will be possible " + \
                                      " for TokenMixer input #3 and #4 to merge into a single token, but not input #6 and #8 (assuming an input exists on #7 , if " + \
                                      " input #7 is empty then #6 and #8 can merge). \n \n Order of operations for the selected Modes are as follows : \n " + \
                                      " 1) Similar Mode (if selected) \n 2) Interpolate Mode (if selected) \n 3) Merge Mode (if selected, otherwise Concat Mode will be performed)")

                                with gr.Accordion('Save Settings',open=False):
                                    self.inputs.settings.five_sets_mode = gr.Checkbox(value=False,label="Make 5 embeddings at once", interactive = True)   
                                    self.inputs.settings.autoselect = gr.Checkbox(value=False,label="Autoselect sub_name", interactive = True)                            
                                    self.inputs.sub_name = gr.Textbox(label="Embedding sub-name",lines=1,placeholder='enter something short like "x" ')                                                      
                                    with gr.Accordion('Tutorial : What is this?',open=False , visible= False) as tutorial_0 :
                                      gr.Markdown("The output when 'Make 5 embeddings at once' is selected will, for embedding name 'example' , be in the format " + \
                                        "{example|example2|example3|example4|example5}. \n \n " + \
                                        "With dynamic prompt extension installed, you can paste the {example|example2|example3|example4|example5} string " + \
                                        "output into the regular prompt box. \n \n Then one of the embeddings variants will be selected at random among the 5. \n \n" + \
                                        "Embeddings generated by the TokenMixer will" + \
                                        "be saved in a separate folder under embeddings/TokenMixer/* , meaning you can generate hundreds of embeddings " + \
                                        "with the TokenMixer extension without having to worry about sorting them out from your important embeddings " + \
                                        "stored in the embeddings/* folder. \n \n Enabling 'Autoselect sub_name' mode will automaticlly assign " + \
                                        "a sub-name to the embedding without requiring any input in the sub-name textbox. \n \n")
                                    
                                      with gr.Accordion("What is a 'Sub-name'?",open=False):
                                        gr.Markdown(
                                        "This sequence can be shortened by selecting a 'sub-name' , like 'x' . The output will when the sub-name 'x' is selected be " + \
                                        "{example|x2|x3|x4|x5}' . \n \n The sub-embeddings are saved as normal embeddings, hence " + \
                                        "you cannot use the same sub-embedding name for two different embedding types , or they will " +\
                                        "get overwritten after each batch.")

                                      with gr.Accordion("What is a 'Autoselect'?",open=False):
                                        gr.Markdown("Autoselect will automatically assign a subname without requiring you to type " + \
                                        "anything into the textbox between iterations. \n \n " + \
                                        "The list is the alphabet a-z. Once the autoselect output reaches 'z' , the sub-name will revert " + \
                                        "back to 'a' and overwrite the embedding previously saved under the sub-name 'a'. \n \n" + \
                                        "NOTE! : The autoselector will overwrite the sub-name stored under 'a' even when the 'Enable Overwrite' " + 
                                        " checkbox is de-selected! If you find a cool embedding that is stored in the Embeddings/TokenMixer/* folder " + \
                                        "which is saved under an autoselector name ,  RENAME it or assume that the autoselector will at some point overwrite it. \n \n " + \
                                        "The autoselector is a counter that outputs the strings 'a'-'z' in sequence as it is called. It does NOT check " + \
                                        "if an embedding is already saved under the output name. \n \n Take care not to save anything important " + \
                                        " in the Embeddings/TokenMixer/* folder under an autoselector name. ")

                                with gr.Accordion('Experimental',open=False , visible = False): #Experimental                                             
                                  with gr.Accordion('Negatives',open=False): 
                                    self.inputs.sliders.positive_weight = gr.Slider(minimum=0, maximum=100, step=0.1, label="Positive weight %", default=0 , interactive = True)
                                    self.inputs.posbox = gr.Textbox(label="Encourage similarity with the following tokens", lines=3 , interactive = False)
                                with gr.Accordion('Override',open=False , visible = False): #Experimental 
                                  self.inputs.override_box = gr.Textbox(label="Similarity settings override",lines=1,placeholder='(costheta|length|rand|interp|iters|gain)')
                                  with gr.Accordion('Tutorial : What is this?',open=False , visible= False) as tutorial_2 : 
                                    gr.Markdown("Memorizing the position of all the sliders can be difficult. \n \n " + \
                                    "In this field you can paste a string to re-create the " +  \
                                    "settings from a previous session. ")  
                                      
                          with gr.Column(): 
                              self.buttons.reset = gr.Button(value="Clear")
                              self.outputs.embedding_box = gr.Textbox(label="------> Output",lines=1,placeholder='Embedding name output')
                              gr.Markdown("### TokenMixer Operation Settings")
                              with gr.Accordion('General Settings',open=True):
                                self.inputs.sliders.negative_weight = gr.Slider(value = 50 , minimum=0, maximum=100, step=0.1, \
                                label="Negative token strength % ", default=50 , interactive = True) 
                                self.inputs.sliders.randomize = gr.Slider(value = 50 , minimum=0, maximum=100, step=0.1, \
                                label="Sample randomization %", default=50 , interactive = True)

                                self.inputs.sliders.gain = gr.Slider(minimum=0, maximum=20, step=0.1, label="Vector gain multiplier", default=1 , interactive = True)
                                self.inputs.sliders.iterations = gr.Slider(minimum=0, maximum=6000, step=1, label="Similarity samples max", default=1000 , interactive = True)
                               
                                
                                with gr.Accordion('Tutorial : What is this?',open=False , visible= False) as tutorial_3 : 
                                  gr.Markdown("These sliders set general values for the 'Similar Mode' , 'Merge Mode' and " + \
                                  "'Interpolate Mode' operations on the TokenMixer. These operations can be enabled " + \
                                  "under the ← 'TokenMixer modes of operation' tab, whare further information about their use " + \
                                  "is provided.  \n \n The 'Vector gain multiplier' slider multiplies the output vector length " + \
                                  "by the slider value. \n \n The 'Similarity samples max' slider sets the maximum number of " + \
                                  "samples the TokenMixer is allowed to create when finding similar tokens. " + \
                                  "The 'Randomization %' slider sets the randomness of the samples , where 100% is completely random samples and " + \
                                  "0% will return the input vector on every sample. \n \n For a randomization rate of 'r' for a sample 'X' , the samples  " + \
                                  "will be computed as 'gain * |X| * norm(X*(1 - r) + r*R)' , where R is a random vector with values ranging from 1 to -1 , " + \
                                  "gain is the value of the 'Vector gain multiplier' slider , |X| is the vector length of input token X and " + \
                                  "norm() is a function which normalizes a given vector , i.e sets its length of the given vector to 1. ") 

                              with gr.Accordion('Similar Mode Settings',open=False , visible = False): # Deprecated
                                self.inputs.sliders.angle = gr.Slider(minimum=0, maximum=100, step=0.1, label="Req. vector cos(theta) similarity %", default=60 , interactive = True)
                                self.inputs.sliders.length = gr.Slider(minimum=0, maximum=100, step=0.1, label="Req. vector length similarity %", default=80 , interactive = True)          
                                self.inputs.settings.allow_negative_gain = gr.Checkbox(value=False,label="Allow negative gain", interactive = True , visible = False)         
                                with gr.Accordion('Tutorial : What is this?',open=False , visible= False) as tutorial_4 : 
                                  gr.Markdown("These sliders control the allowed range for the similar vector output. \n \n" + \
                                  "The 'Req. vector cos(theta) similarity' slider sets the allowed angle deviation between the TokenMixer input token " + \
                                  "and the generated similar token ,  where 100% means the vectors are parallell and 0% means that they " + \
                                  "are perpendicular to each other. \n \n The 'Req. vector length similarity' sets the allowed length " + \
                                  "ratio between the input token length R and similar output token length P, where 100% means that their " + \
                                  "lengths are the same , i.e R = P , 80% means that P can be in the range R/0.8 and 0.8 * R , " + \
                                  "and 0% means that P can be in the range between R/0.01 to R * 0.01")  

                                  with gr.Accordion("What is a 'Allow negative gain'?",open=False):
                                      gr.Markdown("This option will enable the sampler to output tokens that have negative " + \
                                      "similarity to the input token. For a similarity value of -100% , the similar vector " + \
                                      "will be parallell to the  input vector , but oriented in the opposite direction." + \
                                      "If you wish to experiment with tokens of opposite polarity , " + \
                                      "it is advisable to use the token Calculator instead. \n \n Assuming you want to " + \
                                      "investigate the opposite polarity of the token 'example' , you can write  ' - example ' " + \
                                      "in the Token Calculator to get the desired output. \n \n " + \
                                      "Enabling 'Allow Negative Gain' option will have unpredictible effects on the output of the TokenMixer")

                              with gr.Accordion('Interpolate Mode Settings',open=False):
                                self.inputs.sliders.interpolate = gr.Slider(minimum=0, maximum=100, step=0.1, label="Req. similarity to merge %", default=50 , interactive = True)
   
                              with gr.Accordion('Output Log',open=False) as logs : 
                                self.outputs.log = gr.Textbox(label="Log", lines=10)
                                #vector_save_button_press = gr.Button(value="Save vector to text file")
          ### End of UI
          self.tutorials = []
          self.tutorials.append(tutorial_0)
          self.tutorials.append(tutorial_1)
          self.tutorials.append(tutorial_2)
          self.tutorials.append(tutorial_3)
          self.tutorials.append(tutorial_4)

          self.show = [show]
          self.logs = [logs]
                  
          #set default values
          self.inputs.sliders.angle.value = 60
          self.inputs.sliders.gain.value = 1
          self.inputs.sliders.length.value = 80
          self.inputs.sliders.randomize.value = 0
          self.inputs.sliders.interpolate.value = 50
          self.inputs.sliders.iterations.value = 2000


          self.buttons.save.style(size="lg")
          self.buttons.reset.style(size="lg")
          self.setupIO_with(self)
          
## end  of class TokenMixer 
