import gradio as gr
from modules import script_callbacks, shared, sd_hijack
from modules.shared import cmd_opts
from pandas import Index
from pandas.core.groupby.groupby import OutputFrameOrSeries
from pydantic import NoneStrBytes
import torch, os
from modules.textual_inversion.textual_inversion import Embedding
import collections, math, random , numpy
import re #used to parse string to int
import copy
from torch.nn.modules import ConstantPad1d, container

from lib.toolbox.constants import MAX_NUM_MIX

from lib.data import dataStorage

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn


class Synonymizer:

  def Reset (self , mini_input , tokenbox) : 
    return '' , ''

  def _add(self, language):
    if self.languages != '': 
      self.languages = self.languages + " , "
    self.languages = self.languages + language

  def _get_languages(self, *args) :
    english         = args[0]          
    standard_Arabic = args[1]
    bulgarian       = args[2]        
    catalan         = args[3]       
    danish          = args[4]          
    greek           = args[5]           
    basque          = args[6]         
    finnish         = args[7]         
    french          = args[8]         
    galician        = args[9]         
    hebrew          = args[10]          
    croatian        = args[11]       
    indian          = args[12]        
    islandic        = args[13]      
    italian         = args[14]       
    japanese        = args[15]       
    lithuanian      = args[16]    
    dutch           = args[17]    
    norwegian       = args[18]   
    polish          = args[19]       
    portuguese      = args[20]    
    spanish         = args[21]       
    swedish         = args[22]
    #########
    english_id = ['eng']
    standard_Arabic_id = ['arb']
    bulgarian_id = ['bul']
    catalan_id = ['cat']
    danish_id = ['dan']
    greek_id = ['ell']
    basque_id = ['eus']
    finnish_id = ['fin']
    french_id = ['fra']
    galician_id = ['glg']
    hebrew_id = ['heb']
    croatian_id = ['hrv']
    indian_id = ['ind']
    islandic_id = ['isl']
    italian_id = ['ita']
    japanese_id = ['jpn']
    lithuanian_id = ['lit']
    dutch_id = ['nld']
    norwegian_id = ['nno']
    polish_id = ['pol']
    portuguese_id = ['por']
    spanish_id = ['spa']
    swedish_id = ['swe']
    ########   
    self.languages = ''
    langs = []
    #######
    if english : 
      self._add("English")
      langs = langs + english_id
    if standard_Arabic : 
      self._add("Standard_Arabic")
      langs = langs + standard_Arabic_id
    if bulgarian : 
      self._add("Bulgarian")
      langs = langs + bulgarian_id      
    if catalan : 
      self._add("Catalan")
      langs = langs + catalan_id   
    if danish : 
      self._add("Danish")
      langs = langs + danish_id        
    if greek : 
      self._add("Greek")
      langs = langs + greek_id        
    if basque : 
      self._add("Basque")
      langs = langs + basque_id      
    if finnish : 
      self._add("Finnish")
      langs = langs + finnish_id       
    if french : 
      self._add("French")
      langs = langs + french_id    
    if galician : 
      self._add("Galician")
      langs = langs + galician_id     
    if hebrew : 
      self._add("Hebrew")
      langs = langs + hebrew_id       
    if croatian : 
      self._add("Croatian")
      langs = langs + croatian_id 
    if indian : 
      self._add("Indian")
      langs = langs + indian_id   
    if islandic : 
      self._add("Islandic")
      langs = langs + islandic_id     
    if italian : 
      self._add("Italian")
      langs = langs + italian_id   
    if japanese : 
      self._add("Japanese")
      langs = langs + japanese_id     
    if lithuanian : 
      self._add("Lithuanian")
      langs = langs + lithuanian_id    
    if dutch : 
      self._add("Dutch")
      langs = langs + dutch_id
    if norwegian : 
      self._add("Norwegian")
      langs = langs + norwegian_id  
    if polish : 
      self._add("Polish")
      langs = langs + polish_id
    if portuguese : 
      self._add("Portuguese")
      langs = langs + portuguese_id 
    if spanish : 
      self._add("Spanish")
      langs = langs + spanish_id     
    if swedish : 
      self._add("Swedish")
      langs = langs + swedish_id
    ######
    self.langs = langs
    return self.languages , self.langs

  def Synonymize (self  , *args) :
    #Get the inputs
    mini_input = args[0]
    sendtomix = args[1]
    definition_mode = args[2]
    send_to_negatives = args[3]
    random_token_length = args[4]
    mix_input = args[6]
    stack_mode = args[7]
    ######
    hypernym_mode = args[5]
    hyponym_mode = args[8]
    meronym_mode = args[9]
    holonym_mode = args[10]
    entailment_mode = args[11]
    antonym_mode = args[12]
    ######################
    english         = args[13]          
    standard_Arabic = args[14]
    bulgarian       = args[15]        
    catalan         = args[16]       
    danish          = args[17]          
    greek           = args[18]           
    basque          = args[19]         
    finnish         = args[20]         
    french          = args[21]         
    galician        = args[22]         
    hebrew          = args[23]          
    croatian        = args[24]       
    indian          = args[25]        
    islandic        = args[26]      
    italian         = args[27]       
    japanese        = args[28]       
    lithuanian      = args[29]    
    dutch           = args[30]    
    norwegian       = args[31]   
    polish          = args[32]       
    portuguese      = args[33]    
    spanish         = args[34]       
    swedish         = args[35]
    #########
    languages , langs = self._get_languages(
    english , standard_Arabic , bulgarian , catalan , danish  , \
    greek , basque , finnish , french , galician  , hebrew  , \
    croatian , indian , islandic , italian  , japanese  , \
    lithuanian , dutch , norwegian , polish , portuguese , \
    spanish , swedish)          
    ######
    compact_mode      = args[36]   
    inspect_mode      = args[37]
    no_of_suggestions = args[38]   
    ######

    #######
    definitions , list_of_synonyms , list_of_hyponyms , \
    list_of_hypernyms , list_of_meronyms , list_of_holonyms , \
    list_of_entailments , list_of_antonyms , list_of_everything , \
    tokenbox = self.get( 
    mini_input , langs , languages , compact_mode , \
    definition_mode , hypernym_mode , hyponym_mode , \
    meronym_mode , holonym_mode  , entailment_mode , \
    antonym_mode , inspect_mode , no_of_suggestions)
    ######

    if inspect_mode : 
      tokenbox.append('##########################################################')
      tokenbox.append(' ')

    for suggestion in range(no_of_suggestions):
      text = ''
      for everything in list_of_everything:
        text = text + random.choice(everything) + '   '
      tokenbox.append(text)
      tokenbox.append('  ')

    return '' , '\n'.join(tokenbox)

  def get(self, *args) :
    mini_input      = args[0]
    langs           = args[1]
    languages       = args[2]
    compact_mode    = args[3]
    definition_mode = args[4]
    hypernym_mode   = args[5]
    hyponym_mode    = args[6]
    meronym_mode    = args[7]
    holonym_mode    = args[8]
    entailment_mode = args[9]
    antonym_mode    = args[10]
    inspect_mode    = args[11]
    no_of_suggestions = args[12]

    words = []
    definitions = []
    list_of_synonyms = []
    list_of_hyponyms = []
    list_of_hypernyms = []
    list_of_meronyms = []
    list_of_holonyms = []
    list_of_entailments = []
    list_of_antonyms = []
    list_of_everything = []
    tokenbox = []
    tokenmixer_vectors= ''
    text = ''
    found = False
    first = True
    defcount = 0

    tokenbox.append('Running Token Synonymizer...')
    tokenbox.append(' ')
    tokenbox.append('Languages :  ' + languages)
    tokenbox.append(' ')
    tokenbox.append('Number of synonym prompt suggestions :  ' + str(no_of_suggestions))
    tokenbox.append(' ')
    tokenbox.append('##########################################################')

    for word in mini_input.strip().split() :  
      synonyms  = []
      hyponyms  = []
      hypernyms = []
      holonyms  = []
      meronyms  = []
      entailments = []
      antonyms = []
      everything = []
      words.append(word)
      ######
      definition = None
      for synset in wn.synsets(word) :
        ##################
        definition = "'" + synset.definition() + "'"
        definitions.append(definition)
        ##################
        for entailment in synset.entailments() :
            if not entailment_mode: break
            for lang in langs:
              for name in entailment.lemma_names(lang):
                entailments.append(name)
                everything.append(name)        
        ##################
        for meronym in synset.part_meronyms() :
            if not meronym_mode: break
            for lang in langs:
              for name in meronym.lemma_names(lang):
                meronyms.append(name)
                everything.append(name)  
        ###########################
        for holonym in synset.part_holonyms() :
            if not holonym_mode: break
            for lang in langs:
              for name in holonym.lemma_names(lang):
                holonyms.append(name)
                everything.append(name)  
        ###########################
        for hyponym in synset.hyponyms() :
            if not hyponym_mode: break
            for lang in langs:
              for name in hyponym.lemma_names(lang):
                hyponyms.append(name)
                everything.append(name)  
        ###########################
        for hypernym in synset.hypernyms() :
            if not hypernym_mode: break
            for lang in langs:
              for name in hypernym.lemma_names(lang):
                hypernyms.append(name)
                everything.append(name)  
        ############################
        for lang in langs:
          for synonym in synset.lemma_names(lang) :
            synonyms.append(synonym)
            everything.append(synonym)  
        ############################
        for lang in langs:
          if not antonym_mode: break
          for lemma in synset.lemmas(lang) :
            for antonym in lemma.antonyms():
              antonyms.append(antonym.name())
              everything.append(antonym.name()) 
        ############################
        defcount +=1
        if not first and inspect_mode :
          tokenbox.append(' ')  
          tokenbox.append('##########################################################')
        else : first = False
        
        if definition_mode and inspect_mode :
            tokenbox.append("Interpretation no. " + str(defcount) + " of '" + word + "' :")
            tokenbox.append("definition :   " + "'" + definition + "'")
            found = True 
        
        if inspect_mode : 
          tokenbox.append('------------------------------------------------' + \
          '---------------------------------------------')
        ######################
        if inspect_mode :
          prev = found 
          text = ''
          found = False
          for name in synonyms : 
            text = text  + name + '     '
            found = True
          if found:
            if not prev and not compact_mode:  
              tokenbox.append("............." + \
              "................................." + \
              "................................." + \
              ".....................................")
            tokenbox.append("Synonyms of this version of '" + word  + "' : ")
            tokenbox.append(text)
          elif not compact_mode : tokenbox.append("No known synonyms for this version of'" + word  + "'")
          if found :
            tokenbox.append("............." + \
            "................................." + \
            "................................." + \
            ".....................................")
        #######################
        if meronym_mode and inspect_mode :
            prev = found
            text = ''
            found = False
            for name in meronyms:
              text = text  + name + '     '
              found = True
            if found:
              if not prev and not compact_mode: 
                tokenbox.append("............." + \
                "................................." + \
                "................................." + \
                ".....................................")
              tokenbox.append("Meronyms of this version of '" + word + "' : ")
              tokenbox.append(text)
            elif not compact_mode : tokenbox.append("No known meronyms for this version of'" + word  + "'")
            if found : 
                tokenbox.append("............." + \
                "................................." + \
                "................................." + \
                ".....................................") 
        #######################
        if antonym_mode and inspect_mode :
            prev = found
            text = ''
            found = False
            for name in antonyms:
              text = text  + name + '     '
              found = True
            if found:
              if not prev and not compact_mode:  
                  tokenbox.append("............." + \
                  "................................." + \
                  "................................." + \
                  ".....................................")
              tokenbox.append("Antonyms of this version of '" + word + "' : ")
              tokenbox.append(text)
            elif not compact_mode : tokenbox.append("No known antonyms for this version of'" + word  + "'")
            if found :  
                tokenbox.append("............." + \
                "................................." + \
                "................................." + \
                ".....................................")
        #######################
        if entailment_mode and inspect_mode :
            prev = found
            text = ''
            found = False
            for name in entailments:
              text = text  + name + '     '
              found = True
            if found:
              if not prev and not compact_mode:  
                  tokenbox.append("............." + \
                  "................................." + \
                  "................................." + \
                  ".....................................")
              tokenbox.append("Entailments of this version of '" + word + "' : ")
              tokenbox.append(text)
            elif not compact_mode : tokenbox.append("No known entailments for this version of'" + word  + "'")
            if found :
                tokenbox.append("............." + \
                "................................." + \
                "................................." + \
                ".....................................")
        #######################
        if holonym_mode and inspect_mode :
            prev = found
            text = ''
            found = False
            for name in holonyms:
              text = text  + name + '     '
              found = True
            if found:
              if not prev and not compact_mode:  
                  tokenbox.append("............." + \
                  "................................." + \
                  "................................." + \
                  ".....................................")
              tokenbox.append("Holonym of this version of '" + word + "' : ")
              tokenbox.append(text)
            elif not compact_mode : tokenbox.append("No known holonyms for this version of'" + word  + "'")
            if found :
                tokenbox.append("............." + \
                "................................." + \
                "................................." + \
                ".....................................")
        #######################
        if hyponym_mode and inspect_mode :
            prev = found
            text = ''
            found = False
            for name in hyponyms:
              text = text  + name + '     '
              found = True
            if found:
              if not prev and not compact_mode:  
                tokenbox.append("............." + \
                "................................." + \
                "................................." + \
                ".....................................")
              tokenbox.append("Hyponyms of this version of '" + word + "' : ")
              tokenbox.append(text)
            elif not compact_mode : tokenbox.append("No known hyponyms for this version of'" + word  + "'")
            if found :
                tokenbox.append("............." + \
                "................................." + \
                "................................." + \
                ".....................................")
        #######################
        if hypernym_mode and inspect_mode :
            prev = found
            text = ''
            found = False
            for name in hypernyms:
              text = text  + name + '     '
              found = True
            if found:
              if not prev and not compact_mode:  
                tokenbox.append("............." + \
                "................................." + \
                "................................." + \
                ".....................................")
              tokenbox.append("Hypernym of this version of '" + word + "' : ")
              tokenbox.append(text)
            elif not compact_mode : tokenbox.append("No known hypernyms for this version of'" + word  + "'")
      ##############################
      list_of_entailments.append(list(set(entailments)))
      list_of_meronyms.append(list(set(meronyms)))
      list_of_holonyms.append(list(set(holonyms))) 
      list_of_synonyms.append(list(set(synonyms)))
      list_of_hyponyms.append(list(set(hyponyms)))
      list_of_hypernyms.append(list(set(hypernyms)))
      list_of_antonyms.append(list(set(antonyms)))
      list_of_everything.append(list(set(everything)))
    ######
    return  definitions , \
    list_of_synonyms , \
    list_of_hyponyms , \
    list_of_hypernyms , \
    list_of_meronyms , \
    list_of_holonyms , \
    list_of_entailments , \
    list_of_antonyms , \
    list_of_everything , \
    tokenbox 

  def setupIO_with (self, module):
    #Tell the buttons in this class what to do when 
    #pressed by the user

    input_list = []
    output_list = []

    if (module.ID == "Synonymizer") :
      output_list.append(self.inputs.mini_input)
      output_list.append(self.outputs.tokenbox)
      self.buttons.reset.click(fn = self.Reset, inputs = input_list , outputs = output_list)


    if (module.ID == "TokenMixer") :
      input_list.append(self.inputs.mini_input)       #0
      input_list.append(self.inputs.sendtomix)        #1
      input_list.append(self.inputs.definition_mode)  #2
      input_list.append(self.inputs.negatives)        #3
      input_list.append(self.inputs.randlen)          #4
      input_list.append(self.inputs.hypernym_mode)    #5
      input_list.append(module.inputs.mix_input)      #6
      input_list.append(self.inputs.stack_mode)       #7
      input_list.append(self.inputs.hyponym_mode)     #8
      input_list.append(self.inputs.meronym_mode)     #9
      input_list.append(self.inputs.holonym_mode)     #10
      input_list.append(self.inputs.entailment_mode)  #11
      input_list.append(self.inputs.antonym_mode)     #12
      #######
      input_list.append(self.lang.english)            #13
      input_list.append(self.lang.standard_Arabic)    #14
      input_list.append(self.lang.bulgarian)          #15
      input_list.append(self.lang.catalan)            #16
      input_list.append(self.lang.danish)             #17
      input_list.append(self.lang.greek)              #18
      input_list.append(self.lang.basque)             #19
      input_list.append(self.lang.finnish)             #20
      input_list.append(self.lang.french)             #21
      input_list.append(self.lang.galician)           #22
      input_list.append(self.lang.hebrew)             #23
      input_list.append(self.lang.croatian)           #24
      input_list.append(self.lang.indian)             #25
      input_list.append(self.lang.islandic)           #26
      input_list.append(self.lang.italian)            #27
      input_list.append(self.lang.japanese)           #28
      input_list.append(self.lang.lithuanian)         #29
      input_list.append(self.lang.dutch)              #30
      input_list.append(self.lang.norwegian)          #31
      input_list.append(self.lang.polish)             #32
      input_list.append(self.lang.portuguese)         #33
      input_list.append(self.lang.spanish)            #34
      input_list.append(self.lang.swedish)            #35
      ######
      input_list.append(self.inputs.compact_mode)     #36
      input_list.append(self.inputs.inspect_mode)     #37
      input_list.append(self.inputs.suggestions)      #38
      #######
      output_list.append(module.inputs.mix_input) #0
      output_list.append(self.outputs.tokenbox)   #1
      #######
      self.buttons.tokenize.click(fn=self.Synonymize , inputs = input_list , outputs = output_list)

  #End of setupIO_with()


  def __init__(self, label , vis = False):

    #Pass reference to global object "dataStorage" to class
    self.data = dataStorage 

    class Outputs :
      def __init__(self):
        Outputs.tokenbox = []

    class Inputs :
      def __init__(self):
        Inputs.mini_input = []
        Inputs.sendtomix = []
        Inputs.definition_mode = []
        Inputs.randlen = []
        Inputs.hypernym_mode = []
        Inputs.stack_mode = []
        Inputs.hyponym_mode = []
        Inputs.holonym_mode = []
        Inputs.meronym_mode = []
        Inputs.entailment_mode = []
        Inputs.antonym_mode = []
        Inputs.compact_mode = []
        Inputs.inspect_mode = []
        Inputs.suggestions = []

    class Lang :
      def __init__(self):
        Lang.english = []
        Lang.standard_Arabic = []
        Lang.bulgarian = []
        Lang.catalan = []
        Lang.danish = []
        Lang.greek = []
        Lang.basque = []
        Lang.finnish = []
        Lang.french = []
        Lang.galician = []
        Lang.hebrew = []
        Lang.croatian = []
        Lang.indian = []
        Lang.islandic = []
        Lang.italian = []
        Lang.japanese = []
        Lang.lithuanian = []
        Lang.dutch = []
        Lang.norwegian = []
        Lang.polish = []
        Lang.portuguese = []
        Lang.spanish = []
        Lang.swedish = []

    class Buttons :
      def __init__(self):
        Buttons.tokenize = []
        Buttons.reset = []

    self.outputs= Outputs()
    self.inputs = Inputs()
    self.buttons= Buttons()
    self.lang = Lang()
    self.ID = "Synonymizer"
    self.languages = None
    self.langs = []


    #test = gr.Label('Prompt MiniTokenizer' , color = "red")
    #create UI
    with gr.Accordion(label , open=False , visible = vis) as show :
      gr.Markdown("Get CLIP tokens and embedding vectors")
      with gr.Row() :
        self.inputs.suggestions = gr.Slider(label="No. of suggestions",value=5, \
                                minimum=0, maximum=100, step=1 , interactive = True)
      with gr.Row() :  
          self.buttons.tokenize = gr.Button(value="Synonymize", variant="primary")
          self.buttons.reset = gr.Button(value="Reset", variant = "secondary")
      with gr.Row() : 
        self.inputs.mini_input = gr.Textbox(label='', lines=2, \
        placeholder="Enter a short prompt or name of embedding" , interactive = True)
      with gr.Row() : 
        self.outputs.tokenbox = gr.Textbox(label="Output", lines=8, interactive = False)
      with gr.Row():
          self.inputs.sendtomix = gr.Checkbox(value=False, label="Send to input", interactive = True)
          self.inputs.negatives = gr.Checkbox(value=False, label="Send to negatives", interactive = True , visible = True) 
          self.inputs.stack_mode = gr.Checkbox(value=True, label="Stack Mode", interactive = True)
      with gr.Row():
          with gr.Accordion("Output options" ,open=False , visible = True):
            with gr.Row():
              self.inputs.compact_mode = gr.Checkbox(value=False, label="Ignore empty sets", interactive = True)
              self.inputs.definition_mode = gr.Checkbox(value=True, label="Include definition", interactive = True)
              self.inputs.inspect_mode = gr.Checkbox(value=True, label="Inspect Mode", interactive = True)
              self.inputs.holonym_mode = gr.Checkbox(value=True, label="Less specific (Holonyms)", interactive = True , visible = True) 
              self.inputs.meronym_mode = gr.Checkbox(value=True, label="More specific (Meronyms)", interactive = True , visible = True) 
              self.inputs.hypernym_mode = gr.Checkbox(value=True, label="Less specific (Hypernyms)", interactive = True , visible = True) 
              self.inputs.hyponym_mode = gr.Checkbox(value=True, label="More specific (Hyponyms)", interactive = True , visible = True) 
              self.inputs.entailment_mode = gr.Checkbox(value=True, label="More specific (Entailments)", interactive = True , visible = True)
              self.inputs.antonym_mode = gr.Checkbox(value=True, label="Opposites (Antonyms)", interactive = True , visible = True)
          
            with gr.Row():
              with gr.Accordion("Include language" ,open=False , visible = True):
                with gr.Row(): 
                  self.lang.english = gr.Checkbox(value= True, label="English", interactive = True)
                  self.lang.standard_Arabic = gr.Checkbox(value= False, label="Standard_Arabic", interactive = True)
                  self.lang.bulgarian = gr.Checkbox(value= False, label="Bulgarian", interactive = True)
                  self.lang.catalan = gr.Checkbox(value= False, label="Catalan", interactive = True)
                  self.lang.danish = gr.Checkbox(value= False, label="Danish", interactive = True)
                  self.lang.greek = gr.Checkbox(value= False, label="Greek", interactive = True)
                  self.lang.basque = gr.Checkbox(value= False, label="Basque", interactive = True)
                  self.lang.finnish = gr.Checkbox(value= False, label="Finish", interactive = True)
                  self.lang.french = gr.Checkbox(value= False, label="French", interactive = True)
                  self.lang.galician = gr.Checkbox(value= False, label="Galician", interactive = True)
                  self.lang.hebrew = gr.Checkbox(value= False, label="Hebrew", interactive = True)
                  self.lang.croatian = gr.Checkbox(value= False, label="Croatian", interactive = True)
                  self.lang.indian = gr.Checkbox(value= False, label="Indian", interactive = True)
                  self.lang.islandic = gr.Checkbox(value= False, label="Islandic", interactive = True)
                  self.lang.italian = gr.Checkbox(value= False, label="Italian", interactive = True)
                  self.lang.japanese = gr.Checkbox(value= False, label="Japanese", interactive = True)
                  self.lang.lithuanian = gr.Checkbox(value= False, label="Lithuanian", interactive = True)
                  self.lang.dutch = gr.Checkbox(value= False, label="Dutch", interactive = True)
                  self.lang.norwegian = gr.Checkbox(value= False, label="Norwegian", interactive = True)
                  self.lang.polish = gr.Checkbox(value= False, label="Polish", interactive = True)
                  self.lang.portuguese = gr.Checkbox(value= False, label="Portuguese", interactive = True)
                  self.lang.spanish = gr.Checkbox(value= False, label="Spanish", interactive = True)
                  self.lang.swedish = gr.Checkbox(value= False, label="Swedish", interactive = True)
          
      with gr.Accordion("Random ' _ ' token settings" ,open=False , visible = False) as randset : 
        self.inputs.randlen = gr.Slider(minimum=0, maximum=10, step=0.01, label="Randomized ' _ ' token length", default=0.35 , interactive = True)
      
      with gr.Accordion('Tutorial : What is this?',open=False , visible= False) as tutorial_0 :
          gr.Markdown("The Synonymizer uses the python Natural Language ToolKit (NLTK) " + \
          "To find words that semantically similar to the given token.  \n \n " +  \
          "It can also translate your token into different languages." )
    #end of UI

    self.tutorials = [tutorial_0]
    self.show = [show]
    self.randset = [randset]

    self.inputs.randlen.value = 0.35
    self.buttons.tokenize.style(size="sm")
    self.buttons.reset.style(size="sm")

    self.setupIO_with(self)
## End of class MiniTokenizer--------------------------------------------------#