#TOOLS.PY
import torch, os
import copy
import collections
import gradio as gr
from modules import shared
from modules import sd_hijack , sd_hijack_open_clip
from modules.shared import cmd_opts
from modules.textual_inversion.textual_inversion import Embedding
######
from library.toolbox.constants import MAX_NUM_MIX 
from library.toolbox.constants import TENSOR_DEVICE_TYPE , TENSOR_DATA_TYPE
choosen_device = TENSOR_DEVICE_TYPE
datatype = TENSOR_DATA_TYPE
#------------------------------------------------------------
from library.toolbox.constants import START_OF_TEXT_ID , END_OF_TEXT_ID
start_of_text_ID = START_OF_TEXT_ID
end_of_text_ID = END_OF_TEXT_ID
#------------------------------------------------------------

from modules import devices, sd_hijack_optimizations, shared, script_callbacks, errors, sd_unet
from modules.hypernetworks import hypernetwork
from modules.shared import cmd_opts
from modules import sd_hijack_clip, sd_hijack_open_clip, sd_hijack_unet, sd_hijack_xlmr, xlmr
import ldm.modules.encoders.modules
import random , numpy

from library.toolbox.constants import Token

class Tools :
  #The class tools contain built in functions for handling 
  #tokens and embeddings

      def get_cond_stage_model_from (self, model):
        conditioner = getattr(model, 'conditioner', None)
        text_cond_models = []
        cond_stage_model = None
        
        if conditioner:
            for i in range(len(conditioner.embedders)):
                embedder = conditioner.embedders[i]
                typename = type(embedder).__name__
                if typename == 'FrozenOpenCLIPEmbedder':
                    conditioner.embedders[i] = sd_hijack_open_clip.\
                    FrozenOpenCLIPEmbedderWithCustomWords(embedder, self)
                    text_cond_models.append(conditioner.embedders[i])
                if typename == 'FrozenCLIPEmbedder':
                    model_embeddings = embedder.transformer.text_model.embeddings
                    conditioner.embedders[i] = sd_hijack_clip.\
                    FrozenCLIPEmbedderForSDXLWithCustomWords(embedder, self)
                    text_cond_models.append(conditioner.embedders[i])
                if typename == 'FrozenOpenCLIPEmbedder2':
                    conditioner.embedders[i] = sd_hijack_open_clip.\
                    FrozenOpenCLIPEmbedder2WithCustomWords(embedder, self)
                    text_cond_models.append(conditioner.embedders[i])

            if len(text_cond_models) == 1: 
              cond_stage_model = text_cond_models[0]
            else: cond_stage_model = conditioner

        if type(cond_stage_model) == xlmr.BertSeriesModelWithTransformation:
            cond_stage_model = sd_hijack_xlmr.FrozenXLMREmbedderWithCustomWords(cond_stage_model, self)

        elif type(cond_stage_model) == ldm.modules.encoders.modules.FrozenCLIPEmbedder:
            cond_stage_model = sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords(cond_stage_model, self)

        elif type(cond_stage_model) == ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder:
            cond_stage_model = sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords(cond_stage_model, self)

        return cond_stage_model   
      ###### End of get_cond_stage_model_from() 

      # Create a folder in the UI to store 
      # embeddings created by the TokenMixer extension
      def make_emb_folder(self , name):
        savepath = os.path.join(cmd_opts.embeddings_dir, name)
        try: os.makedirs(savepath)
        except: pass
        try: modules.sd_hijack.model_hijack.embedding_db.add_embedding_dir(savepath)
        except: pass
        return savepath
      #### End of make_emb_folder()

      # SD-Next compatibility (work in progress)
      def get_diffusers(self):
        return None
        #model = shared.sd_model
        #tokenizer = model.tokenizer
        #tokenizer1280 = model.tokenizer_2
        #loaded_embs = collections.OrderedDict(
        #sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.items(),
        #    key=lambda x: str(x[0]).lower())) #doesn't actually work with diffusers
        #internal_embs = model.text_encoder.get_input_embeddings().weight
        #is_sdxl = True
        #internal_embs1280 = model.text_encoder_2.get_input_embeddings().weight
        #return tokenizer , internal_embs , loaded_embs , is_sdxl , internal_embs1280 , tokenizer1280
      #### End of get_diffusers()

      #Check if a valid model is loaded
      def model_is_loaded(self):
        if not hasattr(shared, 'sd_model'): return False
        else: return (shared.sd_model != None)
      #### End of model_is_loaded()

      # Create a random tensor
      def random(self , use_1280_dim = False):

        # Fetch stuff
        distance = torch.nn.PairwiseDistance(p=2)
        random_token_length = self.random_token_length
        random_token_length_randomization = self.random_token_length_randomization
        ######
        
        origin = self.origin768\
        .to(device = choosen_device , dtype = datatype)
        if use_1280_dim : origin = self.origin1280\
        .to(device = choosen_device , dtype = datatype)

        size = self.size768
        if use_1280_dim : size = self.size1280
        emb_vec = torch.rand(size)\
        .to(device = choosen_device , dtype = datatype)

        dist = distance(emb_vec , origin).numpy()[0]
        rdist = random_token_length * \
        (1 - random_token_length_randomization*random.random())
        emb_vec = (rdist/dist)*emb_vec
        #######
        return emb_vec.to(device = choosen_device , dtype = datatype)
      ### End of random()

      # Get boolean flags for the sd_model
      def get_flags(self):
        assert self.model_is_loaded(), "Model is not loaded"
        model = shared.sd_model
        is_sdxl = hasattr(model, 'conditioner')
        is_sd2 = hasattr(model.cond_stage_model, 'model') and not is_sdxl
        is_sd1 = not is_sd2
        return is_sdxl , is_sd2 , is_sd1
      ### End of get_flags()

      #Fetch the loaded embeddings
      def get_loaded_embs(self):
        is_sdxl , is_sd2 , is_sd1 = self.get_flags()
        loaded_embs = collections.OrderedDict(
        sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.items(),
            key=lambda x: str(x[0]).lower()))
        return loaded_embs
      #### End of get_loaded_embs()

      # Fetch internal_embs for use in SD 1.5
      def get_internal_embs(self):
        is_sdxl , is_sd2 , is_sd1 = self.get_flags()
        #######
        embedder = None
        if is_sdxl : embedder = self.cond_stage_models.embedders[0].wrapped
        else : embedder = shared.sd_model.cond_stage_model.wrapped
        assert embedder != None , "embedder is NoneType!"
        ######
        internal_emb_dir = None
        if is_sd2 : internal_emb_dir = embedder.model
        else : internal_emb_dir = embedder.transformer.text_model.embeddings
        assert internal_emb_dir != None , "internal_emb_dir is NoneType!"
        ######
        internal_embs = None
        internal_embs = internal_emb_dir.token_embedding.wrapped.weight
        assert internal_embs != None , "internal_embs is NoneType!"
        return internal_embs
      ########## End of get_internal_embs()

     #Fetch the tokenizer
      def get_tokenizer(self):
        is_sdxl , is_sd2 , is_sd1 = self.get_flags()

        embedder = None
        if is_sdxl : embedder = self.cond_stage_models.embedders[0].wrapped
        else : embedder = shared.sd_model.cond_stage_model.wrapped

        tokenizer = None
        if is_sd2 :
          from modules.sd_hijack_open_clip import tokenizer as open_clip_tokenizer
          tokenizer = open_clip_tokenizer
        else: tokenizer = embedder.tokenizer
        return tokenizer
      ######## End of get_tokenizer()

      # Fetch the loaded embeddings again
      def update_loaded_embs(self):
        Tools.loaded_embs = self.get_loaded_embs()
      #### End of update_loaded_embs()

      # Get a letter from the list and 
      # return it (for use in a embedding savename)
      def get_subname(self):
        self.count = copy.copy(self.count + 1)
        if self.count >= len(self.letter):
          self.count = 1
        return self.letter[self.count]
      ### End of get_subname()

      # Get the IDs of the tokens with greatest similarity 
      # to the input token (for use in the Embedding inspector module)
      def get_best_ids (self , emb_id , similarity , max_similar_embs , vector) :
        internal_embs = self.internal_embs.to(device = choosen_device , dtype = datatype)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        all_sim = cos(internal_embs , vector)
        scores = torch.mul(all_sim , \
        torch.ge(similarity * torch.ones(all_sim.shape) , 100*all_sim))
        sorted_scores, sorted_ids = torch.sort(scores, descending=True)
        best_ids = sorted_ids[0:max_similar_embs].detach().numpy()
        return  best_ids , sorted_scores
      #### End of get_best_ids()

      # Get embedding IDs from text
      def tokenize(self, text , max_length = False):

        text_input = None
        if max_length: 
          text_input = self.tokenizer(\
          text , truncation=False , \
          padding="max_length", \
          max_length=self.tokenizer.model_max_length , \
           return_tensors="pt")
        else:
          text_input = self.tokenizer(text , \
          truncation=False , return_tensors="pt")
        ######
        emb_ids = text_input.input_ids[0]

        return emb_ids.to(device = choosen_device , dtype = torch.int)
      #### End of get_emb_ids_from()

      # Pass the text through the first layer in the SDXL model
      # Outputs a tensor with the format [A , B  , C]
      # Not sure if this is useful to anything in TokenMixer
      def encode_embedding_init_text(self , text , use_1280_dim = False):
        nvpt = self.tokenizer.model_max_length
        embedded = None
        if use_1280_dim : 
          embedded = self.FrozenOpenCLIPEmbedder2WithCustomWords\
          .encode_embedding_init_text(text , nvpt).to(choosen_device)
        else : 
          embedded = self.FrozenCLIPEmbedderForSDXLWithCustomWords\
          .encode_embedding_init_text(text , nvpt).to(choosen_device)
        #######
        return embedded
    
      @staticmethod
      def concat(input , target):
        assert type(input) == torch.Tensor , "input is not a " + \
        "torch.Tensor, it is a " + str(type(input)) + " !"
        tensor = input.to(device = choosen_device , dtype = datatype)
        if target == None: return tensor
        else : return torch.cat([tensor , target] , dim = 0)\
        .to(device = choosen_device , dtype = datatype)
    
      # Get name from an ID
      def get_name_from(self, _ID , using_index = 0):

        #Fetch functions
        tokenizer = self.tokenizer
        token = self.token
        #####

        emb_name_utf8 = tokenizer.decoder.get(_ID)
        if emb_name_utf8 != None:
            byte_array_utf8 = bytearray([tokenizer.byte_decoder[c] for c in emb_name_utf8])
            emb_name = byte_array_utf8.decode("utf-8", errors='backslashreplace')
        else: emb_name = '!Unknown ID!'

        if token.is_random_type(_ID):
          emb_name = "Random_" + str(using_index)

        return emb_name # return embedding name for embedding ID
      # End of emb_id_to_name()

      def process(self, input , to = 'tensors' , \
      use_1280_dim = False , max_length = False , using_index = 0):
        
        # Do some checks
        assert type(to) == str , "value for 'to' param is not string , it is " + \
        " a " + str(type(to)) + " !"
        ######
        
        # Fetch the input depending on type
        text = None # string
        _ID_tensor = None # torch.Tensor 
        _ID_int32 = None # torch.int32
        if type(input) == str : text = input
        elif type(input) == torch.Tensor : _ID_tensor = input
        else : _ID_int32 = input
        #######

        # Fetch other stuff
        internal_embs = self.internal_embs768
        if use_1280_dim :  internal_embs = self.internal_embs1280
        token = self.token
        #####

        #Initialize output params
        emb_vecs = None
        emb_ids = None
        name = None
        ######

        # If input is text string , tokenize it and fetch vectors
        if text != None:
          emb_vec = None 
          _ID_tensor = self.tokenize(text , max_length)
          emb_ids = _ID_tensor.to(choosen_device , dtype = torch.int32)
          for _ID in _ID_tensor:
            if token.is_normal_type(_ID): 
              emb_vec = internal_embs[_ID]
            elif token.is_random_type(_ID): 
              emb_vec = self.random(use_1280_dim)
            elif token.is_start_of_text_type(_ID): 
              emb_vec = internal_embs[start_of_text_ID]
            elif token.is_end_of_text_type(_ID):
              emb_vec = internal_embs[end_of_text_ID] 
            else : assert False , "_ID within tokenized _ID_tensor batch " + \
            " with type " + str(type(_ID_int32)) + \
            " could not be identified!"
            ######
            emb_vec = emb_vec.unsqueeze(0)
            if emb_vecs == None : emb_vecs = emb_vec
            else: emb_vecs = self.concat(emb_vec , emb_vecs)
          #### End of for loop
        #######

        # If input is ID_tensor , do same as above but 
        # skip the tokenization step since we already have the IDs
        if _ID_tensor != None:
          emb_vec = None
          emb_ids = _ID_tensor.to(choosen_device , dtype = torch.int32)
          for _ID in _ID_tensor:
            if token.is_normal_type(_ID): 
              emb_vec = internal_embs[_ID]
            elif token.is_random_type(_ID): 
              emb_vec = self.random(use_1280_dim)
            elif token.is_start_of_text_type(_ID): 
              emb_vec = internal_embs[start_of_text_ID]
            elif token.is_end_of_text_type(_ID):
              emb_vec = internal_embs[end_of_text_ID] 
            else : assert False , "_ID within _ID_tensor batch " + \
            " with type " + str(type(_ID_int32)) + \
            " could not be identified!"
            ######
            emb_vec = emb_vec.unsqueeze(0)
            if emb_vecs == None : emb_vecs = emb_vec
            else: emb_vecs = self.concat(emb_vec , emb_vecs)
          #### End of for loop
        #########

        # If input is single ID , fetch the vector directly
        if _ID_int32 != None : 
          emb_ids = _ID_int32
          if token.is_normal_type(_ID_int32): 
            emb_vecs = internal_embs[_ID_int32]
          elif token.is_random_type(_ID_int32): 
            emb_vecs = self.random(use_1280_dim)
          elif token.is_start_of_text_type(_ID_int32): 
            emb_vecs = internal_embs[start_of_text_ID]
          elif token.is_end_of_text_type(_ID_int32):
            emb_vec = internal_embs[end_of_text_ID] 
          else : assert False , "single _ID with type " + \
          str(type(_ID_int32)) + " could not be identified!"
        #########

        if to == 'tensors':
          return emb_vecs.squeeze(0)\
          .to(device = choosen_device , dtype = datatype)
        elif to == 'ids':
          return emb_ids
        elif to == 'name':
          if _ID_int32 != None: 
            name = self.get_name_from(_ID_int32 , using_index)
            assert type(name) == str , "name is not string , it is " + \
            " a " + str(type(name)) + " !"
          return name
        else : assert False , "parameter to = " + to + " could not be interpreted!" 
      ##### End of get_emb_vecs_from()





      def __init__(self , count=1):
        
        #Default values if no model is loaded
        Tools.token = Token()
        Tools.get_token_type_from = self.token.type_from
        Tools.tokenizer = None
        Tools.internal_embs = None
        Tools.internal_embs768 = None
        Tools.internal_embs1280 = None
        Tools.loaded_embs = None
        Tools.no_of_internal_embs = 0
        Tools.no_of_internal_embs768 = 0
        Tools.no_of_internal_embs1280 = 0
        Tools.is_sdxl = False
        Tools.loaded = False
        Tools.text_encoder = None
        is_sdxl = False
        is_sd2 = False
        is_sd1 = False
        Tools.FrozenCLIPEmbedderForSDXLWithCustomWords = None #0 for SDXL (768)
        Tools.FrozenOpenCLIPEmbedder2WithCustomWords = None   #1 for SDXL (1280)
        #######
        Tools.FrozenCLIPEmbedderForSDXL = None  #0 for SDXL (768)
        Tools.FrozenOpenCLIPEmbedder2 = None #1 for SDXL (1280)
        #######
        Tools.embedder768 = None #FrozenCLIPEmbedderForSDXLWithCustomWords
        Tools.embedder1280 = None #FrozenOpenCLIPEmbedder2WithCustomWords
        ######
        Tools.internal_embs786 = None
        Tools.internal_embs1280 = None
        ######
        Tools.size768 = 0
        Tools.size1280 = 0
        Tools.origin768 = None
        Tools.origin1280 = None
        #####
        Tools.random_token_length = None
        Tools.random_token_length_randomization = None
        ###### End of default values

        # Check if a valid SD model is loaded (SD 1.5 , SD2 or SDXL)
        model_is_loaded = self.model_is_loaded()
        if model_is_loaded : is_sdxl , is_sd2 , is_sd1 = self.get_flags()
        from pprint import pprint
        if is_sdxl : pprint("MODEL IS SDXL")
        else : pprint("MODEL IS SD15")
        ######

        #Add values to Tools.py
        #if a valid sd-model is loaded
        if model_is_loaded : 
          pprint("MODEL IS LOADED")
          Tools.cond_stage_models = self.get_cond_stage_model_from(shared.sd_model)
          Tools.is_sdxl = is_sdxl
          Tools.emb_savepath = self.make_emb_folder('TokenMixer') 
          Tools.tokenizer = self.get_tokenizer()
          Tools.loaded_embs = self.get_loaded_embs()
          Tools.internal_embs = self.get_internal_embs()
          assert self.internal_embs != None , "internal embeddings are NoneType!"
          Tools.internal_embs786 = self.get_internal_embs() #default value (overwritten for SDXL)
          Tools.internal_embs1280 = self.get_internal_embs() #default value (overwritten for SDXL)
          Tools.no_of_internal_embs = len(self.internal_embs)
          assert self.tokenizer != None , "tokenizer is NoneType!"
          Tools.loaded = True
        ########

        # SDXL Text encoders resources
        # In A1111 they are contained within another class
        # that allows them to process texts in "chunks" of
        # 77 tokens , rather then being limited to 77 tokens
        # as is usually the case
        if model_is_loaded and is_sdxl:

          #CLIPTextModel (768 Dimension)
          Tools.embedder768 = self.cond_stage_models.embedders[0]
          Tools.FrozenCLIPEmbedderForSDXLWithCustomWords = self.cond_stage_models.embedders[0]
          Tools.FrozenCLIPEmbedderForSDXL = self.cond_stage_models.embedders[0].wrapped
          self.internal_embs768 =\
          self.FrozenCLIPEmbedderForSDXL.transformer.text_model.\
          embeddings.token_embedding.wrapped.weight 
          Tools.no_of_internal_embs768 = len(self.internal_embs768)
          #######
          
          #CLIPTextModelWithProjection (1280 Dimension)
          Tools.embedder1280 = self.cond_stage_models.embedders[1]
          Tools.FrozenOpenCLIPEmbedder2WithCustomWords = self.cond_stage_models.embedders[1]
          Tools.FrozenOpenCLIPEmbedder2 = self.cond_stage_models.embedders[1].wrapped
          Tools.internal_embs1280 = \
          self.FrozenOpenCLIPEmbedder2.model.token_embedding.wrapped.weight
          Tools.no_of_internal_embs1280 = len(self.internal_embs1280)
          #######
          #BaseModelOutputWithPooling

        if model_is_loaded:
          sample_ID = self.tokenize(',')
          emb_vec768 = self.get_emb_vecs_from(',')
          emb_vec1280 = self.get_emb_vecs_from(',' , use_1280_dim=True)
          Tools.size768 = emb_vec768.shape[1]
          Tools.size1280 = emb_vec1280.shape[1]
          Tools.origin768 = torch.zeros(self.size768)\
          .to(choosen_device , dtype = datatype)
          Tools.origin1280 = torch.zeros(self.size1280)\
          .to(choosen_device , dtype = datatype)
        #########

        Tools.count = count 
        Tools.letter = ['easter egg' , 'a' , 'b' , 'c' , 'd' , 'e' , 'f' , 'g' , 'h' , 'i' , \
        'j' , 'k' , 'l' , 'm' , 'n' , 'o' , 'p' , 'q' , 'r' , 's' , 't' , 'u' , \
        'v' , 'w' , 'x' , 'y' , 'z']
#End of Tools class