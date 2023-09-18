#TOOLS.PY
import torch, os
import copy
from types import NoneType
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
        model = shared.sd_model
        tokenizer = model.tokenizer
        tokenizer1280 = model.tokenizer_2
        loaded_embs = collections.OrderedDict(
        sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.items(),
            key=lambda x: str(x[0]).lower())) #doesn't actually work with diffusers
        internal_embs = model.text_encoder.get_input_embeddings().weight
        is_sdxl = True
        internal_embs1280 = model.text_encoder_2.get_input_embeddings().weight
        return tokenizer , internal_embs , loaded_embs , is_sdxl , internal_embs1280 , tokenizer1280
      #### End of get_diffusers()

      #Check if a valid model is loaded
      def model_is_loaded(self):
        if not hasattr(shared, 'sd_model'): return False
        else: return (shared.sd_model != None)
      #### End of model_is_loaded()

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
        else : embedder = self.cond_stage_models.embedders.wrapped
        ######
        internal_emb_dir = None
        if is_sd2 : internal_emb_dir = embedder.model
        else : internal_emb_dir = embedder.transformer.text_model.embeddings
        ######
        internal_embs = internal_emb_dir.token_embedding.wrapped.weight 
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

      # Get embedding vectors from text
      def get_emb_vecs_from(self, text , use_1280_dim = False):
        assert type(text) == str , "input text is not string!"
        embedded = None
        ########
        if use_1280_dim:  embedded = self._get_emb_vecs1280_from(text)
        else: embedded = self._get_emb_vecs768_from(text)
        return embedded.to(choosen_device)
      ##### End of get_emb_vecs_from()

      def __init__(self , count=1):
        #Default values if no model is loaded
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
          Tools.cond_stage_models = self.get_cond_stage_model_from(shared.sd_model)
          Tools.is_sdxl = is_sdxl
          Tools.emb_savepath = self.make_emb_folder('TokenMixer') 
          Tools.tokenizer = self.get_tokenizer()
          Tools.loaded_embs = self.get_loaded_embs()
          Tools.internal_embs = self.get_internal_embs()
          Tools.no_of_internal_embs = len(self.internal_embs)
          assert self.tokenizer != None , "tokenizer is NoneType!"
          Tools.loaded = True
        ########
#tokenize
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

        Tools.count = count 
        Tools.letter = ['easter egg' , 'a' , 'b' , 'c' , 'd' , 'e' , 'f' , 'g' , 'h' , 'i' , \
        'j' , 'k' , 'l' , 'm' , 'n' , 'o' , 'p' , 'q' , 'r' , 's' , 't' , 'u' , \
        'v' , 'w' , 'x' , 'y' , 'z']
#End of Tools class