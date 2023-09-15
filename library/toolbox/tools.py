#TOOLS.PY
from types import NoneType
import gradio as gr
from modules import script_callbacks, shared, \
sd_hijack , sd_models , sd_hijack_open_clip , textual_inversion , xlmr
from modules.shared import cmd_opts
#######
from pandas import Index
from pandas.core.groupby.groupby import OutputFrameOrSeries
import torch, os
from modules.textual_inversion.textual_inversion import Embedding
#######
import collections, math, random , numpy
import copy
from torch.nn.modules import ConstantPad1d, container
from library.toolbox.constants import MAX_NUM_MIX 
#------------------------------------------------------------
from library.toolbox.constants import TENSOR_DEVICE_TYPE , TENSOR_DATA_TYPE
choosen_device = TENSOR_DEVICE_TYPE
datatype = TENSOR_DATA_TYPE

from library.toolbox.constants import START_OF_TEXT_ID , END_OF_TEXT_ID
start_of_text_ID = START_OF_TEXT_ID
end_of_text_ID = END_OF_TEXT_ID

# Check that MPS is available (for MAC users)
#if torch.backends.mps.is_available(): 
#  choosen_device = torch.device("mps")
#else : choosen_device = torch.device("cpu")
######

class Tools :
  #The class tools contain built in functions for handling 
  #tokens and embeddings

      def isCutoff(self, ID):
        return ((ID == start_of_text_ID) or (ID == end_of_text_ID))    

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
        from pprint import pprint
        pprint('test')
        pprint(internal_embs.shape)
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

      #Helper function
      def _get_embedder(self):
        is_sdxl , is_sd2 , is_sd1 = self.get_flags()
        model = shared.sd_model
        embedder = None
        if is_sdxl: embedder = model.cond_stage_model.embedders[0].wrapped
        else: embedder = model.cond_stage_model.wrapped
        return embedder
      ## End of _get_embedder()

      def get_FrozenOpenCLIPEmbedder2WithCustomWords(self):
        is_sdxl , is_sd2 , is_sd1 = self.get_flags()
        model = shared.sd_model
        FrozenOpenCLIPEmbedder2WithCustomWords = None
        if is_sdxl : FrozenOpenCLIPEmbedder2WithCustomWords = \
        model.cond_stage_model.embedders[1]
        return FrozenOpenCLIPEmbedder2WithCustomWords

      def get_FrozenOpenCLIPEmbedder2(self):
        is_sdxl , is_sd2 , is_sd1 = self.get_flags()
        model = shared.sd_model
        FrozenOpenCLIPEmbedder2 = None
        if is_sdxl : FrozenOpenCLIPEmbedder2 = \
        model.cond_stage_model.embedders[1].wrapped
        return FrozenOpenCLIPEmbedder2

      def get_text_encoder1280(self):
        FrozenOpenCLIPEmbedder2 = self.get_FrozenOpenCLIPEmbedder2()
        text_encoder1280 = None
        if FrozenOpenCLIPEmbedder2 != None: text_encoder1280 = \
        FrozenOpenCLIPEmbedder2.model.token_embedding.wrapped
        return text_encoder1280

      # Helper function
      def _get_internal_emb_dir(self):
        is_sdxl , is_sd2 , is_sd1 = self.get_flags()
        embedder = self._get_embedder()
        model = shared.sd_model
        ########
        internal_emb_dir = None
        internal_embs = None
        internal_embs1280 = None
        if is_sd2 : internal_emb_dir = embedder.model
        else : internal_emb_dir = embedder.transformer.text_model.embeddings
        return internal_emb_dir
      ###########

      #Fetch text_encoder for dimension 768
      # Works for all models (SDXL , SD2 , SD1.5)
      def get_text_encoder(self):
        is_sdxl , is_sd2 , is_sd1 = self.get_flags()
        internal_emb_dir = self._get_internal_emb_dir()
        text_encoder768 = internal_emb_dir.token_embedding.wrapped
        return text_encoder768
      #######

      #Fetch text_encoder for dimension 768
      # Works for all models (SDXL , SD2 , SD1.5)
      def get_text_encoder768(self):
        is_sdxl , is_sd2 , is_sd1 = self.get_flags()
        internal_emb_dir = self._get_internal_emb_dir()
        text_encoder768 = internal_emb_dir.token_embedding.wrapped
        return text_encoder768
      #######

      # Fetch internal_embs for dimension 768
      # Works for all models (SDXL , SD2 , SD1.5)
      def get_internal_embs(self):
        is_sdxl , is_sd2 , is_sd1 = self.get_flags()
        internal_emb_dir = self._get_internal_emb_dir()
        internal_embs = internal_emb_dir.token_embedding.wrapped.weight 
        return internal_embs
      ##########

      # Fetch internal_embs for dimension 768
      # Works for all models (SDXL , SD2 , SD1.5)
      def get_internal_embs768(self):
        is_sdxl , is_sd2 , is_sd1 = self.get_flags()
        internal_emb_dir = self._get_internal_emb_dir()
        internal_embs768 = internal_emb_dir.token_embedding.wrapped.weight 
        return internal_embs768
      ##########

      # Fetch internal_embs for dimension 1280
      # (SDXL only)
      def get_internal_embs1280(self):
        FrozenOpenCLIPEmbedder2 = self.get_FrozenOpenCLIPEmbedder2()
        if FrozenOpenCLIPEmbedder2 != None : internal_embs1280 = \
        FrozenOpenCLIPEmbedder2.model.token_embedding.wrapped.weight
        return internal_embs1280

      #Fetch the tokenizer
      def get_tokenizer(self):
        is_sdxl , is_sd2 , is_sd1 = self.get_flags()
        embedder = self._get_embedder()
        tokenizer = None
        if is_sd2 :
          from modules.sd_hijack_open_clip import tokenizer as open_clip_tokenizer
          tokenizer = open_clip_tokenizer
        else: tokenizer = embedder.tokenizer
        return tokenizer
      ########

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
      def get_best_ids (self , emb_id , similarity , max_similar_embs , emb_vec = None) :
        max_sim = copy.copy(max_similar_embs)
        simil = copy.copy(similarity)
        ID = copy.copy(emb_id)
        internal_embs = self.internal_embs.to(device = choosen_device , dtype = datatype)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        vector = None
        if emb_vec != None:
          vector = emb_vec.to(device = choosen_device , dtype = datatype)
        elif isinstance(ID, int):
          vector = internal_embs[ID].to(device = choosen_device , dtype = datatype)
        else: return None , None

        all_sim = cos(internal_embs , vector)
        scores = torch.mul(all_sim , torch.ge(simil * torch.ones(all_sim.shape) , 100*all_sim))
        sorted_scores, sorted_ids = torch.sort(scores, descending=True)
        best_ids = sorted_ids[0:max_sim].detach().numpy()
        return  best_ids , sorted_scores
      #### End of get_best_ids()

      def get_CLIPTextModel(self):
        is_sdxl , is_sd2 , is_sd1 = self.get_flags()
        embedder = self._get_embedder()
        CLIPTextModel = embedder.transformer
        return CLIPTextModel


        #text_input = tokenizer(text , \
        #truncation=False, padding="max_length", \
        #max_length=tokenizer.model_max_length , return_tensors="pt")



      # Get embedding IDs from text
      def get_emb_ids_from(self, text , use_1280_dim = False):
        text_input = None
        tokenizer = self.tokenizer768 # Use the same tokenizer
        text_input = tokenizer(text ,truncation=False , return_tensors="pt")
        emb_ids = text_input.input_ids[0].to(device = choosen_device)
        return emb_ids
      #### End of get_emb_ids_from()


            #emb_vecs = self.text_encoder768(\
            #torch.asarray(emb_ids, device="cuda")\
            #).to(device=choosen_device , dtype = datatype).squeeze(0)

      # Get embedding vectors from text
      def get_emb_vecs_from(self, text , use_1280_dim = False):
        assert type(text) == str , "input text is not string!"
        emb_ids_list = self.get_emb_ids_from(text , use_1280_dim).numpy()
        #######
        text_encoder = None
        if use_1280_dim : text_encoder = self.text_encoder1280
        else: text_encoder = self.text_encoder768
        
        emb_vecs = text_encoder(\
        torch.asarray(emb_ids_list, device="cuda" , dtype = torch.int)\
        ).to(device=choosen_device , dtype = datatype).squeeze(0)

        return emb_vecs
      ##### End of get_emb_vecs_from()

      def __init__(self , count=1):
        #Default values if no model is loaded
        Tools.CLIPTextModel = None
        Tools.tokenizer = None
        Tools.tokenizer768 = None
        Tools.tokenizer1280 = None
        Tools.internal_embs = None
        Tools.internal_embs768 = None
        Tools.internal_embs1280 = None
        Tools.FrozenOpenCLIPEmbedder2 = None
        Tools.loaded_embs = None
        Tools.no_of_internal_embs = 0
        Tools.no_of_internal_embs768 = 0
        Tools.no_of_internal_embs1280 = 0
        Tools.is_sdxl = False
        Tools.loaded = False
        Tools.text_encoder = None
        Tools.text_encoder768 = None
        Tools.text_encoder1280 = None
        is_sdxl = False
        is_sd2 = False
        is_sd1 = False
        #######

        # Check if a valid SD model is loaded (SD 1.5 , SD2 or SDXL)
        model_is_loaded = self.model_is_loaded()
        if model_is_loaded : is_sdxl , is_sd2 , is_sd1 = self.get_flags()
        ######

        #Add values to Tools.py
        #if a valid sd-model is loaded
        if model_is_loaded : 
          Tools.is_sdxl = is_sdxl
          Tools.emb_savepath = self.make_emb_folder('TokenMixer') 
          Tools.tokenizer = self.get_tokenizer()
          Tools.loaded_embs = self.get_loaded_embs()
          Tools.text_encoder = self.get_text_encoder()
          Tools.internal_embs = self.get_internal_embs()
          Tools.no_of_internal_embs = len(self.internal_embs)
          assert self.tokenizer != None , "tokenizer is NoneType!"
          assert self.text_encoder != None , "text_encoder is NoneType!"
          Tools.loaded = True
        ########

        # SDXL Internal embeddings
        if model_is_loaded and is_sdxl:
          Tools.internal_embs768 = self.get_internal_embs768()
          Tools.internal_embs1280 = self.get_internal_embs1280()
          Tools.no_of_internal_embs768 = len(self.internal_embs768)
          Tools.no_of_internal_embs1280 = len(self.internal_embs1280)
        #######

        # SDXL Text encoders
        if model_is_loaded and is_sdxl:
          self.text_encoder768 = self.get_text_encoder768()
          self.text_encoder1280 = self.get_text_encoder1280()
          assert self.text_encoder768 != None , "encoder768 is NoneType!"
          assert self.text_encoder1280 != None , "encoder1280 is NoneType!"
        ################

        # SDXL Tokenizers (same for both)
        if model_is_loaded and is_sdxl:
          Tools.tokenizer768 = self.get_tokenizer()
          Tools.tokenizer1280 = self.get_tokenizer()
          assert self.tokenizer768 != None , "tokenizer768 is NoneType!"
          assert self.tokenizer1280 != None , "tokenizer1280 is NoneType!"
        ##########

        Tools.count = count 
        Tools.letter = ['easter egg' , 'a' , 'b' , 'c' , 'd' , 'e' , 'f' , 'g' , 'h' , 'i' , \
        'j' , 'k' , 'l' , 'm' , 'n' , 'o' , 'p' , 'q' , 'r' , 's' , 't' , 'u' , \
        'v' , 'w' , 'x' , 'y' , 'z']
#End of Tools class
