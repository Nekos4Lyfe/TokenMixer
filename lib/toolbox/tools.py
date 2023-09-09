import gradio as gr
from modules import script_callbacks, shared, sd_hijack , sd_models , sd_hijack_open_clip , textual_inversion , xlmr


from transformers import XLMRobertaModel,XLMRobertaTokenizer

from modules.shared import cmd_opts
from pandas import Index
from pandas.core.groupby.groupby import OutputFrameOrSeries
import torch, os
from modules.textual_inversion.textual_inversion import Embedding
from pprint import pprint

import collections, math, random , numpy
import re #used to parse string to int
import copy
from torch.nn.modules import ConstantPad1d, container

import warnings

from lib.toolbox.constants import MAX_NUM_MIX 
#-------------------------------------------------------------------------------

class Tools :
  #The class tools contain built in functions for handling 
  #tokens and embeddings
      def make_emb_folder(self , name):
        savepath = os.path.join(cmd_opts.embeddings_dir, name)
        try: os.makedirs(savepath)
        except: pass
        try: modules.sd_hijack.model_hijack.embedding_db.add_embedding_dir(savepath)
        except: pass
        return savepath

      def get(self):

        if not hasattr(shared, 'sd_model'):
          return None , None , None , None , None , None

        #Check if a valid model is loaded
        model = shared.sd_model
        if model == None : 
          return None , None , None , None , None , None

        is_sdxl = hasattr(model, 'conditioner')
        is_sd2 = hasattr(model.cond_stage_model, 'model')
        is_sd1 = not is_sd2 and not is_sdxl

        valid_model = is_sd2 or is_sd1 or is_sdxl
        if not valid_model:
          return None , None , None , None , None , None
        ########

        #Fetch the loaded embeddings
        loaded_embs = collections.OrderedDict(
        sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.items(),
            key=lambda x: str(x[0]).lower()))
        ######

        #Fetch the internal_embeddings
        embedder = None
        if is_sdxl: embedder = model.cond_stage_model.embedders[0].wrapped
        else: embedder = model.cond_stage_model.wrapped
        #####
        internal_emb_dir = None
        if is_sd2 : internal_emb_dir = internal_emb_dir = embedder.model
        else : internal_emb_dir = embedder.transformer.text_model.embeddings
        ####
        internal_embs = internal_emb_dir.token_embedding.wrapped.weight 
        #########

        #Fetch the tokenizer
        tokenizer = None
        if is_sd2 :
          from modules.sd_hijack_open_clip import tokenizer as open_clip_tokenizer
          tokenizer = open_clip_tokenizer
        else: tokenizer = embedder.tokenizer
        ########

        #fetch the internal sdxl embs
        internal_sdxl_embs = None
        sdxl_tokenizer = None
        if is_sdxl : 
          FrozenOpenCLIPEmbedder2 = model.cond_stage_model.embedders[1].wrapped
          internal_sdxl_embs = FrozenOpenCLIPEmbedder2.model.token_embedding.wrapped.weight
          sdxl_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        ########

        #Fetch the SDXL extra stuff (if SDXL model is loaded)
        #sdxl_tokenizer = None
        
        #sdxl_internal_embs = None
        #pprint("####model.cond_stage_model.embedders.wrapped[0]####")
        #pprint(vars(model.cond_stage_model.embedders[0].wrapped))
        #pprint("####model.cond_stage_model.embedders[1]####")
        #pprint(vars(model.cond_stage_model.embedders[1]))
        #sdxl_embedder = model.cond_stage_model.embedders[1].wrapped
        ####
        #if False:
        #  sdxl_internal_emb_dir = sdxl_embedder.transformer.text_model.embeddings
        #  sdxl_internal_embs = sdxl_internal_emb_dir.token_embedding.wrapped.weight
        #  sdxl_tokenizer = sdxl_embedder.tokenizer
        #######

        return tokenizer , internal_embs , loaded_embs , is_sdxl , internal_sdxl_embs , sdxl_tokenizer

      def update_loaded_embs(self):
        tokenizer , internal_embs ,  loaded_embs , is_sdxl , \
        internal_sdxl_embs , sdxl_tokenizer = self.get()
        Tools.loaded_embs = loaded_embs

      def get_subname(self):
        self.count = copy.copy(self.count + 1)
        if self.count >= len(self.letter):
          self.count = 1
        return self.letter[self.count]

      #def sdxl_encode(self,c):
      #  return self.sdxl.encode(c)

      def get_best_ids (self , emb_id , similarity , max_similar_embs , emb_vec = None) :
        
        max_sim = copy.copy(max_similar_embs)
        simil = copy.copy(similarity)
        ID = copy.copy(emb_id)
        internal_embs = self.internal_embs.to(device='cpu',dtype=torch.float32)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        all_sim = None
        scores = None
        sorted_scores = None
        sorted_ids = None
        best_ids = None
        vector = None

        if emb_vec != None:
          vector = emb_vec.to(device='cpu',dtype=torch.float32)
        elif isinstance(ID, int):
          vector = internal_embs[ID]
        else: return None , None

        all_sim = cos(internal_embs , vector)
        scores = torch.mul(all_sim , torch.ge(simil * torch.ones(all_sim.shape) , 100*all_sim))
        sorted_scores, sorted_ids = torch.sort(scores, descending=True)
        best_ids = sorted_ids[0:max_sim].detach().numpy()
        return  best_ids , sorted_scores

      def __init__(self , count=1):

        #Tools.sdxl = xlmr.BertSeriesModelWithTransformation()

        #Fetch embeddings and tokenizer(s)
        Tools.tokenizer = None
        Tools.sdxl_tokenizer = None
        Tools.internal_embs = None
        Tools.internal_sdxl_embs = None
        Tools.loaded_embs = None
        Tools.no_of_internal_embs = 0
        Tools.is_sdxl = False
        tokenizer , internal_embs ,  loaded_embs , is_sdxl , \
        internal_sdxl_embs , sdxl_tokenizer = self.get()
        #####

        #Check if embeddings and tokenizers were loaded properly
        Tools.loaded = True
        if tokenizer == None or internal_embs == None:
          print("TokenMixer could not load model params")
          self.loaded = False
        #####

        #Setup the Tools class for later use
        if self.loaded:
          Tools.tokenizer = tokenizer
          Tools.internal_embs = internal_embs
          Tools.internal_sdxl_embs = internal_sdxl_embs
          Tools.loaded_embs = loaded_embs
          Tools.emb_savepath = self.make_emb_folder('TokenMixer') 
          Tools.no_of_internal_embs = len(internal_embs)
          Tools.no_of_sdxl_internal_embs = len(internal_sdxl_embs)
        ######

        assert sdxl_tokenizer != None , "sdxl_tokenizer is NoneType"
        self.is_sdxl = is_sdxl
        self.sdxl_tokenizer = tokenizer #<<<<--- NOTE

        #SDXL quick fix
        if is_sdxl: pass
        ########

        #Quick Hack
        #if self.loaded and is_sdxl: 
          #self.internal_embs = internal_sdxl_embs
          #self.no_of_internal_embs = self.no_of_sdxl_internal_embs
          #pprint("#####  internal_embs ######")
          #pprint(vars(self.internal_embs))
          #pprint("##########")
          #pprint("#####  internal_sdxl_embs ######")
          #pprint(vars(self.internal_sdxl_embs))
          #pprint("##########") 
        ######

        Tools.count = count 
        Tools.letter = ['easter egg' , 'a' , 'b' , 'c' , 'd' , 'e' , 'f' , 'g' , 'h' , 'i' , \
        'j' , 'k' , 'l' , 'm' , 'n' , 'o' , 'p' , 'q' , 'r' , 's' , 't' , 'u' , \
        'v' , 'w' , 'x' , 'y' , 'z']
#End of Tools class