MAX_NUM_MIX = 400 # number of embeddings that can be mixed
VEC_SHOW_TRESHOLD = 1 # change to 10000 to see all values
VEC_SHOW_PROFILE = 'default' #change to 'full' for more precision
SEP_STR = '-'*60 # separator string
SHOW_SIMILARITY_SCORE = False # change to True to enable
ENABLE_GRAPH = False
GRAPH_VECTOR_LIMIT = 8 # max number of vectors to draw in graph
ENABLE_SHOW_CHECKSUM = False #slows down listing loaded embeddings
REMOVE_ZEROED_VECTORS = False #optional
EMB_SAVE_EXT = '.pt' #'.bin'


import torch
TENSOR_DEVICE_TYPE = "cpu"
TENSOR_DATA_TYPE = torch.float32

START_OF_TEXT_ID = 49406
END_OF_TEXT_ID = 49407