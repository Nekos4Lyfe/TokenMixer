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
import torch , copy
TENSOR_DEVICE_TYPE = "cpu"
TENSOR_DATA_TYPE = torch.float32
START_OF_TEXT_ID = 49406
END_OF_TEXT_ID = 49407

class Token : 

  @staticmethod
  def is_random_type(input) :
        
    _ID = copy.copy(input)

    # Underscore IDs (replace with random vector)
    if _ID == 318 : return True  # = "_</w>" : 318
    if _ID == 62 : return True    #"_": 62
    if _ID == 2355 : return True  #"__": 2355
    if _ID == 4727 : return True  #"____": 4727
    if _ID == 9549 : return True  #"________": 9549
    if _ID == 14423 : return True #"^_": 14423
    if _ID == 20115 : return True #"________________": 20115
    if _ID == 42718 : return True #"-__": 42718
    if _ID == 43812 : return True #"___": 43812
    if _ID == 44144 : return True #">_": 44144
    if _ID == 13109 : return True #"_:</w>": 13109
    if _ID == 13530 : return True #"___</w>": 13530
    if _ID == 23193 : return True #"_-</w>": 23193
    if _ID == 25350 : return True #"____</w>": 25350
    if _ID == 26841 : return True #"_.</w>": 26841
    if _ID == 31417 : return True #"_,</w>": 31417
    if _ID == 36237 : return True #"_*</w>": 36237
    if _ID == 36951 : return True #"_(</w>": 36951
    if _ID == 37393 : return True # "_)</w>": 37393
    if _ID == 37647 : return True #"_/": 37647
    if _ID == 38803 : return True #"_____</w>": 38803
    if _ID == 44701 : return True #"_'</w>": 44701
    if _ID == 47043 : return True #"__:</w>": 47043
    return False
    ########### End of is_underscore_token()

  @staticmethod
  def is_start_of_text_type(input):
        
    _ID = copy.copy(input)

    if _ID == START_OF_TEXT_ID : return True # actual start-of-text token
    # '<' symbol IDs (replace with start-of-text token)
    if _ID == 27 : return True  #"<": 27
    if _ID == 283 : return True #"<</w>": 283
    if _ID == 16482 : return True #"<<</w>": 16482
    if _ID == 24588 : return True #"<<": 24588
    if _ID == 34308 : return True #"</</w>": 34308
    if _ID == 35054 : return True #"<<<</w>": 35054
    if _ID == 47280 : return True #"<-</w>": 47280
    return False
  ####### End of is_start_of_text_token()

  @staticmethod
  def is_end_of_text_type(input):
        
    _ID = copy.copy(input)

    if _ID == END_OF_TEXT_ID : return True # Actual end-of-text-token
    # '>' symbol IDs (replace with end-of-text token)
    if _ID == 29 : return True #">": 29
    if _ID == 285 : return True #"></w>": 285
    if _ID == 2988 : return True #">></w>": 2988
    if _ID == 6395 : return True #">>></w>": 6395
    if _ID == 8270 : return True #">>": 8270
    if _ID == 18435 : return True #">>>></w>": 18435
    if _ID == 18461 : return True #">>>>": 18461
    if _ID == 28015 : return True #"><</w>": 28015
    if _ID == 32972 : return True #">>>>></w>": 32972
    if _ID == 36196 : return True #">:</w>": 36196
    if _ID == 38507 : return True #">.<</w>": 38507
    if _ID == 41947 : return True #">>>>>>>>": 41947
    if _ID == 44144 : return True #">_": 44144
    if _ID == 48947 : return True #">>>>>></w>": 48947
    return False
  ####### End of is_end_of_text_token()

  def type_from(self , input):
    
    _ID = copy.copy(input)

    is_random = self.is_random_type(_ID)
    is_start_of_text = self.is_start_of_text_type(_ID)
    is_end_of_text = self.is_end_of_text_type(_ID)

    # Do some checks
    if is_random :
      assert not is_start_of_text , " logic error!"
      assert not is_end_of_text , " logic error!"
  
    if is_start_of_text : 
      assert not is_random , " logic error!"
      assert not is_end_of_text , " logic error!"

    if is_end_of_text :
      assert not is_start_of_text , " logic error!"
      assert not is_random , " logic error!" 
    ######

    return is_random , is_start_of_text , is_end_of_text

  # Returns False if _ID is a special type
  def is_normal_type(self, input) : 
    
    _ID = copy.copy(input)

    is_random , is_start_of_text , is_start_of_text = \
    self.type_from(_ID)
    is_special =  is_random or is_start_of_text or is_start_of_text
    if is_special : return False
    return True
  #######

  def __init__(self): 
    self.MAX_NUM_MIX = MAX_NUM_MIX 
    self.VEC_SHOW_TRESHOLD = VEC_SHOW_TRESHOLD
    self.VEC_SHOW_PROFILE = VEC_SHOW_PROFILE
    self.SEP_STR = SEP_STR
    self.SHOW_SIMILARITY_SCORE = SHOW_SIMILARITY_SCORE
    self.ENABLE_GRAPH = ENABLE_GRAPH
    self.GRAPH_VECTOR_LIMIT = GRAPH_VECTOR_LIMIT
    self.ENABLE_SHOW_CHECKSUM = ENABLE_SHOW_CHECKSUM
    self.REMOVE_ZEROED_VECTORS = REMOVE_ZEROED_VECTORS
    self.EMB_SAVE_EXT = EMB_SAVE_EXT
    self.TENSOR_DEVICE_TYPE = TENSOR_DEVICE_TYPE
    self.TENSOR_DATA_TYPE = TENSOR_DATA_TYPE
    self.START_OF_TEXT_ID = START_OF_TEXT_ID
    self.END_OF_TEXT_ID = END_OF_TEXT_ID


