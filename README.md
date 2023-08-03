# TokenMixer
The TokenMixer is an StableDiffusion extension for Automatic1111 webui for modifying embedding vectors and/or tokens. The TokenMixer consists of several modules in an integrated and adjustable interface. 

The main feature of the TokenMixer is the ability to find similar tokens to given input tokens. This can be done either as similar latent vectors or as semantically similar tokens using the python Natural Language Toolkit (NLTK) toolset. The latter feature includes the ability to translate tokens into other languages, as well as finding tokens that are more specific ("hypernyms") or less specific ("hyponyms") then the given input token.

Prior to running this extension, check that the Python NLTK package is installed on your device. 

For Mac/Linux users : 'pip install --user -U nltk'

For Windows users, please refer to this guide : https://www.nltk.org/install.html

This extension is still an early-access version. Informal discussions about the TokenMixer can be found here: https://discord.com/channels/1101998836328697867/1133883185101541548

![TokenMixer2](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/0285fba9-78d6-49fb-981b-92084786aa51)

The TokenMixer consists of 7 integrated modules: 
 - Embedding Generator
 - MiniTokenizer
 - Embedding Inspector
 - Token Calculator
 - Token Extrapolator
 - Cross Attention Visualizer
 - Token Synonymizer
 
All of which are integrated into a single hub. Tokens can be transferred freely between the modules, allowing the artist to freely experiment with various vectors to find the exact type of token configuration they need for the current project. 

![Sample_mode2](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/f95aa750-cd80-410d-b852-fff382c42d04)

# Feature summary:

Embedding Generator
- Creating exact blends between CLIP tokens , embeddings and other kinds of latent vectors

- Creating completely new tokens using semi-random sampling of a token or embedding vector

- Creating similar tokens/vectors of an existing vector/token
  
- Merging multiple tokens/vectors into a vector with the greatest similarity to all inputs

- Assigning token negatives that must be perpendicular to the generated token
  
- Concactinate tokens or embeddings into new embeddings
  
- Create batches of different embeddings from a given input

Minitokenizer
- Convert any prompt to tokens
- Split embeddings into individual vectors
- Send output to the Embedding generator

Token Calculator
- Perform numerical addition or subraction of tokens
- Remove or add traits to a given token, i.e
  'King' - 'Man' + 'Woman' to get a 'Queen' token

Embedding inspector
- Find all similar tokens to a given token or embedding vector
- Find all similar IDs to a given token or embedding vector
- Credit to: https://github.com/tkalayci71/embedding-inspector

Token Synonymizer
- Find all similar words in 20 languages using the NLTK Toolkit
- Find Holonyms, Hypernyms , Meronyms , synonyms and Entailments to a given word

Token Extrapolator
- Expand a single or a pair of tokens into multiple tokens
- Navigate the ID list in the CLIP tokenizer
- Randomly sample the ID list in the CLIP tokenizer

Embedding Visualizer
- See which areas of a given input image is targeted by a given embeddings
- Credit : https://github.com/benkyoujouzu/stable-diffusion-webui-visualize-cross-attention-extension

![Synonymizer_2 0](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/5e7d9c1d-4189-4a10-af99-31780299b9ba)

This ReadMe file is a work in progress





