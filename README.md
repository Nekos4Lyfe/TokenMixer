# TokenMixer
The TokenMixer is an StableDiffusion extension for Automatic1111 webui for modifying embedding vectors and/or tokens. The TokenMixer consists of several modules in an integrated and adjustable interface. 

The main feature of the TokenMixer is the ability to find similar tokens to given input tokens. This can be done either as similar latent vectors or as semantically similar tokens using the python Natural Language Toolkit (NLTK) toolset. The latter feature includes the ability to translate tokens into other languages, as well as finding tokens that are more specific ("hypernyms") or less specific ("hyponyms") then the given input token.

This extension is still an early-access version. Informal discussions about the TokenMixer can be found here: https://discord.com/channels/1101998836328697867/1133883185101541548

![TokenMixer2](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/0285fba9-78d6-49fb-981b-92084786aa51)

The TokenMixer consists of 7 modules : 
 - Embedding Generator
 - MiniTokenizer
 - Embedding Inspector
 - Token Calculator
 - Token Extrapolator
 - Cross Attention Visualizer
 - Token Synonymizer
 
All of which are integrated into a single hub. Tokens can be transferred freely between the modules, allowing the artist to freely experiment with various vectors to find the exact type of token configuration they need for the current project. 


![Sample_mode2](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/f95aa750-cd80-410d-b852-fff382c42d04)


Feature summary:

Embedding Generator
- Creating exact blends between CLIP tokens , embeddings and other kinds of latent vectors

- Creating completely new tokens using semi-random sampling of a token or embedding vector

- Creating similar tokens/vectors of an existing vector/token
  
- Merging multiple tokens/vectors into a vector with the greatest similarity to all inputs

- Assigning token negatives that must be perpendicular to the generated token
  
- Concactinate tokens or embeddings into new embeddings

Minitokenizer
- Transform any prompt to tokens

- Split embeddings into individual vectors

- Send output to the Embedding generator

Token Calculator
- Perform numerical addition or subraction of tokens

Embedding inspector
- Find all similar tokens to embedding

Token Synonymizer
- Find all similar words in 20 languages using the NLTK Toolkit

![TokenMixer3](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/48b1b3e7-d2f3-40c4-81c6-d282326df130)




![Tokenmixer2](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/acd006f2-3e5a-4f2e-af1e-3f6d7e834385)





