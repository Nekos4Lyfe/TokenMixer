# TokenMixer
The TokenMixer is an StableDiffusion extension for Automatic1111 webui for modifying embedding vectors and/or tokens. The TokenMixer consists of several modules in an integrated and adjustable interface. 

The main feature of the TokenMixer is the ability to find similar tokens to given input tokens. This can be done either as similar latent vectors or as semantically similar tokens using the python Natural Language Toolkit (NLTK) toolset. The latter feature includes the ability to translate tokens into other languages, as well as finding tokens that are more specific ("hypernyms") or less specific ("hyponyms") then the given input token.

This extension is still an early-access version. Informal discussions about the TokenMixer can be found here: https://discord.com/channels/1101998836328697867/1133883185101541548

![TokenMixer2](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/0285fba9-78d6-49fb-981b-92084786aa51)

Disclaimer : This extension is not compatible with SDXL yet


The TokenMixer consists of 7 integrated modules: 
 - Embedding Generator
 - MiniTokenizer
 - Embedding Inspector
 - Token Calculator
 - Token Extrapolator
 - Cross Attention Visualizer
 - Token Synonymizer
 - CLIP image encoder
 
All of which are integrated into a single hub. Tokens can be transferred freely between the modules, allowing the artist to freely experiment with various vectors to find the exact type of token configuration they need for the current project. 

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

CLIP image encoder
-Get the cross modal embedding which is used by CLIP to find similar tokens

# Demonstration

The following image will be used as a baseline for this demonstration:
![ref2](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/15ce5e27-c224-4c81-80a1-c9a3c88adb2b)

Prompt here is ´blonde girl with red bikini´ . 

Some key problems with this prompt:
- Keywords are very generic
- We get the same (boring) result for every type of seed 

We can create an embedding for this prompt using the MiniTokenizer.

![minit1](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/a8d54c2e-e64e-44d5-90ba-8eedea25b9cc)

The image above can be created by writing "reference" into the prompt (same as you would do with any other embedding)

By feeding the "reference" back into the Minitokenizer , we can see that it is an embedding with 4 vectors:

![minit2](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/f1d65aca-2d16-42c1-a885-b47f35389d4e)

We can extract vectors from the embedding using the [a:b] indexing, where 'a' is the number of first token we want to include and 'b' the end token.
By writing reference[1:3] we get:

![minit3](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/f068d203-dab5-48b8-86b2-36b269405d6d)

This split can be done with any embedding or textual inversion. We can investigate the embedding "reference"  by feeding it into the the Embedding Inspector:

![embin1](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/0348f7fb-fdc9-4d20-88ba-66bcf94a6077)

So far so good! But how can we make this prompt more interesting? The conventional approach would be to add more "stuff" in the prompt input. 
Or, alternatively replace the prompt words with synonyms. The TokenMixer is capable of doing both of these things automatically, but it's 
main feature is the ability to alter the input tokens themselves. 

The most straight forward approach is to use the 'Sample Mode' feature. For a given vector X we calculate a new vector using the formula given below:

![Sample Mode](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/61ebe7c1-f7ba-46c3-8c7c-295e93fcfd8c)

The results can be seen below:

![Sample_mode2](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/f95aa750-cd80-410d-b852-fff382c42d04)

---------

![Synonymizer_2 0](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/5e7d9c1d-4189-4a10-af99-31780299b9ba)

This ReadMe file is a work in progress





