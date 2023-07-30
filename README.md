# TokenMixer
The TokenMixer is an StableDiffusion extension for Automatic1111 webui
![TokenMixer2](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/0285fba9-78d6-49fb-981b-92084786aa51)

This extension will allow you to modify tokens at will and take full control of the latent space. 

The TokenMixer consists of 7 modules : 
 - Embedding Generator
 - MiniTokenizer
 - Embedding Inspector
 - Token Calculator
 - Token Extrapolator
 - Cross Attention Visualizer
 - Token Synonymizer
 
All of which are integrated into a single hub. Tokens can be transferred freely between the modules, allowing the artist to freely experiment with various vectors to find the exact type of token configuration they need for the current project. 

Feature summary:

--Embedding Generator
*Creating exact blends between CLIP tokens , embeddings and other kinds of latent vectors

*Creating completely new tokens using semi-random sampling of a token or embedding vector

*Creating similar tokens/vectors of an existing vector/token

*Merging multiple tokens/vectors into a vector with the greatest similarity to all inputs

*Assigning token negatives that must be perpendicular to the generated token

*Concactinate tokens or embeddings into new embeddings

--Minitokenizer
*Transform any prompt to tokens

*Split embeddings into individual vectors

*Send output to the Embedding generator

--Token Calculator
*Perform numerical addition or subraction of tokens

*Send output to the Embedding generator

--Embedding inspector
*Find all similar tokens to embedding


![TokenMixer3](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/48b1b3e7-d2f3-40c4-81c6-d282326df130)

An A1111 extension for interpolating tokens into embeddings. 
If you are reading then you are seeing an early version of this extension. 


![Tokenmixer2](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/acd006f2-3e5a-4f2e-af1e-3f6d7e834385)





