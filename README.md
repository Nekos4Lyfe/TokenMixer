# TokenMixer
The TokenMixer is an StableDiffusion extension for Automatic1111 webui for modifying embedding vectors and/or tokens. The TokenMixer consists of several modules in an integrated and adjustable interface. 

The main feature of the TokenMixer is the ability to find similar tokens to given input tokens. This can be done either as similar latent vectors or as semantically similar tokens using the python Natural Language Toolkit (NLTK) toolset. The latter feature includes the ability to translate tokens into other languages, as well as finding tokens that are more specific ("hypernyms") or less specific ("hyponyms") then the given input token. 

##Example of "Sample Mode" is shown below:

![new Sample Mode](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/2b0adb78-49fa-4bfe-8c65-c4a3af306f15)

This extension is still an early-access version. Bug reports are welcome. Informal discussions about the TokenMixer can be found here (You will need to navigate to the "resources" Discord tab to find the "TokenMixer" section): https://discord.gg/NFUmXn4W

Disclaimer : This extension is not compatible with SDXL yet

If you have ideas/suggestions/questions about the user interface , please make a post under the "Issues" tab. 

Here is an example of "Roll Mode". This feature performs torch.roll on the input tensors to generate new embeddings.
The process is incredibly quick. It takes roughly 5 minutes to generate 768 embeddings:

![RM](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/81bd38cc-6f7b-40dd-9c85-ccbe59fed03e)

Choosing the embedding at index 3 we get:

![RM2](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/63b0c875-afce-4972-ab2d-835b693ecbdc)


Here is an image showing the UI in its current format (12th of August):

![TMUI2](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/71da0bb7-79d3-41b5-bda4-0d7cf22fe848)

# Feature summary:

The TokenMixer consists of 7 integrated modules: 
 - Embedding Generator
 - MiniTokenizer
 - Embedding Inspector
 - Token Calculator
 - Token Extrapolator
 - Cross Attention Visualizer
 - Token Synonymizer
 - CLIP image encoder
 
 Tokens can be transferred freely between the modules, allowing the artist to freely experiment with various vectors to find the exact type of token configuration they need for the current project. 


###Embedding Generator
- Creating exact blends between CLIP tokens , embeddings and other kinds of latent vectors

- Creating completely new tokens using semi-random sampling of a token or embedding vector

- Creating similar tokens/vectors of an existing vector/token
  
- Merging multiple tokens/vectors into a vector with the greatest similarity to all inputs

- Assigning token negatives that must be perpendicular to the generated token
  
- Concactinate tokens or embeddings into new embeddings
  
- Create batches of different embeddings from a given input

###Minitokenizer
- Convert any prompt to tokens
- Split embeddings into individual vectors
- Send output to the Embedding generator

###Token Calculator
- Perform numerical addition or subraction of tokens
- Remove or add traits to a given token, i.e
  'King' - 'Man' + 'Woman' to get a 'Queen' token

###Embedding inspector
- Find all similar tokens to a given token or embedding vector
- Find all similar IDs to a given token or embedding vector
- Credit to: https://github.com/tkalayci71/embedding-inspector

###Token Synonymizer
- Find all similar words in 20 languages using the NLTK Toolkit
- Find Holonyms, Hypernyms , Meronyms , synonyms and Entailments to a given word

###Token Extrapolator
- Expand a single or a pair of tokens into multiple tokens
- Navigate the ID list in the CLIP tokenizer
- Randomly sample the ID list in the CLIP tokenizer

###Embedding Visualizer
- See which areas of a given input image is targeted by a given embeddings
- Credit : https://github.com/benkyoujouzu/stable-diffusion-webui-visualize-cross-attention-extension

###CLIP image encoder
-Get the cross modal embedding which is used by CLIP to find similar tokens

# How to use

##Installation 
![Step_1](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/21647bd9-11f4-4b3c-999b-a5aee6e0249a)

First paste the string for the git repository in the "Extension Hub" . While you are at it , you can install the following extensions as well which work will with TokenMixer  (optional) : 
https://github.com/ashen-sensored/sd_webui_SAG
https://github.com/adieyal/sd-dynamic-prompts
https://github.com/hnmr293/sd-webui-cutoff
https://github.com/Bing-su/adetailer 

Before reloading the UI ,  it is recommended that you verify that values are upcast to float32 
![Step_3](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/55e499d7-3749-4c80-ab6d-4852b58aafd7)

You do this under the "Settings" tab for the UI
(TokenMixer still works for the lower settings , although results can be a bit "duller" , if that makes sense)

![Step_2](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/8534c860-ac45-4950-a05e-7aa664007d4e)

Now you can reload the UI. You will now see TokenMixer at the tab.

![Step_7](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/addde7b4-262c-46bf-bb49-8c2f59df0532)

This is what the UI looks like

![Step_8](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/30b42860-2d3d-44b5-977d-341fde9fdf1c)

You can input any prompt you like to the Minitokenizer on the left. Press the orange button. 
The Tokens will now be sent to the embedding generator on the right

![Step_9](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/1195debc-a4da-434f-92d3-b24862afe610)

Press the button on the embedding generator to make the embedding. 
You can now input the embedding in the prompt box to get the desired image

![Step10 ](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/5a364d1b-1715-4697-9aec-fb14e31273cb)

Next, we can look at how to automate the process. Select the Token Extrapolator.
It can be found under the tab "tokex" in the menu (module can be added to any tab using the "Add Moduel" tab)

![Step11](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/52635fdb-480b-40e7-bc6f-6f97ac0399c1)

Input the first token in the extrapolator and press generate. You can set the pursuit strength of the extrapolation at the top bar. Leaving the "End token" input blank will place a random token

![Step12](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/4f865303-3aeb-4121-b82f-bd7265bd21de)

By pressing generate, the extrapolator output can be sent to the generator. It can be saved as an embedding like before

![Step13](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/2fbcaec9-665a-42ca-bf8e-d09c2e163c0c)

Result will be an image that has words which often appear close together with "action

![Step14](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/fe78a244-7a69-4ef3-bf25-c2d7c3ca209e)

If we wish to extrapolate the tokens "action" and "movie" into 5 tokens , we can set the pursuit strength to a more normal value. This will create tokens which appear in between the words "action" and "movie". 

![Step15](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/d5356df4-0063-4047-833f-fe7ccc271edc)

And the output is what you see above.

##Roll Mode

Next , the "Roll Mode" feature and how this can be used with the Token extrapolator previously mentioned
Open the generator again and select "Roll Mode" , "Full range" and if you wish the % to curb the full range of embeddings (768 in total for SD 1.5)

![Step_16](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/72974c59-20cc-43fd-b699-d57d2f33e6f8)

Hit the generate button and you will see the result in the image above. 
You can now open the XYZ feature in text to image , and paste the output string from the Embedding generator. Select "Prompt S/R"

![Step_17](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/07e9ac17-28c2-41de-9011-76b93340eede)

These are the settings you can use for prompt rolling. As the vectors generated are often "obscure" , it is advised to NOT match the rolled tokens against a normal prompt , as the normal prompt will "overpower" the rolled embedding.

![Step_18](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/ef0c160a-0e1a-4f07-9a53-b4288d43b4b1)

For solver I recommend starting with DPM2 Karras. This is a second-order solver that is slower then the other solvers but more accurate, meaning it can better "follow" the curve of the obscure embedding tokens. Using an achestral sampler (DPM2 a , Euler a etc) is useful to remove defects like bad fingers etc. Drawback is that it will limit the effects of prompt switching, or "from : to : steps" as they are currently referred as.
You can read more about samplers and their benefits here : https://stable-diffusion-art.com/samplers/
Link to stable diffusion wiki where "from : to : steps" are explained more detail : https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features

![grid !](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/a4ccdffb-621d-4662-9e78-6b062cb1a918)

Here we see the result of the prompt roll. The prompt roll is the torch.roll() function for the input vectors for N steps, where N is the number in the XYZ plot

What you need to know is that the the closer the roll number is to the original prompt (N = 0) , the more "similar in nature" the output will be. 
(Though this is latent space, so things do not always make sense in terms of structure)

Due to the "weakness" of the rolled embeddings , they are highly malleable by the SD model.  You might have heard the term "LoRa" . This is a "Low-Rank-Adaptation-Layer" which is a fancy term for a set of matrices which are placed between the matrices in the SD model , a "layer"
A quirk in the SD 1.5 model is that is dislikes light contrast. It has plenty of training data that contains high contrast settings , but it still wants to keep the "light level" evenly distributed throughout the whole image
The same quirk applies for color. You will note colors in these photos are very "plain" so to speak
First , we will add a detailer LoRa with offset noise to see the change in output:

![grid 2](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/e19203d8-9fc4-41b1-a28c-c6fcd9b21a80)

The malleability on the input tokens on the LoRa allows us to create excellent detail without having to resort to the usual "masterpiece" nonsense
The light constrast is still an issue however. We solve this by using so called "prompt switching"
This is a term for first prompting a setting with high contrast in color or light , and then switching the prompt to the real prompt past roughly 20% of the steps
My preference is "dark cave with x illumination" where x is a color

![grid 3](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/9aa487f2-7a5e-40c0-be8d-77302af3acc0)

"dark cave with green illumination"

![grid 4](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/bf58e91c-3f45-4f33-91e4-b55863e9f373)

"dark cave with azure illumination"

![grid 5](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/7e96c1ca-69c2-438f-921b-174edf92c6c5)

"dark cave with red illumination"

![grid 6](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/167b7bd7-3b0e-4bc1-8002-7d07dc22c5d2)

"dark cave with yellow illumination"

![grid 7](https://github.com/Nekos4Lyfe/TokenMixer/assets/130230016/f0aca2a0-6115-4e7d-9226-623f7185d23a)

"candy color splash swirl"












