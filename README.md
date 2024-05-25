# Mind-map

When we learn something new, we learn them in discrete but seldom we know how these components in the big picture. My goal is to create a model that takes in text input and generates a mind map out of it. 

Why another mindmap model?
1. Market models ranks tree structure higher than graph structure. Mind maps are great because they show the interdependecies.
2. AI is used to create branches based on its knowledge, but I want the AI to create these nodes and edges based on the text provided.

I have a very little idea to manifest this. But I will gather enough resources to push this forward. 

Tips for moving forward:
How mindmaps are usaully created?
What the best/ optimal way to create a mindmap?

Future of mindmaps:
Take multimodal input.
A toggle switch to turn the computer generated words to human writing in appearance.
A slider to select the number of keywords, so the mind map can vary from simple to complex. 
Merge exisiting mindmaps into a major mindmap.


Inorder to create an effective mindmap, this video helped me to initiate my journey:
https://youtu.be/5zT_2aBP6vM?si=S0Hgtv_lCAqFGfXz

The key points are 6-step mnemonic called GRIND: Grouped, Reflective, Interconnected, Non-verbal, Directional, and Emphasized.

1. Grouped: Organize your ideas into categories. This will help you see the relationships between different concepts.

Now preparing the data to do so, is nearlly impossble, because mind maps are subjective and acquiring so called data is not happening as of now. So I'm changing my approach from training the neural network on input text to output mind map to extracting key words on from data, finding their dependecies and ranking the words.

During my hunt for solution this step, I came across KeyBERT and top2Vec. Initially I will implement KeyBERT. top2Vec is planned for future version. Followed by them I will cluster them so that I can find the sub topics this solves the first one - grouped.

I've successfully implemented the keyword extractor, but loading the model directly is creating impact, to the storage, next use API's.

2. Reflective: Your mind map should reflect how you think, not how you write linearly. Don't be afraid to experiment with different layouts.

For this I will use top2vec in future.

When it comes to reflective, I want it to be a graph rather than tree also try different ones so I can find better version. I can take reference from random forest. But this is where humans excel over machines.

3. Interconnected: Look for connections between different ideas, even if they seem distant at first. This will help you create a more holistic understanding of the topic.

Interconnected - this is somewhat similar to graph I need to connect them based on cause-effect, who, where etc., there can n types or so.

4. Non-verbal: Use images, symbols, and doodles instead of words whenever possible. This will help you to remember information more easily.

Non verbal - this is where hugging face models comes handy, any models that creates doodles or stable diffusion, emojis, and symbols.

5. Directional: Arrange your ideas in a way that shows the flow of information. This will help you to see the cause-and-effect relationships between different concepts.

Direction - I should make the edges convey more info, they can different types not just arrow, double weighted arrows

6. Emphasized: Use visual cues, such as color and font size, to highlight the most important points in your mind map. This will help you to focus your attention when you are reviewing your notes.

Emphasied - color, font size ( this is finding the characterstics)