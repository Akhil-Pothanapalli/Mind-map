Mind Map Generator 🧠✨

Introduction
Welcome to the Mind Map Generator project! This application is designed to transform text input into a dynamic mind map, showcasing the relationships between key concepts. Unlike traditional mind map tools, our model uses AI to create nodes and edges based on the provided text, aiming to improve understanding through visual interconnections.

Why Another Mind Map Model?
Graph Structure Superiority: Traditional mind maps often prioritize tree structures. Our approach emphasizes graph structures to better display interdependencies.
Text-Based Node Creation: Instead of relying solely on predefined knowledge, our AI generates nodes and edges directly from your text input.

Features & Goals
Multimodal Input: Future versions will support different types of input, not just text.
Human-like Appearance: A toggle to switch between computer-generated and human-like text.
Customizable Complexity: A slider to adjust the number of keywords for varying complexity.
Mind Map Merging: Merge existing mind maps into a comprehensive one.

Approach and Development Journey
Initial Concept 💡
Initially, the goal was to train a neural network to generate mind maps directly from text. However, due to the subjective nature of mind maps and the difficulty in acquiring enough training data, the approach shifted.

Keyword Extraction
1. Tools Used:
KeyBERT: For keyword extraction.
WordNetLemmatizer: To merge similar keywords.
Spacy: For text processing and generating embedding vectors.
2. Challenges:
Direct model loading impacted storage.
Alternatives like top2vec are planned for future enhancements.

Visualization and Interaction 🖼️🖱️
Interactive Nodes: Users can interact with the nodes to see connections.
Emojis and Images: Adding visual elements to nodes.
Edge Visualization: Displaying edge values and their information through color, thickness, and shape.
Editable Graph: Modifying the output graph will help fine-tune the model.

Current Status 🛠️
The application successfully generates a graph from the input text with some limitations:
Relations are based on word embeddings; an improved algorithm is needed.
Nodes are not fully interactive yet.
Emojis and images need better integration.
Edge values require clearer visualization.
The ability to add/remove edges is under development.
Edits to the graph should update the model, enhancing its learning.

Implementation Details
File Structure
1. app.py: Main application file.
2. discardedapp.py: Previous version highlighting the development journey and changes in approach.
Usage
1. Run the Application:
bash
Copy code
python app.py
2. Enter Text:
Input the text for which you want to generate a mind map.
3. View Output:
The generated mind map will be saved as an HTML file (keyword_similarity_graph.html).

Key Technologies
Python: Core programming language.
KeyBERT: Keyword extraction.
BERT: Pre-trained language model.
NetworkX: Graph creation and manipulation.
PyVis: Visualization of the graph.
NLTK: Natural Language Toolkit for text processing.

Future Enhancements
Improved Node Interaction: Making nodes draggable and more interactive.
Enhanced Visual Elements: Better integration of images and emojis.
Advanced Edge Information: Adding more details to the edges.
Graph Editing: Allowing users to modify the graph and update the model accordingly.

Resources
Mind Map Creation Video: Watch Here
GRIND Mnemonic: Grouped, Reflective, Interconnected, Non-verbal, Directional, and Emphasized.