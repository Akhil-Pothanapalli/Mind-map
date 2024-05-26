from keybert import KeyBERT
from transformers import BertModel, BertTokenizer
import torch
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import sys
from logger import logging
from bokeh.plotting import figure, show
from bokeh.models import CustomJS


kw_model = KeyBERT()

text = "Steve Rogers is the superhero called Captain America. He was given super soldier serum that make him bulk and improving all his physical attributes. His girlfriend name is Peggy Carter. He carries a shield made of Vibranium. Captain America is the leader for the band of superheroes called Avengers."
logging.info("Extracting keywords from text")
keywords = kw_model.extract_keywords(text, top_n=10)

# Print extracted keywords
logging.info("These are the keywords")
print(keywords)


# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to get embeddings
logging.info("Preparing embedding for the keywords")
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Get embeddings for keywords
keyword_embeddings = {kw: get_embedding(kw) for kw, _ in keywords}

# Calculate cosine similarities

similarity_matrix = cosine_similarity(
    [embedding.mean(dim=0).detach().numpy() for embedding in keyword_embeddings.values()]
)
logging.info("Cosine similarity matrix for embeddings calculated")


# Create a graph
logging.info("Initialized a empty graph")
G = nx.Graph()

# Add nodes
for kw, _ in keywords:
    G.add_node(kw)

# Add edges with similarity scores as weights
threshold = 0.8  # You can adjust this threshold
for i, kw1 in enumerate(keyword_embeddings):
    for j, kw2 in enumerate(keyword_embeddings):
        if i != j and similarity_matrix[i, j] > threshold:
            G.add_edge(kw1, kw2, weight=similarity_matrix[i, j])

logging.info(f'Graph created with a threshold of {threshold}')

# Print the graph nodes and edges

print("------------------------------------------------------------")
print(G.nodes())

'''
# Draw the graph
pos = nx.spring_layout(G)
weights = [G[u][v]['weight'] for u,v in G.edges()]
nx.draw(G, pos, with_labels=True, width=weights)
plt.show()
'''

# Function to calculate initial node positions (consider using a layout algorithm)
def get_initial_positions(G):
    # Replace this with your preferred layout algorithm (e.g., nx.spring_layout)
    pos = nx.spring_layout(G)
    return pos

# Create a Bokeh plot with initial node positions
pos = get_initial_positions(G)
p = figure(width=500, height=400, toolbar_location=None)

#p.circle(list(pos.values())[:, 0], list(pos.values())[:, 1], size=10, color='skyblue')
for node, (x, y) in pos.items():
    p.scatter(x, y, size=10, color='skyblue')

# Add edges to the plot
for edge in G.edges():
    u, v = edge
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    p.segment(x0=x1, y0=y1, x1=x2, y1=y2, line_width=2, line_color='gray')

# Define a JavaScript callback to update node positions on drag
node_drag_callback = CustomJS(
    args=dict(G=G, pos=pos),  # Pass the graph and initial positions
    code="""
        const graph = G;
        const positions = pos;
        const node_renderer = cb_obj.source.data['x'];  // Access node data

        const index = cb_obj.xdata.indices[0];  // Get the index of dragged node

        // Update node position in both graph and positions data
        graph.nodes[Object.keys(graph.nodes)[index]]['x'] = cb_obj.x;
        graph.nodes[Object.keys(graph.nodes)[index]]['y'] = cb_obj.y;
        positions[Object.keys(positions)[index]] = [cb_obj.x, cb_obj.y];

        // Update renderer data with new node positions
        node_renderer['x'][index] = cb_obj.x;
        node_renderer['y'][index] = cb_obj.y;
    """,
)

# Assign the callback to the node renderer
# Assuming your nodes are represented by a scatter renderer:
p.add_tools(CustomJS(args=dict(renderer=p.scatter, G=G, pos=pos), code=node_drag_callback))  # Add CustomJS tool

# Display the interactive graph
show(p)