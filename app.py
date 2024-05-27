from keybert import KeyBERT
from transformers import BertModel, BertTokenizer
import torch
import networkx as nx
#import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from logger import logging
from bokeh.plotting import figure, show
from bokeh.models import CustomJS, ColumnDataSource


kw_model = KeyBERT()

text = "Steve Rogers is the superhero called Captain America. He was given super soldier serum that make him bulk and improving all his physical attributes. His girlfriend name is Peggy Carter. He carries a shield made of Vibranium. Captain America is the leader for the band of superheroes called Avengers."
logging.info("Extracting keywords from text")
keywords = kw_model.extract_keywords(text, top_n=10)

# Print extracted keywords
logging.info("keywords extracted")
#print(keywords)

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
#print("------------------------------------------------------------")
#print(G.nodes())

# Function to calculate initial node positions (consider using a layout algorithm)
def get_initial_positions(G):
    pos = nx.spring_layout(G)
    return pos

# Create a Bokeh plot with initial node positions
pos = get_initial_positions(G)
p = figure(width=1000, height=600, toolbar_location=None)

# Prepare data for Bokeh
node_indices = list(pos.keys())
x = [pos[node][0] for node in node_indices]
y = [pos[node][1] for node in node_indices]

# Create ColumnDataSource for nodes
source = ColumnDataSource(data=dict(x=x, y=y, name=node_indices))

# Add nodes to plot
p.scatter('x', 'y', size=10, color='skyblue', source=source)

# Add edges to the plot
for edge in G.edges():
    u, v = edge
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    p.segment(x0=[x1], y0=[y1], x1=[x2], y1=[y2], line_width=2, line_color='gray')

# Define JavaScript callback code as a string
node_drag_callback_code = """
    const positions = pos;
    const renderer = cb_obj;
    const node_renderer = renderer.data_source;

    const index = node_renderer.selected.indices[0];  // Get the index of dragged node

    // Update node position in positions data
    const node_name = node_renderer.data['name'][index];
    positions[node_name] = [node_renderer.data['x'][index], node_renderer.data['y'][index]];

    // Update renderer data with new node positions
    renderer.data_source.change.emit();
"""

# Define the CustomJS callback
node_drag_callback = CustomJS(
    args=dict(pos=pos, renderer=source),
    code=node_drag_callback_code
)

# Assign the callback to the node renderer
source.js_on_change('data', node_drag_callback)

# Display the interactive graph
show(p)
