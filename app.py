from keybert import KeyBERT
from transformers import BertModel, BertTokenizer
import torch
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Initialize KeyBERT
kw_model = KeyBERT()

text = "Steve Rogers is the superhero called Captain America. He was given super soldier serum that make him bulk and improving all his physical attributes. His girlfriend name is Peggy Carter. He carries a shield made of Vibranium. Captain America is the leader for the band of superheroes called Avengers."

# Extract keywords
keywords = kw_model.extract_keywords(text, top_n=10)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to get embeddings
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

# Create a graph
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

# Function to calculate initial node positions
def get_initial_positions(G):
    pos = nx.spring_layout(G)
    return pos

# Create a plot with initial node positions
pos = get_initial_positions(G)
plt.figure(figsize=(12, 8))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')

# Draw edges
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

# Display the graph
plt.title('Keyword Similarity Graph')
plt.show()
