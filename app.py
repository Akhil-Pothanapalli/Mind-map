from keybert import KeyBERT
from transformers import BertModel, BertTokenizer
import torch
import sklearn
import networkx as nx
import matplotlib.pyplot as plt


kw_model = KeyBERT()

text = "Your input text goes here."
keywords = kw_model.extract_keywords(text, top_n=10)

# Print extracted keywords
print(keywords)


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
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(
    [embedding.detach().numpy() for embedding in keyword_embeddings.values()]
)


# Create a graph
G = nx.Graph()

# Add nodes
for kw, _ in keywords:
    G.add_node(kw)

# Add edges with similarity scores as weights
threshold = 0.5  # You can adjust this threshold
for i, kw1 in enumerate(keyword_embeddings):
    for j, kw2 in enumerate(keyword_embeddings):
        if i != j and similarity_matrix[i, j] > threshold:
            G.add_edge(kw1, kw2, weight=similarity_matrix[i, j])

# Print the graph nodes and edges
print(G.nodes())
print(G.edges(data=True))


# Draw the graph
pos = nx.spring_layout(G)
weights = [G[u][v]['weight'] for u,v in G.edges()]
nx.draw(G, pos, with_labels=True, width=weights)
plt.show()