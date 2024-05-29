import os
import networkx as nx
from collections import defaultdict
from keybert import KeyBERT
from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import nltk
import emoji
from pyvis.network import Network

# Constants
FONT_PATH = r"fonts\NotoColorEmoji.ttf"
HTML_FILE = "keyword_similarity_graph.html"

# Delete existing HTML file if it exists
if os.path.exists(HTML_FILE):
    os.remove(HTML_FILE)

# Initialize KeyBERT
kw_model = KeyBERT()

# User input for text
text = input("Enter your text: ")

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

# Use a lemmatizer to merge similar keywords
lemmatizer = WordNetLemmatizer()
lemmatized_keywords = defaultdict(list)
for kw, score in keywords:
    lemma = lemmatizer.lemmatize(kw)
    lemmatized_keywords[lemma].append((kw, score))

# Choose the highest scored keyword for each lemma
final_keywords = {max(values, key=lambda x: x[1])[0]: get_embedding(max(values, key=lambda x: x[1])[0]) 
                  for values in lemmatized_keywords.values()}

# Calculate cosine similarities
similarity_matrix = cosine_similarity(
    [embedding.mean(dim=0).detach().numpy() for embedding in final_keywords.values()]
)

# Create a graph
G = nx.Graph()

# Add nodes
for kw in final_keywords:
    G.add_node(kw)

# Add edges with similarity scores as weights
threshold = 0.65  # Adjusted threshold
keywords_list = list(final_keywords.keys())
for i, kw1 in enumerate(keywords_list):
    for j, kw2 in enumerate(keywords_list):
        if i != j and similarity_matrix[i, j] > threshold:
            G.add_edge(kw1, kw2, weight=similarity_matrix[i, j])

# Function to calculate initial node positions
def get_initial_positions(G):
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    return pos

# Create a mapping of predefined emojis
predefined_emoji_mapping = {
    "cricket": "🏏",
    "batsman": "🏏",
    "captain": "👨‍✈️",
    "leader": "👨‍✈️",
    "team": "👥",
    "match": "🏟️",
    "win": "🏆",
    "run": "🏃",
    "ball": "⚾",
    "score": "📊",
    # Add more mappings as necessary
}

# Function to get emoji for a keyword
def get_emoji(keyword):
    if keyword in predefined_emoji_mapping:
        return predefined_emoji_mapping[keyword]
    else:
        # Fallback: Use the first emoji returned for the keyword
        emj = emoji.emojize(f':{keyword}:')
        if emj != f':{keyword}:':
            return emj
        else:
            return ""

# Create a network
net = Network(height="100%", width="100%", bgcolor="#222222", font_color="white")

# Add nodes and edges
for node in G.nodes():
    net.add_node(node, label=f"{node} {get_emoji(node)}")
for edge in G.edges():
    net.add_edge(edge[0], edge[1], value=G.edges[edge]['weight'])

# Save as HTML file
net.write_html(HTML_FILE)
