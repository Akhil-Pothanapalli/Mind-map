from keybert import KeyBERT
from transformers import BertModel, BertTokenizer
import torch
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import sys
from exception import CustomException
from logger import logging

try:
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
    threshold = 0.5  # You can adjust this threshold
    for i, kw1 in enumerate(keyword_embeddings):
        for j, kw2 in enumerate(keyword_embeddings):
            if i != j and similarity_matrix[i, j] > threshold:
                G.add_edge(kw1, kw2, weight=similarity_matrix[i, j])

    logging.info(f'Graph created with a threshold of {threshold}')

    # Print the graph nodes and edges

    print("------------------------------------------------------------/n")
    print(G.nodes())


    # Draw the graph
    pos = nx.spring_layout(G)
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    nx.draw(G, pos, with_labels=True, width=weights)
    plt.show()

except Exception as e:
    raise CustomException(e, sys)