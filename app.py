import os
import torch
from logger import logging

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gensim.models import Word2Vec  # Example using Word2Vec

def extract_keywords(text):
    """
    Extracts keywords from a given text using KeyBART.

    Args:
      text (str): The text input from which to extract keywords.

    Returns:
      list: A list of strings representing the extracted keywords.
    """

    # Preprocess text (optional, consider domain-specific cleaning)
    # You can add steps like removing stop words, stemming/lemmatization here
    # processed_text = text.lower()  # Example: convert to lowercase

    logging.info("Extracting keywords")

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt")

    # Generate keyphrases using KeyBART
    with torch.no_grad():
        logging.info("Using KeyBART model")
        outputs = model.generate(**inputs)

    # Decode the generated tokens (assuming first output is the most likely)
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract keywords from the generated text (replace with your selection logic)
    keywords = decoded_text.split(" ")  # Simple splitting by space (modify as needed)
    logging.info("Keywords extracted")

    return keywords


# Generate a set of word embeddings

# Load pre-trained word embeddings (replace with your model path)
word_embedding_model = Word2Vec.load("word2vec_model.bin")


def extract_word_similarities(keywords, word_embedding_model):
  """
  Calculates pairwise similarities between keywords using a word embedding model.

  Args:
      keywords (list): A list of strings representing the keywords.
      word_embedding_model (gensim.models.Word2Vec): A loaded word embedding model.

  Returns:
      set: A set containing tuples of (word1, word2, similarity) representing word pairs and their similarities.
  """
  logging.info("creating word embedding pairs for keywords")


  similarities = set()
  for i in range(len(keywords)):
    for j in range(i + 1, len(keywords)):
      word1, word2 = keywords[i], keywords[j]
      # Check if both words are in the word embedding vocabulary
      logging.info("Using Word2Vec model")
      if word1 in word_embedding_model.wv and word2 in word_embedding_model.wv:
        similarity = word_embedding_model.wv.similarity(word1, word2)
        similarities.add((word1, word2, similarity))
      else:
        print(f"Warning: Skipping similarity calculation for {word1} and {word2} (not in vocabulary)")

  logging.info("Word embeddings created")

  return similarities


# Example usage
text = "This is a sample document about natural language processing."
keywords = extract_keywords(text)  # Replace with your actual extract_keywords function

word_similarities = extract_word_similarities(keywords, model)

# Print the word similarities
for word1, word2, similarity in word_similarities:
  print(f"{word1} - {word2} similarity:", similarity)


if __name__ == "__main__":
    
    logging.info("Initiating the model")

    # Load KeyBART tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART")
    model = AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART")

    # Get user input text
    user_text = input("Enter the text to extract keywords from: ")

    # Extract keywords from the user input
    keywords = extract_keywords(user_text)

    # Print the extracted keywords
    print("Extracted keywords:", keywords)