
!pip install transformers torch

# Import necessary modules
import torch
from transformers import AutoTokenizer, AutoModel

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get sentence embeddings
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # Using the [CLS] token embedding
    return embeddings

# Function to calculate similarity
def calculate_similarity(sentence1, sentence2):
    embedding1 = get_sentence_embedding(sentence1)
    embedding2 = get_sentence_embedding(sentence2)
    cosine_similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return cosine_similarity.item()

# Example sentences
original_text = "The quick brown fox jumps over the lazy dog."
suspected_text1 = "A fast brown fox leaps over a sleepy dog."  # Similar but reworded
suspected_text2 = "Artificial intelligence is transforming the world."  # Completely different

# Calculate similarity
similarity1 = calculate_similarity(original_text, suspected_text1)
similarity2 = calculate_similarity(original_text, suspected_text2)

# Define a plagiarism threshold (e.g., 0.8)
threshold = 0.8

# Print results
print(f"Similarity with suspected_text1: {similarity1:.4f}")
print(f"Similarity with suspected_text2: {similarity2:.4f}")
print(f"Is suspected_text1 plagiarized? {'Yes' if similarity1 > threshold else 'No'}")
print(f"Is suspected_text2 plagiarized? {'Yes' if similarity2 > threshold else 'No'}")
