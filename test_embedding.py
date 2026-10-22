import ollama
from similarity import cosine_similarity



def create_embedding(content):
    embedding1 = ollama.embeddings(
        model="nomic-embed-text",
        prompt=content
    )
    return embedding1["embedding"]





