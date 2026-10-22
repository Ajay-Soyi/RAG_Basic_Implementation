from ollama import Client
from dotenv import load_dotenv
import os
from retriever import load_documents

from test_embedding import create_embedding

from similarity import cosine_similarity

documents = load_documents()

document_embeddings = {}

for filename, content in documents.items():

    document_embeddings[filename] = create_embedding(
        content
    )

# Load configuration
load_dotenv()

# Create AI client
client = Client()

print("=" * 40)
print("      My AI Assistant")
print("=" * 40)

while True:

    user_input = input("\nYou : ")

    if user_input.lower() == "quit":
        print("\nAI  : Goodbye! Have a great day.")
        break

    question_embedding = create_embedding(
    user_input)

    best_score = -1

    best_document = None

    for filename, embedding in document_embeddings.items():

        score = cosine_similarity(
        question_embedding,
        embedding
        )

        if score > best_score:

            best_score = score

            best_document = filename

    context = documents[best_document]

    prompt = f"""
Answer the question using only
the following information.

Context:

{context}

Question:

{user_input}
"""


    response = client.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\nAI :", response["message"]["content"])

