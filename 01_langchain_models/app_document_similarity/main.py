# I'm using Google's Embedding

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

fruit_docs = [
    "Orange is orange",
    "Banana is yellow",
    "Apple is red"
]


def generate_query_embedding(user_query, embedder:GoogleGenerativeAIEmbeddings):
    return embedder.embed_query(user_query)


def get_fruit_document_embedding(fruit_docs, embedder:GoogleGenerativeAIEmbeddings):
    return embedder.embed_documents(fruit_docs)

def calculate_cosine_similarity_score(user_query_embedding, fruit_docs_embedding):
    return cosine_similarity([user_query_embedding], fruit_docs_embedding) # type: ignore


if __name__ == "__main__":
    user_query = "Banana is very tasty orange!"
    user_query_embedding = generate_query_embedding(user_query, embeddings)
    fruit_docs_embedding = get_fruit_document_embedding(fruit_docs, embeddings)
    score = calculate_cosine_similarity_score(user_query_embedding, fruit_docs_embedding)

    score = score[0]

    index, max_score = sorted(list(enumerate(score)), key = lambda x:x[1])[-1]

    print(score)
    print(user_query)
    print(fruit_docs[index])
    print("Similarity score", max_score)








