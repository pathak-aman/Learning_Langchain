from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedder = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

text = "Banana is yellow!"

vector = embedder.embed_query(text)

print(str(vector))
documents = [
    "Orange is orange",
    "Banana is yellow",
    "Apple is red"
]

vector = embedder.embed_documents(documents)
print(str(vector))
