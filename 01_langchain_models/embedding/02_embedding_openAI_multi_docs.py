from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

documents = [
    "Orange is orange",
    "Banana is yellow",
    "Apple is red"
]

embbeder = OpenAIEmbeddings(model = "text-embedding-3-large", dimensions = 8)

embeddings = embbeder.embed_documents(documents)

print(str(embeddings))