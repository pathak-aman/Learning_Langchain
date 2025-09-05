from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

embbeder = OpenAIEmbeddings(model = "text-embedding-3-large", dimensions=32)
# embbeder = OpenAIEmbeddings(model = "text-embedding-ada-002")

text_embedding = embbeder.embed_query("Delhi is the capital of India")
print(str(text_embedding))
