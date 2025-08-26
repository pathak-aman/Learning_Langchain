from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm_obj = OpenAI(model = "gpt-3.5-turbo-instruct")

result = llm_obj.invoke("What is the captal of India")

print(result)