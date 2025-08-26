from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt3.5", temperature=0, max_completion_tokens=20)

print("Model invoking")
result = model.invoke("Brainstorm the future of AI in 5 bullet points!")
print("Model invoked")
# print(result)
print(result.content)
