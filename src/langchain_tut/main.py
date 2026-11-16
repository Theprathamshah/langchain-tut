import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_AI_API_KEY"),
)

prompt = PromptTemplate.from_template("Explain {topic} in one sentence.")
chain = prompt | llm
print(chain.invoke({"topic": "LangChain"}).content)
