from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
load_dotenv()
HF_KEY = os.getenv('HF_KEY')

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

# Initialize LLM
model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3-0324",
    api_key=HF_KEY,
    base_url="https://router.huggingface.co/v1",
    temperature=0.7,
    max_tokens=500
)
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of Peru?"})
print(result)
