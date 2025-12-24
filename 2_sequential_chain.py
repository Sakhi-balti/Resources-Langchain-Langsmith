from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import os

load_dotenv()
HF_KEY = os.getenv('HF_KEY')

os.environ['LANGCHAIN_PROJECT'] = 'Sequential chain'

# Define prompts
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# Initialize LLM
model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3-0324",
    api_key=HF_KEY,
    base_url="https://router.huggingface.co/v1",
    temperature=0.7,
    max_tokens=500
)

parser = StrOutputParser()

CONFIG = {
    'run_name': 'Sequential chain',
    'tags': ['llm', 'report generation', 'summary'],
    'metadata': {'model': 'deepseek', 'api': 'hugging face'}
}

def wrap_in_dict(text):
    """Convert string output to dict with 'text' key"""
    return {"text": text}

chain = (
    prompt1 
    | model 
    | parser 
    | RunnableLambda(wrap_in_dict)  # Transform string to dict
    | prompt2 
    | model 
    | parser
)

result = chain.invoke({'topic': 'Tourism in Pakistan'}, config=CONFIG)

print("\n=== Final Summary ===")
print(result)
