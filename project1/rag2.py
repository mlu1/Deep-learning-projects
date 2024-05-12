import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
os.environ['OPEN_API_KEY'] = 'sk-proj-96ybTcbM9qH7wcwv4GkgT3BlbkFJBGqScPkhjVrapJiObdE0'
 # Get environment variables
OPEN_API_KEY = os.getenv('OPEN_API_KEY')
#MODEL ='gpt-3.5-turbo'
#MODEL = 'mixtral:8x7b'
MODEL = 'llama2'
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

if MODEL.startswith('gpt'):
    model = ChatOpenAI(api_key=OPEN_API_KEY,model=MODEL)
    embeddings =OpenAIEmbeddings(api_key=OPEN_API_KEY)
else:
    model = Ollama(model=MODEL)
    embeddings =OllamaEmbeddings()
 
parser = StrOutputParser()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

loader = PyPDFLoader("human-nutrition.pdf")
p1 = loader.load_and_split()
pages = p1[0:10]
from langchain.prompts import PromptTemplate

template = """
Answer the question based on the context below . If you can't answer the question
,reply "I dont know"
context{context}
Question :{question}
"""
prompt = PromptTemplate.from_template(template)
prompt.format(context = "Here is context",question = "Here is context")
chain = prompt | model | parser

print(chain.invoke({
                "context":"The name I was given was Mluleki",
                "question":"whats my name?"
            }))

print(chain.input_schema.schema())

##SEARCH RELEVANT DOCUMENTS
vectorstore  = DocArrayInMemorySearch.from_documents(pages,embedding = embeddings)
retriever = vectorstore.as_retriever()
retriever.invoke('healthy')

chain = ({
          "context":itemgetter("question")|retriever,
          "question":itemgetter("question")
          }
         | prompt | model | parser
        )
chain.invoke({"question": "What is food"})
questions = ["what is the purpose of food","how much nutrition in this"]

for question in questions:
    print(f"Question:{question}")    
    print(f"Answer: {chain.invoke({'question':question})}")
    print("\n")