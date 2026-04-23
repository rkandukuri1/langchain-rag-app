from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY_FREE"), model="gpt-4.1-nano")

src_file = "VigilantCorp_Company_Profile.pdf"
loader = PyPDFLoader(src_file)
text_data = loader.load()

prompt_template = '''
You are a helpful chat assistant. Use following document and answer the below questions. Dont use you own knowledge.
If you dont find the answer in the below document, reply saying - Not Found

Question : {qq}
Document : {doc}
'''

initial_prompt = PromptTemplate.from_template(prompt_template)

st.subheader("Rag Application - Langchain")

user_qq = st.chat_input("Enter your prompt ")

if user_qq:
    final_prompt = initial_prompt.format(qq = user_qq, doc = text_data)
    response = client.invoke(final_prompt)
    st.write(response.text)