import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import WebBaseLoader #to load website ie data injestion
from langchain_text_splitters import RecursiveCharacterTextSplitter #tu create chunks of data
from langchain_openai import OpenAIEmbeddings #to create embeddings
from langchain_openai import ChatOpenAI #llm
from langchain_community.vectorstores import FAISS
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#step 1 : function to load the data and generate chunks, embed them and store into a vecorestore
# the user will insert a website link
def load_website(str_link):
    web_loader = WebBaseLoader(web_path = str_link)
    raw_doc = web_loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 300)
    docs=splitter.split_documents(raw_doc)
    embedder = OpenAIEmbeddings()
    db = FAISS.from_documents(docs,embedder)
    return db

#step 2: create a global docuemnt chain and a prompt and a llm
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(
    """You are an AI assistant.
Answer the question using ONLY the context below.

<context>
{context}
</context>

Question:
{input}
"""
)

llm = ChatOpenAI(model= "gpt-4o")
doc_chain = create_stuff_documents_chain(llm,prompt)

#step 3: create a function to generate document chain and retireval chain
#so the user at begin will enter a web link and its context baxed on which user will get
#result
def retrieval_chain(str_link,query):
    from langchain_core.documents import Document #it is a wrapper which dtructures the data
    from langchain_core.documents import Document
    #load the vector store with vectors in it
    vector_store = load_website(str_link)
    from langchain_classic.chains import create_retrieval_chain
    retriever = vector_store.as_retriever() # db - faiss
    retrieval_chain = create_retrieval_chain(retriever,doc_chain)
    response = retrieval_chain.invoke({
     "input": query
        })
    return response['answer']


#main

import streamlit as st
st.title("Context-Based Data Reciever using Langchain and OpenAI (Intro to document and retrieval chaining)")
st.write("Enter a website link and your query and you will get the answer")

user_query = st.text_area("Query", height=150)
user_url = st.text_area("URL",height=140)

if st.button("Give Answer"):
    if user_url.strip() == "" or user_query.strip()=="":
        st.warning("Please enter both the credentials")
    else:
        result = retrieval_chain(user_url,user_query)
        st.subheader("Answer")
        st.success(result)

