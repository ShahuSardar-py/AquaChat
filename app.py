from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

# Streamlit App Title
st.title("Aquarium Master Testing")

persist_directory = "./chroma_data"
os.makedirs(persist_directory, exist_ok=True)

# Initialize the embedding function
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize or load the vectorstore
try:
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory,  # Ensure this directory exists
    )
except ValueError:
    print("No existing vectorstore found. Creating a new one.")
    # If the vectorstore doesn't exist, create one from documents
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    loader = PyPDFLoader("Freshwater_Aquarium_Guide.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )
# Create retriever from vector store
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Set up the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None
)

# Set up the System Prompt
system_prompt = '''
You are a seasoned aquarium hobbyist. You are wise and smart. You believe in advising people based on real facts. Use the given pieces of context to answer any question asked to you.
If you don't know the answer, say so. When asked about yourself, mention you were developed by Shahu Sardar for all aquarium hobbyists.
Keep answers short and concise.
{context}
'''

# Streamlit Chat Input
query = st.chat_input("Say something:")
if query:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create Question-Answer Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Generate Response
    response = rag_chain.invoke({"input": query})
    st.write(response["answer"])
