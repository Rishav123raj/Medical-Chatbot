from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from src.helper import load_pdf, text_splitter, download_hugging_face_embeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import os

app = Flask(__name__)
load_dotenv()


extracted_data = load_pdf("C:\\Users\\Lenovo\\Desktop\\PROJECTS\\LLM\\Medical-Chatbot\\data")

text_chunks = text_splitter(extracted_data)

embeddings = download_hugging_face_embeddings()

#Creating Embeddings for each text chunk using HuggingFace Embeddings and storing in ChromaDb vector database
persist_directort = 'db'
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(documents=text_chunks, embedding=embeddings,
                                  persist_directory=persist_directort)

vectordb.persist()
vectordb=None
vectordb = Chroma(persist_directory=persist_directort, embedding_function=embeddings)
retriever = vectordb.as_retriever()

prompt_template = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": prompt_template}

model=genai.GenerativeModel("gemini-2.0-flash")
llm=ChatGoogleGenerativeAI(google_api_key=os.getenv('GOOGLE_API_KEY'), model="gemini-2.0-flash", temperature=0.8)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs=chain_type_kwargs,
    return_source_documents=True,
)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg")
    print("User input : ", msg)
    result = qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(debug = True)

