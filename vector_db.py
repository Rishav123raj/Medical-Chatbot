from src.helper import load_pdf, text_splitter, download_hugging_face_embeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

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