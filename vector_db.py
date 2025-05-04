from src.helper import load_pdf, text_splitter, download_hugging_face_embeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

