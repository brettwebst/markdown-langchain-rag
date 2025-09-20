from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

class EmbeddingManager:
    def __init__(self, all_sections, persist_directory='db'):
        self.all_sections = all_sections
        self.persist_directory = persist_directory
        self.vectordb = None
        
    # Method to create and persist embeddings
    def create_and_persist_embeddings(self):
        # Creating an instance of HuggingFaceEmbeddings with a local model
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have a GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        # Creating an instance of Chroma with the sections and the embeddings
        self.vectordb = Chroma.from_documents(documents=self.all_sections, embedding=embedding, persist_directory=self.persist_directory)
         # Persisting the embeddings
         # self.vectordb.persist() # no longer needed as document manager now persists the embeddings