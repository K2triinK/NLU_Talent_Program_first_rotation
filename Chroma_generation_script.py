import pyodbc
import pandas as pd
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from helpers import trans
from easynmt import EasyNMT

DATABASE_NAME = "name_of_your_choice"
DATA_FILE_NAME = "example_chroma.xlsx" #write path

# Choose a model that works for your language
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Load client
print('Loading client.')
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="db" # Optional, defaults to .chromadb/ in the current directory
))

# Delete a collection and all associated embeddings, documents, and metadata, for UPDATE. ⚠️ This is destructive and not reversible
try:
    client.delete_collection(name=DATABASE_NAME)
except IndexError:
    pass

# Load data
print('Loading data.')
data = pd.read_excel(DATA_FILE_NAME)
# It is also possible to load data from SQL, for example, just write a query here

df = data['TEXT_ID', 'TITLE', 'TEXT', 'PORTFOLIO']
df['TEXT_CONCAT'] = df.TITLE + ' ' + df.TEXT
df['ID'] = df.TEXT_ID.apply(lambda x: str(x))

# Generate your vector database
print('Generating vector database.')

# Organizing lists that will go in the database
ids = [df.id]
metadatas = [{'title': d.TITLE, 'portfolio': d.PORTFOLIO} for i,d in df.iterrows()]
documents = df.TEXT_CONCAT.tolist()
print('Data prepared')

# Create collection with the same name, for UPDATE
collection = client.create_collection(name=DATABASE_NAME, embedding_function=sentence_transformer_ef)

collection.add(
    documents = documents,
    metadatas = metadatas,
    ids = ids[0].tolist()
)

client.persist()

print('All done.')