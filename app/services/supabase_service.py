import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFMinerLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import psycopg2
import numpy as np

load_dotenv()

DB_CONNECTION = os.getenv("DB_CONNECTION")

# Ajustar la ruta del archivo PDF
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/airbnb-faq.pdf"))

# Verificar si la ruta del archivo es correcta
print(f"Loading file from: {file_path}")

# --------------------------------------------------------------
# Load the documents
# --------------------------------------------------------------

loader = PDFMinerLoader(file_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

# --------------------------------------------------------------
# Create a PGVector Store
# --------------------------------------------------------------

def create_pgvector_table(conn, table_name):
    with conn.cursor() as cursor:
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                content TEXT,
                vector VECTOR(1536)
            );
        """)
        conn.commit()

def insert_documents(conn, table_name, docs, embeddings):
    with conn.cursor() as cursor:
        for doc in docs:
            vector = embeddings.embed_query(doc.page_content)
            vector_str = ','.join(map(str, vector))
            cursor.execute(f"""
                INSERT INTO {table_name} (content, vector)
                VALUES (%s, %s::vector);
            """, (doc.page_content, f'[{vector_str}]'))
        conn.commit()

conn = psycopg2.connect(DB_CONNECTION)
COLLECTION_NAME = "menu_magdalena_rooftop"

create_pgvector_table(conn, COLLECTION_NAME)
insert_documents(conn, COLLECTION_NAME, docs, embeddings)

conn.close()
