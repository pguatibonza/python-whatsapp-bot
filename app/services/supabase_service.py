import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFMinerLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import psycopg2
import numpy as np

load_dotenv()

DB_CONNECTION = os.getenv("DB_CONNECTION")

# --------------------------------------------------------------
# Creación de la tabla de conversaciones en Supabase
# --------------------------------------------------------------

def create_conversation_table():
    conn = psycopg2.connect(DB_CONNECTION)
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                phone_number VARCHAR(20),
                message TEXT,
                response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
    conn.close()

def store_message(phone_number, message, response):
    conn = psycopg2.connect(DB_CONNECTION)
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO conversations (phone_number, message, response)
            VALUES (%s, %s, %s);
        """, (phone_number, message, response))
        conn.commit()
    conn.close()

def get_stored_messages(phone_number):
    conn = psycopg2.connect(DB_CONNECTION)
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT message, response
            FROM conversations
            WHERE phone_number = %s
            ORDER BY timestamp;
        """, (phone_number,))
        results = cursor.fetchall()
    conn.close()
    return results

# Crear la tabla de conversaciones
create_conversation_table()

# --------------------------------------------------------------
# Código para cargar documentos y vectorizarlos (comentado)
# --------------------------------------------------------------

# Ajustar la ruta del archivo PDF
# file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/airbnb-faq.pdf"))

# Verificar si la ruta del archivo es correcta
# print(f"Loading file from: {file_path}")

# # Cargar los documentos
# loader = PDFMinerLoader(file_path)
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Reducir chunk_size y aumentar chunk_overlap
# docs = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings()

# Crear una tabla para PGVector
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

# Descomentar estas líneas si necesitas cargar los documentos
# conn = psycopg2.connect(DB_CONNECTION)
# COLLECTION_NAME = "menu_magdalena_rooftop"

# create_pgvector_table(conn, COLLECTION_NAME)
# insert_documents(conn, COLLECTION_NAME, docs, embeddings)

# conn.close()
