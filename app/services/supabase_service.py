import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFMinerLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from supabase import create_client, Client
from langchain_community.vectorstores import SupabaseVectorStore
import psycopg2
import logging
from langchain_community.document_loaders import DirectoryLoader

# Cargar variables de entorno
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME=os.getenv("TABLE_NAME")

# Crear cliente de Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print(f"Supabase client initialized: {supabase}")

# Función para crear la tabla en PostgreSQL si no existe
def create_pgvector_table(conn, table_name):
    with conn.cursor() as cursor:
        cursor.execute(f"""
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding VECTOR(1536)
            );
        """)
        conn.commit()

# Conectar a PostgreSQL y crear la tabla si no existe
conn = psycopg2.connect(DB_CONNECTION)
COLLECTION_NAME = "menu_magdalena_rooftop"
create_pgvector_table(conn, COLLECTION_NAME)

# Verificar si ya hay documentos en la tabla
def documents_exist(conn, table_name):
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cursor.fetchone()[0]
    return count > 0

# Verificar si ya hay documentos y cargar si no existen
if not documents_exist(conn, COLLECTION_NAME):
    # Cargar los documentos
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/MENUMAGDALENAROOFTOP_removed.pdf"))
    print(f"Loading file from: {file_path}")

    loader = PDFMinerLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    # Insertar los documentos en Supabase utilizando SupabaseVectorStore
    vector_store = SupabaseVectorStore.from_documents(
        docs,
        embeddings,
        client=supabase,
        table_name=COLLECTION_NAME,
        query_name="match_documents"
    )

    print("Documentos vectorizados e insertados en Supabase con éxito.")
else:
    print("Los documentos ya existen en la base de datos, no se realizó una nueva carga.")
conn.close()

def load_directory(directory):
    #Carga los archivos desde un directorio 
    loader=DirectoryLoader(directory,show_progress=True)
    documents=loader.load()
    logging.info("documentos cargados en langchain")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    # Insertar los documentos en Supabase utilizando SupabaseVectorStore
    vector_store = SupabaseVectorStore.from_documents(
        docs,
        embeddings,
        client=supabase,
        table_name=TABLE_NAME,
        query_name="match_documents"
    )
    logging.info("Documentos insertados correctamente en la base de datos")
    return vector_store




# Funciones comentadas para la tabla de conversaciones en Supabase
# def create_conversation_table():
#     conn = psycopg2.connect(DB_CONNECTION)
#     with conn.cursor() as cursor:
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS conversations (
#                 id SERIAL PRIMARY KEY,
#                 phone_number VARCHAR(20),
#                 message TEXT,
#                 response TEXT,
#                 timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             );
#         """)
#         conn.commit()
#     conn.close()

# def store_message(phone_number, message, response):
#     conn = psycopg2.connect(DB_CONNECTION)
#     with conn.cursor() as cursor:
#         cursor.execute("""
#             INSERT INTO conversations (phone_number, message, response)
#             VALUES (%s, %s, %s);
#         """, (phone_number, message, response))
#         conn.commit()
#     conn.close()

# def get_stored_messages(phone_number):
#     conn = psycopg2.connect(DB_CONNECTION)
#     with conn.cursor() as cursor:
#         cursor.execute("""
#             SELECT message, response
#             FROM conversations
#             WHERE phone_number = %s
#             ORDER BY timestamp;
#         """, (phone_number,))
#         results = cursor.fetchall()
#     conn.close()
#     return results

# # Crear la tabla de conversaciones
# create_conversation_table()
