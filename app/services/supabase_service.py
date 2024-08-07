import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFMinerLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from supabase import create_client, Client
from langchain_community.vectorstores import SupabaseVectorStore
import logging
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.indexes import SQLRecordManager, index

# Cargar variables de entorno
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME=os.getenv("TABLE_NAME")
embeddings=OpenAIEmbeddings()

# Crear cliente de Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print(f"Supabase client initialized: {supabase}")

#Crear indice para evitar duplicados
namespace=f"supabaseParra/{TABLE_NAME}"
record_manager=SQLRecordManager(namespace,db_url="sqlite:///record_manager_cache.sql")

# # Función para crear la tabla en PostgreSQL si no existe
# def create_pgvector_table(conn, table_name):
#     with conn.cursor() as cursor:
#         cursor.execute(f"""
#             CREATE EXTENSION IF NOT EXISTS vector;
#             CREATE TABLE IF NOT EXISTS {table_name} (
#                 id SERIAL PRIMARY KEY,
#                 content TEXT,
#                 metadata JSONB,
#                 embedding VECTOR(1536)
#             );
#         """)
#         conn.commit()

# # Conectar a PostgreSQL y crear la tabla si no existe
# conn = psycopg2.connect(DB_CONNECTION)
# COLLECTION_NAME = "menu_magdalena_rooftop"
# create_pgvector_table(conn, COLLECTION_NAME)

# # Verificar si ya hay documentos en la tabla
# def documents_exist(conn, table_name):
#     with conn.cursor() as cursor:
#         cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
#         count = cursor.fetchone()[0]
#     return count > 0

# # Verificar si ya hay documentos y cargar si no existen
# if not documents_exist(conn, COLLECTION_NAME):
#     # Cargar los documentos
#     file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/MENUMAGDALENAROOFTOP_removed.pdf"))
#     print(f"Loading file from: {file_path}")

#     loader = PDFMinerLoader(file_path)
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     docs = text_splitter.split_documents(documents)

#     embeddings = OpenAIEmbeddings()

#     # Insertar los documentos en Supabase utilizando SupabaseVectorStore
#     vector_store = SupabaseVectorStore.from_documents(
#         docs,
#         embeddings,
#         client=supabase,
#         table_name=COLLECTION_NAME,
#         query_name="match_documents"
#     )

#     print("Documentos vectorizados e insertados en Supabase con éxito.")
# else:
#     print("Los documentos ya existen en la base de datos, no se realizó una nueva carga.")
# conn.close()

def clean_data(documents):
    for document in documents:
        document.page_content=document.page_content.replace("\u0000","")
    return documents

#Carga el vector store de la base de datos
def load_vector_store():
    vector_store = SupabaseVectorStore(
        embedding=embeddings,
        table_name=TABLE_NAME,
        client=supabase,
        query_name="match_documents",
    )
    return vector_store


#Carga un directorio con archivos a la base de datos
def load_directory(directory):
    #Carga los archivos desde un directorio 
    loader=DirectoryLoader(directory,show_progress=True)
    documents=loader.load()
    documents=clean_data(documents)
    logging.info("documentos cargados en langchain")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    #Obtener la base de datos
    vector_store = load_vector_store()
    
    #Crear el indice para evitar duplicados
    print(index(docs,record_manager,vector_store,cleanup="incremental",source_id_key="source"))

    logging.info("Documentos insertados correctamente en la base de datos")
    return vector_store

#Añade documento a la base de datos
def load_file(file_path):
    loader=UnstructuredFileLoader(file_path)
    documents=loader.load()
    documents=clean_data(documents)

    logging.info("Documento cargado en langchain")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    #Obtener la base de datos
    vector_store = load_vector_store()
    
    #Crear el indice para evitar duplicados
    print(index(docs,record_manager,vector_store,cleanup="incremental",source_id_key="source"))
    logging.info("Documento insertado correctamente en la base de datos")
    return vector_store


