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
namespace=f"concesionarios/parra-arango/{TABLE_NAME}"
record_manager=SQLRecordManager(namespace,db_url="sqlite:///record_manager_cache.sql")
#record_manager.create_schema() ##Siempre hay que crear esquema cuando hay un nuevo record manager


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


#load_directory("data/DocumentosParraArango/")