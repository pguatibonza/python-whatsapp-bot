import os
from dotenv import load_dotenv
from fastapi import Request
from fastapi import Depends
from langchain_community.document_loaders import PDFMinerLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from supabase import create_client, Client
from langchain_community.vectorstores import SupabaseVectorStore
import logging
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.indexes import SQLRecordManager, index
from langchain_experimental.text_splitter import SemanticChunker
from config import Settings

settings=Settings()


class SupabaseService:
    """
    Class to manage the connection and operations with Supabase.
    Ensures that the client is initialized only once and reused.
    """

    _client: Client = None
    _embeddings = None
    _vector_store=None
    

    @classmethod
    def get_client(cls) -> Client:
        """
        Returns a singleton instance of the Supabase client.
        If it doesn't exist, it creates and stores it for reuse.

        Returns:
            Client: Supabase client instance.
        """
        if cls._client is None:
            
            cls._client=create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            logging.info("Supabase client initialized successfully.")
        return cls._client
    
    @classmethod 
    def get_embeddings(cls):
        if cls._embeddings is None:
            cls._embeddings = OpenAIEmbeddings()
        return cls._embeddings
    

    @classmethod
    def load_vector_store(cls) -> SupabaseVectorStore:
        """
        Loads the vector store from the Supabase database.

        Returns:
            SupabaseVectorStore: Initialized vector store.
        """
        

        table_name = settings.TABLE_NAME
        supabase = cls.get_client()
        embeddings = cls.get_embeddings()
        if cls._vector_store is None:
            cls._vector_store=SupabaseVectorStore(
                embedding=embeddings,
                table_name=table_name,
                client=supabase,
                query_name="match_documents",
            )
        return cls._vector_store

    @classmethod
    def load_vehicle_brands_models(cls) -> list[dict]:
        """
        Loads vehicle brands and models from the database.

        Returns:
            list[dict]: List of vehicle brands and models.
        """
        table_name_vehicles = settings.TABLE_NAME_VEHICLES
        supabase = cls.get_client()
        try:
            response = supabase.table(table_name_vehicles).select('marca, modelo, id').execute()
            logging.info("Successfully retrieved vehicle brands and models.")
            return response.data
        except Exception as e:
            logging.error(f"Error loading vehicle brands and models: {e}")
            return []

    @classmethod
    def load_vehicle_info_by_id(cls, vehicle_id: int) -> list[dict]:
        """
        Loads information for a specific vehicle by ID.

        Args:
            vehicle_id (int): ID of the vehicle.

        Returns:
            list[dict]: Vehicle information.
        """
        table_name_vehicles = settings.TABLE_NAME_VEHICLES
        supabase = cls.get_client()
        try:
            response = supabase.table(table_name_vehicles).select('*').eq("id", vehicle_id).execute()
            logging.info(f"Successfully retrieved vehicle information for ID {vehicle_id}.")
            return response.data
        except Exception as e:
            logging.error(f"Error loading vehicle info by ID {vehicle_id}: {e}")
            return []
    @classmethod
    def load_dealership_info(cls) -> list[dict]:
        """
        Loads dealership information from the database.

        Returns:
            list[dict]: List of dealerships with columns id, nombre, and description.
        """
        table_name_concesionarios = settings.TABLE_NAME_DEALERSHIP
        supabase = cls.get_client()
        try:
            response = supabase.table(table_name_concesionarios).select('*').execute()
            logging.info("Successfully retrieved dealership information.")
            return response.data
        except Exception as e:
            logging.error(f"Error loading dealership information: {e}")
            return []

    @classmethod
    def clean_data(cls, documents: list) -> list:
        """
        Cleans document data by removing unwanted characters.

        Args:
            documents (list): List of documents to clean.

        Returns:
            list: Cleaned documents.
        """
        for document in documents:
            document.page_content = document.page_content.replace("\u0000", "")
        return documents

    @classmethod
    def process_documents(cls, loader, path_or_dir: str) -> list:
        """
        Generalized document processing for files or directories.

        Args:
            loader: Loader class for documents.
            path_or_dir (str): Path to the file or directory.

        Returns:
            list: Processed and split documents.
        """
        documents = loader(path_or_dir).load()
        documents = cls.clean_data(documents)
        embeddings = OpenAIEmbeddings()
        text_splitter = SemanticChunker(embeddings)
        return text_splitter.split_documents(documents)

    @classmethod
    def load_directory(cls, directory: str):
        """
        Loads all documents from a directory into the vector store.

        Args:
            directory (str): Directory path.

        Returns:
            SupabaseVectorStore: Updated vector store.
        """
        docs = cls.process_documents(DirectoryLoader, directory)
        vector_store = cls.load_vector_store()
        namespace = f"concesionarios/los-coches/{settings.TABLE_NAME}"
        record_manager = SQLRecordManager(namespace, db_url="sqlite:///record_manager_cache.sql")
        index(docs, record_manager, vector_store, cleanup="incremental", source_id_key="source")
        logging.info("Documents successfully inserted into the database.")
        return vector_store

    @classmethod
    def load_file(cls, file_path: str):
        """
        Loads a single document file into the vector store.

        Args:
            file_path (str): Path to the file.

        Returns:
            SupabaseVectorStore: Updated vector store.
        """
        docs = cls.process_documents(UnstructuredFileLoader, file_path)
        vector_store = cls.load_vector_store()
        namespace = f"concesionarios/los-coches/{settings.TABLE_NAME}"
        record_manager = SQLRecordManager(namespace, db_url="sqlite:///record_manager_cache.sql")
        index(docs, record_manager, vector_store, cleanup="incremental", source_id_key="source")
        logging.info("Document successfully inserted into the database.")
        return vector_store
