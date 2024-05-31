from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import UnstructuredPDFLoader, PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from supabase import create_client
import logging
import os
import psycopg2

from . import tools_restaurant

# Configuración inicial
load_dotenv()
DB_CONNECTION = os.getenv("DB_CONNECTION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

chat = ChatOpenAI(model="gpt-4o", temperature=0)
tools = tools_restaurant.TOOLS

# Crear cliente de Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings()  # Inicializar embeddings
vector_store = SupabaseVectorStore(client=supabase, table_name="menu_magdalena_rooftop", embedding=embeddings)

# Plantilla del sistema
SYSTEM_TEMPLATE = """
Eres un chatbot de asistencia al cliente para el restaurante ROOFTOP magdalena

Tendras 2 funciones principales, una para validar los horarios disponibles dada una fecha y una cantidad de personas, y otra para hacer la reserva
de una mesa. 

1. Obtener los horarios disponibles para hacer reservas dada una fecha y una cantidad de personas, y opcionalmente un area(solo si el usuario pregunta)
2. Hacer una reserva. Cuando el usuario quiera hacer una reserva vas a seguir los siguientes pasos :
    a.  Le preguntas la fecha y la cantidad de personas. Tomas la fecha y la conviertes al formato YYYY-MM-DD. Si el usuario no te da el año, asume que es 2024.
    b. Le preguntas si tiene preferencia por alguna zona (main o terraza)
    b. Validas si esa fecha esta disponible y le muestras los horarios.
    c.  Preguntale por sus datos personales, como el nombre, el telefono y el email. 
        Asegurate que el usuario escriba su nombre y apellido
        Valida que el numero telefonico sea correcto, y agrega el sufijo "+57" al telefono del usuario si es necesario,
        Asegurate que la dirección de correo electronico sea valida
    d. Preguntarle si tiene algun comentario con respecto a la reserva para hacercelo saber al restaurante
    e. Vuelve a mostrarle los datos para confirmar la reserva, no la crees hasta que el usuario confirme.No le puedes dar el id de la reserva

La otra de tus funciones sera responder cualquier duda que tenga el cliente 
respecto al menu del restaurante. Para ello podras usar el contexto que está abajo.
<context>
{context}
</context>
"""

def get_db_connection():
    """ Establece conexión con la base de datos PostgreSQL. """
    return psycopg2.connect(DB_CONNECTION)

# # No carga bien
# def upload_file_unstructured_pdf(file_path):
#     loader = UnstructuredPDFLoader(file_path)
#     data = loader.load()
#     return data

# # Mejor, segmenta mejor la info
# def upload_file_miner(file_path):
#     loader = PDFMinerLoader(file_path)
#     data = loader.load()
#     return data

# # Load pdfs and create text splits
# def create_from_directory(file_directory):
#     data = []
#     for file in os.listdir(file_directory):
#         path = os.path.join(file_directory, file)
#         loader = PDFMinerLoader(path)
#         data += loader.load()
#         logging.info(f"Documento cargado desde el archivo {path}")

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     all_splits = text_splitter.split_documents(data)
#     logging.info("Documentos spliteados")
#     return all_splits

# # Take a directory and create/add vectors to the vector store
# def add_pdfs_from_directory(file_directory):
#     all_splits = create_from_directory(file_directory)
#     vector_store.add_documents(all_splits)
#     logging.info("Documentos añadidos")
#     return vector_store

# # db = add_pdfs_from_directory("../../data/")
# db = add_pdfs_from_directory("data/")

store = {}

# Gets the chat history based on the cellphone number (id)
# For production use cases, you will want to use a persistent implementation of chat message history, such as RedisChatMessageHistory
def get_session_history_local(session_id: str) -> BaseChatMessageHistory:
    """Fetch the chat history"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Production use case
def get_session_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(session_id, url=REDIS_URL)

# Create a Prompt template with a LLM model
def create_chain_agent():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(chat, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

def store_message(session_id, message, response):
    """ Crea y almacena un mensaje en la base de datos """
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO conversations (phone_number, message, response, timestamp)
            VALUES (%s, %s, %s, NOW())
        """, (session_id, message, response))
    conn.commit()
    conn.close()
    logging.info(f"Stored message for session {session_id}")

def get_chat(chain):
    """ Configura la cadena con historial de mensajes. """
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

# Generates a response based on an input
def run_chain(message_body, wa_id, context, conversation_chain):
    message = conversation_chain.invoke({"input": message_body, "context": context}, {"configurable": {"session_id": wa_id}})
    logging.info(f"Generated message: {message['output']}")
    return message['output']

# Delete messages from the message history
def trim_messages(messages, conversation_limit=10):
    if len(messages) > conversation_limit:
        return messages[-conversation_limit:]
    return messages

# Create chain
agent_executor = create_chain_agent()

# Create chain with trimming
agent_executor_with_message_trimming = (RunnablePassthrough.assign(chat_history=lambda x: trim_messages(x["chat_history"]))
 | agent_executor)

# Get the session history
conversation_chain = get_chat(agent_executor_with_message_trimming)

def generate_response(message_body, wa_id, name):
    """ Generates a response using the chat model and stores the conversation in the database. """
    context = vector_store.similarity_search(message_body)
    message = run_chain(message_body, wa_id, context, conversation_chain)
    store_message(wa_id, message_body, message)
    logging.info(f"Generated message: {message}")
    return message
