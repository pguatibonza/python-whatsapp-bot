from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import psycopg2
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_openai_tools_agent
import logging
import os

from . import tools_restaurant 

# Configuración inicial
load_dotenv()
DB_CONNECTION = os.getenv("DB_CONNECTION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(model="gpt-4o", temperature=0.2)
tools=tools_restaurant.TOOLS

# Plantilla del sistema
SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context. 
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":
>>>>>>> persistent_db

Una de tus funciones sera responder cualquier duda que tenga el cliente 
respecto al menu del restaurante. Para ello podras usar el contexto que está abajo.
<context>
{context}
</context>

Por otro lado, tambien tendras a la mano 2 funciones, una para validar los horarios disponibles dada una fecha y una cantidad de personas, y otra para hacer la reserva
de una mesa. 

1. Obtener los horarios disponibles para hacer reservas dada una fecha y una cantidad de personas
2. hacer una reserva. Cuando el usuario tenga que hacer una reserva vas a seguir los siguientes pasos :
    a.  le preguntas la fecha y la cantidad de personas. Tomas la fecha y la conviertes al formato YYYY-MM-DD, si el usuario no te da el año, asume que es 2024.
    b. Validas si esa fecha esta disponible y le muestras los horarios.
    c.  Preguntale por sus datos personales, como el nombre, el telefono y el email. 
        Asegurate que tenga nombre y apellido
        Valida que el numero telefonico sea correcto, y agrega el sufijo "+57" al telefono del usuario si es necesario,
        Asegurate que la dirección de correo electronico sea valida
    d. Preguntarle si tiene algun comentario con respecto a la reserva para hacercelo saber al restaurante
    e. Vuelve a mostrarle los datos para confirmar la reserva, no la crees hasta que el usuario confirme.No le puedes dar el id de la reserva
"""

def get_db_connection():
    """ Establece conexión con la base de datos PostgreSQL. """
    return psycopg2.connect(DB_CONNECTION)


#No carga bien
def upload_file_unstructured_pdf(file_path):
    loader=UnstructuredPDFLoader(file_path)
    data=loader.load()
    return data

#Mejor, segementa mejor la info
def upload_file_miner(file_path):
    loader=PDFMinerLoader(file_path)
    data=loader.load()
    return data
#Load pdfs and create text splits
def create_from_directory(file_directory):
    embeddings=OpenAIEmbeddings()
    data=[]
    for file in os.listdir(file_directory):
        path=os.path.join(file_directory,file)
        loader=PDFMinerLoader(path)
        data+=loader.load()

        logging.info(f"Documento cargado desde el archivo {path}")

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)

    all_splits=text_splitter.split_documents(data)

    logging.info("Documentos spliteados")
    return all_splits

#Take a directory and create/add vectors to the vector store
def add_pdfs_from_directory(file_directory):
    embeddings=OpenAIEmbeddings()
    try:
        vector_store=FAISS.load_local("db",embeddings)
        logging.info("Vector store cargada")
        all_splits=create_from_directory(file_directory)
        vector_store.add_documents(all_splits,embeddings)
        logging.info("Documentos añadidos")
    except :
        all_splits=create_from_directory(file_directory)
        logging.info("Documentos añadidos y creados")
        vector_store=FAISS.from_documents(all_splits,embeddings)
        logging.info("Vector store creada")
    vector_store.save_local("db")

    logging.info("Vector store guardada")
    return vector_store


#db=add_pdfs_from_directory("../../data/")
db=add_pdfs_from_directory("data/")

store = {}
#Gets the chat history based on the cellphone number (id)
#For production use cases, you will want to use a persistent implementation of chat message history, such as RedisChatMessageHistory

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """ Fetches or initializes chat message history for a given session ID. """
    conn = get_db_connection()
    history = ChatMessageHistory()
    with conn.cursor() as cursor:
        cursor.execute("SELECT message, response FROM conversations WHERE phone_number = %s ORDER BY timestamp DESC", (session_id,))
        records = cursor.fetchall()
        for record in records:
            # Combine message and response into one string or adjust as necessary
            combined_message = f"User: {record[0]}, Bot: {record[1]}"
            history.add_message(combined_message)  # Adjusted to pass one parameter
    conn.close()

    if not records:
        logging.info(f"Creating new session history for {session_id}")
    else:
        logging.info(f"Session history for {session_id} retrieved with {len(records)} messages")
    
    return history




#Create a Prompt template  with a LLM model
def create_chain_agent():

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent= create_openai_tools_agent(chat,tools,prompt)
    agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)
    #chain=create_stuff_documents_chain(chat,prompt)
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


#Generates a response based on an input
def run_chain(message_body,wa_id,context,conversation_chain):
    message=conversation_chain.invoke({"input": message_body,"context":context},{"configurable":{"session_id" : wa_id}})
    logging.info(f"Generated message :  {message['output']}")
    return message['output']

#Delete messages from the database
def trim_messages(chain_input,wa_id,conversation_limit=4):

    stored_messages = store[wa_id].messages
    if len(stored_messages) <= conversation_limit:
        return False

    store[wa_id].clear()

    for message in stored_messages[-conversation_limit:]:
        store[wa_id].add_message(message)
    return True


#Create chain
agent_executor = create_chain_agent()

#Get the session history
conversation_chain=get_chat(agent_executor)

def query_pgvector(conn, table_name, query, k=4):
    """ Consulta vectores utilizando pgvector para obtener documentos relevantes. """
    embeddings = OpenAIEmbeddings()
    query_vector = embeddings.embed_query(query)
    query_vector_str = ','.join(map(str, query_vector))
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT content FROM {table_name} ORDER BY vector <-> %s::vector LIMIT %s;
        """.format(table_name=table_name), (f'[{query_vector_str}]', k))
        results = cursor.fetchall()
    return [result[0] for result in results]

def generate_response(message_body, wa_id, name):
    """ Generates a response using the chat model and stores the conversation in the database. """


    conn = get_db_connection()
    context = query_pgvector(conn, "menu_magdalena_rooftop", message_body)
    conn.close()


    response_message = conversation_chain.invoke({
        "input": message_body,
        "context": context
    }, {"configurable": {"session_id": wa_id}})

    # Store the received message and the generated response in the database
    store_message(wa_id, message_body, response_message.content)

    logging.info(f"Generated message: {response_message.content}")
    return response_message.content

