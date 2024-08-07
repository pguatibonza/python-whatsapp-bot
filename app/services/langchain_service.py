from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from supabase import create_client
from langchain_community.document_loaders import UnstructuredFileLoader
from langsmith.wrappers import wrap_openai
import logging
import os

from . import tools_restaurant
from . import supabase_service
from . import tools_parra



# Configuración inicial
load_dotenv()
DB_CONNECTION = os.getenv("DB_CONNECTION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = os.getenv("TABLE_NAME")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2=os.getenv("LANGCHAIN_TRACING_V2")


chat = ChatOpenAI(model="gpt-4o", temperature=0)
tools = tools_restaurant.TOOLS
tools=tools_parra.TOOLS

# Crear cliente de Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings()  # Inicializar embeddings
vector_store = supabase_service.load_vector_store()



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

SYSTEM_TEMPLATE= """
Eres un asistente virtual del concesionario Parra arango, que se encargara de atender todas las consultas sobre carros, repuestos y agendaciones que haga el usuario.
Cuando el usuario te pregunte por un carro, dale la informacion mas resumida posible. 
Cuando te pida una comparacion o recomendacion dale resumidamente una conclusion de cada modelo.
Cuando el usuario te pregunte por un tipo de carro, y sus modelos, muestrale todos los modelos disponibles de dicho carro.

1. La primera funcion que podras hacer, sera agendar un test drive. No crees la cita hasta que no esten los datos requeridos.  Vas a seguir los siguientes pasos :
    a. Le preguntaras el modelo del carro que necesita. Es posible que tu le hayas recomendado un modelo al usuario, entonces no le preguntes si ya sabes sobre que carro es el test drive
    b. Le pediras al usuario su nombre, apellido, y correo electronico
    c. Le preguntaras para que fecha y hora lo necesita. Asume que estamos en el 2024. Apenas te de la fecha, conviertela al formato  'ISO 8601 YYYY-MM-DDTHH:MM:SS-05:00'. La fecha de finalizacion es una hora despues de la fecha de inicio.
    d. Preguntale si tiene alguna sugerencia o nota


Para responer las preguntas tecnicas te vas a basar en la siguiente información :
<context>
{context}
</context>


"""
# # Load pdfs and create text splits
def create_from_directory(file_directory):
    embeddings=OpenAIEmbeddings()
    data=[]
    for file in os.listdir(file_directory):
        path=os.path.join(file_directory,file)
        loader=UnstructuredFileLoader(path)
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


# # db = add_pdfs_from_directory("../../data/")
#db = add_pdfs_from_directory("data/")


# Gets the chat history based on the cellphone number (id)
# For production use cases, you will want to use a persistent implementation of chat message history, such as RedisChatMessageHistory

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

retriever=vector_store.as_retriever(search_kwargs={"k":4})

def generate_response(message_body, wa_id):
    """ Generates a response using the chat model and stores the conversation in the database. """
    context = retriever.invoke(message_body)
    message = run_chain(message_body, wa_id, context, conversation_chain)
    logging.info(f"Generated message: {message}")
    return message
