from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import psycopg2
from dotenv import load_dotenv
import logging
import os

# Configuración inicial
load_dotenv()
DB_CONNECTION = os.getenv("DB_CONNECTION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(model="gpt-4o", temperature=0.2)

# Plantilla del sistema
SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context. 
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""

def get_db_connection():
    """ Establece conexión con la base de datos PostgreSQL. """
    return psycopg2.connect(DB_CONNECTION)

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


def create_chain():
    """ Crea y configura la cadena de procesamiento con Langchain. """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    return prompt | chat

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
    chain = create_chain()
    conversation_chain = get_chat(chain)

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


# Uso de ejemplo:
# response = generate_response("me llamo Pablo", "123", "Pablo")
# print(get_session_history("123"))
