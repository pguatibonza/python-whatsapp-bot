from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
import psycopg2
import numpy as np
from dotenv import load_dotenv
import logging
import os

SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context. 
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""

load_dotenv()
DB_CONNECTION = os.getenv("DB_CONNECTION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(model="gpt-4o", temperature=0.2)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        logging.info(f"Creating new session history for {session_id}")
        store[session_id] = ChatMessageHistory()
        logging.info(f"Session history for {session_id} created")
    return store[session_id]

def create_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | chat
    return chain

def get_chat(chain):
    chain_with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return chain_with_message_history

def run_chain(message_body, wa_id, context, conversation_chain):
    message = conversation_chain.invoke({"input": message_body, "context": context}, {"configurable": {"session_id": wa_id}})
    logging.info(f"Generated message: {message.content}")
    return message.content

def trim_messages(chain_input, wa_id, conversation_limit=4):
    stored_messages = store[wa_id].messages
    if len(stored_messages) <= conversation_limit:
        return False
    store[wa_id].clear()
    for message in stored_messages[-conversation_limit:]:
        store[wa_id].add_message(message)
    return True

def query_pgvector(conn, table_name, query, embeddings, k=4):
    query_vector = embeddings.embed_query(query)
    query_vector_str = ','.join(map(str, query_vector))
    with conn.cursor() as cursor:
        cursor.execute(f"""
            SELECT content
            FROM {table_name}
            ORDER BY vector <-> %s::vector
            LIMIT %s;
        """, (f'[{query_vector_str}]', k))
        results = cursor.fetchall()
        return [result[0] for result in results]

def generate_response(message_body, wa_id, name):
    chain = create_chain()
    conversation_chain = get_chat(chain)

    conn = psycopg2.connect(DB_CONNECTION)
    context = query_pgvector(conn, "menu_magdalena_rooftop", message_body, OpenAIEmbeddings())
    conn.close()

    response_message = run_chain(message_body, wa_id, context, conversation_chain)
    return response_message

# Example usage:
# response = generate_response("me llamo Pablo", "123", "Pablo")
# print(get_session_history("123"))
