from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import logging
import os


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

store = {}
#Gets the chat history based on the cellphone number (id)
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    #Si no existe un historial de mensajes, crea uno
    if session_id not in store:
        logging.info(f"Creating new session history for {session_id}")
        store[session_id] = ChatMessageHistory()
        logging.info(f"Session history for {session_id} created")
    return store[session_id]

#Create a Prompt template  with a LLM model
def create_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | chat
    return chain

# Get the chain with the session history
def get_chat(chain):
    chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    )
    return chain_with_message_history

#Generates a response based on an input
def run_chain(message_body,wa_id,conversation_chain):
    message=conversation_chain.invoke({"input": message_body, },{"configurable":{"session_id" : wa_id}})
    logging.info(f"Generated message :  {message.content}")
    return message.content

def trim_messages(chain_input,wa_id,conversation_limit):

    stored_messages = store[wa_id].messages
    if len(stored_messages) <= conversation_limit:
        return False

    store[wa_id].clear()

    for message in stored_messages[-conversation_limit:]:
        store[wa_id].add_message(message)
    return True



def generate_response(message_body,wa_id,name):
    #Create chain
    chain = create_chain()

    #Get the session history
    conversation_chain=get_chat(chain)

    wa_id="3227077343"
    #Delete messages if they exceed the conversation limit
    #conversation_chain_with_trimming=(RunnablePassthrough.assign(messages_trimmed=trim_messages) | {"chain_input":conversation_chain,"wa_id":wa_id})

    #Create response
    response_message=run_chain(message_body,wa_id,conversation_chain)

    return response_message



