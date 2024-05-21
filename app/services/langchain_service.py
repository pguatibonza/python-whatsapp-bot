from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import shelve
from langchain_community.document_loaders import PDFMinerLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import logging
import os

SYSTEM_TEMPLATE="""
Answer the user's questions based on the below context. 
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(model="gpt-4o", temperature=0.2)



def upload_file_pypdf(file_path):
    loader=PyPDFLoader(file_path)
    pages=loader.load_and_split()
    return pages
#No carga bien
def upload_file_unstructured_pdf(file_path):
    loader=UnstructuredPDFLoader(file_path)
    data=loader.load()
    return data
#mas rapido
def upload_file_pymu(file_path):
    loader=PyMuPDFLoader(file_path)
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

    if session_id not in store:
        logging.info(f"Creating new session history for {session_id}")
        store[session_id]=ChatMessageHistory()
        logging.info(f"Session history for {session_id} created")

    return store[session_id]
        



#Create a Prompt template  with a LLM model
def create_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | chat
    #chain=create_stuff_documents_chain(chat,prompt)
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
def run_chain(message_body,wa_id,context,conversation_chain):
    message=conversation_chain.invoke({"input": message_body,"context":context },{"configurable":{"session_id" : wa_id}})
    logging.info(f"Generated message :  {message.content}")
    return message.content

#Delete messages from the database
def trim_messages(chain_input,wa_id,conversation_limit=4):

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

    #Create retriever
    retriever=db.as_retriever(k=4)
    context=retriever.invoke(message_body)

    #Delete messages if they exceed the conversation limit
    #conversation_chain_with_trimming=(RunnablePassthrough.assign(messages_trimmed=trim_messages) | conversation_chain)

    #Create response
    response_message=run_chain(message_body,wa_id,context,conversation_chain)

    return response_message

#response=generate_response("me llamo Pablo","123","Pablo")
#print(get_session_history("123"))

