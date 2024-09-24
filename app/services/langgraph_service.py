from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from supabase import create_client
#from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.output_parsers import StrOutputParser
from typing import Annotated, Callable
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import Document
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Literal
from typing import List
from typing_extensions import TypedDict
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pprint import pprint
from typing import Any,  Literal, Union
from langchain_core.messages import  AnyMessage
from langchain.schema import AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from typing import Annotated, Literal, Optional
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
import logging
import os
from . import tools_restaurant
from . import supabase_service
from . import tools

load_dotenv()
DB_CONNECTION = os.getenv("DB_CONNECTION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = os.getenv("TABLE_NAME")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2=os.getenv("LANGCHAIN_TRACING_V2")




chat = ChatOpenAI(model="gpt-4o", temperature=0)
# Crear cliente de Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings()  # Inicializar embeddings
vector_store = supabase_service.load_vector_store()
retriever=vector_store.as_retriever(search_kwargs={"k":20})
#memory = SqliteSaver.from_conn_string(":memory:") #despues se conecta a bd
memory=MemorySaver()






#LLM with function call
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_tools=tools.TOOLS
agent= llm.bind_tools([tools.tool_create_event_test_drive,tools.toRagAssistant])


#Prompt

system="""
YYou are a customer support assistant at Los Coches, a car dealership offering Volkswagen and Renault vehicles. 
Your role is to assist customers by:

Answering Questions: Provide accurate and helpful information about the vehicles available, always chech for availability. 
This includes specifications, features, pricing, availability, and financing options. (Always check context)
Only the specialized assistant is given the permission to do this for the customer. 

The customer is not aware of the different specialized assistant, so do not mention them; just quietly delegate 
through function calls.

Scheduling Appointments: Help customers set up appointments for test drives. 
***Collect necessary information such as their name, email, preferred date and time, and the specific vehicle models they are interested in. If the user want to add comments, let him, but it is optional
The dates must be in the following format :  YYYY-MM-DDTHH:MM:SS-05:00. Quietly add the UTC format without asking the user.
Don't show the user the format you are using, only ask day and hour. 

Make sure your output is in WhatsApp format to have bold titles and everything. For example, instead of using ### use * to bold the text.

The length of your answers shouldn't surpass 200 words, only surpass if its absolutely necessary.

Ask the customer their name on the first message and ALWAYS refer to them by their name. Your first message should be "¡Hola! Bienvenido a Los Coches, ¿Con quien tengo el gusto de hablar?"

When comparing vehicles give a brief description of both vehicles and their prices, at the end give the conclusion of what is the strengths of each car.

When a customer gives a budget, always try to give 2 options if possible. Option #1 should be the option that suits perfectly within the customers conditions. Option number 2 should be a car that is between 10%-15% outside their budget, and your purpose is to try to upsell the vehicle by giving a better sales pitch and giving finance options. 

You can always answer to car related questions, except when the customer tries to compare or look for vehicle information that we don't have.

Your goal is to enhance the customer experience by providing excellent service and facilitating their journey towards purchasing a vehicle from Los Coches.

Professional Interaction: Communicate in a friendly, professional, and courteous manner. 
Ensure that all customer inquiries are addressed promptly and thoroughly.

You must answer in Spanish

Current time = {time}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{input}"),
        ("placeholder","{messages}"),
        
    ]
).partial(time=datetime.now())

main_agent=prompt | agent



### Generate
#llm 
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools([tools.CompleteOrEscalate])

system="""
You are a specialized customer support assistant for Los Coches, a car dealership that offers Volkswagen and Renault. 
Your main function is to answer any requests customers have about Los Coches and the cars they offer.

Access to Context: You have comprehensive knowledge and access to detailed information about all vehicles in the Los Coches inventory. 
This includes specifications, features, pricing, availability, customer reviews, and current promotions or financing options. 
Use the  context provided below to give accurate and helpful responses to customer inquiries. 

You can only answer questions based on the context below.

When a customer gives a budget, always try to give 2 options if possible. Option #1 should be the option that suits perfectly within the customers conditions. Option number 2 should be a car that is between 10%-15% outside their budget, and your purpose is to try to upsell the vehicle by giving a better sales pitch and giving finance options. 

Customer Inquiries: Assist customers with any questions they may have about specific car models, compare different vehicles, 
provide recommendations based on their preferences and needs, and inform them about additional services offered by Los Coches.
When answering the user, you must first analyze the information we have(ALWAYS) and after that give him a summarized, and concise response.
Only answer with DETAILED explanations if the customer asked for it.

Make sure your output is in WhatsApp format to have bold titles and everything. For example, never use ###, instead use * to bold the text.

Only give information about the cars we have, never give information about cars Los Coches doesn't sell, EXTREMEY IMPORTANT.

Ask the customer their name on the first message and ALWAYS refer to them by their name. Your first message should be "¡Hola! Bienvenido a Los Coches, ¿Con quien tengo el gusto de hablar?"

If the context provided is not enough to answer the user inquiries, then CompleteOrEscalate
If the customer changes their mind, escalate the task back to the main assistant.
If the customer needs help and your function is not appropriate to answer him, then CompleteOrEscalate.
If the customer input is not about inquiries related to the car dealership, you must CompleteOrEscalate.
You must answer in Spanish
Customer request : {user_input}

Context : {context}

time : {time}

"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{user_input}"),
        ("placeholder", "{messages}")
    ]
).partial(time=datetime.now())

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_agent = prompt | llm 


### Question Re-writer

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Prompt
system = """Usted es un reformulador de preguntas que convierte una pregunta de entrada en una versión mejorada y optimizada
para la recuperación de información en un vectorstore. Analice la entrada e intente razonar sobre la intención / significado semántico subyacente. 
Tenga en cuenta el historial de mensajes del usuario para completar la pregunta, y mantenga el significado semantico subyacente"""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("placeholder","{messages}"),
        (
            "human",
            "Aqui esta la pregunta inicial: \n\n {question} \n Formule una respuesta mejorada.",
        )
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
#question_rewriter.invoke({"question": question})



###Graph state

def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    user_input:str
    messages : Annotated[list, add_messages]
    dialog_state : Annotated[list[Literal["primary_assistant","rag_assistant"]],update_dialog_stack]



### Nodes

def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: GraphState) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node

def primary_assistant(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the input, it will decide to use any tool, retrieve info, or keep chatting.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    
    user_input=state['messages'][-1]
    
    response=main_agent.invoke({"input":user_input,"messages":state['messages']})

    return {"messages": [response],"user_input":user_input.content}

def rag_assistant(state):

    if state['messages'][-2].tool_calls:
        user_input=state['messages'][-2].tool_calls[0]['args']['request']
    else :
        user_input = state['messages'][-1].content
     #Extrae contexto segun el query
    context= retriever.invoke(user_input)

    #Responde de acuerdo al contexto
    response= rag_agent.invoke({"user_input":user_input,"messages":state["messages"],"context":context})

    return {"messages": [response]}





### Edges ###


#Define tool routing


def route_primary_assistant(
    state: GraphState,
) -> Literal[
    "enter_rag_assistant",
    "tools",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == tools.toRagAssistant.__name__:
            return "enter_rag_assistant"
        elif tool_calls[0]["name"] == "create_event_test_drive":
            return "tools"
    raise ValueError("Invalid route")

def route_to_workflow(
    state: GraphState,
) -> Literal[
    "primary_assistant",
    "rag_assistant",
]:
    """If we are in a delegated state, route directly to the appropriate assistant."""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]

def route_assistants(
    state: GraphState,
) -> Literal[
    "leave_skill",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == tools.CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"

# This node will be shared for exiting all specialized assistants
def pop_dialog_state(state: GraphState) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the student as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }




workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_conditional_edges(START,route_to_workflow)
workflow.add_node("primary_assistant",primary_assistant)

workflow.add_node("enter_rag_assistant", create_entry_node("RAG assistant", "rag_assistant"))  
workflow.add_node("rag_assistant", rag_assistant)
workflow.add_edge("enter_rag_assistant","rag_assistant")

workflow.add_node("tools",ToolNode([tools.create_event_test_drive]))
workflow.add_edge("tools","primary_assistant")

workflow.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    {
        "enter_rag_assistant": "enter_rag_assistant",
        "tools": "tools",
        END: END,
    },
)
workflow.add_conditional_edges("rag_assistant", route_assistants)
workflow.add_node("leave_skill",pop_dialog_state)
workflow.add_edge("leave_skill", "primary_assistant")

# Compile
app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "6"}}

# from pprint import pprint

# Run

def generate_response(message_body,wa_id):
    config={"configurable": {"thread_id":"1111113"}}
    inputs={"messages": message_body}
    messages_output=[]
    for output in app.stream(inputs,config=config):
        for key, value in output.items():
            pprint(f"Node '{key}':")
            pprint(value, indent=2, width=80, depth=None)
            if 'messages' in value:
                output_message=value["messages"][-1]
                if isinstance(output_message,AIMessage) and output_message.content!='':
                    messages_output.append(output_message.content)
        pprint("\n---\n")
    return messages_output
