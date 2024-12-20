from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pprint import pprint
from typing import Any,  Literal, Union
from langchain_core.messages import  AnyMessage
from langchain.schema import AIMessage
from typing import Annotated, Literal, Optional
from langchain_core.messages import ToolMessage, HumanMessage, RemoveMessage,SystemMessage
from langgraph.prebuilt import ToolNode
import logging
import os
from dotenv import load_dotenv
load_dotenv()
# import tools
# import prompts 
# from graphrag_service import search_engine

from . import tools
from . import prompts 
from .graphrag_service import search_engine

DB_CONNECTION = os.getenv("DB_CONNECTION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = os.getenv("TABLE_NAME")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2=os.getenv("LANGCHAIN_TRACING_V2")
TABLE_NAME_VEHICLES=os.getenv("TABLE_NAME_VEHICLES")

chat = ChatOpenAI(model="gpt-4o", temperature=0)
# Crear cliente de Supabase
embeddings = OpenAIEmbeddings()  # Inicializar embeddings
#vector_store = supabase_service.load_vector_store()
#retriever=vector_store.as_retriever(search_kwargs={"k":3})
#memory = SqliteSaver.from_conn_string(":memory:") #despues se conecta a bd
memory=MemorySaver()


#LLM with function call
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent= llm.bind_tools([tools.toAppointmentAssistant,tools.toRagAssistant])


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompts.PRIMARY_ASSISTANT_PROMPT),
        ("human", "{input}"),
        ("placeholder","{messages}"),
        
    ]
).partial(time=datetime.now())

main_agent=prompt | agent

###Appointment Assistant

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent= llm.bind_tools([tools.tool_create_event_test_drive,tools.tool_get_available_time_slots,tools.tool_is_time_slot_available])

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompts.APPOINTMENT_ASSISTANT_PROMPT),
        ("human", "{input}"),
        ("placeholder","{messages}"),
        
    ]
).partial(time=datetime.now())

appointment_agent=prompt | agent

### RAG ASSISTANT
llm = ChatOpenAI(model="gpt-4o", temperature=0)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompts.CONTEXTUAL_ASSISTANT_PROMPT),
        ("human", "{user_input}"),
        ("placeholder", "{messages}")
    ]
).partial(time=datetime.now())

rag_agent = prompt | llm 


# Post-processing
def format_docs(docs):
    """
    Formats a list of documents into a single string.

    Args:
        docs (List[Document]): List of Document objects to format.

    Returns:
        str: Formatted string containing the content of all documents.
    """
    return "\n\n".join(doc.page_content for doc in docs)


### Context Router

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompts.QUERY_IDENTIFIER_PROMPT),
        ("human", "{user_input}"),
        ("placeholder", "{messages}")
    ]
)


context_router = prompt | llm.bind_tools([tools.CompleteOrEscalate,tools.QueryIdentifier,tools.MultimediaIdentifier])

### Get multimedia info

llm = ChatOpenAI(model="gpt-4o",temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompts.MULTIMEDIA_ASSISTANT_PROMPT),
        ("human", "{user_input}"),
        ("placeholder", "{messages}")
    ]
)

multimedia_agent = prompt | llm.bind_tools([tools.tool_get_car_technical_info])

###Graph state

def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """
    Updates the dialog stack by either pushing or popping states.

    Args:
        left (list[str]): Current dialog stack.
        right (Optional[str]): The new state to add or "pop" to remove the last state.

    Returns:
        list[str]: Updated dialog stack.
    """

    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]

class GraphState(TypedDict):
    """
    Defines the structure for maintaining graph state during assistant execution.

    Attributes:
        user_input (str): Input provided by the user.
        messages (list): List of messages exchanged during the dialog.
        dialog_state (list[str]): Current dialog state stack.
        summary (str): Summary of the conversation.
        context (str): Additional context information.
    """

    user_input:str
    messages : Annotated[list, add_messages]
    dialog_state : Annotated[list[Literal["primary_assistant","rag_assistant"]],update_dialog_stack]
    summary : str = ""
    context : str =""



### Nodes

def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    """
    Creates an entry node for an assistant.

    Args:
        assistant_name (str): Name of the assistant.
        new_dialog_state (str): State to transition to.

    Returns:
        Callable: Function to create an entry node.
    """
    def entry_node(state: GraphState) -> dict:
        messages=[]
        for tool_call in state["messages"][-1].tool_calls:
            tool_call_id = tool_call["id"]
            message =ToolMessage(
                        content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user.",
                        tool_call_id=tool_call_id,
                    )
            messages.append(message)
        return {
            "messages" : messages,
            "dialog_state": new_dialog_state,
        }

    return entry_node

async def primary_assistant(state):
    """
    Generates a response based on the current state using the primary assistant.

    Args:
        state (GraphState): Current state of the conversation.

    Returns:
        dict: Updated state with appended response.
    """
    logging.debug("Entering primary_assistant node")
    user_input=state["user_input"]
    
    response=await main_agent.ainvoke({"input":user_input,"messages":state['messages'], "summary":state.get("summary","")})
    
    logging.debug(f"Response from primary_assistant: {response.content}")
    return {"messages": [response],"user_input":user_input}

async def appointment_assistant(state):
    """
    Generates a response based on the current state using the .

    Args:
        state (GraphState): Current state of the conversation.

    Returns:
        dict: Updated state with appended response.
    """
    logging.debug("Entering appointment_assistant node")

    user_input=state["user_input"]
    response=await appointment_agent.ainvoke({"input":user_input,"messages":state["messages"],"summary":state.get("summary","")})
    
    logging.debug(f"Response from appointment_assistant : {response.content}")
    return {"messages":[response]}

async def rag_router(state):
    logging.debug("Entering  rag router")
    
    user_input=state["user_input"]
    response=await context_router.ainvoke({"user_input":user_input,"messages":state["messages"],"summary":state.get("summary","")})
    
    logging.debug(f"Tool called : {response}")
    return {"messages" : [response]}


async def graph_rag(state):
    logging.debug("Entering graph rag node")

    tool_call=state["messages"][-1].tool_calls[0]
    tool_call_id=tool_call["id"]
    query=tool_call["args"]["query"]
    message=ToolMessage(content="Now accessing to the graph rag database." ,tool_call_id=tool_call_id)

    response = await search_engine.asearch(query)
    logging.debug(f"Response from graph rag : {response.response}")
    return {"context":response.response,"messages":[message]}

async def rag_assistant(state):
    logging.debug("Entering rag answering node")
    
    user_input=state["user_input"]
    context=state.get("context","")
    response=await  rag_agent.ainvoke({"user_input":user_input,"messages":state["messages"],"context":context ,"summary":state.get("summary","")})
    
    logging.debug(f"Response from rag assistant : {response.content}")
    return {"messages": [response]}

async def multimedia_assistant(state):
    logging.debug("Entering multimedia assistant node")
    
    user_input=state["user_input"]
    response = await multimedia_agent.ainvoke({"user_input":user_input,"messages":state["messages"],"summary":state.get("summary","")})
    
    logging.debug(f"Response from multimedia assistant : {response.content}")
    return {"messages":[response]}

async def summarize_conversation(state: GraphState):
    # Get any existing summary
    summary = state.get("summary", "")
    
    # Create summarization prompt
    if summary:
        # A summary already exists
        summary_message = (
            f"This is the summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    
    # Add prompt to the history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = await llm.ainvoke(messages)

    # Identify the last 2 messages, including their tool call pairs if necessary
    last_two_messages = state["messages"][-2:]  # Get the last two messages
    final_messages = []  # To store the final messages we want to keep

    for message in last_two_messages:
        
        final_messages.append(message)
        print(message)
        # If the message is a ToolMessage, make sure to include the following AI response
        if isinstance(message, ToolMessage):
            print("tool")
            tool_index = state["messages"].index(message)
            if tool_index - 1 < len(state["messages"]):
                i=1
                while  isinstance(state["messages"][tool_index - i],ToolMessage):
                    final_messages.append(state["messages"][tool_index - i])
                    i+=1   
                final_messages.append(state["messages"][tool_index - i])
    # Now prepare the list of messages to delete (all messages except the final ones)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"] if m not in final_messages]

    return {"summary": response.content, "messages": delete_messages}



### Edges ###


#Define tool routing
def should_summarize(state:GraphState):
        
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 20:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END


def route_primary_assistant(
    state: GraphState,
) -> Literal[
    "enter_rag_assistant",
    "enter_appointment_assistant",
    "summarize_conversation",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        summarize=should_summarize(state)
        return summarize
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == tools.toRagAssistant.__name__:
            return "enter_rag_assistant"
        elif tool_calls[0]["name"]==tools.toAppointmentAssistant.__name__:
            return "enter_appointment_assistant"
        else:
            raise ValueError("Invalid tool call")
    raise ValueError("Invalid route")

def route_to_workflow(
    state: GraphState,
) -> Literal[
    "primary_assistant",
    "router_rag_assistant",
    "appointment_assistant"
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
    "summarize_conversation",
    "__end__",
]:
    #route = tools_condition(state)
    #if route == END:
    summarize=should_summarize(state)
    return summarize
    # tool_calls = state["messages"][-1].tool_calls
    # did_cancel = any(tc["name"] == tools.CompleteOrEscalate.__name__ for tc in tool_calls)
    # if did_cancel:
    #     return "leave_skill"
    
def route_rag_assistant(state:GraphState) -> Literal["leave_skill","graph_rag","rag_assistant","enter_multimedia_assistant"]:
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == tools.CompleteOrEscalate.__name__:
            return "leave_skill"
        elif tool_calls[0]["name"] == tools.QueryIdentifier.__name__:
            return "graph_rag"
        elif tool_calls[0]["name"] ==tools.MultimediaIdentifier.__name__:
            return "enter_multimedia_assistant"
    return "rag_assistant"

def route_multimedia_assistant(state):
    tool_calls=state["messages"][-1].tool_calls
    if tool_calls:
        return "tools_multimedia"
    return should_summarize(state)

def route_appointment_assistant(state):
    tool_calls=state["messages"][-1].tool_calls
    if tool_calls:
        return "tools_appointment"
    return should_summarize(state)

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

#Entry
workflow.add_conditional_edges(START,route_to_workflow)

#Primary assistant
workflow.add_node("primary_assistant",primary_assistant)
workflow.add_conditional_edges("primary_assistant",route_primary_assistant)

#Appointment assistant
workflow.add_node("enter_appointment_assistant",create_entry_node("Appointment assistant","appointment_assistant"))
workflow.add_node("appointment_assistant",appointment_assistant)
workflow.add_conditional_edges("appointment_assistant",route_appointment_assistant)

workflow.add_edge("enter_appointment_assistant","appointment_assistant")
workflow.add_node("tools_appointment",ToolNode([tools.tool_create_event_test_drive,tools.tool_is_time_slot_available,tools.tool_get_available_time_slots]))
workflow.add_edge("tools_appointment","appointment_assistant")

#Router rag
workflow.add_node("router_rag_assistant",rag_router)
workflow.add_conditional_edges("router_rag_assistant", route_rag_assistant)

#Rag assistant
workflow.add_node("enter_rag_assistant", create_entry_node("RAG assistant", "router_rag_assistant"))  
workflow.add_node("rag_assistant", rag_assistant)
workflow.add_edge("enter_rag_assistant","router_rag_assistant")

#Multimedia assistant
workflow.add_node("enter_multimedia_assistant",create_entry_node("Multimedia Assistant","router_rag_assistant"))
workflow.add_edge("enter_multimedia_assistant","multimedia_assistant")

workflow.add_node("multimedia_assistant",multimedia_assistant)
workflow.add_node("tools_multimedia",ToolNode([tools.tool_get_car_technical_info]))
workflow.add_edge("tools_multimedia","multimedia_assistant")
workflow.add_conditional_edges("multimedia_assistant", route_multimedia_assistant)


#Graph rag
workflow.add_node("graph_rag",graph_rag)
workflow.add_edge("graph_rag","rag_assistant")

workflow.add_conditional_edges("rag_assistant", route_assistants)
#Utilities
workflow.add_node("leave_skill",pop_dialog_state)
workflow.add_edge("leave_skill", "primary_assistant")

workflow.add_node("summarize_conversation",summarize_conversation)
workflow.add_edge("summarize_conversation",END)

# Compile
app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "6"}}

# from pprint import pprint

# Run

async def generate_response(message_body,wa_id):
    config={"configurable": {"thread_id":wa_id}}
    inputs={"messages": message_body, "user_input": message_body}
    messages_output=[]
    async for output in app.astream(inputs,config=config):
        for key, value in output.items():
            pprint(f"Node '{key}':")
            #pprint(value, indent=2, width=80, depth=None)
            if 'messages' in value:
                output_message=value["messages"][-1]
                if isinstance(output_message,AIMessage) and output_message.content!='':
                    messages_output.append(output_message.content)
        pprint("\n---\n")
    return messages_output
