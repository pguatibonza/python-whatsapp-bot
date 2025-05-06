"""
langgraph_service.py

This module implements the core workflow of the customer service chatbot using LangGraph.
It defines the state, nodes, and edges that govern the interaction between multiple specialized agents,
such as the primary assistant, appointment assistant, RAG assistant, multimedia assistant, and final response assistant.
The state graph is used to control conversation routing and delegation based on user queries.

The module also defines functions for:
    - Updating the dialog state.
    - Creating entry nodes for specialized assistants.
    - Asynchronous invocation of agents.
    - Routing logic to delegate queries to the appropriate agent.
    - Summarizing the conversation when the history grows too long.
    - Compiling the workflow and generating the final response.

External dependencies include LangChain, LangGraph, and environment configuration.
"""
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from typing import Annotated, Callable
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import Document
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Literal, List, Union, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pprint import pprint
from langchain_core.messages import AnyMessage
from langchain.schema import AIMessage
from langchain_core.messages import ToolMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
import logging
import os
from dotenv import load_dotenv
import asyncio
from asyncio import WindowsSelectorEventLoopPolicy
import sys

# Load environment variables
load_dotenv()
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

# Import our custom modules
from . import tools, prompts, agents
from .graphrag_service import search_engine



# Load environment configuration
DB_CONNECTION = os.getenv("DB_CONNECTION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = os.getenv("TABLE_NAME")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
TABLE_NAME_VEHICLES = os.getenv("TABLE_NAME_VEHICLES")
DB_URI = os.getenv("DB_URI")

# Initialize LLM and Embeddings (if needed)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()  # For potential future use (e.g., vector searches)

# -------------------------
# Instantiate Specialized Agents
# -------------------------
# Each agent is instantiated from the agents module.
main_agent = agents.PrimaryAgent()
appointment_agent = agents.AppointmentAgent()
rag_agent = agents.RagAgent()
response_agent = agents.FinalResponseAgent()
context_router = agents.ContextRouterAgent()
multimedia_agent = agents.MultimediaAgent()


# -------------------------
# Graph State and Utility Functions
# -------------------------
def update_dialog_stack(left: List[str], right: Optional[str]) -> List[str]:
    """
    Updates the dialog state stack by either pushing a new state or popping the last state.
    
    Args:
        left (List[str]): Current dialog state stack.
        right (Optional[str]): New state to add, or "pop" to remove the last state.
        
    Returns:
        List[str]: Updated dialog state stack.
    """
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]

class GraphState(TypedDict):
    """
    Defines the structure for maintaining the conversation state during assistant execution.
    
    Attributes:
        user_input (str): The latest input provided by the user.
        messages (list): A list of messages exchanged during the conversation.
        dialog_state (list[str]): The current state stack tracking active agents.
        summary (str): A running summary of the conversation.
        context (str): Additional context information (e.g., aggregated tool responses).
    """
    user_input: str
    messages: Annotated[List, add_messages]
    dialog_state: Annotated[List[Literal["primary_assistant", "rag_assistant"]], update_dialog_stack]
    summary: str
    context: str

def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    """
    Creates an entry node function for a specialized assistant.
    
    This function generates a ToolMessage to indicate the delegation to a specialized assistant.
    
    Args:
        assistant_name (str): The name of the assistant (e.g., "Appointment assistant").
        new_dialog_state (str): The dialog state to transition to after delegation.
        
    Returns:
        Callable: A function that creates an entry node with the appropriate message and state update.
    """
    def entry_node(state: GraphState) -> dict:
        messages = []
        # Iterate over tool calls in the last message and create corresponding ToolMessages.
        for tool_call in state["messages"][-1].tool_calls:
            tool_call_id = tool_call["id"]
            message = ToolMessage(
                content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user.",
                tool_call_id=tool_call_id,
            )
            messages.append(message)
        return {
            "messages": messages,
            "dialog_state": new_dialog_state,
        }
    return entry_node

# -------------------------
# Node Functions (Agent Invocations)
# -------------------------
async def primary_assistant(state: GraphState) -> dict:
    """
    Processes user input using the PrimaryAgent (general queries) with robust error handling.
    
    Args:
        state (GraphState): The current conversation state.
    
    Returns:
        dict: Updated state containing the agent's response or a fallback error message.
    """
    logging.debug("Entering primary_assistant node")
    user_input = state["user_input"]
    try:
        response = await main_agent.ainvoke({
            "user_input": user_input,
            "messages": state["messages"],
            "summary": state.get("summary", "")
        })
    except Exception as e:
        logging.error("Error in primary_assistant: %s", e)
        response = SystemMessage(content="Lo siento, ocurrió un error al procesar tu consulta general. Por favor, inténtalo de nuevo.")
    logging.debug(f"Response from primary_assistant: {response.content}")
    return {"messages": [response], "user_input": user_input}

async def appointment_assistant(state: GraphState) -> dict:
    """
    Schedules test drive appointments using the AppointmentAgent with robust error handling.
    
    Args:
        state (GraphState): The current conversation state.
    
    Returns:
        dict: Updated state with the appointment agent's response or fallback error message.
    """
    logging.debug("Entering appointment_assistant node")
    user_input = state["user_input"]
    try:
        response = await appointment_agent.ainvoke({
            "user_input": user_input,
            "messages": state["messages"],
            "summary": state.get("summary", "")
        })
    except Exception as e:
        logging.error("Error in appointment_assistant: %s", e)
        response = SystemMessage(content="Lo siento, no se pudo procesar la solicitud de cita. Por favor, inténtalo nuevamente.")
    logging.debug(f"Response from appointment_assistant: {response.content}")
    return {"messages": [response]}

async def rag_router(state: GraphState) -> dict:
    """
    Determines the appropriate routing for technical queries using the ContextRouterAgent with error handling.
    
    Args:
        state (GraphState): The current conversation state.
    
    Returns:
        dict: Updated state with the router's response or fallback error message.
    """
    logging.debug("Entering rag router node")
    user_input = state["user_input"]
    try:
        response = await context_router.ainvoke({
            "user_input": user_input,
            "messages": state["messages"],
            "summary": state.get("summary", "")
        })
        # Clear content to indicate that the router does not provide a final answer.
        response.content = ""
    except Exception as e:
        logging.error("Error in rag_router: %s", e)
        response = SystemMessage(content="Lo siento, ocurrió un error al determinar la ruta. Por favor, inténtalo de nuevo.")
    logging.debug(f"Router response: {response}")
    return {"messages": [response]}

async def graph_rag(state: GraphState) -> dict:
    """
    Aggregates technical data responses using tool calls in the RAG flow with error handling.
    
    Args:
        state (GraphState): The current conversation state.
    
    Returns:
        dict: Contains aggregated context and tool response messages, or fallback message on error.
    """
    logging.debug("Entering graph rag node")
    final_response = ""
    messages = []
    try:
        for tool_call in state["messages"][-1].tool_calls:
            tool_call_id = tool_call["id"]
            if tool_call["name"] == tools.QueryIdentifier.__name__:
                query = tool_call["args"]["query"]
                response = await search_engine.search(query)
                message = ToolMessage(content=f"{response.response}", tool_call_id=tool_call_id)
                final_response += response.response
            else:
                message = ToolMessage(content="Not valid tool call", tool_call_id=tool_call_id)
            messages.append(message)
    except Exception as e:
        logging.error("Error in graph_rag: %s", e)
        final_response = "Lo siento, ocurrió un error al obtener la información técnica."
        messages = [SystemMessage(content=final_response)]
    logging.debug(f"Response from graph rag: {final_response}")
    return {"context": final_response, "messages": messages}

async def db_context(state: GraphState) -> dict:
    """
    Retrieves general dealership information via a database lookup with error handling.
    
    Args:
        state (GraphState): The current conversation state.
    
    Returns:
        dict: Updated state with dealership context and corresponding messages, or fallback error message.
    """
    logging.debug("Entering db context node")
    final_response = ""
    messages = []
    try:
        for tool_call in state["messages"][-1].tool_calls:
            tool_call_id = tool_call["id"]
            if tool_call["name"] == tools.DealershipInfoIdentifier.__name__:
                query = tool_call["args"]["query"]
                response = tools.get_dealership_description(query)
                message = ToolMessage(content=f"{response}", tool_call_id=tool_call_id)
                final_response += response
            else:
                message = ToolMessage(content="Not valid tool call", tool_call_id=tool_call_id)
            messages.append(message)
    except Exception as e:
        logging.error("Error in db_context: %s", e)
        final_response = "Lo siento, no se pudo recuperar la información de concesionarios."
        messages = [SystemMessage(content=final_response)]
    logging.debug(f"Response from db context: {final_response}")
    return {"context": final_response, "messages": messages}

async def rag_assistant(state: GraphState) -> dict:
    """
    Generates a detailed technical response using the RagAgent with error handling.
    
    Args:
        state (GraphState): The current conversation state.
    
    Returns:
        dict: Updated state containing the RAG agent's response or a fallback message.
    """
    logging.debug("Entering rag assistant node")
    user_input = state["user_input"]
    context = state.get("context", "")
    try:
        response = await rag_agent.ainvoke({
            "user_input": user_input,
            "messages": state["messages"],
            "context": context,
            "summary": state.get("summary", "")
        })
    except Exception as e:
        logging.error("Error in rag_assistant: %s", e)
        response = SystemMessage(content="Lo siento, hubo un problema al procesar tu consulta técnica. Por favor, inténtalo de nuevo.")
    logging.debug(f"Response from rag assistant: {response.content}")
    return {"messages": [response]}

async def multimedia_assistant(state: GraphState) -> dict:
    """
    Fetches multimedia content or technical details using the MultimediaAgent with error handling.
    
    Args:
        state (GraphState): The current conversation state.
    
    Returns:
        dict: Updated state with the multimedia agent's response or fallback message.
    """
    logging.debug("Entering multimedia assistant node")
    user_input = state["user_input"]
    try:
        response = await multimedia_agent.ainvoke({
            "user_input": user_input,
            "messages": state["messages"],
            "summary": state.get("summary", "")
        })
    except Exception as e:
        logging.error("Error in multimedia_assistant: %s", e)
        response = SystemMessage(content="Lo siento, no se pudo obtener el contenido multimedia. Inténtalo nuevamente.")
    logging.debug(f"Response from multimedia assistant: {response.content}")
    return {"messages": [response]}

async def response_assistant(state: GraphState) -> dict:
    """
    Synthesizes a final answer using the FinalResponseAgent with robust error handling.
    
    Args:
        state (GraphState): The current conversation state.
    
    Returns:
        dict: Updated state with the final consolidated response or fallback error message.
    """
    logging.debug("Entering final response node")
    user_input = state["user_input"]
    try:
        last_response = state["messages"][-1].content  # Latest message as context
        response = await response_agent.ainvoke({
            "user_input": user_input,
            "messages": state["messages"],
            "summary": state.get("summary", ""),
            "last_response": last_response
        })
    except Exception as e:
        logging.error("Error in response_assistant: %s", e)
        response = SystemMessage(content="Lo siento, ocurrió un error al generar la respuesta final. Por favor, inténtalo de nuevo.")
    logging.debug(f"Response from Response assistant: {response.content}")
    
    return {"messages": [response]}

async def summarize_conversation(state: GraphState) -> dict:
    """
    Summarizes the conversation history to reduce context length and update the conversation summary,
    with error handling.
    
    Args:
        state (GraphState): The current conversation state.
    
    Returns:
        dict: Contains the updated summary and a list of messages to be removed, or fallback summary.
    """
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is the summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    try:
        response = await llm.ainvoke(messages)
    except Exception as e:
        logging.error("Error in summarize_conversation: %s", e)
        response = SystemMessage(content=summary or "No summary available.")
    
    # Determine final messages to keep.
    last_two_messages = state["messages"][-2:]
    final_messages = []
    for message in last_two_messages:
        final_messages.append(message)
        if isinstance(message, ToolMessage):
            tool_index = state["messages"].index(message)
            if tool_index - 1 < len(state["messages"]):
                i = 1
                while isinstance(state["messages"][tool_index - i], ToolMessage):
                    final_messages.append(state["messages"][tool_index - i])
                    i += 1
                final_messages.append(state["messages"][tool_index - i])
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"] if m not in final_messages]
    return {"summary": response.content, "messages": delete_messages}


# -------------------------
# Routing Functions (Edges)
# -------------------------
def should_summarize(state: GraphState) -> str:
    """
    Determines whether the conversation should be summarized based on the number of messages.
    
    Args:
        state (GraphState): The current conversation state.
    
    Returns:
        str: The next node key ("summarize_conversation") or END if no summarization is needed.
    """
    messages = state["messages"]
    if len(messages) > 20:
        return "summarize_conversation"
    return END

def route_primary_assistant(state: GraphState) -> Literal["enter_rag_assistant", "enter_appointment_assistant", "final_response", "__end__"]:
    """
    Routes the conversation from the primary assistant based on tool calls.
    
    Args:
        state (GraphState): The current conversation state.
    
    Returns:
        Literal: A string indicating the next node in the workflow.
    """
    route = tools_condition(state)
    if route == END:
        return "final_response"
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == tools.toRagAssistant.__name__:
            return "enter_rag_assistant"
        elif tool_calls[0]["name"] == tools.toAppointmentAssistant.__name__:
            return "enter_appointment_assistant"
        else:
            raise ValueError("Invalid tool call")
    raise ValueError("Invalid route")

def route_to_workflow(state: GraphState) -> Literal["primary_assistant", "router_rag_assistant", "appointment_assistant"]:
    """
    Routes directly to a delegated assistant if the dialog state indicates a transfer.
    
    Args:
        state (GraphState): The current conversation state.
    
    Returns:
        Literal: The target node key for the next agent.
    """
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]

def route_rag_assistant(state: GraphState) -> Literal["leave_skill", "graph_rag", "rag_assistant", "enter_multimedia_assistant", "db_context"]:
    """
    Routes the conversation for technical queries by examining tool calls in the last message.
    
    Args:
        state (GraphState): The current conversation state.
    
    Returns:
        Literal: The next node key for the appropriate specialized agent.
    """
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == tools.CompleteOrEscalate.__name__:
            return "leave_skill"
        elif tool_calls[0]["name"] == tools.QueryIdentifier.__name__:
            return "graph_rag"
        elif tool_calls[0]["name"] == tools.DealershipInfoIdentifier.__name__:
            return "db_context"
        elif tool_calls[0]["name"] == tools.MultimediaIdentifier.__name__:
            return "enter_multimedia_assistant"
    return "rag_assistant"

def route_multimedia_assistant(state) -> str:
    """
    Routes the conversation from the multimedia assistant based on tool calls.
    
    Args:
        state: The current conversation state.
    
    Returns:
        str: Next node key ("tools_multimedia" or "final_response").
    """
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        return "tools_multimedia"
    return "final_response"

def route_appointment_assistant(state) -> str:
    """
    Routes the conversation from the appointment assistant based on tool calls.
    
    Args:
        state: The current conversation state.
    
    Returns:
        str: Next node key ("leave_skill", "tools_appointment", or "final_response").
    """
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == tools.CompleteOrEscalate.__name__:
            return "leave_skill"
        return "tools_appointment"
    return "final_response"

def pop_dialog_state(state: GraphState) -> dict:
    """
    Pops the current dialog state, returning control to the primary assistant.
    
    This function creates a ToolMessage to signal the resumption of the host assistant.
    
    Args:
        state (GraphState): The current conversation state.
    
    Returns:
        dict: Updated state with a "pop" signal and corresponding message.
    """
    messages = []
    if state["messages"][-1].tool_calls:
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

# -------------------------
# Build the State Graph
# -------------------------
workflow = StateGraph(GraphState)

# Define the nodes and edges for the conversation flow.

# Entry node: Determine routing based on the current state.
workflow.add_conditional_edges(START, route_to_workflow)

# Primary assistant node and its routing.
workflow.add_node("primary_assistant", primary_assistant)
workflow.add_conditional_edges("primary_assistant", route_primary_assistant)

# Appointment assistant nodes.
workflow.add_node("enter_appointment_assistant", create_entry_node("Appointment assistant", "appointment_assistant"))
workflow.add_node("appointment_assistant", appointment_assistant)
workflow.add_conditional_edges("appointment_assistant", route_appointment_assistant)
workflow.add_edge("enter_appointment_assistant", "appointment_assistant")
workflow.add_node("tools_appointment", ToolNode([tools.tool_create_event_test_drive, tools.tool_is_time_slot_available, tools.tool_get_available_time_slots, tools.CompleteOrEscalate]))
workflow.add_edge("tools_appointment", "appointment_assistant")

# Router node for RAG.
workflow.add_node("router_rag_assistant", rag_router)
workflow.add_conditional_edges("router_rag_assistant", route_rag_assistant)

# RAG assistant nodes.
workflow.add_node("enter_rag_assistant", create_entry_node("RAG assistant", "router_rag_assistant"))
workflow.add_node("rag_assistant", rag_assistant)
workflow.add_edge("enter_rag_assistant", "router_rag_assistant")
workflow.add_edge("rag_assistant", "final_response")

# Multimedia assistant nodes.
workflow.add_node("enter_multimedia_assistant", create_entry_node("Multimedia Assistant", "router_rag_assistant"))
workflow.add_edge("enter_multimedia_assistant", "multimedia_assistant")
workflow.add_node("multimedia_assistant", multimedia_assistant)
workflow.add_node("tools_multimedia", ToolNode([tools.tool_get_car_technical_info]))
workflow.add_edge("tools_multimedia", "multimedia_assistant")
workflow.add_conditional_edges("multimedia_assistant", route_multimedia_assistant)

# Graph RAG node.
workflow.add_node("graph_rag", graph_rag)
workflow.add_edge("graph_rag", "rag_assistant")

# DB context node.
workflow.add_node("db_context", db_context)
workflow.add_edge("db_context", "rag_assistant")

# Final response node.
workflow.add_node("final_response", response_assistant)
workflow.add_conditional_edges("final_response", should_summarize)

# Utility nodes.
workflow.add_node("leave_skill", pop_dialog_state)
workflow.add_edge("leave_skill", "primary_assistant")
workflow.add_node("summarize_conversation", summarize_conversation)
workflow.add_edge("summarize_conversation", END)

# Initialize memory checkpointer for the state graph
# (Here using in-memory storage; consider switching to a persistent database for production)
# Optional: tune these for your workload
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}
async def _init_graph():

    # 1. create async pool
    pool = AsyncConnectionPool(
        conninfo=DB_URI,
        max_size=20,
        kwargs=connection_kwargs,
    )

    await pool.open()
    # 2. make async saver and ensure table exists
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    # 3. compile once with this saver
    return workflow.compile(checkpointer=saver)
    # at module import, block until Graph is ready
   
# Compile the state graph with memory checkpointer.
app = asyncio.get_event_loop().run_until_complete(_init_graph())
config = {"configurable": {"thread_id": "6"}}

# -------------------------
# Generate Response Function
# -------------------------
async def generate_response(message_body, wa_id):
    """
    Asynchronously generates a final response based on the conversation state.
    
    This function iterates over the compiled state graph's asynchronous stream,
    filters for messages from the final response node, and returns the final output.
    
    Args:
        message_body: The initial message or conversation history.
        wa_id: The unique thread ID for the conversation.
    
    Returns:
        List[str]: A list containing the final response message(s).
    """
    config = {"configurable": {"thread_id": wa_id}}
    inputs = {"messages": message_body, "user_input": message_body}
    messages_output = []
    async for output in app.astream(inputs, config=config):
        for key, value in output.items():
            if key == 'final_response' and 'messages' in value:
                output_message = value["messages"][-1]
                if isinstance(output_message, AIMessage) and output_message.content != '':
                    messages_output.append(output_message.content)
    return messages_output
