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
from langchain_core.messages import ToolMessage, HumanMessage, RemoveMessage,SystemMessage
from langgraph.prebuilt import ToolNode
import logging
import os
from . import tools_restaurant
from . import supabase_service
from . import tools
from .prompts import PRIMARY_ASSISTANT_PROMPT,CONTEXTUAL_ASSISTANT_PROMPT,QUERY_IDENTIFIER_PROMPT

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


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PRIMARY_ASSISTANT_PROMPT),
        ("human", "{input}"),
        ("placeholder","{messages}"),
        
    ]
).partial(time=datetime.now())

main_agent=prompt | agent



### RAG ASSISTANT
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools([tools.CompleteOrEscalate])


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUAL_ASSISTANT_PROMPT),
        ("human", "{user_input}"),
        ("placeholder", "{messages}")
    ]
).partial(time=datetime.now())

rag_agent = prompt | llm 


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


### Context Router

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QUERY_IDENTIFIER_PROMPT),
        ("human", "{user_input}"),
        ("placeholder", "{messages}")
    ]
)


context_router = prompt | llm.with_structured_output(tools.QueryIdentifier)


###Graph state

def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]

class GraphState(TypedDict):


    user_input:str
    messages : Annotated[list, add_messages]
    dialog_state : Annotated[list[Literal["primary_assistant","rag_assistant"]],update_dialog_stack]
    summary : str



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

    user_input=state["user_input"]
    
    response=main_agent.invoke({"input":user_input,"messages":state['messages']})

    return {"messages": [response],"user_input":user_input}

def rag_assistant(state):

    user_input=state["user_input"]
    response=context_router.invoke({"user_input":user_input,"messages":state["messages"]})

    context=""
    #Si el user input necesia extraer info de la base de datos
    if response.database : 
        #Si el user input es suficiente para meterlo a la bd
        if response.sufficient : 
            context=retriever.invoke(response.query)
        # Si no, pregunta una follow-up question
        else : 
            return {"messages":[AIMessage(content=response.follow_up)]}
    
    response= rag_agent.invoke({"user_input":user_input,"messages":state["messages"],"context":context})

    return {"messages": [response]}

def get_summary(state: GraphState):
    summary = state.get("summary", "")
    if summary:
        
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        messages=[SystemMessage(content=system_message)] + state['messages']

        
    else:
        messages = state["messages"]
    
    return {"messages": messages}

def summarize_conversation(state: GraphState):
    # First, we get any existing summary
    summary = state.get("summary", "")
    # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )  
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)
    

    # Delete all but the 3 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}



### Edges ###


#Define tool routing
def should_summarize(state:GraphState):
        
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 10:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END


def route_primary_assistant(
    state: GraphState,
) -> Literal[
    "enter_rag_assistant",
    "tools",
    "summarize_conversation"
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
    "summarize_conversation",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        summarize=should_summarize(state)
        return summarize
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
workflow.add_node("get_summary", get_summary)

workflow.add_edge(START,"get_summary")
workflow.add_conditional_edges("get_summary",route_to_workflow)

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
        "summarize_conversation":"summarize_conversation",
        END: END,
    },
)
workflow.add_conditional_edges("rag_assistant", route_assistants)
workflow.add_node("leave_skill",pop_dialog_state)
workflow.add_edge("leave_skill", "primary_assistant")

workflow.add_node("summarize_conversation",summarize_conversation)
workflow.add_edge("summarize_conversation",END)

# Compile
app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "6"}}

# from pprint import pprint

# Run

def generate_response(message_body,wa_id):
    config={"configurable": {"thread_id":"1111113"}}
    inputs={"messages": message_body, "user_input": message_body}
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
