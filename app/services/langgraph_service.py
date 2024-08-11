from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from supabase import create_client
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.output_parsers import StrOutputParser
from typing import Annotated
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
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pprint import pprint
from typing import Any,  Literal, Union
from langchain_core.messages import  AnyMessage
import logging
import os
from . import tools_restaurant
from . import supabase_service
from . import tools_parra

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
retriever=vector_store.as_retriever(search_kwargs={"k":4})
memory = SqliteSaver.from_conn_string(":memory:") #despues se conecta a bd

retriever_tool=create_retriever_tool( 
    retriever, 
    'retrieve_info',
    ' Busca y devuelve informacion sobre los vehiculos del concesionario, repuestos e informacion general'
    ,document_separator="\n\n\n"
)   


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm_tools=[retriever_tool,tool]


#LLM with function call
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent= llm.bind_tools(llm_tools)

#Prompt
system = """Eres un asistente de servicio al cliente del concesionario Parra arango. 
Debes comunicarte amablemente con el usuario y mantener precisa y concisa la conversación.
Tambien debes identificar cuando haya una llamda a una herramienta correctamente.
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{input}"),
    ]
)

agent_router=route_prompt | agent

### Retrieval Grader
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documentos son relevante para la pregunta, 'si' o 'no'"
    )
structured_llm_grader=llm.with_structured_output(GradeDocuments)

# Prompt
system = """
Usted es un evaluador que está valorando la relevancia de un documento recuperado respecto a una pregunta del usuario.
Si el documento contiene palabra(s) clave o un significado semántico relacionado con la pregunta del usuario, califíquelo como relevante.El objetivo es filtrar recuperaciones erróneas.
Asigne una puntuación binaria 'si' o 'no' para indicar si el documento es relevante para la pregunta.
"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)
retrieval_grader = grade_prompt | structured_llm_grader
# question = "cual es el precio de la torres supreme 2025"
# docs = retriever.invoke(question)
# doc_txt = docs[1].page_content
# print(retrieval_grader.invoke({"question": question, "document": doc_txt}))


### Generate
#llm 
llm = ChatOpenAI(model="gpt-4o", temperature=0)

system="""
Eres un analista experto del concesionario parra-arango cuya funcion es responder preguntas a los clientes.
Usa las siguientes piezas de información contextual para responder la pregunta. Si no conoces la respuesta, di que
esa información no la tienes disponible.
Manten una respuesta concisa
Question: {question} 

Context: {context} 

Answer:"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt | llm #| StrOutputParser()

# #Run 
# generation = rag_chain.invoke({"context": docs, "question": question})
# print(generation)

### Hallucination Grader

# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'si' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """Usted es un evaluador que está valorando si una generación de LLM está fundamentada en / apoyada por un conjunto de hechos recuperados.
Asigne una puntuación binaria 'si' o 'no'. 'si' significa que la respuesta está fundamentada en / apoyada por el conjunto de hechos."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Conjunto de hechos: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
#hallucination_grader.invoke({"documents": docs, "generation": generation})

### Answer Grader


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'si' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """Usted es un evaluador que está valorando si una respuesta aborda / resuelve una pregunta.
Asigne una puntuación binaria 'si' o 'no'. 'si' significa que la respuesta resuelve la pregunta."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Pregunta del usuario: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
#answer_grader.invoke({"question": question, "generation": generation})

### Question Re-writer

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Prompt
system = """Usted es un reformulador de preguntas que convierte una pregunta de entrada en una versión mejorada y optimizada
para la recuperación de información en un vectorstore. Analice la entrada e intente razonar sobre la intención / significado semántico subyacente. 
Tenga en cuenta el historial de mensajes del usuario para completar la pregunta, y mantenga el significado semantico subyacente"""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Aqui esta la pregunta inicial: \n\n {question} \n Formule una respuesta mejorada.",
        ),
         ("placeholder","{messages}")
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
#question_rewriter.invoke({"question": question})



###Graph state
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]
    messages : Annotated[list, add_messages]


### Nodes
def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the input, it will decide to use any tool, retrieve info, or keep chatting.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    message=state['messages']
    response=agent_router.invoke({"input":message})
    

    return {"messages": [response]}

     

# def retrieve(state):
#     """
#     Retrieve documents

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): New key added to state, documents, that contains retrieved documents
#     """
#     print("---RETRIEVE---")
#     question = state["question"]

#     # Retrieval
#     documents = retriever.invoke(question)
#     return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "messages": [generation],"generation":generation.content}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["messages"][-2].tool_calls[0]['args']['query']#penultimo mensaje es el AI message que se manda para la tool call
    raw_document = state["messages"][-1].content #Ulimo mensaje siempte es lo que devuelve el retrieve tool

    documents=raw_document.split("\n\n\n")
    filtered_documents =[]
    for document in documents:
        score = retrieval_grader.invoke(
        {"question": question, "document": document}
        )
        grade = score.binary_score
        if grade == "si":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_documents.append(document)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    return {"documents": filtered_documents, "question": question}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question,"messages":state["messages"]})
    retrieve_call=agent_router.invoke({"input":better_question})
    return {"documents": documents, "question": better_question,"messages":[retrieve_call]}


### Edges ###



def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "si":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "si":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

#Define tool routing

def tools_condition_modified(
    state: Union[list[AnyMessage], dict[str, Any]],
) -> Literal["tools", "__end__"]:
    """Use in the conditional_edge to route to the ToolNode if the last message

    has tool calls. Otherwise, route to the end.

    Args:
        state (Union[list[AnyMessage], dict[str, Any]]): The state to check for
            tool calls. Must have a list of messages (MessageGraph) or have the
            "messages" key (StateGraph).

    Returns:
        The next node to route to.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        for tool_call in ai_message.tool_calls:
            if tool_call["name"]=="retrieve_info":
                return "retrieve"
        return "tools"
    return "__end__"

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("agent",agent)
retrieve=ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # retrieve
tool_node=ToolNode(tools)
workflow.add_node("tools",tool_node)
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
workflow.add_edge(START,"agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition_modified,
    {
        "retrieve": "retrieve",
        "tools":"tools",
        END:END
    },
)
workflow.add_edge("tools","agent")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "5"}}

# from pprint import pprint

# Run
inputs = {
    "messages": "Cual es el precio de las tivoli XLV"
}
for output in app.stream(inputs,config=config):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        pprint(value, indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value["generation"])