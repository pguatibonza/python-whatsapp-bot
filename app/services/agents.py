"""
agents.py

This module defines a set of specialized agent classes for a multi-agent customer service chatbot.
Each agent encapsulates a specific role (e.g., general customer support, appointment scheduling,
technical details retrieval, multimedia content, and query routing) by extending the BaseAgent class.
This design improves modularity, maintainability, and scalability of the chatbot.

Classes:
    BaseAgent: Base class for all agents. Initializes the LLM, binds tools (if provided), 
               sets up prompt templates, and combines the prompt with the LLM pipeline.
    PrimaryAgent: Handles general customer queries and delegates tasks to specialized agents.
    AppointmentAgent: Manages test drive scheduling by binding appointment-related tools.
    RagAgent: Processes contextual vehicle-related queries such as specifications, pricing,
              and availability.
    FinalResponseAgent: Synthesizes responses from specialized agents into a unified final answer.
    MultimediaAgent: Provides multimedia content (images, videos, technical info) for vehicles.
    ContextRouterAgent: Routes customer queries by determining which tool or specialized agent 
                        should handle the request.
"""

from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from . import prompts, tools

class BaseAgent:
    """
    BaseAgent is the core class that encapsulates the common functionality for all agents.
    
    It initializes a ChatOpenAI model, optionally binds tools, and creates a prompt template
    using the provided message tuples. The combined agent is then built by chaining the prompt 
    with the LLM (and tools) via the pipe (|) operator.
    
    Attributes:
        llm (ChatOpenAI): The language model instance.
        agent: The LLM instance after optionally binding tools.
        prompt (ChatPromptTemplate): The prompt template generated from the provided messages.
        combined_agent: The combined pipeline of prompt and agent ready for invocation.
    """
    def __init__(self, prompt_messages, model="gpt-4o", tools_list=None):
        self.llm = ChatOpenAI(model=model, temperature=0)
        if tools_list:
            # Bind tools if provided.
            self.agent = self.llm.bind_tools(tools_list)
        else:
            self.agent = self.llm
        self.prompt = ChatPromptTemplate.from_messages(prompt_messages).partial(time=datetime.now())
        self.combined_agent = self.prompt | self.agent

    async def ainvoke(self, input_data: dict):
        """
        Asynchronously invokes the combined agent pipeline with the provided input data.
        
        Args:
            input_data (dict): A dictionary containing keys such as 'user_input', 'messages', 
                               and 'summary' to be used in the prompt.
        
        Returns:
            The response from the agent.
        """
        return await self.combined_agent.ainvoke(input_data)

class PrimaryAgent(BaseAgent):
    """
    PrimaryAgent handles general customer support queries. It gathers initial user input and,
    based on the conversation, delegates tasks to specialized assistants (e.g., Appointment or RAG).
    
    Uses the PRIMARY_ASSISTANT_PROMPT and binds delegation tools: toAppointmentAssistant and toRagAssistant.
    """
    def __init__(self):
        prompt_messages = [
            ("system", prompts.PRIMARY_ASSISTANT_PROMPT),
            ("placeholder", "{messages}"),
            ("human", "{user_input}")
        ]
        super().__init__(prompt_messages, model="gpt-4o", 
                         tools_list=[tools.toAppointmentAssistant, tools.toRagAssistant])

class AppointmentAgent(BaseAgent):
    """
    AppointmentAgent handles scheduling of test drives.
    
    Uses the APPOINTMENT_ASSISTANT_PROMPT and binds tools for scheduling:
        - tool_create_event_test_drive
        - tool_get_available_time_slots
        - tool_is_time_slot_available
        - CompleteOrEscalate (for delegation/fallback)
    """
    def __init__(self):
        prompt_messages = [
            ("system", prompts.APPOINTMENT_ASSISTANT_PROMPT),
            ("placeholder", "{messages}"),
            ("human", "{user_input}")
        ]
        super().__init__(prompt_messages, model="gpt-4o",
                         tools_list=[tools.tool_create_event_test_drive,
                                     tools.tool_get_available_time_slots,
                                     tools.tool_is_time_slot_available,
                                     tools.CompleteOrEscalate])

class RagAgent(BaseAgent):
    """
    RagAgent processes contextual queries related to vehicle details including specifications,
    pricing, availability, and promotions. Uses the CONTEXTUAL_ASSISTANT_PROMPT.
    """
    def __init__(self):
        prompt_messages = [
            ("system", prompts.CONTEXTUAL_ASSISTANT_PROMPT),
            ("placeholder", "{messages}"),
            ("human", "{user_input}")
        ]
        super().__init__(prompt_messages, model="gpt-4o")

class FinalResponseAgent(BaseAgent):
    """
    FinalResponseAgent synthesizes and consolidates responses from the specialized agents
    to generate a final, cohesive answer for the customer.
    
    Uses the FINAL_RESPONSE_ASSISTANT_PROMPT.
    """
    def __init__(self):
        prompt_messages = [
            ("system", prompts.FINAL_RESPONSE_ASSISTANT_PROMPT),
            ("placeholder", "{messages}"),
            ("human", "{user_input}")
        ]
        super().__init__(prompt_messages, model="gpt-4o")

class MultimediaAgent(BaseAgent):
    """
    MultimediaAgent provides multimedia content (such as images, videos, or technical information)
    about vehicles when requested by the user.
    
    Uses the MULTIMEDIA_ASSISTANT_PROMPT and binds the tool for technical information retrieval.
    """
    def __init__(self):
        prompt_messages = [
            ("system", prompts.MULTIMEDIA_ASSISTANT_PROMPT),
            ("placeholder", "{messages}"),
            ("human", "{user_input}")
        ]
        super().__init__(prompt_messages, model="gpt-4o",
                         tools_list=[tools.tool_get_car_technical_info])

class ContextRouterAgent(BaseAgent):
    """
    ContextRouterAgent acts as a routing system. It evaluates customer queries to determine 
    the most appropriate specialized agent or tool to handle the request.
    
    Uses the QUERY_IDENTIFIER_PROMPT and binds multiple tools:
        - CompleteOrEscalate
        - QueryIdentifier
        - MultimediaIdentifier
        - DealershipInfoIdentifier
    """
    def __init__(self):
        prompt_messages = [
            ("system", prompts.QUERY_IDENTIFIER_PROMPT),
            ("placeholder", "{messages}"),
            ("human", "{user_input}")
        ]
        super().__init__(prompt_messages, model="gpt-4o",
                         tools_list=[tools.CompleteOrEscalate,
                                     tools.QueryIdentifier,
                                     tools.MultimediaIdentifier,
                                     tools.DealershipInfoIdentifier])
