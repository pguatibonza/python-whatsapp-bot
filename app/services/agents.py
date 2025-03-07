# agents.py
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from . import prompts, tools

class BaseAgent:
    def __init__(self, prompt_messages, model="gpt-4o", tools_list=None):
        self.llm = ChatOpenAI(model=model, temperature=0)
        if tools_list:
            # Bind tools if provided
            self.agent = self.llm.bind_tools(tools_list)
        else:
            self.agent = self.llm
        self.prompt = ChatPromptTemplate.from_messages(prompt_messages).partial(time=datetime.now())
        self.combined_agent = self.prompt | self.agent

    async def ainvoke(self, input_data: dict):
        return await self.combined_agent.ainvoke(input_data)

class PrimaryAgent(BaseAgent):
    def __init__(self):
        # This agent gathers general customer queries and delegates when needed.
        prompt_messages = [
            ("system", prompts.PRIMARY_ASSISTANT_PROMPT),
            ("placeholder", "{messages}"),
            ("human", "{user_input}")
        ]
        # It binds tools for delegation to appointment or RAG assistants.
        super().__init__(prompt_messages, model="gpt-4o", tools_list=[tools.toAppointmentAssistant, tools.toRagAssistant])

class AppointmentAgent(BaseAgent):
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
    def __init__(self):
        prompt_messages = [
            ("system", prompts.CONTEXTUAL_ASSISTANT_PROMPT),
            ("placeholder", "{messages}"),
            ("human", "{user_input}")
        ]
        super().__init__(prompt_messages, model="gpt-4o")

class FinalResponseAgent(BaseAgent):
    def __init__(self):
        prompt_messages = [
            ("system", prompts.FINAL_RESPONSE_ASSISTANT_PROMPT),
            ("placeholder", "{messages}"),
            ("human", "{user_input}")
        ]
        super().__init__(prompt_messages, model="gpt-4o")

class MultimediaAgent(BaseAgent):
    def __init__(self):
        prompt_messages = [
            ("system", prompts.MULTIMEDIA_ASSISTANT_PROMPT),
            ("placeholder", "{messages}"),
            ("human", "{user_input}")
        ]
        super().__init__(prompt_messages, model="gpt-4o",
                         tools_list=[tools.tool_get_car_technical_info])

class ContextRouterAgent(BaseAgent):
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
