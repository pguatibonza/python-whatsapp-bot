"""
tools.py

This module implements the external integration tools and related helper functions used by the multi-agent chatbot.
It provides functionality for:
    - Interacting with Google Calendar (for scheduling test drives).
    - Retrieving available time slots.
    - Checking the availability of specific time slots.
    - Assigning available workers for appointments.
    - Creating test drive events in the calendar.
    - Fetching vehicle technical details and dealership information via Supabase.
    
Additionally, the module defines several Pydantic models for structured outputs and wraps functions as 
StructuredTool instances for safe integration into the LangGraph workflow.
"""

import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.supabase_service import SupabaseService
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from config import Settings
import logging

# -------------------------
# Google Calendar Integration
# -------------------------
settings=Settings()
SCOPES = ['https://www.googleapis.com/auth/calendar']
CREDENTIALS_PATH=settings.GOOGLE_CREDENTIALS_PATH
TOKEN_PATH=settings.GOOGLE_TOKEN_PATH
print(TOKEN_PATH)
def get_calendar_service():
    """
    Initializes and returns a Google Calendar service object.
    
    Uses credentials stored in 'token.pickle' or initiates a new OAuth flow if necessary.
    
    Returns:
        A Google Calendar service instance.
    """
    creds = None
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        logging.info("Access token expired, refreshing...")
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing credentials...")
            creds.refresh(Request())
            print("Credentials Refreshed.")
        else:
            logging.info("No valid token found; starting console OAuth flow...")
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, 'wb') as token:
            pickle.dump(creds, token)
    service = build('calendar', 'v3', credentials=creds)
    return service

# Initialize the calendar service
service = get_calendar_service()

# -------------------------
# Appointment Scheduling Tools
# -------------------------
WORKERS = ["pabloalejandrogb1@gmail.com", "p.guatibonza@uniandes.edu.co", "andrew.molina.m@gmail.com"]
WORKING_HOURS = (8, 17)
MAX_APPOINTMENTS_PER_SLOT = len(WORKERS)

def get_available_time_slots(date: str):
    """
    Retrieves available time slots for a given date within defined working hours.
    
    Args:
        date (str): The date for which to retrieve available slots, formatted as 'YYYY-MM-DD'.
    
    Returns:
        list: A list of available time slots as strings formatted in 'HH:MM'.
              Only slots with fewer than the maximum allowed appointments are included.
    """
    start_of_day = datetime.strptime(date, '%Y-%m-%d').replace(hour=WORKING_HOURS[0], minute=0, second=0)
    end_of_day = datetime.strptime(date, '%Y-%m-%d').replace(hour=WORKING_HOURS[1], minute=0, second=0)
    events_result = service.events().list(
        calendarId='primary',
        timeMin=start_of_day.isoformat() + 'Z',
        timeMax=end_of_day.isoformat() + 'Z',
        singleEvents=True,
        orderBy='startTime'
    ).execute()
    events = events_result.get('items', [])
    
    availability = {start_of_day + timedelta(hours=i): 0 for i in range(WORKING_HOURS[1] - WORKING_HOURS[0])}
    for event in events:
        event_start = datetime.fromisoformat(event['start']['dateTime']).replace(tzinfo=None)
        slot = event_start.replace(minute=0, second=0, microsecond=0)
        if slot in availability:
            availability[slot] += 1
    available_slots = [slot.strftime('%H:%M') for slot, count in availability.items() if count < MAX_APPOINTMENTS_PER_SLOT]
    return available_slots

tool_get_available_time_slots = StructuredTool.from_function(
    func=get_available_time_slots, 
    name="get_available_time_slots", 
    description="Devuelve una lista con los horarios disponibles dada una fecha",
    handle_tool_error=True
)

def is_time_slot_available(date: str, time_slot: str):
    """
    Checks if a specific time slot is available for a given date.
    
    Args:
        date (str): The date to check, formatted as 'YYYY-MM-DD'.
        time_slot (str): The time slot to verify, formatted as 'HH:MM'.
    
    Returns:
        bool: True if the time slot is available, False otherwise.
    """
    available_slots = get_available_time_slots(date)
    return time_slot in available_slots

tool_is_time_slot_available = StructuredTool.from_function(
    func=is_time_slot_available, 
    name="is_time_slot_available", 
    description="True si el time slot está disponible, falso de lo contrario",
    handle_tool_error=True
)

def assign_available_worker(date: str, time_slot: str):
    """
    Assigns an available worker for a given date and time slot.
    
    Args:
        date (str): The date in 'YYYY-MM-DD' format.
        time_slot (str): The time slot in 'HH:MM' format.
    
    Returns:
        str: The email address of the assigned worker.
    
    Raises:
        ValueError: If no workers are available for the given time slot.
    """
    start_time = datetime.strptime(f"{date}T{time_slot}:00-05:00", '%Y-%m-%dT%H:%M:%S%z')
    end_time = start_time + timedelta(hours=1)
    events_result = service.events().list(
        calendarId='primary',
        timeMin=start_time.isoformat(),
        timeMax=end_time.isoformat(),
        singleEvents=True,
        orderBy='startTime'
    ).execute()
    events = events_result.get('items', [])
    current_appointments = []
    for event in events:
        if 'attendees' in event:
            current_appointments.extend([att['email'] for att in event['attendees'] if 'email' in att])
    for worker in WORKERS:
        if worker not in current_appointments:
            return worker
    raise ValueError(f"No workers are available for the time slot {time_slot} on {date}.")

def create_event_test_drive(car_model: str, name: str, lastname: str, customer_email: str, date_begin: str, date_finish: str, notes=""):
    """
    Creates a test drive appointment event for a vehicle.
    
    Args:
        car_model (str): The car model for the test drive.
        name (str): The first name of the customer.
        lastname (str): The last name of the customer.
        customer_email (str): The customer's email address.
        date_begin (str): The start datetime in ISO 8601 format (YYYY-MM-DDTHH:MM:SS-05:00).
        date_finish (str): The end datetime in ISO 8601 format.
        notes (str, optional): Additional appointment notes.
    
    Returns:
        dict: The details of the created event.
    
    Raises:
        ValueError: If no worker is available for the requested time slot.
    """
    date = date_begin.split('T')[0]
    time_slot = date_begin.split('T')[1][:5]
    assigned_worker = assign_available_worker(date, time_slot)
    event = {
        'summary': f"Test drive del vehiculo {car_model} para la persona {name} {lastname}",
        'location': "Concesionario principal Los Coches",
        'description': f"Test drive del vehiculo {car_model} para la persona {name} {lastname} con notas: {notes}",
        'start': {
            'dateTime': date_begin,
            'timeZone': 'America/Bogota'
        },
        'end': {
            'dateTime': date_finish,
            'timeZone': 'America/Bogota'
        },
        'attendees': [
            {'email': customer_email},
            {'email': assigned_worker}
        ]
    }
    event = service.events().insert(calendarId='primary', body=event).execute()
    return event

tool_create_event_test_drive = StructuredTool.from_function(
    func=create_event_test_drive, 
    name="create_event_test_drive", 
    description="Crea una cita para hacer el test drive de un vehiculo",
    handle_tool_error=True
)

# -------------------------
# Pydantic Models for Delegation and Data Retrieval
# -------------------------
class CompleteOrEscalate(BaseModel):
    """
    Represents a tool used to mark a task as complete or to escalate control of the dialogue 
    back to the main assistant.
    
    Attributes:
        cancel (bool): Indicates if the task is being canceled.
        reason (str): The reason for cancellation or escalation.
    """
    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "User wanted to schedule a test drive.",
            },
            "example 3": {
                "cancel": False,
                "reason": "User wants to know more information about the car/car dealersip.",
            },
        }

class toRagAssistant(BaseModel):
    """
    Transfers work to a specialized assistant for addressing conceptual or technical inquiries
    related to vehicle details and dealership information.
    
    Attributes:
        request (str): Follow-up questions the primary assistant should clarify.
    """
    request: str = Field(description="Any necessary follow-up questions the primary assistant should clarify before proceeding. The request must be related to the car dealership 'los coches'.")

class toMultimediaAssistant(BaseModel):
    """
    Transfers work to a specialized assistant for extracting technical information and multimedia content.
    
    Attributes:
        request (str): Follow-up questions regarding multimedia or technical data.
    """
    request: str = Field(description="Any necessary follow-up questions the primary assistant should clarify before proceeding. The request must be related to the car dealership 'los coches'.")

class toAppointmentAssistant(BaseModel):
    """
    Transfers work to the appointment assistant to handle scheduling of test drive appointments.
    
    Attributes:
        request (str): Follow-up question needed by the primary assistant.
    """
    request: str = Field(description="Any necessary follow-up question the primary assistant should clarify before proceeding.")

class QueryIdentifier(BaseModel):
    """
    Identifies if access to the vector database is required.
    
    Attributes:
        query (str): A precise search string combining the current request with relevant context.
    """
    query: str = Field(description="Precise search string combining current request with necessary context from conversation history. Example: 'Price of ModelX, ModelY' when following up on previous model list.")

class MultimediaIdentifier(BaseModel):
    """
    Identifies if multimedia extraction is needed based on the user's request.
    
    Attributes:
        query (str): The query to process by the multimedia assistant.
    """
    query: str = Field(description='Query that is going to enter into the multimedia assistant.')

class DealershipInfoIdentifier(BaseModel):
    """
    Identifies if retrieval of general dealership information is needed.
    
    Attributes:
        query (str): A search string combining dealership-specific context with the current request.
    """
    query: str = Field(
        description="Precise search string combining current request with dealership-specific information from conversation history. For example, 'What financing options do you offer at Los Coches?' or 'Current promotions and offers at Los Coches.'"
    )

class CarModel(BaseModel):
    """
    Represents a vehicle identifier used to fetch technical details.
    
    Attributes:
        id (int): The unique identifier of the car.
    """
    id: int = Field(description="Car id")

def get_car_technical_info(brand: str, model: str):
    """
    Retrieves the technical details and multimedia for a vehicle based on the provided brand and model.
    
    This function uses a smaller model (gpt-4o-mini) with structured output to determine the correct car ID,
    and then fetches detailed information via the SupabaseService.
    
    Args:
        brand (str): The brand of the vehicle.
        model (str): The model of the vehicle.
    
    Returns:
        The technical information and media for the selected vehicle.
    """
    vehicles_info = SupabaseService.load_vehicle_brands_models()
    print("vehiculos cargados")
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(CarModel)
    SYSTEM = """" 
    Given a list of specyfications of a car (model, brand, id) and a user brand and model, you must select
    which one the user is referring to. Try to return the closest answer. If there is no close answer, dont return anything.
    ###
    vehicles info : {vehicles_info}
    ###
    user brand : {brand}
    ###
    user model : {model}
    """
    prompt = ChatPromptTemplate.from_messages(
        ("system", SYSTEM)
    )
    llm = prompt | chat
    id = llm.invoke({"vehicles_info": vehicles_info, "brand": brand, "model": model}).id
    print("id obtenido correctamente", id)
    vehicle_info = SupabaseService.load_vehicle_info_by_id(id)
    print("info vehiculo obtenida correctamente")
    return vehicle_info

tool_get_car_technical_info = StructuredTool.from_function(
    func=get_car_technical_info, 
    name="get_car_info", 
    description="Obtiene la ficha tecnica y fotos/videos de un carro",
    handle_tool_error=True
)

def get_dealership_description(query: str) -> str:
    """
    Retrieves the dealership description that best matches the user's query.
    
    Args:
        query (str): The user's query regarding dealership details (e.g., financing, offers).
    
    Returns:
        str: The description of the dealership that answers the query.
    """
    dealerships = SupabaseService.load_dealership_info()

    class DealershipDescription(BaseModel):
        description: str = Field(..., description="Description of the dealership that answers the query")
    
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(DealershipDescription)
    SYSTEM = """
    Given a list of dealership information (each entry contains 'id', 'nombre', and 'description')
    and a user query regarding dealership details (such as financing options, offers, or services),
    select the dealership that best matches the query and return only the 'description' field that answers the query.
    ###
    Dealership info: {dealership_info}
    ###
    User query: {query}
    """
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM)])
    llm = prompt | chat
    result = llm.invoke({"dealership_info": dealerships, "query": query})
    return result.description
