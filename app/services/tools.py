import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .supabase_service import SupabaseService

#from supabase_service import load_vehicle_brands_models,load_vehicle_info_by_id
# Autenticación y autorización
SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_calendar_service():
    creds = None
    # El archivo token.pickle almacena las credenciales del usuario
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # Si no hay credenciales válidas disponibles, solicita al usuario que inicie sesión
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("pp")
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Guarda las credenciales para la próxima ejecución
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('calendar', 'v3', credentials=creds)
    return service
service=get_calendar_service()


WORKERS=["pabloalejandrogb1@gmail.com","p.guatibonza@uniandes.edu.co","andrew.molina.m@gmail.com"]
WORKING_HOURS = (8, 17)
MAX_APPOINTMENTS_PER_SLOT = len(WORKERS)

def get_available_time_slots(date: str):
    """
    Retrieve available time slots for a given date within defined working hours.
    Args:
        date (str): The date for which to retrieve available time slots, formatted as 'YYYY-MM-DD'.

    Returns:
        list: A list of available time slots as strings formatted in 'HH:MM'.
        Only slots with less than the maximum number of allowed appointments are included.
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
    
    # Initialize the availability dictionary with time slots set to zero appointments
    availability = {start_of_day + timedelta(hours=i): 0 for i in range(WORKING_HOURS[1] - WORKING_HOURS[0])}

    for event in events:
        event_start = datetime.fromisoformat(event['start']['dateTime']).replace(tzinfo=None)
        slot = event_start.replace(minute=0, second=0, microsecond=0)
        if slot in availability:
            availability[slot] += 1

    # Filter the time slots to include only those with less than the maximum allowed appointments
    # and format them as strings in 'HH:MM' format
    available_slots = [
        slot.strftime('%H:%M') for slot, count in availability.items() if count < MAX_APPOINTMENTS_PER_SLOT
    ]
    return available_slots

tool_get_available_time_slots=StructuredTool.from_function(
    func=get_available_time_slots, 
    name = "get_available_time_slots", 
    description="Devuelve una lista con los horarios disponibles dada una fecha ",
    handle_tool_error=True)

def is_time_slot_available(date: str, time_slot: str):
    """
    Check if a specific time slot is available for a given date.

    Args:
        date (str): The date to check, formatted as 'YYYY-MM-DD'.
        time_slot (str): The time slot to check, formatted as 'HH:MM'.

    Returns:
        bool: True if the time slot is available, False otherwise.
    """
    available_slots = get_available_time_slots(date)
    return time_slot in available_slots
tool_is_time_slot_available=StructuredTool.from_function(
    func=is_time_slot_available, 
    name = "is_time_slot_available", 
    description="True si el time slot está disponible, falso de lo contrario",
    handle_tool_error=True)

def assign_available_worker(date: str, time_slot: str):
    """
    Assign an available worker for a given date and time slot.

    Args:
        date (str): The date in 'YYYY-MM-DD' format.
        time_slot (str): The time slot in 'HH:MM' format.

    Returns:
        str: The email of the assigned worker.

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
    Create a test drive appointment for a car.

    Args:
        car_model (str): The car model for the test drive.
        name (str): The first name of the customer.
        lastname (str): The last name of the customer.
        customer_email (str): The email address of the customer.
        date_begin (str): The start datetime of the appointment in ISO 8601 format. YYYY-MM-DDTHH:MM:SS-05:00
        date_finish (str): The end datetime of the appointment in ISO 8601 format.
        notes (str, optional): Additional notes for the appointment.

    Returns:
        dict: The created event details.

    Raises:
        ValueError: If the time slot is not available.
    """
    date = date_begin.split('T')[0]
    time_slot = date_begin.split('T')[1][:5]  # Extract time in 'HH:MM' format

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

tool_create_event_test_drive=StructuredTool.from_function(
    func=create_event_test_drive, 
    name = "create_event_test_drive", 
    description="Crea una cita para hacer el test drive de un vehiculo",
    handle_tool_error=True)

class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the customer needs."""

    cancel: bool =True
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
                "reason": "User wants to know more information about the car/car dealersip .",
            },
        }

class toRagAssistant(BaseModel):
    """
    Transfers work to a specialized assistant  to handle any conceptual doubts/inquiries about the vehicles available and the general information of the car dealership. 
    This includes specifications, features, pricing, availability, and any current promotions or financing options.
    Only give information about vehicles Los Coches have, never give information about another brand they don't sell.
    If general information about car concepts are asked, answer.
    """
    request: str=Field(description="Any necessary follow-up questions the primary assistant  should clarify  before proceeding. The request must be related to the  car dealership 'los coches'. ")
class toMultimediaAssistant(BaseModel):
    """
    Transfers work to a specialized assistant to extract the technical card and/or videos/images from a car model to handle any conceptual doubts/inquiries about the vehicles available OR schedule test drives.
    """
    request: str=Field(description="Any necessary follow-up questions the primary assistant  should clarify  before proceeding. The request must be related to the  car dealership 'los coches'. ")
class toAppointmentAssistant(BaseModel):
    """
    Tranfers work to a specialized assistant to handle the generation of the test drive appointments
    """
    request: str=Field(description="Any necessary follow-up question the primary assistant should clarify before proceeding")
class QueryIdentifier(BaseModel):
    """Identifies if vector database access is needed. When creating the query:
    - Maintain context from previous interactions
    - Combine previous context with new request parameters"""
    
    query: str = Field(description="Precise search string combining current request with necessary context from conversation history. Example: 'Price of ModelX, ModelY' when following up on previous model list")
class MultimediaIdentifier(BaseModel):
    """ Identify if the model needs to extract any multimedia(technical cards, videos, images) given the user request"""
    query : str = Field(description='query that is going to enter into the multimedia assistant')
class DealershipInfoIdentifier(BaseModel):
    """
    Identifies if retrieval of general dealership information is needed.
    When creating the query, maintain context from previous interactions and combine it with the new request parameters related to dealership data.
    Example: "Financing options, promotions, and services available at Los Coches."
    """
    query: str = Field(
        description="Precise search string combining current request with dealership-specific information from conversation history. For example, 'What financing options do you offer at Los Coches?' or 'Current promotions and offers at Los Coches.'"
    )
class CarModel(BaseModel):
    id : int = Field (description = "Car id ")

def get_car_technical_info(brand:str,model:str):
    """
    Obtiene la ficha tecnica y videos de un carro dado su modelo y marca
    """
    vehicles_info = SupabaseService.load_vehicle_brands_models()
    print("vehiculos cargados")
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(CarModel)
    SYSTEM = """"
    Given a list of specyfications of a car (model, brand, id ) and a user brand and model, you must select
    which one the user is referring to. Try to return the closest  answer.  If there is no close answer, dont return anything
    ###
    vehicles info : {vehicles_info}
    ###
    user brand : {brand}
    ###
    user model : {model}
    """
    prompt = ChatPromptTemplate.from_messages(
        ("system",SYSTEM)
    )
    llm = prompt | chat

    id=llm.invoke({"vehicles_info":vehicles_info, "brand":brand,"model":model}).id
    print("id obtenido correctamente" ,id)
    vehicle_info = SupabaseService.load_vehicle_info_by_id(id)
    print("info vehiculo obtenida correctamente")
    return vehicle_info

tool_get_car_technical_info=StructuredTool.from_function(
    func=get_car_technical_info, 
    name = "get_car_info", 
    description="Obtiene la ficha tecnica y fotos/videos de un carro",
    handle_tool_error=True)

def get_dealership_description(query: str) -> str:
    """
    Given a user query and the list of dealerships loaded from the database,
    returns the 'description' field of the dealership that best answers the query.
    
    Args:
        query (str): The user's query regarding dealership details (e.g., financing options, offers, services).
        
    Returns:
        str: The description of the selected dealership.
    """
    # Load dealership information from the database
    dealerships = SupabaseService.load_dealership_info()

    # Define a structured output model for the description
    class DealershipDescription(BaseModel):
        description: str = Field(..., description="Description of the dealership that answers the query")
    
    # Setup the LLM with structured output using a smaller model
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(DealershipDescription)
    
    # Define the system prompt
    SYSTEM = """
    Given a list of dealership information (each entry contains 'id', 'nombre', and 'description')
    and a user query regarding dealership details (such as financing options, offers, or services),
    select the dealership that best matches the query and return only the 'description' field that answers the query.
    ###
    Dealership info: {dealership_info}
    ###
    User query: {query}
    """
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM)])
    llm = prompt | chat
    
    # Invoke the LLM with the dealership info and the query
    result = llm.invoke({"dealership_info": dealerships, "query": query})

    
    return result.description


