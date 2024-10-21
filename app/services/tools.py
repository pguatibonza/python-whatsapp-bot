import os
import pickle
from typing import Literal, Optional
import google.auth
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.tools import ToolException
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
#from .supabase_service import load_vehicle_brands_models,load_vehicle_info_by_id

from supabase_service import load_vehicle_brands_models,load_vehicle_info_by_id
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



def create_event_test_drive(car_model : str , name : str, lastname : str, email : str, date_begin :str ,date_finish :str,  notes="" ):
    
    """
        Crea una cita para realizar el test drive del vehiculo
    """

    #date_finish=date_begin + timedelta(hours=1)
    #date_finish=date_finish.isoformat()
    #date_begin=date_begin.isoformat()

    event={
        'summary': f"Test drive  del vehiculo {car_model} para la persona {name} {lastname} ",
        'location': "Concesionario principal Los Coches",
        'description' : f"Test drive del vehiculo {car_model} para la persona {name} {lastname} con notas  : {notes} ",
        'start':{
            'dateTime': date_begin ,
            'timeZone':'America/Bogota'
        },
        'end':{
            'dateTime':date_finish,
            'timezone' : 'America/Bogota'
        },
        'attendees':[
            {'email':email}
        ]

    }
    event=service.events().insert(calendarId='primary',body=event).execute()
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
    Transfers work to a specialized assistant  to handle any conceptual doubts/inquiries about the vehicles available. 
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
class QueryIdentifier(BaseModel):
    """Identify if the model needs to extract info from the vector database to answer the user and if it does, 
    it identifies if the user input is sufficient to search in the database. If not, then a follow-up question is asked"""

    query : str = Field(description=' query that is going to enter into the vector store to retrieve the information the user needs')

class MultimediaIdentifier(BaseModel):
    """ Identify if the model needs to extract any multimedia(technical cards, videos, images) given the user request"""
    query : str = Field(description='query that is going to enter into the multimedia assistant')
class CarModel(BaseModel):
    id : int = Field (description = "Car id ")

def get_car_technical_info(brand:str,model:str):
    """
    Obtiene la ficha tecnica y videos de un carro dado su modelo y marca
    """
    vehicles_info = load_vehicle_brands_models()
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
    vehicle_info = load_vehicle_info_by_id(id)
    print("info vehiculo obtenida correctamente")
    return vehicle_info

tool_get_car_technical_info=StructuredTool.from_function(
    func=get_car_technical_info, 
    name = "get_car_info", 
    description="Obtiene la ficha tecnica y fotos/videos de un carro",
    handle_tool_error=True)
    

