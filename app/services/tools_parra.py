import os
import pickle
import google.auth
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.tools import ToolException
from langchain_openai import ChatOpenAI

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
class EventTestDrive(BaseModel):
    car_model : str=Field("Nombre del modelo del vehiculo al que se le hara test drive")
    name : str=Field(description="Es el nombre de quien va a hacer la cita")
    lastname : str=Field(description="Es el apellido de quien va a hacer la cita")
    email : str=Field(description="Es el correo electronico de quien va a hacer la cita")
    date_begin : str=Field(description="Es la fecha en la cual se hará la cita. Está en el formato ISO especificando la compensacion horaria con respecto al UTC -07:00")
    date_finish : str=Field(description="Es la fecha en la cual se acaba la cita. Es una hora despues de que empieza. Está en el formato ISO especificando la compensacion horaria con respecto al UTC -07:00" )
    notes : str=Field(description=" Notas adicionales que deja el usuario")
def create_event_test_drive(car_model : str , name : str, lastname : str, email : str, date_begin :str ,date_finish :str,  notes="" ):
    
    """
        Crea una cita para realizar el test drive del vehiculo
    """

    #date_finish=date_begin + timedelta(hours=1)
    #date_finish=date_finish.isoformat()
    #date_begin=date_begin.isoformat()

    event={
        'summary': f"Test drive  del vehiculo {car_model} para la persona {name} {lastname} ",
        'location': "Concesionario principal Parra Arango",
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

TOOLS=[tool_create_event_test_drive]