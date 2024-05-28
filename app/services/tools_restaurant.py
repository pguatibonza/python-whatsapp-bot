import base64
import requests
from dotenv import load_dotenv
import os
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.tools import ToolException
from langchain_openai import ChatOpenAI

load_dotenv()
BOOKING_API_KEY=os.getenv('BOOKING_API_KEY')
encoded_key=base64.b64encode(BOOKING_API_KEY.encode()).decode()
BASE_URL="https://api.resos.com/v1/"
headers={
    'Authorization': f'Basic {encoded_key}'
}



#Obtener las areas disponibles 
def get_areas():
    url=BASE_URL + "tables"
    response=requests.get(url,headers=headers)
    return response.json()

def get_area_by_name(name:str)->dict:
    areas=get_areas()
    for area in areas :
        if name == area['name']:
            return area["_id"]
    return None


class AvailableTimes(BaseModel):
    date : str =Field(description="Es una fecha en el formato YYYY-MM-DD, debe ser mayor o igual a la fecha actual")
    people : int =Field(description="Cantidad de personas que necesita el cliente" )


def available_times_at_date(date: str ,people=4,area=None):
    """Retorna los horarios disponibles para una reserva, dada una fecha y una cantidad de personas"""

    url=BASE_URL + "bookingFlow/times"

    #parameters for the request
    params={
        'date' :date ,
        'people' : people
    }

    if area != None :
        params['area']=area

    #Make the request
    response=requests.get(url,headers=headers,params=params)

    if response.status_code == 200:
        results=response.json()
    else: 
        raise ToolException(f"Ha ocurrido un error con status  : {response.status_code} y contenido {response.content}")

    if len(results)>0:
        result=results[0]
    else :
        return f"No tenemos reservas disponibles para este dia"

    if len(result['unavailableTimes'] ) == 0:
        return " El dia entero esta disponible para reservas. Recuerde que abrimos a las 12 pm y cerramos a las 9pm"
    else : 

        return f" Los horarios disponibles para reservar en la fecha  {date} son: {result['availableTimes']} "



class createBooking(BaseModel):
    date : str =Field(description="Es una fecha en el formato YYYY-MM-DD, debe ser mayor o igual a la fecha actual")
    people : int =Field(description="Cantidad de personas que se necesitan para la reserva" )
    time : str =Field(description="Hora en la que se realizara la reserva")
    people : int=Field(description="Cantidad de personas que van a entrar en la reserva")
    phone : str=Field(description="El numero telefonico de la persona que reserva")
    email : str=Field(description="El correo electronico de la persona que hace la reserva")
    name : str=Field(description="Nombre de la persona que hace la reserva")
    area_name : str=Field(description="Nombre del area donde se va a hacer la reserva")
    comment : str=Field(description="Comentario adicional que hace el cliente para tener en cuenta el dia de la reserva")
    duration : str=Field(description="Duracion de la reserva")
    tables : str=Field(description="Cantidad de mesas que usara el cliente")

    

def create_booking(date : str , time : str, people : int , phone : str ,email : str, name : str ,area_name='main', comment = "",duration = 120 ,tables=1):
    """Crea una reserva dada una fecha, horario, numero de personas y area, si lo desea"""
    #Obtiene el id del area dado un nombre
    area_id=get_area_by_name(area_name)
    if area_id==None:
        return " No existe un area con ese nombre"

    url=BASE_URL+'bookings'
    params= {
        "date" : date,
        "time" : time,
        "people" : people,
        "duration" :duration,
        "guest" : {
            "name" : name,
            "phone" : phone,
            "email" : email,
        },
        "areaId" : area_id,
        "status" : "approved",
        "comment" : comment 
    }
    headers_post=headers
    headers_post['Content-Type']='application/json'

    response=requests.post(url,headers=headers_post,json=params)
    if response.status_code==200:
        booking_id=response.content
        return f"Reserva creada exitosamente, el id de la reserva es  : {response.content}"
    else : 
        raise ToolException(f"Reserva fallida debido a  {response.content}")

tool_create_booking=StructuredTool.from_function(
    func=create_booking,
    name="create_booking",
    description="Crea una nueva reserva ",
    handle_tool_error=True
)

tool_available_time=StructuredTool.from_function(
    func=available_times_at_date,
    name="available_times_at_date",
    description=" Retorna los horarios disponibles para hacer una reserva dada una fecha y un numero de personas",
    handle_tool_error=True
)

TOOLS= [tool_create_booking, tool_available_time]

#tools=[tool_create_booking,tool_available_time]
#from langchain import hub
#from langchain.agents import AgentExecutor, create_tool_calling_agent
#prompt = hub.pull("hwchase17/openai-tools-agent")
#agent=create_tool_calling_agent(llm,tools,prompt)
#agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True,handle_parsing_errors=True)


