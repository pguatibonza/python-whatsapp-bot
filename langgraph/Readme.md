# Los Coches AI Chatbot

Este repositorio contiene un chatbot potenciado por IA para concesionarios de vehículos que ofrece:

* Programación de citas de prueba de manejo y revisiones técnicas
* Consulta de información técnica avanzada (fichas técnicas) de los vehículos disponibles
* Información general sobre el concesionario

---

## Requisitos previos

* **Podman** y **Podman Compose** instalados
* **Python 3.11+**
* **Podman secrets** configurados para las credenciales de Google Calendar
* Proyecto de **Supabase** con tablas para vector store, vehículos y concesionarios
* Proyecto en **Google Cloud** con la API de Calendar activada y credenciales OAuth 2.0

---

## 1. Configurar la API de Google Calendar

1. **Crear proyecto y habilitar API**: En Google Cloud Console, crea un proyecto nuevo y habilita la **Google Calendar API**. ([developers.google.com](https://developers.google.com/workspace/calendar/api/guides/overview?utm_source=chatgpt.com))

2. **Crear Service Account**: Navega a **IAM & Admin > Service Accounts**, crea una cuenta de servicio y genera una clave en formato JSON. Descarga el archivo `service-account.json`. ([developers.google.com](https://developers.google.com/identity/protocols/oauth2/service-account?utm_source=chatgpt.com))

3. **Domain-wide delegation** (opcional, para acceder a calendarios de usuarios de Workspace):

   * Al crear la Service Account, marca la opción **Enable G Suite Domain-wide Delegation**. ([developers.google.com](https://developers.google.com/cloud-search/docs/guides/delegation?utm_source=chatgpt.com))
   * En tu **Admin Console** (Workspace), ve a **Security > Access and data control > API Controls > Domain-wide Delegation > Manage domain-wide Delegation** y haz clic en **Add new**.
   * Ingresa el **Client ID** de la Service Account y los scopes requeridos, por ejemplo:

     * `https://www.googleapis.com/auth/calendar` (acceso completo) ([developers.google.com](https://developers.google.com/identity/protocols/oauth2/service-account?utm_source=chatgpt.com))
   * Haz clic en **Authorize** para finalizar. ([support.google.com](https://support.google.com/a/answer/162106?hl=en&utm_source=chatgpt.com))

4. **Compartir un calendario específico** (alternativa si no usas Workspace o quieres acceso puntual):

   * En Google Calendar web, abre **Settings** del calendario deseado.
   * En **Share with specific people**, añade el correo de la Service Account y asigna permiso **See all event details**. ([medium.com](https://medium.com/product-monday/accessing-google-calendar-api-with-service-account-a99aa0f7f743?utm_source=chatgpt.com))

5. **Registrar secretos en Podman**:

   ```bash
   podman secret create calendar-json secrets/service-account.json
   ```

6. **Autenticación en Python** (ejemplo en `src/tools.py`):

   ```python
   from google.oauth2 import service_account
   from googleapiclient.discovery import build

   SCOPES = ['https://www.googleapis.com/auth/calendar']
   SERVICE_ACCOUNT_FILE = '/run/secrets/calendar-json'
   # Para domain-wide delegation:
   DELEGATED_USER = 'usuario@tudominio.com'

   creds = service_account.Credentials.from_service_account_file(
       SERVICE_ACCOUNT_FILE, scopes=SCOPES)
   creds = creds.with_subject(DELEGATED_USER)

   service = build('calendar', 'v3', credentials=creds)
   ```

---

## 2. Configurar Supabase

1. Regístrate en [supabase.com](https://supabase.com) y crea un proyecto.
2. En **Project Settings > API**, copia tu `SUPABASE_URL` y `SUPABASE_KEY`.
3. Configura las variables de entorno en `.env`:

   ```ini
   SUPABASE_URL=https://<tu-proyecto>.supabase.co
   SUPABASE_KEY=<tu-anon-key>
   ```

(Omitimos la creación manual de tablas, ya que se generan automáticamente al iniciar el servicio.)

---

## 3. Variables de entorno

Copia el archivo de ejemplo `.env.example` a `.env` y rellena las claves:

```ini
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://<tu-proyecto>.supabase.co
SUPABASE_KEY=<tu-anon-key>
TABLE_NAME=documents
TABLE_NAME_VEHICLES=multimedia_vehiculos
TABLE_NAME_DEALERSHIP=concesionario
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=los-coches
LANGCHAIN_TRACING_V2=true
DB_URI=postgresql://postgres:postgres@host.docker.internal:5432/langgraph?sslmode=disable


GRAPHRAG_API_KEY=sk-...
GRAPHRAG_LLM_MODEL=gpt-4o
GRAPHRAG_EMBEDDING_MODEL=text-embedding-3-small
```

---

## 4. Levantar Postgres con Podman Compose

En la raíz del proyecto:

```bash
podman-compose -f podman-compose.yml up -d
```

Esto iniciará un contenedor `postgres:16-alpine` en el puerto `5432` y persistirá los datos en el volumen `pgdata`.

---

## 5. Construir y ejecutar el servicio

```bash
podman build -t langgraph-service .

podman run -d \
  -p 8000:8000 \
  --env-file .env \
  --name langgraph-service \
  langgraph-service
```

---

## 6. Uso

* **Health check**:

  ```bash
  curl http://localhost:8000/health
  # → { "status": "ok" }
  ```

* **Endpoint /chat**:

````bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{ "wa_id": "test", "message": "3227077343 para mañana a las 2 pm" }'
```bash
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"wa_id":"usuario123","message":"Hola, quiero agendar un test drive"}'
````

La respuesta será un JSON con el campo `reply`, conteniendo la respuesta generada por el chatbot.

---

## 7. Carga de datos

La carga de datos se realiza manualmente desde la interfaz o directamente en la base de datos, por lo que no se requiere configuración adicional.

---
