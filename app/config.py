import sys
import os
from dotenv import load_dotenv
import logging


def load_configurations(app):
    load_dotenv()
    app.config["ACCESS_TOKEN"] = os.getenv("ACCESS_TOKEN")
    app.config["YOUR_PHONE_NUMBER"] = os.getenv("YOUR_PHONE_NUMBER")
    app.config["APP_ID"] = os.getenv("APP_ID")
    app.config["APP_SECRET"] = os.getenv("APP_SECRET")
    app.config["RECIPIENT_WAID"] = os.getenv("RECIPIENT_WAID")
    app.config["VERSION"] = os.getenv("VERSION")
    app.config["PHONE_NUMBER_ID"] = os.getenv("PHONE_NUMBER_ID")
    app.config["VERIFY_TOKEN"] = os.getenv("VERIFY_TOKEN")
    app.config["DB_CONNECTION"] = os.getenv("DB_CONNECTION")
    app.config["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    app.config["SUPABASE_URL"] = os.getenv("SUPABASE_URL")
    app.config["SUPABASE_KEY"] = os.getenv("SUPABASE_KEY")
    app.config["TABLE_NAME"] = os.getenv("TABLE_NAME")
    app.config["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
    app.config["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    app.config["LANGCHAIN_TRACING_V2"]=os.getenv("LANGCHAIN_TRACING_V2")
    app.config["TABLE_NAME_VEHICLES"]=os.getenv("TABLE_NAME_VEHICLES")
    app.config["TABLE_NAME_DEALERSHIP"]=os.getenv("TABLE_NAME_DEALERSHIP")


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
