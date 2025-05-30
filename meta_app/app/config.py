import sys
import os
from dotenv import load_dotenv
import logging
import logging.config



def load_configurations(app):
    load_dotenv()
    app.config["ACCESS_TOKEN"] = os.getenv("ACCESS_TOKEN")
    app.config["APP_ID"] = os.getenv("APP_ID")
    app.config["APP_SECRET"] = os.getenv("APP_SECRET")
    app.config["RECIPIENT_WAID"] = os.getenv("RECIPIENT_WAID")
    app.config["VERSION"] = os.getenv("VERSION")
    app.config["PHONE_NUMBER_ID"] = os.getenv("PHONE_NUMBER_ID")
    app.config["VERIFY_TOKEN"] = os.getenv("VERIFY_TOKEN")
    app.config["LANGGRAPH_URL"]=os.getenv("LANGGRAPH_URL")

def configure_logging():

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "DEBUG",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": "DEBUG",
        },
        "loggers": {
            "httpx": {
                "level": "WARNING",  # Only log warnings or higher from httpx
                "propagate": True,
            },
            "httpcore": {
                "level": "WARNING",  # Only log warnings or higher from httpcore
                "propagate": True,
            },
            "openai._base_client": {
                "level": "WARNING",  # Silence debug messages from OpenAI's base client
                "propagate": True,
            },
            # You can add additional loggers here if needed.
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)
