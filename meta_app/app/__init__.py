from quart import Quart
from app.config import load_configurations, configure_logging
from .views import webhook_blueprint
from .utils.whatsapp_utils import cleanup_dedup_store
import asyncio
import logging

def create_app():
    app = Quart(__name__)

    # Load configurations and logging settings
    load_configurations(app)
    configure_logging()

    # Register blueprints
    app.register_blueprint(webhook_blueprint)

    # Start the deduplication cleanup task
    @app.before_serving
    async def startup():
        asyncio.create_task(cleanup_dedup_store())
        logging.info("Started deduplication cleanup task.")

    return app
