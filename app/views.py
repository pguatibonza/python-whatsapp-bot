import logging
import json
import asyncio

from quart import Blueprint, request, jsonify, current_app

from .decorators.security import signature_required
from .utils.whatsapp_utils import (
    process_whatsapp_message,
    is_valid_whatsapp_message,
)

webhook_blueprint = Blueprint("webhook", __name__)

async def handle_message(body: dict):
    """
    Asynchronously handle incoming webhook events from WhatsApp.
    """
    if is_valid_whatsapp_message(body):
        logging.info("Valid WhatsApp message received, processing message...")
        response = await process_whatsapp_message(body)
        logging.info("Message processed and response sent.")
        return response
    else:
        # If the request is not a WhatsApp API event, return an error
        return (
            jsonify({"status": "error", "message": "Not a WhatsApp API event"}),
            404,
        )

# Required webhook verification for WhatsApp
def verify():
    # Parse params from the webhook verification request
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    # Check if a token and mode were sent
    if mode and token:
        # Check the mode and token sent are correct
        if mode == "subscribe" and token == current_app.config["VERIFY_TOKEN"]:
            # Respond with 200 OK and challenge token from the request
            logging.info("WEBHOOK_VERIFIED")
            return challenge, 200
        else:
            # Responds with '403 Forbidden' if verify tokens do not match
            logging.info("VERIFICATION_FAILED")
            return jsonify({"status": "error", "message": "Verification failed"}), 403
    else:
        # Responds with '400 Bad Request' if verify tokens do not match
        logging.info("MISSING_PARAMETER")
        return jsonify({"status": "error", "message": "Missing parameters"}), 400

@webhook_blueprint.route("/webhook", methods=["GET"])
def webhook_get():
    return verify()

@webhook_blueprint.route("/webhook", methods=["POST"])
@signature_required
async def webhook_post():
    body = await request.get_json()
    #logging.info(f"Received webhook: {json.dumps(body)}")
    
    # Offload processing to a background task
    asyncio.create_task(handle_message(body))
    
    # Respond immediately to prevent WhatsApp retries
    return jsonify({"status": "received"}), 200
