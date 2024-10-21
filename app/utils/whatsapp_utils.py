import logging
import json
import re
import asyncio
from datetime import datetime, timedelta
from collections import deque

from quart import current_app, jsonify
import httpx  # Use httpx for asynchronous HTTP requests

# Import your response generators
from app.services.langgraph_service import generate_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# In-memory deduplication store with deque and set
dedup_store = deque()
dedup_set = set()
DEDUP_TTL_SECONDS = 300  # 5 minutes
dedup_lock = asyncio.Lock()

def log_http_response(response: httpx.Response):
    logging.info(f"Status: {response.status_code}")
    logging.info(f"Content-type: {response.headers.get('content-type')}")
    logging.debug(f"Body: {response.text}")

def get_text_message_input(recipient: str, text: str) -> str:
    message_payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient,
        "type": "text",
        "text": {"preview_url": False, "body": text},
    }
    return json.dumps(message_payload)

def process_text_for_whatsapp(text: str) -> str:
    # Remove brackets 【】
    text = re.sub(r"\【.*?\】", "", text).strip()
    # Replace **bold** with *italic* for WhatsApp
    return re.sub(r"\*\*(.*?)\*\*", r"*\1*", text)

async def send_message(data: str) -> httpx.Response:
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {current_app.config['ACCESS_TOKEN']}",
    }

    url = f"https://graph.facebook.com/{current_app.config['VERSION']}/{current_app.config['PHONE_NUMBER_ID']}/messages"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, content=data, headers=headers, timeout=10.0)
            response.raise_for_status()
        except httpx.TimeoutException:
            logging.error("Timeout occurred while sending message")
            raise
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logging.error(f"Request failed: {e}")
            raise
        else:
            log_http_response(response)
            return response

async def is_duplicate_message(message_id: str) -> bool:
    async with dedup_lock:
        if message_id in dedup_set:
            return True
        else:
            expiration_time = datetime.now() + timedelta(seconds=DEDUP_TTL_SECONDS)
            dedup_store.append((message_id, expiration_time))
            dedup_set.add(message_id)
            return False

async def cleanup_dedup_store():
    while True:
        await asyncio.sleep(60)  # Run cleanup every minute
        async with dedup_lock:
            now = datetime.now()
            while dedup_store and dedup_store[0][1] < now:
                expired_id, _ = dedup_store.popleft()
                dedup_set.discard(expired_id)
            if len(dedup_store) % 100 == 0:  # Log every 100 cleanups
                logging.info(f"Dedup store size: {len(dedup_store)}")

async def process_whatsapp_message(body: dict):
    wa_id = body["entry"][0]["changes"][0]["value"]["contacts"][0]["wa_id"]

    message = body["entry"][0]["changes"][0]["value"]["messages"][0]
    message_body = message["text"]["body"]

    # Deduplication: Check if message ID has already been processed
    message_id = message.get("id")
    if not message_id:
        logging.warning("No message ID found.")
        return jsonify({"status": "error", "message": "No message ID found"}), 400

    try:
        if await is_duplicate_message(message_id):
            logging.info(f"Duplicate message detected: {message_id}")
            return jsonify({"status": "duplicate", "message": "Message already processed"}), 200
    except Exception as e:
        logging.error(f"Error during deduplication check: {e}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

    if not message_body:
        logging.warning("Empty message body.")
        return jsonify({"status": "error", "message": "Empty message body"}), 400

    try:
        # Generate responses asynchronously
        responses = await generate_response(message_body, wa_id)
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return jsonify({"status": "error", "message": "Failed to generate response"}), 500

    # Aggregate all AI messages into a single response
    aggregated_response = "\n".join(set(responses))  # Using set to avoid exact duplicates

    # Send the aggregated response
    processed_response = process_text_for_whatsapp(aggregated_response)
    data = get_text_message_input(current_app.config["RECIPIENT_WAID"], processed_response)
    
    try:
        await send_message(data)
    except Exception as e:
        logging.error(f"Failed to send message: {e}")
        return jsonify({"status": "error", "message": "Failed to send message"}), 500

    return jsonify({"status": "success", "message": "Message sent"}), 200


def is_valid_whatsapp_message(body: dict) -> bool:
    """
    Check if the incoming webhook event has a valid WhatsApp message structure.
    """
    return (
        body.get("object")
        and body.get("entry")
        and body["entry"][0].get("changes")
        and body["entry"][0]["changes"][0].get("value")
        and body["entry"][0]["changes"][0]["value"].get("messages")
        and body["entry"][0]["changes"][0]["value"]["messages"][0]
    )
