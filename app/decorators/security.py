from functools import wraps
from quart import current_app, jsonify, request
import logging
import hashlib
import hmac


async def validate_signature(payload, signature):
    """
    Validate the incoming payload's signature against our expected signature
    """
    # Use the App Secret to hash the payload
    expected_signature = hmac.new(
        bytes(current_app.config["APP_SECRET"], "latin-1"),
        msg=payload.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

    # Check if the signature matches
    return hmac.compare_digest(expected_signature, signature)


def signature_required(f):
    """
    Decorator to ensure that the incoming requests to our webhook are valid and signed with the correct signature.
    """

    @wraps(f)
    async def decorated_function(*args, **kwargs):
        # Access the header asynchronously and remove 'sha256=' prefix
        signature = request.headers.get("X-Hub-Signature-256", "")[7:]

        # Await getting the data asynchronously in Quart
        payload = (await request.get_data()).decode("utf-8")

        # Validate the signature using the updated async function
        if not await validate_signature(payload, signature):
            logging.info("Signature verification failed!")
            return jsonify({"status": "error", "message": "Invalid signature"}), 403
        
        # Await the wrapped function to ensure async compatibility
        return await f(*args, **kwargs)

    return decorated_function