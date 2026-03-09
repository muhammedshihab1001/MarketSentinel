import time
import os
import logging
from fastapi import Request, HTTPException
from collections import defaultdict, deque

logger = logging.getLogger("marketsentinel.security")

API_KEY = os.getenv("API_KEY")

RATE_LIMIT = int(os.getenv("API_RATE_LIMIT_PER_MIN", "60"))
WINDOW = 60

requests_store = defaultdict(lambda: deque())


def verify_api_key(request: Request):

    if not API_KEY:
        return

    client_key = request.headers.get("X-API-KEY")

    if client_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )


def rate_limit(request: Request):

    ip = request.client.host
    now = time.time()

    queue = requests_store[ip]

    while queue and queue[0] < now - WINDOW:
        queue.popleft()

    if len(queue) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )

    queue.append(now)