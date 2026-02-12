from dotenv import load_dotenv
import os

def init_env():
    load_dotenv()

    required = [
        "MARKETAUX_API_KEY",
        "GNEWS_API_KEY"
    ]

    missing = [k for k in required if not os.getenv(k)]

    if missing:
        raise RuntimeError(
            f"Missing environment variables: {missing}"
        )
