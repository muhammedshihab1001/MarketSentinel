import secrets

def generate_api_key(length: int = 32):
    return secrets.token_hex(length)


if __name__ == "__main__":
    key = generate_api_key(32)
    print("\nGenerated API Key:\n")
    print(key)
    print("\nAdd this to your .env file:\n")
    print(f"API_KEY={key}\n")
