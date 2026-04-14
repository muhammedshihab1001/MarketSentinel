"""
MarketSentinel — Owner Password Hash Generator
================================================
Run this ONCE to generate your owner password hash.
Paste the output into .env as OWNER_PASSWORD_HASH.

Usage:
    python scripts/generate_owner_hash.py

Never store your plain password anywhere.
Only the hash goes into .env.

FIX: OWNER_PASSWORD_HASH is now printed WITH double quotes.
     Without quotes, Docker Compose treats $2b$12$... dollar
     signs as shell variable substitutions and corrupts the
     hash before it reaches the container. The quotes prevent
     this substitution entirely.

FIX: OWNER_USERNAME no longer hardcoded as "shihab".
     Script now asks for username — avoids exposing real
     usernames in source code that is publicly visible on GitHub.
"""

import sys
import secrets


def main():

    print("\n" + "=" * 52)
    print("  MarketSentinel — Owner Setup")
    print("=" * 52)

    try:
        from passlib.context import CryptContext
    except ImportError:
        print("\n[ERROR] passlib not installed.")
        print("Run: pip install passlib[bcrypt]")
        sys.exit(1)

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    print("\nThis script generates a bcrypt hash of your password.")
    print("The hash is stored in .env — your plain password is never saved.\n")

    # FIX: Ask for username instead of hardcoding "shihab"
    username = input("Enter owner username: ").strip()

    if len(username) < 3:
        print("\n[ERROR] Username must be at least 3 characters.")
        sys.exit(1)

    password = input("Enter owner password: ").strip()

    if len(password) < 8:
        print("\n[ERROR] Password must be at least 8 characters.")
        sys.exit(1)

    confirm = input("Confirm password: ").strip()

    if password != confirm:
        print("\n[ERROR] Passwords do not match.")
        sys.exit(1)

    hashed = pwd_context.hash(password)
    jwt_secret = secrets.token_hex(32)

    # Self-verify — confirms hash is valid before you paste it
    verified = pwd_context.verify(password, hashed)
    if not verified:
        print("\n[ERROR] Hash self-verification failed. Do not use this output.")
        sys.exit(1)

    print("\n" + "=" * 52)
    print("  Add these to your .env file:")
    print("=" * 52)
    print(f"\nOWNER_USERNAME={username}")
    print(f'OWNER_PASSWORD_HASH="{hashed}"')   # quoted — prevents Docker $ mangling
    print(f'JWT_SECRET="{jwt_secret}"')
    print("DEMO_REQUESTS_PER_FEATURE=10")
    print("DEMO_BLOCK_DAYS=7")
    print("JWT_OWNER_EXPIRE_DAYS=30")
    print("JWT_DEMO_EXPIRE_HOURS=24")
    print("\n" + "=" * 52)
    print("  IMPORTANT:")
    print("  - Hash is wrapped in double quotes — required for Docker Compose")
    print("  - Never commit .env to git")
    print("  - Never share OWNER_PASSWORD_HASH")
    print("  - Never share JWT_SECRET")
    print("=" * 52)

    print("\n" + "=" * 52)
    print("  Verification:")
    print("=" * 52)
    print(f"  Hash length   : {len(hashed)}  (must be 60)")
    print(f"  Self-verify   : {verified}  (must be True)")
    print(f"  Hash preview  : {hashed[:20]}...")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    main()
