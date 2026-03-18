#!/usr/bin/env python3
"""
MarketSentinel — Owner Password Hash Generator
================================================
Run this ONCE to generate your owner password hash.
Paste the output into .env as OWNER_PASSWORD_HASH.

Usage:
    python scripts/generate_owner_hash.py

Never store your plain password anywhere.
Only the hash goes into .env.
"""

import sys
import os
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

    print("\n" + "=" * 52)
    print("  Add these to your .env file:")
    print("=" * 52)
    print(f"\nOWNER_USERNAME=shihab")
    print(f"OWNER_PASSWORD_HASH={hashed}")
    print(f"JWT_SECRET={jwt_secret}")
    print(f"DEMO_REQUESTS_PER_FEATURE=3")
    print(f"DEMO_BLOCK_DAYS=7")
    print(f"JWT_OWNER_EXPIRE_DAYS=30")
    print(f"JWT_DEMO_EXPIRE_HOURS=24")
    print("\n" + "=" * 52)
    print("  IMPORTANT:")
    print("  - Never commit .env to git")
    print("  - Never share OWNER_PASSWORD_HASH")
    print("  - Never share JWT_SECRET")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    main()