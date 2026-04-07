"""
Tests for owner login, demo login, and JWT token handling.
Covers issue #5 (hash quoting) and auth flow end-to-end.
"""

import pytest
from unittest.mock import patch

from app.core.auth.jwt_handler import (
    verify_password,
    get_password_hash,
    create_owner_token,
    create_demo_token,
    verify_token,
    decode_token,
    authenticate_owner,
)


# =====================================================
# PASSWORD HASHING
# =====================================================

class TestPasswordHashing:

    def test_hash_and_verify_roundtrip(self):
        """Hash generation and verification must be consistent."""
        password = "abcd123"
        hashed = get_password_hash(password)
        assert verify_password(password, hashed) is True

    def test_wrong_password_fails(self):
        """Wrong password must not verify against hash."""
        hashed = get_password_hash("correct_password")
        assert verify_password("wrong_password", hashed) is False

    def test_hash_starts_with_bcrypt_prefix(self):
        """bcrypt hashes must start with $2b$."""
        hashed = get_password_hash("testpassword")
        assert hashed.startswith("$2b$")

    def test_hash_length_is_60(self):
        """bcrypt hashes must be exactly 60 characters."""
        hashed = get_password_hash("testpassword")
        assert len(hashed) == 60

    def test_quoted_hash_same_as_unquoted(self):
        """
        Regression test for issue #5.
        Docker Compose with unquoted hash corrupts it.
        The hash itself must verify regardless of quoting.
        """
        password = "mypassword123"
        hashed = get_password_hash(password)
        # Simulates what Docker sees after corruption
        corrupted = ""
        assert verify_password(password, corrupted) is False
        # Real hash (quoted in .env) must work
        assert verify_password(password, hashed) is True

    def test_empty_password_not_verifiable(self):
        """Empty password must not verify against any hash."""
        hashed = get_password_hash("realpassword")
        assert verify_password("", hashed) is False

    def test_invalid_hash_returns_false(self):
        """verify_password must return False on malformed hash."""
        assert verify_password("password", "not_a_hash") is False
        assert verify_password("password", "") is False
        assert verify_password("password", "$2b$12$corrupted") is False


# =====================================================
# JWT TOKEN CREATION
# =====================================================

class TestTokenCreation:

    def test_create_owner_token_returns_string(self):
        token = create_owner_token("shihab")
        assert isinstance(token, str)
        assert len(token) > 50

    def test_create_demo_token_returns_string(self):
        token = create_demo_token("abc123fingerprint")
        assert isinstance(token, str)
        assert len(token) > 50

    def test_owner_token_payload_role(self):
        token = create_owner_token("shihab")
        payload = verify_token(token)
        assert payload is not None
        assert payload["role"] == "owner"
        assert payload["sub"] == "shihab"

    def test_demo_token_payload_role(self):
        token = create_demo_token("abc123fingerprint")
        payload = verify_token(token)
        assert payload is not None
        assert payload["role"] == "demo"
        assert payload["fingerprint"] == "abc123fingerprint"

    def test_decode_token_raises_on_invalid(self):
        with pytest.raises(Exception):
            decode_token("not.a.valid.token")

    def test_decode_token_raises_on_empty(self):
        with pytest.raises(Exception):
            decode_token("")


# =====================================================
# AUTHENTICATE OWNER
# =====================================================

class TestAuthenticateOwner:

    def test_correct_credentials_return_true(self):
        password = "testpassword123"
        hashed = get_password_hash(password)
        with patch("app.core.auth.jwt_handler.OWNER_PASSWORD_HASH", hashed):
            with patch("app.core.auth.jwt_handler.OWNER_USERNAME", "shihab"):
                result = authenticate_owner("shihab", password)
        assert result is True

    def test_wrong_password_returns_false(self):
        hashed = get_password_hash("correct")
        with patch("app.core.auth.jwt_handler.OWNER_PASSWORD_HASH", hashed):
            with patch("app.core.auth.jwt_handler.OWNER_USERNAME", "shihab"):
                result = authenticate_owner("shihab", "wrong")
        assert result is False

    def test_wrong_username_returns_false(self):
        hashed = get_password_hash("password")
        with patch("app.core.auth.jwt_handler.OWNER_PASSWORD_HASH", hashed):
            with patch("app.core.auth.jwt_handler.OWNER_USERNAME", "shihab"):
                result = authenticate_owner("notshihab", "password")
        assert result is False

    def test_empty_hash_returns_false(self):
        """Regression: corrupted/empty hash must not authenticate."""
        with patch("app.core.auth.jwt_handler.OWNER_PASSWORD_HASH", ""):
            result = authenticate_owner("shihab", "anypassword")
        assert result is False
