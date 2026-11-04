"""Passwords, tokens and the encryption of users' API keys."""

import base64
import hashlib
import hmac
import json
import secrets

from cryptography.fernet import Fernet, InvalidToken

from .config import SECRET_KEY

_fernet = Fernet(base64.urlsafe_b64encode(hashlib.sha256(b"api-keys:" + SECRET_KEY.encode()).digest()))
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${salt.hex()}${digest.hex()}"


def check_password(password: str, stored: str | None) -> bool:
    if not stored:
        return False
    _, n, r, p, salt, digest = stored.split("$")
    got = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p))
    return hmac.compare_digest(got.hex(), digest)


def new_token() -> tuple[str, str]:
    """A random token, and its hash: the token goes to the browser or into an email link, only the
    hash is stored, so the database alone can't be used to sign in."""
    token = secrets.token_urlsafe(32)
    return token, token_hash(token)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def encrypt_keys(keys: dict[str, str]) -> str | None:
    return _fernet.encrypt(json.dumps(keys).encode()).decode() if keys else None


def decrypt_keys(stored: str | None) -> dict[str, str]:
    """{} when there are none, or when they were encrypted with another SECRET_KEY."""
    if not stored:
        return {}
    try:
        return json.loads(_fernet.decrypt(stored.encode()))
    except InvalidToken:
        return {}
