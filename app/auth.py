"""Accounts: sign up with a username, email and password (the email verified by a link), or with
Google. A session is a random token in an HttpOnly cookie; the database keeps only its hash.

Admins (ADMIN_EMAILS) use the server's API keys from .env; everyone else adds their own, which are
stored encrypted (crypto.py). The first admin to sign in gets the library from before accounts.
"""

import asyncio
import base64
import json
import re
import secrets
import time
from typing import Annotated
from urllib.parse import quote, urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from . import crypto, db, mailer
from .config import ADMIN_EMAILS, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, PUBLIC_URL, SERVER_KEYS
from .providers import PROVIDERS, get_provider

router = APIRouter()

SESSION_COOKIE = "session"
SESSION_DAYS = 30
LINK_HOURS = {"verify": 48, "reset": 2}
KEY_NAMES = (*PROVIDERS, "tavily")  # what a user can store a key for
_USERNAME = re.compile(r"[A-Za-z0-9_.-]{3,32}")
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_DUMMY_HASH = crypto.hash_password(secrets.token_urlsafe(16))  # so an unknown login takes as long as a known one


# ---------------------------------------------------------------- the signed-in user

def is_admin(row: dict) -> bool:
    return bool(row["email_verified"]) and row["email"].lower() in ADMIN_EMAILS


def user_keys(row: dict) -> dict[str, str]:
    """{provider id or "tavily": API key} for the user's requests: their own keys, and for an admin
    the server's (which win)."""
    keys = crypto.decrypt_keys(row["api_keys"])
    return {**keys, **SERVER_KEYS} if is_admin(row) else keys


def _user(row: dict) -> dict:
    return {**row, "is_admin": is_admin(row), "keys": user_keys(row)}


def current_user(request: Request) -> dict:
    """The signed-in user (with "is_admin" and "keys"); 401 when nobody is."""
    token = request.cookies.get(SESSION_COOKIE)
    row = token and db.session_user(crypto.token_hash(token))
    if not row:
        raise HTTPException(401, "Sign in first")
    return _user(row)


User = Annotated[dict, Depends(current_user)]


def public(user: dict) -> dict:
    """What the browser is told about the user. Keys show only where they come from and their end."""
    own = crypto.decrypt_keys(user["api_keys"])
    keys = {}
    for name in KEY_NAMES:
        if user["is_admin"] and name in SERVER_KEYS:
            keys[name] = {"source": "server"}
        elif own.get(name):
            keys[name] = {"source": "yours", "hint": "…" + own[name][-4:]}
    return {"id": user["id"], "email": user["email"], "username": user["username"], "name": user["name"],
            "is_admin": user["is_admin"], "tour_done": bool(user["tour_done"]),
            "has_password": bool(user["password_hash"]), "google": bool(user["google_sub"]), "keys": keys}


def base_url(request: Request) -> str:
    """The app's address as the browser sees it (behind Vercel's proxy too)."""
    if PUBLIC_URL:
        return PUBLIC_URL
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    return f"{proto.split(',')[0].strip()}://{host.split(',')[0].strip()}"


def _start_session(response: Response, request: Request, user_id: str):
    token, token_hash = crypto.new_token()
    db.add_session(token_hash, user_id, time.time() + SESSION_DAYS * 86400)
    response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_DAYS * 86400, httponly=True, samesite="lax",
                        secure=base_url(request).startswith("https://"), path="/")
    row = db.get_user(user_id)
    if is_admin(row):
        db.claim_unowned(user_id)


async def _send_link(request: Request, user: dict, purpose: str):
    token, token_hash = crypto.new_token()
    await db.aio.add_email_token(token_hash, user["id"], purpose, time.time() + LINK_HOURS[purpose] * 3600)
    # In the URL's fragment, which browsers don't send to the server, and used by the page with a
    # POST: a mail scanner that opens the link doesn't use it up.
    link = f"{base_url(request)}/#{purpose}={token}"
    if purpose == "verify":
        subject = "Confirm your email for Paper Reader"
        text = (f"Hi {user['username'] or ''},\n\nOpen this link to confirm your email address and start using "
                f"Paper Reader:\n\n{link}\n\nThe link works for {LINK_HOURS[purpose]} hours. If you didn't sign up, "
                "ignore this email.")
    else:
        subject = "Reset your Paper Reader password"
        text = (f"Open this link to choose a new password:\n\n{link}\n\nThe link works for {LINK_HOURS[purpose]} "
                "hours. If you didn't ask for it, ignore this email: your password stays as it is.")
    await mailer.send(user["email"], subject, text)


# ---------------------------------------------------------------- username and password

class SignupIn(BaseModel):
    username: str
    email: str
    password: str


class LoginIn(BaseModel):
    login: str  # username or email
    password: str


class EmailIn(BaseModel):
    email: str


class TokenIn(BaseModel):
    token: str
    password: str | None = None  # a reset's new password


def _check_password(password: str):
    if not 8 <= len(password) <= 200:
        raise HTTPException(400, "Use a password of at least 8 characters")


@router.get("/api/auth/options")
def options():
    """How one can sign in here (shown before anyone has)."""
    return {"google": bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET), "email": mailer.configured()}


@router.post("/api/auth/signup")
async def signup(body: SignupIn, request: Request):
    """Creates the account, unverified until its email's link is opened."""
    username, email = body.username.strip(), body.email.strip()
    if not _USERNAME.fullmatch(username):
        raise HTTPException(400, "A username has 3 to 32 letters, digits, dots, dashes or underscores")
    if len(email) > 254 or not _EMAIL.fullmatch(email):
        raise HTTPException(400, "That doesn't look like an email address")
    _check_password(body.password)
    existing = await db.aio.find_user(email=email)
    if existing and existing["email_verified"]:
        raise HTTPException(409, "There's already an account with this email. Sign in, or reset its password.")
    if existing:
        # Signed up before and not confirmed yet. Replacing that password would let whoever signs
        # up second choose the password of the account the owner then confirms from their inbox.
        # "Forgot password" sends its link to the address, so only the owner can change it.
        raise HTTPException(409, "This email already has an account waiting for confirmation. Open the link we"
                                 " sent it, or use “Forgot password?” to choose a new password.")
    if await db.aio.find_user(username=username):
        raise HTTPException(409, "That username is taken")
    password_hash = await asyncio.to_thread(crypto.hash_password, body.password)
    user = await db.aio.create_user(email=email, username=username, password_hash=password_hash)
    await _send_link(request, user, "verify")
    return {"ok": True, "email": email}


@router.post("/api/auth/login")
def login(body: LoginIn, request: Request):
    name = body.login.strip()
    row = db.find_user(email=name) if "@" in name else db.find_user(username=name)
    if row and not row["password_hash"]:
        raise HTTPException(401, "This account signs in with Google. (To add a password, use “Forgot password”.)")
    if not crypto.check_password(body.password, row["password_hash"] if row else _DUMMY_HASH) or not row:
        raise HTTPException(401, "Wrong username, email or password")
    if not row["email_verified"]:
        return JSONResponse({"detail": f"Confirm your email first: open the link we sent to {row['email']}.",
                             "unverified": True}, status_code=403)
    response = JSONResponse({"ok": True})
    _start_session(response, request, row["id"])
    return response


@router.post("/api/auth/resend")
async def resend(body: LoginIn, request: Request):
    """Sends the verification link again (needs the password, so it can't be used to spam an address)."""
    name = body.login.strip()
    row = await (db.aio.find_user(email=name) if "@" in name else db.aio.find_user(username=name))
    if row and not row["email_verified"] and row["password_hash"] and \
            await asyncio.to_thread(crypto.check_password, body.password, row["password_hash"]):
        await _send_link(request, row, "verify")
    return {"ok": True}


@router.post("/api/auth/verify")
def verify(body: TokenIn, request: Request):
    user_id = db.take_email_token(crypto.token_hash(body.token), "verify")
    if not user_id:
        raise HTTPException(400, "This link has expired or was already used. Sign in to get a new one.")
    db.update_user(user_id, email_verified=1)
    response = JSONResponse({"ok": True})
    _start_session(response, request, user_id)
    return response


@router.post("/api/auth/forgot")
async def forgot(body: EmailIn, request: Request):
    """Emails a link for choosing a new password (also a way to add one to a Google account). The
    answer is the same whether or not there's an account, so it can't be used to find out."""
    row = await db.aio.find_user(email=body.email.strip()) if body.email.strip() else None
    if row:
        await _send_link(request, row, "reset")
    return {"ok": True}


@router.post("/api/auth/reset")
def reset(body: TokenIn, request: Request):
    _check_password(body.password or "")
    user_id = db.take_email_token(crypto.token_hash(body.token), "reset")
    if not user_id:
        raise HTTPException(400, "This link has expired or was already used. Ask for a new one.")
    # The link reached the address, so it's verified too; other sessions end with the old password.
    db.update_user(user_id, password_hash=crypto.hash_password(body.password), email_verified=1)
    db.delete_sessions(user_id=user_id)
    response = JSONResponse({"ok": True})
    _start_session(response, request, user_id)
    return response


@router.post("/api/auth/logout")
def logout(request: Request):
    if token := request.cookies.get(SESSION_COOKIE):
        db.delete_sessions(token_hash=crypto.token_hash(token))
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


# ---------------------------------------------------------------- Google

GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
STATE_COOKIE = "oauth_state"


def _signin_error(message: str) -> RedirectResponse:
    return RedirectResponse(f"/#signin-error={quote(message)}", status_code=303)


@router.get("/auth/google")
def google_start(request: Request):
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
        raise HTTPException(404, "Google sign-in isn't set up (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET)")
    state = secrets.token_urlsafe(24)
    response = RedirectResponse(GOOGLE_AUTH + "?" + urlencode({
        "client_id": GOOGLE_CLIENT_ID, "redirect_uri": base_url(request) + "/auth/google/callback",
        "response_type": "code", "scope": "openid email profile", "state": state, "prompt": "select_account",
    }), status_code=303)
    # Ties Google's answer to this browser: the callback must bring the same state back.
    response.set_cookie(STATE_COOKIE, state, max_age=600, httponly=True, samesite="lax",
                        secure=base_url(request).startswith("https://"), path="/auth/google")
    return response


def _id_token_claims(id_token: str) -> dict:
    """The claims of an ID token that came straight from Google's token endpoint over TLS, which
    Google's docs say needn't have its signature checked; its audience and expiry still are."""
    payload = id_token.split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    if claims.get("aud") != GOOGLE_CLIENT_ID or claims.get("iss") not in ("accounts.google.com",
                                                                          "https://accounts.google.com"):
        raise ValueError("the ID token isn't for this app")
    if claims.get("exp", 0) < time.time():
        raise ValueError("the ID token has expired")
    return claims


@router.get("/auth/google/callback")
async def google_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    if error:
        return _signin_error("Google sign-in was cancelled" if error == "access_denied" else f"Google: {error}")
    if not code or not state or not secrets.compare_digest(state, request.cookies.get(STATE_COOKIE, "")):
        return _signin_error("That sign-in link is stale. Try again.")
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(GOOGLE_TOKEN, data={
                "code": code, "client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": base_url(request) + "/auth/google/callback", "grant_type": "authorization_code",
            })
            r.raise_for_status()
        claims = _id_token_claims(r.json()["id_token"])
    except Exception as e:
        return _signin_error(f"Google sign-in failed: {e}")
    if not claims.get("email") or not claims.get("email_verified"):
        return _signin_error("Google didn't confirm an email address for this account")

    row = await db.aio.find_user(google_sub=claims["sub"])
    if not row and (row := await db.aio.find_user(email=claims["email"])):
        # The address signed up with a password before. Google vouches for the address now; a
        # password (and username) chosen by someone who never confirmed it is dropped.
        fields = {"google_sub": claims["sub"], "email_verified": 1, "name": row["name"] or claims.get("name", "")}
        if not row["email_verified"]:
            fields |= {"password_hash": None, "username": None}
        await db.aio.update_user(row["id"], **fields)
    elif not row:
        row = await db.aio.create_user(email=claims["email"], google_sub=claims["sub"], name=claims.get("name", ""),
                                       email_verified=True)
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(STATE_COOKIE, path="/auth/google")
    await asyncio.to_thread(_start_session, response, request, row["id"])
    return response


# ---------------------------------------------------------------- the account

class AccountPatch(BaseModel):
    tour_done: bool | None = None
    keys: dict[str, str | None] | None = None  # name -> key to store, or None/"" to remove it


@router.get("/api/auth/me")
def me(user: User):
    return public(user)


@router.patch("/api/account")
async def update_account(body: AccountPatch, user: User):
    """Marks the walkthrough seen, and saves or removes the user's API keys. A provider key is
    tried first, so a mistyped one is caught here."""
    if body.tour_done is not None:
        await db.aio.update_user(user["id"], tour_done=int(body.tour_done))
    if body.keys is not None:
        stored = crypto.decrypt_keys(user["api_keys"])
        for name, key in body.keys.items():
            if name not in KEY_NAMES:
                raise HTTPException(400, f"Unknown key: {name}")
            key = (key or "").strip()
            if not key:
                stored.pop(name, None)
                continue
            if name in PROVIDERS:
                try:
                    await asyncio.wait_for(get_provider(name, key).check_key(), 20)
                except Exception as e:
                    raise HTTPException(400, f"{PROVIDERS[name].label} didn't accept that key: "
                                             f"{str(e)[:200] or type(e).__name__}")
            stored[name] = key
        await db.aio.update_user(user["id"], api_keys=crypto.encrypt_keys(stored))
    return public(_user(await db.aio.get_user(user["id"])))
