# Proxy server for Trakt and TMDB using FastAPI
# Forwards requests to the upstream APIs while preserving query params, body, and most headers.
# Adds lightweight in-memory caching for TMDB GET endpoints.

import os
import time
import asyncio
import sqlite3
import json
from typing import Dict, Any
from datetime import datetime

import httpx
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio as aioredis

# Load environment variables from .env (if present) so os.getenv() below picks them up
load_dotenv()

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("nacho-proxy")

app = FastAPI(title="Nacho Time API Proxy")


# Middleware to auto-refresh tokens when needed
class TokenRefreshMiddleware(BaseHTTPMiddleware):
	async def dispatch(self, request: Request, call_next):
		# Only process requests that have an Authorization header for Trakt endpoints
		auth_header = request.headers.get("authorization")
		
		# Skip if no auth header or not a Trakt request
		if not auth_header or not request.url.path.startswith("/trakt"):
			return await call_next(request)
		
		# Extract token from Authorization header
		if auth_header.startswith("Bearer "):
			initial_token = auth_header[7:].strip()
		else:
			# Not a Bearer token, skip
			return await call_next(request)
		
		logger.info("TokenRefresh: Checking token %s... for path %s", initial_token[:10], request.url.path)
		
		# Get current token info from database
		token_info = get_current_token(initial_token)
		
		if not token_info:
			logger.warning("TokenRefresh: No token found for initial_token %s...", initial_token[:10])
			return JSONResponse(
				status_code=401,
				content={"error": "Unauthorized", "message": "Token not found or expired"}
			)
		
		current_token = token_info["current_token"]
		refresh_token_value = token_info["refresh_token"]
		created_at = token_info["created_at"]
		expires_in = token_info["expires_in"]
		
		# Check if token needs refresh (less than half validity remains)
		if needs_refresh(created_at, expires_in):
			logger.info("TokenRefresh: Token needs refresh (age: %s, half-life: %s)", 
				time.time() - created_at, expires_in / 2 if expires_in else 0)
			
			if not refresh_token_value:
				logger.error("TokenRefresh: No refresh token available for initial_token %s...", initial_token[:10])
				return JSONResponse(
					status_code=401,
					content={"error": "Unauthorized", "message": "No valid refresh token available"}
				)
			
			# Refresh the token
			new_token_data = await refresh_token(refresh_token_value)
			
			if not new_token_data:
				logger.error("TokenRefresh: Failed to refresh token for initial_token %s...", initial_token[:10])
				return JSONResponse(
					status_code=401,
					content={"error": "Unauthorized", "message": "Failed to refresh token"}
				)
			
			# Store the new token
			update_current_token(initial_token, new_token_data)
			current_token = new_token_data["access_token"]
			logger.info("TokenRefresh: Token refreshed successfully, new token: %s...", current_token[:10])
		else:
			logger.info("TokenRefresh: Token still valid, using current_token: %s...", current_token[:10])
		
		# Replace the Authorization header with the current token
		# We need to modify the request headers
		headers = dict(request.headers)
		headers["authorization"] = f"Bearer {current_token}"
		
		# Create a new request scope with updated headers
		scope = request.scope
		scope["headers"] = [(k.encode(), v.encode()) for k, v in headers.items()]
		
		# Continue with the modified request
		response = await call_next(request)
		return response


# Middleware to check for Authorization Bearer token and strip it
class AuthorizationMiddleware(BaseHTTPMiddleware):
	async def dispatch(self, request: Request, call_next):
		# Check if Authorization header exists
		auth_header = request.headers.get("X-Nacho-Auth")
		
		if not auth_header:
			logger.warning("Request blocked: Missing Authorization header from %s", request.client.host if request.client else "unknown")
			return JSONResponse(
				status_code=401,
				content={"error": "Unauthorized", "message": "Authorization header required"}
			)
		token = auth_header
		
		if not token:
			logger.warning("Request blocked: Empty Bearer token from %s", request.client.host if request.client else "unknown")
			return JSONResponse(
				status_code=401,
				content={"error": "Unauthorized", "message": "Bearer token cannot be empty"}
			)
		
		logger.info("Authorization validated and stripped for request from %s", request.client.host if request.client else "unknown")
		
		# Process the request (the header will be stripped automatically as we don't forward it)
		response = await call_next(request)
		return response


# Add the middlewares to the app (order matters: last added runs first)
app.add_middleware(AuthorizationMiddleware)
app.add_middleware(TokenRefreshMiddleware)



# Read upstream base URLs and secrets from environment variables
TRAKT_BASE = os.getenv("API_TRAKT_BASEURL", "https://api.trakt.tv").rstrip("/")
TRAKT_CLIENT_ID = os.getenv("API_TRAKT_CLIENT_ID")
TRAKT_CLIENT_SECRET = os.getenv("API_TRAKT_CLIENT_SECRET")

TMDB_BASE = os.getenv("API_TMDB_BASEURL", "https://api.themoviedb.org/3").rstrip("/")
TMDB_READ_ACCESS_TOKEN = os.getenv("API_TMDB_READ_ACCESS_TOKEN")

PROWLARR_BASE = os.getenv("API_PROWLARR_BASEURL", "http://localhost:9696").rstrip("/")
PROWLARR_API_KEY = os.getenv("API_PROWLARR_API_KEY")

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))


# Startup info: log which important env vars were loaded (mask sensitive values)
def _is_set(val: str | None) -> str:
	return "yes" if val else "no"

logger.info("Startup: TRAKT_BASE=%s TRAKT_CLIENT_ID_set=%s TRAKT_CLIENT_SECRET_set=%s TMDB_TOKEN_set=%s PROWLARR_BASE=%s PROWLARR_KEY_set=%s REDIS=%s:%s", 
	TRAKT_BASE, _is_set(TRAKT_CLIENT_ID), _is_set(TRAKT_CLIENT_SECRET), _is_set(TMDB_READ_ACCESS_TOKEN),
	PROWLARR_BASE, _is_set(PROWLARR_API_KEY),
	REDIS_HOST, REDIS_PORT)

# Initialize SQLite database for OAuth tokens
DB_PATH = os.getenv("OAUTH_DB_PATH", "oauth_tokens.db")

def init_db():
	"""Initialize SQLite database for storing OAuth tokens."""
	conn = sqlite3.connect(DB_PATH)
	cursor = conn.cursor()
	cursor.execute("""
		CREATE TABLE IF NOT EXISTS oauth_tokens (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			access_token TEXT NOT NULL,
			refresh_token TEXT,
			token_type TEXT,
			expires_in INTEGER,
			created_at INTEGER NOT NULL,
			scope TEXT,
			revoked_at INTEGER,
			is_deleted INTEGER DEFAULT 0,
			initial_token TEXT,
			current_token TEXT
		)
	""")
	# Create index for faster lookups by initial_token
	cursor.execute("""
		CREATE INDEX IF NOT EXISTS idx_initial_token 
		ON oauth_tokens(initial_token) 
		WHERE is_deleted = 0
	""")
	conn.commit()
	conn.close()
	logger.info("SQLite database initialized at %s", DB_PATH)

# Initialize database on startup
init_db()

def store_oauth_token(token_data: dict):
	"""Store OAuth token in SQLite database."""
	conn = sqlite3.connect(DB_PATH)
	cursor = conn.cursor()
	
	# Store the token with initial_token = current_token (first time storage)
	access_token = token_data.get("access_token")
	cursor.execute("""
		INSERT INTO oauth_tokens (access_token, refresh_token, token_type, expires_in, created_at, scope, initial_token, current_token)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	""", (
		access_token,
		token_data.get("refresh_token"),
		token_data.get("token_type"),
		token_data.get("expires_in"),
		int(time.time()),
		token_data.get("scope"),
		access_token,  # initial_token = access_token
		access_token   # current_token = access_token
	))
	conn.commit()
	token_id = cursor.lastrowid
	conn.close()
	logger.info("Stored OAuth token with ID %s", token_id)
	return token_id

def get_current_token(initial_token: str):
	"""Get the current valid token for a given initial token."""
	conn = sqlite3.connect(DB_PATH)
	cursor = conn.cursor()
	cursor.execute("""
		SELECT access_token, refresh_token, expires_in, created_at, current_token
		FROM oauth_tokens
		WHERE initial_token = ? AND is_deleted = 0
		ORDER BY created_at DESC
		LIMIT 1
	""", (initial_token,))
	row = cursor.fetchone()
	conn.close()
	
	if row:
		return {
			"access_token": row[0],
			"refresh_token": row[1],
			"expires_in": row[2],
			"created_at": row[3],
			"current_token": row[4]
		}
	return None

def update_current_token(initial_token: str, new_token_data: dict):
	"""Update the current token for an initial token with a refreshed token."""
	conn = sqlite3.connect(DB_PATH)
	cursor = conn.cursor()
	
	new_access_token = new_token_data.get("access_token")
	
	# Insert new token record with same initial_token
	cursor.execute("""
		INSERT INTO oauth_tokens (access_token, refresh_token, token_type, expires_in, created_at, scope, initial_token, current_token)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	""", (
		new_access_token,
		new_token_data.get("refresh_token"),
		new_token_data.get("token_type"),
		new_token_data.get("expires_in"),
		int(time.time()),
		new_token_data.get("scope"),
		initial_token,      # Keep the same initial_token
		new_access_token    # Update current_token to new one
	))
	
	conn.commit()
	token_id = cursor.lastrowid
	conn.close()
	logger.info("Updated current token for initial_token %s... with new token ID %s", initial_token[:10], token_id)
	return token_id

def needs_refresh(created_at: int, expires_in: int) -> bool:
	"""Check if token needs refresh (less than half validity remains)."""
	if not expires_in:
		return False
	
	now = time.time()
	age = now - created_at
	half_life = expires_in / 2
	
	needs_it = age >= half_life
	logger.debug("Token age: %s, half-life: %s, needs refresh: %s", age, half_life, needs_it)
	return needs_it

async def refresh_token(refresh_token_value: str) -> dict:
	"""Refresh an OAuth token using the refresh_token."""
	client = get_client()
	upstream_url = f"{TRAKT_BASE}/oauth/token"
	
	body = {
		"refresh_token": refresh_token_value,
		"client_id": TRAKT_CLIENT_ID,
		"client_secret": TRAKT_CLIENT_SECRET,
		"redirect_uri": "urn:ietf:wg:oauth:2.0:oob",
		"grant_type": "refresh_token"
	}
	
	headers = {
		"Content-Type": "application/json",
		"trakt-api-version": "2",
		"User-Agent": "NachoTime/1.0.0",
	}
	
	logger.info("Refreshing token using refresh_token")
	resp = await client.post(upstream_url, json=body, headers=headers)
	
	if resp.status_code == 200:
		token_data = resp.json()
		logger.info("Token refresh successful")
		return token_data
	else:
		logger.error("Token refresh failed with status %s: %s", resp.status_code, resp.content[:200])
		return None

def revoke_oauth_token(access_token: str):
	"""Mark OAuth token as revoked/deleted in the database."""
	conn = sqlite3.connect(DB_PATH)
	cursor = conn.cursor()
	cursor.execute("""
		UPDATE oauth_tokens 
		SET is_deleted = 1, revoked_at = ?
		WHERE access_token = ? AND is_deleted = 0
	""", (int(time.time()), access_token))
	conn.commit()
	rows_affected = cursor.rowcount
	conn.close()
	logger.info("Marked %s token(s) as revoked for access_token %s...", rows_affected, access_token[:10])
	return rows_affected

# HTTP client reused across requests
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
	global _client
	if _client is None:
		_client = httpx.AsyncClient(timeout=60.0)
	return _client


@app.on_event("startup")
async def startup_event():
	"""Initialize Redis cache on startup."""
	redis = aioredis.from_url(
		f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
		encoding="utf-8",
		decode_responses=True
	)
	FastAPICache.init(RedisBackend(redis), prefix="nacho-cache:")
	logger.info("FastAPICache initialized with Redis backend at %s:%s", REDIS_HOST, REDIS_PORT)


@app.on_event("shutdown")
async def shutdown_event():
	global _client
	if _client is not None:
		await _client.aclose()
		_client = None
	# Clear FastAPICache
	await FastAPICache.clear()
	logger.info("FastAPICache cleared on shutdown")


def _make_cache_key(url: str, params: Dict[str, Any]) -> str:
	# stable representation of url + sorted query params
	if not params:
		return url
	parts = [f"{k}={params[k]}" for k in sorted(params.keys())]
	return url + "?" + "&".join(parts)


def _tmdb_ttl_for_path(path: str) -> int:
	# Enforce a 30-minute TTL for all TMDB responses per user requirement.
	# If you later want to restore path-based heuristics, change this logic.
	return 30 * 60


async def _get_from_tmdb_with_cache(url: str, params: Dict[str, Any]):
	"""Fetch from TMDB with Redis caching (30-minute TTL)."""
	key = _make_cache_key(url, params)
	
	# Try to get from Redis cache
	try:
		cached_data = await FastAPICache.get_backend().get(key)
		if cached_data:
			logger.info("TMDB cache HIT %s", key)
			# Parse the cached JSON data
			cached = json.loads(cached_data)
			return cached["status"], cached["headers"], cached["content"].encode(), True
	except Exception as e:
		logger.warning("Cache retrieval error: %s", e)

	logger.info("TMDB cache MISS %s — fetching %s params=%s", key, url, params)
	client = get_client()
	resp = await client.get(url, params=params, headers={"Authorization": f"Bearer {TMDB_READ_ACCESS_TOKEN}"} if TMDB_READ_ACCESS_TOKEN else None)

	content = resp.content
	status = resp.status_code
	# Copy headers we care about
	headers = {k: v for k, v in resp.headers.items()}

	# Cache only successful GET responses (200)
	if resp.status_code == 200:
		ttl = _tmdb_ttl_for_path(url.replace(TMDB_BASE + "/", ""))
		cache_data = {
			"status": status,
			"headers": headers,
			"content": content.decode('utf-8', errors='replace')
		}
		try:
			await FastAPICache.get_backend().set(key, json.dumps(cache_data), ttl)
			logger.info("TMDB response cached %s ttl=%s", key, ttl)
		except Exception as e:
			logger.warning("Cache storage error: %s", e)

	return status, headers, content, False


def _filter_upstream_headers(headers: Dict[str, str]) -> Dict[str, str]:
	# Remove hop-by-hop and headers that should not be forwarded back to client
	hop_by_hop = {
		"connection",
		"keep-alive",
		"proxy-authenticate",
		"proxy-authorization",
		"te",
		"trailer",
		"transfer-encoding",
		"upgrade",
		"Authorization",
		"Encoding",
		"content-encoding",
		"x-rate-limit"
	}
	return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop}


def _mask_sensitive_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Return a copy of headers with sensitive values masked for safe logging."""
    masked = {}
    for k, v in headers.items():
        low = k.lower()
        if low in ("authorization", "trakt-api-key", "x-api-key", "x-api-token") and v:
            masked[k] = v[:4] + "..."  # show prefix only
        else:
            masked[k] = v
    return masked


async def _proxy_request(base: str, full_path: str, request: Request, add_headers: Dict[str, str] | None = None, use_tmdb_cache: bool = False):
	client = get_client()
	method = request.method
	upstream_url = f"{base}/{full_path.lstrip('/')}"

	# Build headers for upstream request: start fresh and only add what we need
	# Don't forward client headers that could cause issues with Cloudflare
	upstream_headers = {}
	
	# Only forward specific safe headers from the client
	safe_headers = ["accept", "accept-encoding", "accept-language"]
	for header_name in safe_headers:
		if header_name in request.headers:
			upstream_headers[header_name] = request.headers[header_name]
	
	# Add our custom headers (API keys, tokens, etc.)
	if add_headers:
		upstream_headers.update({k: v for k, v in add_headers.items() if v is not None})

	params = dict(request.query_params)

	body = None
	if method not in ("GET", "HEAD"):
		body = await request.body()

	logger.info("Incoming request %s %s params=%s from=%s", method, upstream_url, dict(request.query_params), request.client.host if request.client else "unknown")
	logger.info("Incoming request headers: %s", dict(request.headers))
	if body:
		logger.info("Incoming request body: %s", body[:1000])  # Log first 1000 bytes

	# TMDB caching only for GET
	if use_tmdb_cache and method == "GET":
		logger.info("Upstream request headers (for TMDB cache check): %s", upstream_headers)
		status, headers, content, cached = await _get_from_tmdb_with_cache(upstream_url, params)
		headers = _filter_upstream_headers(headers)
		logger.info("Responding (cached=%s) %s %s -> %s", cached, method, upstream_url, status)
		if not cached:
			logger.info("Response body (first 500 bytes): %s", content[:500])
		return Response(content=content, status_code=status, headers=headers)

	# Otherwise forward the request to upstream
	logger.info("Forwarding to upstream %s %s params=%s", method, upstream_url, params)
	logger.info("Upstream request headers (full): %s", upstream_headers)
	if body:
		logger.info("Upstream request body: %s", body[:1000])
	
	resp = await client.request(method, upstream_url, params=params, content=body, headers=upstream_headers)

	headers = _filter_upstream_headers(dict(resp.headers))
	logger.info("Upstream response %s -> %s", upstream_url, resp.status_code)
	logger.info("Upstream response headers: %s", dict(resp.headers))
	logger.info("Upstream response body (first 500 bytes): %s", resp.content[:500])
	return Response(content=resp.content, status_code=resp.status_code, headers=headers)


# Specific Trakt routes MUST come BEFORE the catch-all route

# OAuth device code generation endpoint
@app.post("/trakt/oauth/device/code")
async def trakt_oauth_device_code(request: Request):
	"""Handle OAuth device code generation - injects client_id into the request body."""
	logger.info("Trakt OAuth device code request received")
	
	# Read the incoming body (may be empty or contain other data)
	try:
		body = await request.body()
		if body:
			incoming_data = json.loads(body)
			logger.info("Incoming device code request body: %s", incoming_data)
		else:
			incoming_data = {}
	except json.JSONDecodeError:
		incoming_data = {}
	
	# Replace body with only client_id as required
	outgoing_body = {"client_id": TRAKT_CLIENT_ID}
	
	logger.info("Modified device code request body: %s", outgoing_body)
	
	# Forward to Trakt API
	client = get_client()
	upstream_url = f"{TRAKT_BASE}/oauth/device/code"
	headers = {
		"Content-Type": "application/json",
		"trakt-api-version": "2",
		"User-Agent": "NachoTime/1.0.0",
	}
	
	logger.info("Forwarding device code request to %s", upstream_url)
	resp = await client.post(upstream_url, json=outgoing_body, headers=headers)
	
	content = resp.content
	status = resp.status_code
	response_headers = _filter_upstream_headers(dict(resp.headers))
	
	logger.info("Device code response status: %s", status)
	logger.info("Device code response body: %s", content[:500])
	
	return Response(content=content, status_code=status, headers=response_headers, media_type="application/json")


# OAuth token exchange endpoint
@app.post("/trakt/oauth/device/token")
async def trakt_oauth_device_token(request: Request):
	"""Handle OAuth token exchange - injects client_id and client_secret, stores token on success."""
	logger.info("Trakt OAuth token exchange request received")
	
	# Read the incoming body (should contain device_code, etc.)
	try:
		body = await request.body()
		incoming_data = json.loads(body) if body else {}
		logger.info("Incoming token request body: %s", incoming_data)
	except json.JSONDecodeError:
		logger.error("Failed to parse incoming token request body")
		return JSONResponse(
			status_code=400,
			content={"error": "Invalid JSON in request body"}
		)
	
	# Add client_id and client_secret to the existing body
	outgoing_body = {**incoming_data}
	outgoing_body["client_id"] = TRAKT_CLIENT_ID
	outgoing_body["client_secret"] = TRAKT_CLIENT_SECRET
	
	logger.info("Modified token request body: client_id=%s, client_secret=***", TRAKT_CLIENT_ID)
	logger.info("Full token request body: %s", outgoing_body)
	
	# Forward to Trakt API
	client = get_client()
	upstream_url = f"{TRAKT_BASE}/oauth/device/token"
	headers = {
		"Content-Type": "application/json",
	}
	
	logger.info("Forwarding token request to %s, with headers: %s", upstream_url, _mask_sensitive_headers(headers))
	resp = await client.post(upstream_url, json=outgoing_body, headers=headers)
	
	content = resp.content
	status = resp.status_code
	response_headers = _filter_upstream_headers(dict(resp.headers))
	
	logger.info("Token response status: %s", status)
	
	# If successful (200), store the token in the database
	if status == 200:
		try:
			token_data = json.loads(content)
			logger.info("Token exchange successful, storing in database")
			logger.info("Token data: access_token=%s..., expires_in=%s", 
				token_data.get("access_token", "")[:10], 
				token_data.get("expires_in"))
			
			token_id = store_oauth_token(token_data)
			logger.info("Token stored with ID %s at timestamp %s", token_id, int(time.time()))
		except Exception as e:
			logger.error("Failed to store token in database: %s", e)
	else:
		logger.warning("Token exchange failed with status %s: %s", status, content[:200])
	
	return Response(content=content, status_code=status, headers=response_headers, media_type="application/json")

# OAuth token exchange endpoint
@app.post("/trakt/oauth/revoke")
async def trakt_oauth_revoke(request: Request):
	"""Handle OAuth token revocation - marks token as deleted in database."""
	logger.info("Trakt OAuth token revocation request received")
	
	# Read the incoming body (should contain token, etc.)
	try:
		body = await request.body()
		incoming_data = json.loads(body) if body else {}
		logger.info("Incoming revoke request body: %s", incoming_data)
	except json.JSONDecodeError:
		logger.error("Failed to parse incoming revoke request body")
		return JSONResponse(
			status_code=400,
			content={"error": "Invalid JSON in request body"}
		)
	
	# Extract the token that will be revoked
	token_to_revoke = incoming_data.get("token")
	
	# Add client_id and client_secret to the existing body
	outgoing_body = {**incoming_data}
	outgoing_body["client_id"] = TRAKT_CLIENT_ID
	outgoing_body["client_secret"] = TRAKT_CLIENT_SECRET
	
	logger.info("Modified revoke request body: client_id=%s, client_secret=***, token=%s...", 
		TRAKT_CLIENT_ID, token_to_revoke[:10] if token_to_revoke else "None")
	
	# Forward to Trakt API
	client = get_client()
	upstream_url = f"{TRAKT_BASE}/oauth/revoke"
	headers = {
		"Content-Type": "application/json",
		"trakt-api-version": "2",
		"User-Agent": "NachoTime/1.0.0",
	}

	logger.info("Forwarding revoke request to %s", upstream_url)
	resp = await client.post(upstream_url, json=outgoing_body, headers=headers)
	
	content = resp.content
	status = resp.status_code
	response_headers = _filter_upstream_headers(dict(resp.headers))
	
	logger.info("Revoke response status: %s", status)
	
	# If successful (200 or 204), mark the token as deleted in the database
	if status in (200, 204):
		if token_to_revoke:
			try:
				rows_affected = revoke_oauth_token(token_to_revoke)
				if rows_affected > 0:
					logger.info("Token revoked successfully and marked as deleted in database")
				else:
					logger.warning("Token not found in database or already deleted")
			except Exception as e:
				logger.error("Failed to mark token as deleted in database: %s", e)
		else:
			logger.warning("No token provided in revoke request, skipping database update")
	else:
		logger.warning("Token revocation failed with status %s: %s", status, content[:200])
	
	return Response(content=content, status_code=status, headers=response_headers, media_type="application/json")


@app.get("/trakt/movies/trending")
async def trakt_movies_trending(page: int = 1, limit: int = 20):
	"""Cached endpoint for trending movies - no auth required, results are cached for 5 minutes."""
	cache_key = f"trakt:movies_trending_p{page}_l{limit}"
	
	# Check Redis cache
	try:
		cached_data = await FastAPICache.get_backend().get(cache_key)
		if cached_data:
			logger.info("Trakt trending movies cache HIT: page=%s limit=%s", page, limit)
			cached = json.loads(cached_data)
			return Response(
				content=cached["content"].encode(),
				status_code=cached["status"],
				headers=cached["headers"],
				media_type="application/json"
			)
	except Exception as e:
		logger.warning("Cache retrieval error: %s", e)
	
	logger.info("Trakt trending movies cache MISS: page=%s limit=%s - fetching from upstream", page, limit)
	
	# Fetch from upstream (no auth header)
	client = get_client()
	upstream_url = f"{TRAKT_BASE}/movies/trending"
	headers ={
		"trakt-api-key": TRAKT_CLIENT_ID,
		"trakt-api-version": "2",
		"User-Agent": "NachoTime/1.1.0",
		"Accept": "application/json",
	}
	
	logger.info("Fetching %s?page=%s&limit=%s with headers: %s", upstream_url, page, limit, headers)
	resp = await client.get(upstream_url, params={"page": page, "limit": limit}, headers=headers)
	
	content = resp.content
	status = resp.status_code
	response_headers = _filter_upstream_headers(dict(resp.headers))
	print("DEBUG: Response headers:", response_headers)
	
	logger.info("Trakt trending movies upstream response: status=%s", status)
	logger.info("Response body (first 500 bytes): %s", content[:500])
	
	# Cache for 5 minutes on success
	if status == 200:
		ttl = 5 * 60
		cache_data = {
			"status": status,
			"headers": response_headers,
			"content": content.decode('utf-8', errors='replace')
		}
		try:
			await FastAPICache.get_backend().set(cache_key, json.dumps(cache_data), ttl)
			logger.info("Cached trending movies for %s seconds", ttl)
		except Exception as e:
			logger.warning("Cache storage error: %s", e)

	print(Response(content=content, status_code=status, headers=response_headers, media_type="application/json"))

	return Response(content=content, status_code=status, headers=response_headers, media_type="application/json")


@app.get("/trakt/shows/trending")
async def trakt_shows_trending(page: int = 1, limit: int = 20):
	"""Cached endpoint for trending shows - no auth required, results are cached for 5 minutes."""
	cache_key = f"trakt:shows_trending_p{page}_l{limit}"
	
	# Check Redis cache
	try:
		cached_data = await FastAPICache.get_backend().get(cache_key)
		if cached_data:
			logger.info("Trakt trending shows cache HIT: page=%s limit=%s", page, limit)
			cached = json.loads(cached_data)
			return Response(
				content=cached["content"].encode(),
				status_code=cached["status"],
				headers=cached["headers"],
				media_type="application/json"
			)
	except Exception as e:
		logger.warning("Cache retrieval error: %s", e)
	
	logger.info("Trakt trending shows cache MISS: page=%s limit=%s - fetching from upstream", page, limit)
	
	# Fetch from upstream (no auth header)
	client = get_client()
	upstream_url = f"{TRAKT_BASE}/shows/trending"
	headers = {
		"trakt-api-key": TRAKT_CLIENT_ID,
		"trakt-api-version": "2",
		"User-Agent": "NachoTime/1.1.0",
		"Accept": "application/json",
	}
	
	logger.info("Fetching %s?page=%s&limit=%s with headers: %s", upstream_url, page, limit, headers)
	resp = await client.get(upstream_url, params={"page": page, "limit": limit}, headers=headers)
	
	content = resp.content
	status = resp.status_code
	response_headers = _filter_upstream_headers(dict(resp.headers))
	
	logger.info("Trakt trending shows upstream response: status=%s", status)
	logger.info("Response body (first 500 bytes): %s", content[:500])
	
	# Cache for 5 minutes on success
	if status == 200:
		ttl = 5 * 60
		cache_data = {
			"status": status,
			"headers": response_headers,
			"content": content.decode('utf-8', errors='replace')
		}
		try:
			await FastAPICache.get_backend().set(cache_key, json.dumps(cache_data), ttl)
			logger.info("Cached trending shows for %s seconds", ttl)
		except Exception as e:
			logger.warning("Cache storage error: %s", e)
	
	return Response(content=content, status_code=status, headers=response_headers, media_type="application/json")


# Catch-all route for other Trakt endpoints (must be defined AFTER specific routes)
@app.api_route("/trakt/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def trakt_proxy(full_path: str, request: Request):
	"""Proxy all requests under /trakt/... to the Trakt.tv API.

	Adds required Trakt headers (API key, API version, User-Agent) while preserving
	the incoming Authorization header if present.
	"""
	add_headers = {
		"trakt-api-key": TRAKT_CLIENT_ID,
		"trakt-api-version": "2",
		"User-Agent": "NachoTime/1.0.0",
		"Accept": "application/json",
	}
	# If the client provided Authorization (Bearer access token) preserve it so upstream
	# sees the user's token.
	if "authorization" in request.headers:
		add_headers["Authorization"] = request.headers["authorization"]

	# Ensure content-type for non-GET requests if missing
	if request.method not in ("GET", "HEAD") and "content-type" not in request.headers:
		add_headers["Content-Type"] = "application/json"

	logger.info("Proxying to Trakt: %s", full_path)
	return await _proxy_request(TRAKT_BASE, full_path, request, add_headers=add_headers, use_tmdb_cache=False)


@app.api_route("/tmdb/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def tmdb_proxy(full_path: str, request: Request):
	"""Proxy all requests under /tmdb/... to TheMovieDB API.

	The proxy will attach the read-access bearer token from env and will apply
	a simple in-memory cache for GET requests using heuristics.
	"""
	add_headers = {}
	if TMDB_READ_ACCESS_TOKEN:
		add_headers["Authorization"] = f"Bearer {TMDB_READ_ACCESS_TOKEN}"

	logger.info("Proxying to TMDB: %s", full_path)
	# only GETs use the cache
	return await _proxy_request(TMDB_BASE, full_path, request, add_headers=add_headers, use_tmdb_cache=True)


@app.api_route("/prowlarr/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def prowlarr_proxy(full_path: str, request: Request):
	"""Proxy all requests under /prowlarr/... to Prowlarr API.

	Attaches the API key via X-Api-Key header. No caching is applied to Prowlarr requests.
	"""
	add_headers = {}
	if PROWLARR_API_KEY:
		add_headers["X-Api-Key"] = PROWLARR_API_KEY

	logger.info("Proxying to Prowlarr: %s", full_path)
	# No caching for Prowlarr
	return await _proxy_request(PROWLARR_BASE, full_path, request, add_headers=add_headers, use_tmdb_cache=False)


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("proxy-server.main:app", host="0.0.0.0", port=8000, reload=True)
