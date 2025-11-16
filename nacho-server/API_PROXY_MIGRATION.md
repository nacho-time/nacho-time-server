# API Proxy Migration - SolidStart

The TMDB and Prowlarr proxy functionality has been moved from the Python FastAPI proxy server into the SolidStart application as API routes.

## New API Routes

### TMDB Proxy

**Endpoint:** `/api/tmdb/[...path]`

Proxies all requests to The Movie Database API with:

- Automatic Bearer token authentication
- 30-minute in-memory caching for GET requests
- Support for all HTTP methods (GET, POST, PUT, DELETE, PATCH)

**Example:**

```bash
# Get movie details
curl http://localhost:3000/api/tmdb/movie/550

# Search for movies
curl http://localhost:3000/api/tmdb/search/movie?query=matrix
```

### Prowlarr Proxy

**Endpoint:** `/api/prowlarr/[...path]`

Proxies all requests to Prowlarr API with:

- Automatic API key injection (X-Api-Key header)
- 30-minute caching for search results
- URL rewriting to secure download endpoints
- Support for all HTTP methods

**Example:**

```bash
# Search for content
curl http://localhost:3000/api/prowlarr/api/v1/search?query=ubuntu&type=search

# Get indexers
curl http://localhost:3000/api/prowlarr/api/v1/indexer
```

### Prowlarr Download Proxy

**Endpoint:** `/api/prowlarr/download/[hash]`

Retrieves cached download URLs securely without exposing API keys.

- Handles magnet links
- Handles torrent file downloads
- 1-hour cache TTL

**Example:**

```bash
# Download from cached hash (hash is provided in search results)
curl http://localhost:3000/api/prowlarr/download/abc123def456
```

## Environment Variables

Required environment variables in `.env`:

```bash
# TMDB Configuration
API_TMDB_BASEURL=https://api.themoviedb.org/3
API_TMDB_READ_ACCESS_TOKEN=your_tmdb_read_access_token_here
API_TMDB_IMAGES_BASEURL=https://image.tmdb.org/t/p

# Prowlarr Configuration
API_PROWLARR_BASEURL=http://localhost:9696
API_PROWLARR_API_KEY=your_prowlarr_api_key_here

# Domain name for URL rewriting (used in Prowlarr download links)
DOMAIN_NAME=http://localhost:3000
```

## Features

### TMDB Proxy Features:

- ✅ Bearer token authentication (automatic)
- ✅ 30-minute response caching
- ✅ LRU cache with 1000 entry limit
- ✅ All HTTP methods supported
- ✅ Query parameter forwarding

### Prowlarr Proxy Features:

- ✅ API key authentication (automatic)
- ✅ 30-minute search result caching
- ✅ URL rewriting for secure downloads
- ✅ Download URL caching (1 hour TTL)
- ✅ Magnet link support
- ✅ Torrent file download support
- ✅ All HTTP methods supported

## Cache Management

Both proxies use in-memory caching:

- **TMDB Cache:** 30 minutes TTL, max 1000 entries (LRU)
- **Prowlarr Search Cache:** 30 minutes TTL
- **Prowlarr Download Cache:** 1 hour TTL

Caches are automatically cleaned up based on TTL and LRU policies.

## Migration from Python Proxy

If you were previously using the Python FastAPI proxy server, you can now:

1. Remove the Python proxy server dependency
2. Update your frontend code to call the local API routes directly
3. The SolidStart dev server will handle everything on port 3000

**Before:**

```typescript
fetch("http://localhost:8000/tmdb/movie/550");
```

**After:**

```typescript
fetch("/api/tmdb/movie/550");
```

## Benefits

1. **Simplified Architecture:** No need for separate Python proxy server
2. **Single Port:** Everything runs on port 3000
3. **Better Integration:** Direct access to SolidStart features and session management
4. **TypeScript:** Type-safe proxy code
5. **Easier Deployment:** One application to deploy instead of two

## Development

Run the SolidStart server:

```bash
cd nacho-server
npm run dev
```

Access APIs at:

- TMDB: `http://localhost:3000/api/tmdb/*`
- Prowlarr: `http://localhost:3000/api/prowlarr/*`
- Your app: `http://localhost:3000/`
