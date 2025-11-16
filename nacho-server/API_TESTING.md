# Nacho Time API Test Scripts

## Quick Test Commands

### 1. First, get your API token

Go to http://localhost:3000/tokens and create a new token. Copy it for use below.

### 2. Add a TV Show Episode

Add Breaking Bad S01E01:

```bash
curl -X POST http://localhost:3000/api/history \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "episodes": [
      {
        "imdbID": "tt0959621",
        "season": 1,
        "episode": 1
      }
    ]
  }'
```

### 3. Add Multiple Episodes

Add The Office S01E01-03:

```bash
curl -X POST http://localhost:3000/api/history \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "episodes": [
      {
        "imdbID": "tt0386676",
        "season": 1,
        "episode": 1
      },
      {
        "imdbID": "tt0386676",
        "season": 1,
        "episode": 2
      },
      {
        "imdbID": "tt0386676",
        "season": 1,
        "episode": 3
      }
    ]
  }'
```

### 4. Add a Movie

Add The Matrix:

```bash
curl -X POST http://localhost:3000/api/history \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "movies": [
      {
        "imdbID": "tt0133093"
      }
    ]
  }'
```

### 5. Add Both Movies and Episodes

```bash
curl -X POST http://localhost:3000/api/history \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "movies": [
      {
        "imdbID": "tt0468569"
      }
    ],
    "episodes": [
      {
        "imdbID": "tt0944947",
        "season": 1,
        "episode": 1
      }
    ]
  }'
```

### 6. Add with Custom Timestamps

```bash
curl -X POST http://localhost:3000/api/history \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "episodes": [
      {
        "imdbID": "tt0959621",
        "season": 2,
        "episode": 5,
        "timestampWatched": "2024-11-15T20:30:00Z",
        "timestampAdded": "2024-11-15T20:30:00Z"
      }
    ]
  }'
```

### 7. Get Watch History

Get all history:

```bash
curl -X GET http://localhost:3000/api/history \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

Get last 10 items:

```bash
curl -X GET "http://localhost:3000/api/history?limit=10" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

Get items since a specific date:

```bash
curl -X GET "http://localhost:3000/api/history?since=2024-11-01T00:00:00Z" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

Get last 5 items since yesterday:

```bash
curl -X GET "http://localhost:3000/api/history?limit=5&since=2024-11-15T00:00:00Z" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## Using the Test Script

Run all tests at once:

```bash
./test-api.sh YOUR_TOKEN_HERE
```

This will:

1. Add a Breaking Bad episode
2. Add multiple The Office episodes
3. Add The Matrix movie
4. Add both movies and episodes together
5. Retrieve your watch history

## Common IMDB IDs for Testing

### TV Shows:

- Breaking Bad: `tt0959621`
- The Office (US): `tt0386676`
- Game of Thrones: `tt0944947`
- Stranger Things: `tt4574334`
- The Mandalorian: `tt8111088`
- Friends: `tt0108778`

### Movies:

- The Matrix: `tt0133093`
- The Dark Knight: `tt0468569`
- Inception: `tt1375666`
- Interstellar: `tt0816692`
- The Shawshank Redemption: `tt0111161`
- Pulp Fiction: `tt0110912`

## Response Format

### Success Response (POST):

```json
{
  "success": true,
  "results": {
    "moviesAdded": 1,
    "episodesAdded": 3,
    "errors": []
  }
}
```

### Success Response (GET):

```json
{
  "success": true,
  "data": {
    "movies": [...],
    "episodes": [...]
  },
  "count": {
    "movies": 5,
    "episodes": 10
  }
}
```

### Error Response:

```json
{
  "error": "Invalid or expired token"
}
```
