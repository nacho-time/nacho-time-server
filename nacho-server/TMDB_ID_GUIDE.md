# Using TMDB IDs - Quick Reference

## What Changed?

The API now uses TMDB IDs instead of IMDB IDs for all watch history operations.

## Finding TMDB IDs

### From TMDB Website

1. Go to https://www.themoviedb.org/
2. Search for your movie or TV show
3. The ID is in the URL:
   - Movie: `https://www.themoviedb.org/movie/278-the-shawshank-redemption` → ID is `278`
   - TV Show: `https://www.themoviedb.org/tv/1399-game-of-thrones` → ID is `1399`

### From TMDB API

```bash
# Search for a movie
curl "https://api.themoviedb.org/3/search/movie?api_key=YOUR_KEY&query=Shawshank"

# Search for a TV show
curl "https://api.themoviedb.org/3/search/tv?api_key=YOUR_KEY&query=Game+of+Thrones"
```

### Convert IMDB ID to TMDB ID

```bash
# Using TMDB Find API
curl "https://api.themoviedb.org/3/find/tt0111161?api_key=YOUR_KEY&external_source=imdb_id"
```

## API Usage Examples

### Add a Movie to History

```bash
curl -X POST \
  -H "X-Nacho-Auth: YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "movies": [
      {
        "tmdbID": "278",
        "timestampWatched": "2025-11-15T20:30:00Z"
      }
    ]
  }' \
  "http://localhost:3000/api/history"
```

### Add a TV Episode to History

```bash
curl -X POST \
  -H "X-Nacho-Auth: YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "episodes": [
      {
        "tmdbID": "1399",
        "season": 1,
        "episode": 1,
        "timestampWatched": "2025-11-14T19:00:00Z"
      }
    ]
  }' \
  "http://localhost:3000/api/history"
```

### Popular Movies & TV Shows - TMDB IDs

#### Movies

- The Shawshank Redemption: `278`
- The Godfather: `238`
- The Dark Knight: `155`
- Pulp Fiction: `680`
- Forrest Gump: `13`
- Inception: `27205`
- Fight Club: `550`
- The Matrix: `603`
- Interstellar: `157336`

#### TV Shows

- Game of Thrones: `1399`
- Breaking Bad: `1396`
- The Office (US): `2316`
- Friends: `1668`
- Stranger Things: `66732`
- The Crown: `73375`
- The Mandalorian: `82856`
- House of the Dragon: `94997`
- The Last of Us: `100088`

## Migration Steps

### 1. Run Database Migration

```bash
cd nacho-server
npx prisma migrate deploy  # Production
# OR
npx prisma migrate dev     # Development
```

### 2. Regenerate Prisma Client

```bash
npx prisma generate
```

### 3. (Optional) Clear Existing Data

If your existing data contains IMDB IDs, you may want to clear it:

```bash
# Delete all watch history (irreversible!)
npx prisma studio
# Then manually delete entries from MovieEntry and EpisodeEntry tables
```

### 4. Update Your Clients

- Change all `imdbID` references to `tmdbID`
- Use TMDB IDs (numeric) instead of IMDB IDs (tt followed by numbers)
- Update any hardcoded test data

## Error Messages

### Before

```json
{
  "error": "Movie entry missing imdbID"
}
```

### After

```json
{
  "error": "Movie entry missing tmdbID"
}
```

## Benefits of TMDB IDs

1. **Direct API Access**: No need to convert IDs before querying TMDB API
2. **Better Performance**: One less API call per unique title
3. **Consistency**: Using TMDB throughout the stack
4. **Simpler Code**: Removed ID conversion logic
5. **More Reliable**: TMDB's own IDs are more stable than IMDB lookups
