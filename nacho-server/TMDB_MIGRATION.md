# TMDB ID Migration Summary

## Overview

The project has been converted from using IMDB IDs to TMDB IDs for tracking watch history.

## Database Schema Changes

### Modified Models:

1. **MovieEntry** - `imdbID` → `tmdbID`
2. **EpisodeEntry** - `imdbID` → `tmdbID`
3. **ShowObject** - `imdbID` → `tmdbID`

## Code Changes

### 1. Database Schema (`prisma/schema.prisma`)

- Updated field names from `imdbID` to `tmdbID` in MovieEntry, EpisodeEntry, and ShowObject models

### 2. API Endpoint (`src/routes/api/history.ts`)

- Updated POST endpoint to accept `tmdbID` instead of `imdbID` in request body
- Updated validation error messages to reference `tmdbID`
- Database operations now use `tmdbID` field

### 3. Server Functions (`src/lib/server.ts`)

- **Removed**: `getTmdbIdFromImdb()` function (no longer needed)
- **Modified**: `fetchMediaInfo()` function:
  - Now takes `tmdbID` and `type` ("movie" | "tv") as parameters
  - Directly queries TMDB API with TMDB ID (no IMDB lookup needed)
  - Returns normalized data with `tmdbID` field instead of `imdbID`
  - Cache key now includes type prefix: `${type}:${tmdbID}`

### 4. Client Actions (`src/lib/index.ts`)

- Updated `getWatchHistory()` to work with `tmdbID`
- Modified media info fetching to determine movie vs TV show type
- Updated map keys to use `tmdbID`

### 5. Dashboard UI (`src/routes/dashboard.tsx`)

- Updated `getDisplayTitle()` fallback to use `item.tmdbID`
- Updated unique titles calculation to use `tmdbID`

## Database Migration

A migration file has been created at:
`prisma/migrations/rename_imdb_to_tmdb/migration.sql`

### To apply the migration:

```bash
cd nacho-server
npx prisma migrate deploy
# Or for development:
npx prisma migrate dev
```

### After migration, regenerate Prisma client:

```bash
npx prisma generate
```

## API Changes

### POST /api/history Request Body

**Before:**

```json
{
  "movies": [
    {
      "imdbID": "tt0111161",
      "timestampWatched": "2025-11-15T20:30:00Z"
    }
  ],
  "episodes": [
    {
      "imdbID": "tt0944947",
      "season": 1,
      "episode": 1
    }
  ]
}
```

**After:**

```json
{
  "movies": [
    {
      "tmdbID": "278",
      "timestampWatched": "2025-11-15T20:30:00Z"
    }
  ],
  "episodes": [
    {
      "tmdbID": "1399",
      "season": 1,
      "episode": 1
    }
  ]
}
```

### GET /api/history Response

**Before:**

```json
{
  "success": true,
  "data": {
    "movies": [
      {
        "id": "...",
        "userId": "...",
        "imdbID": "tt0111161",
        "timestampWatched": "...",
        "timestampAdded": "..."
      }
    ]
  }
}
```

**After:**

```json
{
  "success": true,
  "data": {
    "movies": [
      {
        "id": "...",
        "userId": "...",
        "tmdbID": "278",
        "timestampWatched": "...",
        "timestampAdded": "..."
      }
    ]
  }
}
```

## Important Notes

### TMDB ID Format

- **Movies**: Numeric ID (e.g., "278" for The Shawshank Redemption)
- **TV Shows**: Numeric ID (e.g., "1399" for Game of Thrones)
- Unlike IMDB IDs (tt0111161), TMDB IDs are just numbers

### Type Detection

- The `fetchMediaInfo()` function now requires a `type` parameter
- Movies and TV shows are distinguished at the data storage level
- The frontend determines type based on whether the ID appears in movies or episodes array

### Breaking Changes

- All clients must update to send `tmdbID` instead of `imdbID`
- Existing data will be migrated (the migration copies `imdbID` values to `tmdbID`)
- **Important**: Existing data contains IMDB IDs in the database, so you may need to:
  - Clear existing watch history, OR
  - Write a data migration script to convert IMDB IDs to TMDB IDs using TMDB's Find API

## Data Migration Script (Optional)

If you need to convert existing IMDB IDs to TMDB IDs in the database:

```typescript
// Example migration script (not included)
async function migrateImdbToTmdb() {
  const movies = await db.movieEntry.findMany();
  for (const movie of movies) {
    // movie.tmdbID currently contains an IMDB ID
    const tmdbId = await convertImdbToTmdb(movie.tmdbID);
    if (tmdbId) {
      await db.movieEntry.update({
        where: { id: movie.id },
        data: { tmdbID: tmdbId.toString() },
      });
    }
  }
  // Similar for episodes...
}
```

## Testing

Update test files to use TMDB IDs:

- `test-api.sh` - Update all test cases
- `API_TESTING.md` - Update documentation examples
- `prisma/seed.ts` - Update seed data

## Rollback

If you need to roll back:

1. Revert schema changes in `prisma/schema.prisma`
2. Create a reverse migration renaming `tmdbID` back to `imdbID`
3. Revert all code changes
4. Run `npx prisma generate`
