-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;

-- Rename imdbID to tmdbID in MovieEntry
CREATE TABLE "new_MovieEntry" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "userId" TEXT NOT NULL,
    "tmdbID" TEXT NOT NULL,
    "timestampWatched" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "timestampAdded" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "MovieEntry_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

INSERT INTO "new_MovieEntry" ("id", "userId", "tmdbID", "timestampWatched", "timestampAdded")
SELECT "id", "userId", "imdbID", "timestampWatched", "timestampAdded" FROM "MovieEntry";

DROP TABLE "MovieEntry";
ALTER TABLE "new_MovieEntry" RENAME TO "MovieEntry";

-- Rename imdbID to tmdbID in EpisodeEntry
CREATE TABLE "new_EpisodeEntry" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "userId" TEXT NOT NULL,
    "tmdbID" TEXT NOT NULL,
    "season" INTEGER NOT NULL,
    "episode" INTEGER NOT NULL,
    "timestampWatched" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "timestampAdded" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "EpisodeEntry_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

INSERT INTO "new_EpisodeEntry" ("id", "userId", "tmdbID", "season", "episode", "timestampWatched", "timestampAdded")
SELECT "id", "userId", "imdbID", "season", "episode", "timestampWatched", "timestampAdded" FROM "EpisodeEntry";

DROP TABLE "EpisodeEntry";
ALTER TABLE "new_EpisodeEntry" RENAME TO "EpisodeEntry";

-- Rename imdbID to tmdbID in ShowObject
CREATE TABLE "new_ShowObject" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "tmdbID" TEXT NOT NULL
);

INSERT INTO "new_ShowObject" ("id", "tmdbID")
SELECT "id", "imdbID" FROM "ShowObject";

DROP TABLE "ShowObject";
ALTER TABLE "new_ShowObject" RENAME TO "ShowObject";

PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;
