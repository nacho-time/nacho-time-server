import { APIEvent } from "@solidjs/start/server";
import { db } from "~/lib/db";
import { authenticateRequest, isErrorResponse } from "~/lib/api-auth";

// GET /api/history - Retrieve watch history
export async function GET(event: APIEvent) {
  const userOrError = await authenticateRequest(event);

  if (isErrorResponse(userOrError)) {
    return userOrError; // Return error response
  }

  const user = userOrError;
  const url = new URL(event.request.url);

  // Parse query parameters
  const limitNum = url.searchParams.get("limit");
  const sinceTime = url.searchParams.get("since");

  try {
    let movieQuery: any = {
      where: { userId: user.id },
      orderBy: { timestampAdded: "desc" },
    };
    let episodeQuery: any = {
      where: { userId: user.id },
      orderBy: { timestampAdded: "desc" },
    };

    // Apply time-based filter if 'since' parameter exists
    if (sinceTime) {
      const sinceDate = new Date(sinceTime);
      if (isNaN(sinceDate.getTime())) {
        return new Response(
          JSON.stringify({
            error: "Invalid 'since' parameter. Use ISO 8601 format.",
          }),
          {
            status: 400,
            headers: { "Content-Type": "application/json" },
          }
        );
      }
      movieQuery.where.timestampAdded = { gte: sinceDate };
      episodeQuery.where.timestampAdded = { gte: sinceDate };
    }

    // Apply numerical limit if provided
    if (limitNum) {
      const limit = parseInt(limitNum, 10);
      if (isNaN(limit) || limit <= 0) {
        return new Response(
          JSON.stringify({
            error: "Invalid 'limit' parameter. Must be a positive number.",
          }),
          {
            status: 400,
            headers: { "Content-Type": "application/json" },
          }
        );
      }
      movieQuery.take = limit;
      episodeQuery.take = limit;
    }

    // Fetch data
    const [movies, episodes] = await Promise.all([
      db.movieEntry.findMany(movieQuery),
      db.episodeEntry.findMany(episodeQuery),
    ]);

    return new Response(
      JSON.stringify({
        success: true,
        data: {
          movies,
          episodes,
        },
        count: {
          movies: movies.length,
          episodes: episodes.length,
        },
      }),
      {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }
    );
  } catch (error) {
    console.error("Error fetching history:", error);
    return new Response(JSON.stringify({ error: "Internal server error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}

// POST /api/history - Add watch history entries
export async function POST(event: APIEvent) {
  const userOrError = await authenticateRequest(event);

  if (isErrorResponse(userOrError)) {
    return userOrError; // Return error response
  }

  const user = userOrError;

  try {
    const body = await event.request.json();

    if (!body || typeof body !== "object") {
      return new Response(JSON.stringify({ error: "Invalid request body" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    const { movies, episodes } = body;

    // Validate input arrays
    if (movies && !Array.isArray(movies)) {
      return new Response(
        JSON.stringify({ error: "'movies' must be an array" }),
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    if (episodes && !Array.isArray(episodes)) {
      return new Response(
        JSON.stringify({ error: "'episodes' must be an array" }),
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    const results = {
      moviesAdded: 0,
      episodesAdded: 0,
      errors: [] as string[],
    };

    // Add movie entries
    if (movies && movies.length > 0) {
      for (const movie of movies) {
        try {
          if (!movie.tmdbID) {
            results.errors.push("Movie entry missing tmdbID");
            continue;
          }

          const timestampWatched = movie.timestampWatched
            ? new Date(movie.timestampWatched)
            : new Date();

          const timestampAdded = movie.timestampAdded
            ? new Date(movie.timestampAdded)
            : new Date();

          if (isNaN(timestampWatched.getTime())) {
            results.errors.push(
              `Invalid timestampWatched for movie ${movie.tmdbID}`
            );
            continue;
          }

          if (isNaN(timestampAdded.getTime())) {
            results.errors.push(
              `Invalid timestampAdded for movie ${movie.tmdbID}`
            );
            continue;
          }

          await db.movieEntry.create({
            data: {
              userId: user.id,
              tmdbID: String(movie.tmdbID),
              timestampWatched,
              timestampAdded,
            },
          });

          results.moviesAdded++;
        } catch (error) {
          results.errors.push(`Error adding movie ${movie.tmdbID}: ${error}`);
        }
      }
    }

    // Add episode entries
    if (episodes && episodes.length > 0) {
      for (const episode of episodes) {
        try {
          if (
            !episode.tmdbID ||
            episode.season === undefined ||
            episode.episode === undefined
          ) {
            results.errors.push(
              "Episode entry missing required fields (tmdbID, season, episode)"
            );
            continue;
          }

          const timestampWatched = episode.timestampWatched
            ? new Date(episode.timestampWatched)
            : new Date();

          const timestampAdded = episode.timestampAdded
            ? new Date(episode.timestampAdded)
            : new Date();

          if (isNaN(timestampWatched.getTime())) {
            results.errors.push(
              `Invalid timestampWatched for episode ${episode.tmdbID} S${episode.season}E${episode.episode}`
            );
            continue;
          }

          if (isNaN(timestampAdded.getTime())) {
            results.errors.push(
              `Invalid timestampAdded for episode ${episode.tmdbID} S${episode.season}E${episode.episode}`
            );
            continue;
          }

          await db.episodeEntry.create({
            data: {
              userId: user.id,
              tmdbID: String(episode.tmdbID),
              season: episode.season,
              episode: episode.episode,
              timestampWatched,
              timestampAdded,
            },
          });

          results.episodesAdded++;
        } catch (error) {
          results.errors.push(
            `Error adding episode ${episode.tmdbID} S${episode.season}E${episode.episode}: ${error}`
          );
        }
      }
    }

    return new Response(
      JSON.stringify({
        success: true,
        results,
      }),
      {
        status: 201,
        headers: { "Content-Type": "application/json" },
      }
    );
  } catch (error) {
    console.error("Error adding history:", error);
    return new Response(JSON.stringify({ error: "Internal server error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
