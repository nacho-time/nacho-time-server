import { APIEvent } from "@solidjs/start/server";
import { authenticateRequest, isErrorResponse } from "~/lib/api-auth";

const TMDB_BASE =
  process.env.API_TMDB_BASEURL || "https://api.themoviedb.org/3";
const TMDB_READ_ACCESS_TOKEN = process.env.API_TMDB_READ_ACCESS_TOKEN;

// Simple in-memory cache for TMDB requests (30 minute TTL)
const cache = new Map<string, { data: any; timestamp: number }>();
const CACHE_TTL = 30 * 60 * 1000; // 30 minutes

function getCacheKey(path: string, params: URLSearchParams): string {
  const sortedParams = Array.from(params.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${k}=${v}`)
    .join("&");
  return `${path}${sortedParams ? "?" + sortedParams : ""}`;
}

function getFromCache(key: string): any | null {
  const cached = cache.get(key);
  if (!cached) return null;

  const age = Date.now() - cached.timestamp;
  if (age > CACHE_TTL) {
    cache.delete(key);
    return null;
  }

  return cached.data;
}

function setCache(key: string, data: any): void {
  cache.set(key, { data, timestamp: Date.now() });

  // Clean up old entries (simple LRU - keep last 1000 entries)
  if (cache.size > 1000) {
    const entries = Array.from(cache.entries());
    entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
    const toDelete = entries.slice(0, cache.size - 1000);
    toDelete.forEach(([key]) => cache.delete(key));
  }
}

async function proxyToTMDB(event: APIEvent, path: string) {
  const method = event.request.method;
  const url = new URL(event.request.url);
  const params = new URLSearchParams(url.search);

  // Build upstream URL
  const upstreamUrl = `${TMDB_BASE}/${path}${
    params.toString() ? "?" + params.toString() : ""
  }`;

  // Check cache for GET requests
  if (method === "GET") {
    const cacheKey = getCacheKey(path, params);
    const cached = getFromCache(cacheKey);
    if (cached) {
      console.log(`TMDB cache HIT: ${cacheKey}`);
      return new Response(JSON.stringify(cached), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    console.log(`TMDB cache MISS: ${cacheKey}`);
  }

  // Build headers
  const headers: Record<string, string> = {
    Accept: "application/json",
  };

  if (TMDB_READ_ACCESS_TOKEN) {
    headers["Authorization"] = `Bearer ${TMDB_READ_ACCESS_TOKEN}`;
  }

  // Get request body for non-GET requests
  let body: string | undefined;
  if (method !== "GET" && method !== "HEAD") {
    body = await event.request.text();
    if (body) {
      headers["Content-Type"] = "application/json";
    }
  }

  console.log(`Proxying to TMDB: ${method} ${upstreamUrl}`);

  // Make request to TMDB
  const response = await fetch(upstreamUrl, {
    method,
    headers,
    body,
  });

  const data = await response.json();

  // Cache successful GET responses
  if (method === "GET" && response.ok) {
    const cacheKey = getCacheKey(path, params);
    setCache(cacheKey, data);
  }

  // Return response
  return new Response(JSON.stringify(data), {
    status: response.status,
    headers: {
      "Content-Type": "application/json",
    },
  });
}

export async function GET(event: APIEvent) {
  const userOrError = await authenticateRequest(event);
  if (isErrorResponse(userOrError)) {
    return userOrError;
  }

  const path = event.params.path || "";
  console.log("TMDB GET path:", path);
  return proxyToTMDB(event, path);
}

export async function POST(event: APIEvent) {
  const userOrError = await authenticateRequest(event);
  if (isErrorResponse(userOrError)) {
    return userOrError;
  }

  const path = event.params.path || "";
  return proxyToTMDB(event, path);
}

export async function PUT(event: APIEvent) {
  const userOrError = await authenticateRequest(event);
  if (isErrorResponse(userOrError)) {
    return userOrError;
  }

  const path = event.params.path || "";
  return proxyToTMDB(event, path);
}

export async function DELETE(event: APIEvent) {
  const userOrError = await authenticateRequest(event);
  if (isErrorResponse(userOrError)) {
    return userOrError;
  }

  const path = event.params.path || "";
  return proxyToTMDB(event, path);
}

export async function PATCH(event: APIEvent) {
  const userOrError = await authenticateRequest(event);
  if (isErrorResponse(userOrError)) {
    return userOrError;
  }

  const path = event.params.path || "";
  return proxyToTMDB(event, path);
}
