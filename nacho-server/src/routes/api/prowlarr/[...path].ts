import { APIEvent } from "@solidjs/start/server";
import crypto from "crypto";
import { authenticateRequest, isErrorResponse } from "~/lib/api-auth";

const PROWLARR_BASE =
  process.env.API_PROWLARR_BASEURL || "http://localhost:9696";
const PROWLARR_API_KEY = process.env.API_PROWLARR_API_KEY;
const DOMAIN_NAME = process.env.DOMAIN_NAME || "http://localhost:8123";

// In-memory cache for search results (30 minute TTL)
const searchCache = new Map<
  string,
  { data: any; timestamp: number; headers: Record<string, string> }
>();
const SEARCH_CACHE_TTL = 30 * 60 * 1000; // 30 minutes

// In-memory cache for download URLs (1 hour TTL)
const downloadCache = new Map<string, { url: string; timestamp: number }>();
const DOWNLOAD_CACHE_TTL = 60 * 60 * 1000; // 1 hour

function getCacheKey(url: string, params: URLSearchParams): string {
  const sortedParams = Array.from(params.entries())
    .filter(([k]) => k.toLowerCase() !== "apikey")
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${k}=${v}`)
    .join("&");
  return `${url}${sortedParams ? "?" + sortedParams : ""}`;
}

function cacheDownloadUrl(url: string): string {
  const hash = crypto
    .createHash("sha256")
    .update(url)
    .digest("hex")
    .slice(0, 16);
  console.log(`[Prowlarr] Caching download URL with hash: ${hash}`);
  console.log(`[Prowlarr] Full URL being cached: ${url}`);
  downloadCache.set(hash, { url, timestamp: Date.now() });
  return hash;
}

function getCachedDownloadUrl(hash: string): string | null {
  console.log(`[Prowlarr] Retrieving cached download URL for hash: ${hash}`);
  const cached = downloadCache.get(hash);
  if (!cached) {
    console.log(`[Prowlarr] No cached download URL found for hash: ${hash}`);
    return null;
  }

  const age = Date.now() - cached.timestamp;
  if (age > DOWNLOAD_CACHE_TTL) {
    console.log(`[Prowlarr] Cached download URL expired for hash: ${hash}`);
    downloadCache.delete(hash);
    return null;
  }

  console.log(`[Prowlarr] Found cached URL for hash: ${hash}`);
  return cached.url;
}

function rewriteProwlarrUrls(data: any): any {
  const URL_FIELDS = new Set(["guid", "downloadUrl", "magnetUrl"]);

  if (Array.isArray(data)) {
    return data.map((item) => rewriteProwlarrUrls(item));
  }

  if (data && typeof data === "object") {
    const result: any = {};
    for (const [key, value] of Object.entries(data)) {
      if (URL_FIELDS.has(key) && typeof value === "string") {
        // Don't rewrite magnet links
        if (value.startsWith("magnet:")) {
          result[key] = value;
        }
        // Rewrite Prowlarr URLs to use secure cached endpoint
        else if (value.includes(PROWLARR_BASE)) {
          const hash = cacheDownloadUrl(value);
          result[key] = `${DOMAIN_NAME}/api/prowlarr/download/${hash}`;
        } else {
          result[key] = value;
        }
      } else {
        result[key] = rewriteProwlarrUrls(value);
      }
    }
    return result;
  }

  return data;
}

async function proxyToProwlarr(event: APIEvent, path: string) {
  const method = event.request.method;
  const url = new URL(event.request.url);
  const params = new URLSearchParams(url.search);

  // Handle special download endpoint within this route
  if (path.startsWith("download/")) {
    const hash = path.replace("download/", "");
    console.log(`[Prowlarr] Handling download request for hash: ${hash}`);

    if (!hash) {
      return new Response(
        JSON.stringify({ error: "Hash parameter required" }),
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    const cachedUrl = getCachedDownloadUrl(hash);
    console.log(`[Prowlarr] Cached URL for hash ${hash}: ${cachedUrl}`);

    if (!cachedUrl) {
      return new Response(
        JSON.stringify({ error: "Download link not found or expired" }),
        {
          status: 404,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    try {
      // First check if it redirects to a magnet link
      const initialResponse = await fetch(cachedUrl, {
        redirect: "manual",
      });

      if ([301, 302, 303, 307, 308].includes(initialResponse.status)) {
        const location = initialResponse.headers.get("location");
        if (location && location.startsWith("magnet:")) {
          console.log("[Prowlarr] Returning magnet link directly");
          return new Response(location, {
            status: 200,
            headers: { "Content-Type": "text/plain" },
          });
        }
      }

      // Otherwise follow redirects and fetch the content
      const response = await fetch(cachedUrl, {
        redirect: "follow",
      });

      const contentType = response.headers.get("content-type") || "";
      const arrayBuffer = await response.arrayBuffer();
      const content = new Uint8Array(arrayBuffer);

      // Check if it's a magnet link
      const textDecoder = new TextDecoder();
      const text = textDecoder.decode(content.slice(0, 7));
      if (text === "magnet:") {
        return new Response(content, {
          status: 200,
          headers: { "Content-Type": "text/plain" },
        });
      }

      // Check if it's a torrent file
      if (
        contentType.startsWith("application/x-bittorrent") ||
        contentType.startsWith("application/octet-stream")
      ) {
        return new Response(content, {
          status: 200,
          headers: {
            "Content-Type": contentType,
            "Content-Disposition": "attachment",
          },
        });
      }

      // Return whatever we got
      return new Response(content, {
        status: response.status,
        headers: {
          "Content-Type": contentType || "application/octet-stream",
        },
      });
    } catch (error: any) {
      console.error("[Prowlarr] Failed to fetch download:", error);
      return new Response(
        JSON.stringify({
          error: "Failed to fetch download link",
          message: error.message,
        }),
        {
          status: 500,
          headers: { "Content-Type": "application/json" },
        }
      );
    }
  }

  // Build upstream URL for regular Prowlarr requests
  const upstreamUrl = `${PROWLARR_BASE}/${path}${
    params.toString() ? "?" + params.toString() : ""
  }`;

  // Check if this is a search request that should be cached
  const isSearch =
    method === "GET" &&
    (path.includes("search") || path.includes("api/v1/search"));

  if (isSearch) {
    const cacheKey = getCacheKey(upstreamUrl, params);
    const cached = searchCache.get(cacheKey);
    if (cached) {
      const age = Date.now() - cached.timestamp;
      if (age <= SEARCH_CACHE_TTL) {
        console.log(`Prowlarr cache HIT: ${cacheKey}`);
        return new Response(JSON.stringify(cached.data), {
          status: 200,
          headers: cached.headers,
        });
      } else {
        searchCache.delete(cacheKey);
      }
    }
    console.log(`Prowlarr cache MISS: ${cacheKey}`);
  }

  // Build headers
  const headers: Record<string, string> = {
    Accept: "application/json",
  };

  if (PROWLARR_API_KEY) {
    headers["X-Api-Key"] = PROWLARR_API_KEY;
  }

  // Get request body for non-GET requests
  let body: string | undefined;
  if (method !== "GET" && method !== "HEAD") {
    body = await event.request.text();
    if (body) {
      headers["Content-Type"] = "application/json";
    }
  }

  console.log(`Proxying to Prowlarr: ${method} ${upstreamUrl}`);

  // Make request to Prowlarr
  const response = await fetch(upstreamUrl, {
    method,
    headers,
    body,
  });

  const contentType = response.headers.get("content-type") || "";
  let data: any;

  if (contentType.includes("application/json")) {
    data = await response.json();

    // Rewrite URLs in search responses
    if (isSearch && response.ok) {
      data = rewriteProwlarrUrls(data);
    }

    // Cache successful search responses
    if (isSearch && response.ok) {
      const cacheKey = getCacheKey(upstreamUrl, params);
      searchCache.set(cacheKey, {
        data,
        timestamp: Date.now(),
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response(JSON.stringify(data), {
      status: response.status,
      headers: { "Content-Type": "application/json" },
    });
  } else {
    // Return non-JSON responses as-is
    const arrayBuffer = await response.arrayBuffer();
    return new Response(arrayBuffer, {
      status: response.status,
      headers: {
        "Content-Type": contentType || "application/octet-stream",
      },
    });
  }
}

export async function GET(event: APIEvent) {
  const userOrError = await authenticateRequest(event);
  if (isErrorResponse(userOrError)) {
    return userOrError;
  }

  const path = event.params.path || "";
  return proxyToProwlarr(event, path);
}

export async function POST(event: APIEvent) {
  const userOrError = await authenticateRequest(event);
  if (isErrorResponse(userOrError)) {
    return userOrError;
  }

  const path = event.params.path || "";
  return proxyToProwlarr(event, path);
}

export async function PUT(event: APIEvent) {
  const userOrError = await authenticateRequest(event);
  if (isErrorResponse(userOrError)) {
    return userOrError;
  }

  const path = event.params.path || "";
  return proxyToProwlarr(event, path);
}

export async function DELETE(event: APIEvent) {
  const userOrError = await authenticateRequest(event);
  if (isErrorResponse(userOrError)) {
    return userOrError;
  }

  const path = event.params.path || "";
  return proxyToProwlarr(event, path);
}

export async function PATCH(event: APIEvent) {
  const userOrError = await authenticateRequest(event);
  if (isErrorResponse(userOrError)) {
    return userOrError;
  }

  const path = event.params.path || "";
  return proxyToProwlarr(event, path);
}
