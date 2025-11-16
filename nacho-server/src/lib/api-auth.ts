import { APIEvent } from "@solidjs/start/server";
import { validateAuthToken } from "~/lib/server";

/**
 * Authentication middleware for API endpoints
 * Validates the X-Nacho-Auth header and returns the authenticated user
 *
 * @param event - The API event object
 * @returns The authenticated user object or an error Response
 */
export async function authenticateRequest(event: APIEvent) {
  const authHeader = event.request.headers.get("X-Nacho-Auth");

  console.log("Authenticating request with X-Nacho-Auth header:", authHeader);

  if (!authHeader) {
    return new Response(
      JSON.stringify({ error: "Missing X-Nacho-Auth header" }),
      {
        status: 401,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  const token = authHeader.trim();
  const user = await validateAuthToken(token);

  if (!user) {
    return new Response(JSON.stringify({ error: "Invalid or expired token" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  return user;
}

/**
 * Type guard to check if the result is an error Response
 */
export function isErrorResponse(value: any): value is Response {
  return value instanceof Response;
}
