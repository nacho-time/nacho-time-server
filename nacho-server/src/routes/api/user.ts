import { APIEvent } from "@solidjs/start/server";
import { authenticateRequest, isErrorResponse } from "~/lib/api-auth";

// GET /api/user - Get current user information
export async function GET(event: APIEvent) {
  const userOrError = await authenticateRequest(event);

  if (isErrorResponse(userOrError)) {
    return userOrError; // Return error response
  }

  const user = userOrError;

  return new Response(
    JSON.stringify({
      success: true,
      user: {
        id: user.id,
        username: user.username,
        isAdmin: user.isAdmin,
      },
    }),
    {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }
  );
}
