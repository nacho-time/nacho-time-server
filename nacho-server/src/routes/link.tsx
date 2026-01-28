import { createSignal, Show, createEffect } from "solid-js";
import { useSubmission, createAsync, redirect } from "@solidjs/router";
import { A, useSearchParams } from "@solidjs/router";
import { getUser } from "~/lib";
import { action } from "@solidjs/router";
import { createAuthToken, getSession } from "~/lib/server";

const linkDevice = action(async () => {
  "use server";
  const session = await getSession();
  const userId = session.data.userId;

  if (userId === undefined) {
    throw redirect("/login?redirect=/link");
  }

  // Create a token with 1 year expiration (365 days)
  const token = await createAuthToken(userId, 365);

  return {
    success: true,
    token: token.token,
    expiresAt: token.expiresAt,
  };
});

export const route = {
  preload: () => getUser(),
};

export default function Link() {
  const user = createAsync(async () => {
    const userData = await getUser();
    if (!userData) {
      throw redirect("/login?redirect=/link");
    }
    return userData;
  });

  const [searchParams] = useSearchParams();
  const linking = useSubmission(linkDevice);
  const [showToken, setShowToken] = createSignal(false);
  const [tokenValue, setTokenValue] = createSignal("");
  const [copiedToken, setCopiedToken] = createSignal(false);

  // Effect to handle token generation result
  createEffect(() => {
    const result = linking.result;
    if (result && "success" in result && result.success && result.token) {
      if (result.token !== tokenValue()) {
        setShowToken(true);
        setTokenValue(result.token);

        // Open deeplink after token is created
        const serverUrl = window.location.origin;
        const encodedServer = encodeURIComponent(serverUrl);
        const deeplink = `nacho-time://auth?token=${result.token}&server=${encodedServer}`;
        window.location.href = deeplink;
      }
    }
  });

  const linkedToken = () => {
    const result = linking.result;
    if (result && "success" in result && result.success && result.token) {
      return result;
    }
    return null;
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedToken(true);
    setTimeout(() => setCopiedToken(false), 2000);
  };

  const formatDate = (date: string | Date) => {
    return new Date(date).toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  return (
    <main class="min-h-screen bg-gradient-to-br from-indigo-100 via-purple-50 to-pink-100 px-4 py-12">
      <div class="max-w-2xl mx-auto">
        {/* Header */}
        <div class="bg-white rounded-2xl shadow-2xl overflow-hidden">
          <div class="bg-gradient-to-r from-indigo-600 to-purple-600 px-8 py-6">
            <div class="flex items-center justify-between">
              <div>
                <h1 class="text-3xl font-bold text-white">🔗 Link Device</h1>
                <p class="text-indigo-100 mt-2">
                  Connect your device to your account
                </p>
              </div>
              <A
                href="/"
                class="px-4 py-2 bg-white/20 hover:bg-white/30 text-white rounded-lg transition-all text-sm font-medium"
              >
                ← Back
              </A>
            </div>
          </div>

          <div class="px-8 py-8">
            <Show
              when={!showToken()}
              fallback={
                <div>
                  <div class="text-center mb-6">
                    <div class="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mb-4">
                      <svg
                        class="w-8 h-8 text-green-600"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          stroke-linecap="round"
                          stroke-linejoin="round"
                          stroke-width="2"
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    </div>
                    <h2 class="text-2xl font-bold text-gray-800 mb-2">
                      Device Linked Successfully!
                    </h2>
                    <p class="text-gray-600">
                      Your authentication token has been generated
                    </p>
                  </div>

                  <div class="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-6">
                    <div class="flex items-start gap-3">
                      <svg
                        class="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fill-rule="evenodd"
                          d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                          clip-rule="evenodd"
                        />
                      </svg>
                      <div>
                        <p class="text-sm font-semibold text-amber-800 mb-1">
                          Important: Do Not Share This Token
                        </p>
                        <p class="text-xs text-amber-700">
                          This token provides access to your account. Keep it
                          secure and never share it with anyone. If compromised,
                          revoke it immediately from the Tokens page.
                        </p>
                      </div>
                    </div>
                  </div>

                  <Show when={linkedToken()}>
                    {(token) => (
                      <div class="space-y-4">
                        <div>
                          <label class="block text-sm font-medium text-gray-700 mb-2">
                            Your Access Token
                          </label>
                          <div class="flex items-center gap-2">
                            <code class="flex-1 px-4 py-3 bg-gray-50 border border-gray-300 rounded-lg text-sm font-mono break-all">
                              {token().token}
                            </code>
                            <button
                              onClick={() => copyToClipboard(token().token!)}
                              class="px-4 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors font-medium whitespace-nowrap flex items-center gap-2"
                            >
                              {copiedToken() ? (
                                <>
                                  <svg
                                    class="w-5 h-5"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                  >
                                    <path
                                      stroke-linecap="round"
                                      stroke-linejoin="round"
                                      stroke-width="2"
                                      d="M5 13l4 4L19 7"
                                    />
                                  </svg>
                                  Copied!
                                </>
                              ) : (
                                <>
                                  <svg
                                    class="w-5 h-5"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                  >
                                    <path
                                      stroke-linecap="round"
                                      stroke-linejoin="round"
                                      stroke-width="2"
                                      d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                                    />
                                  </svg>
                                  Copy
                                </>
                              )}
                            </button>
                          </div>
                        </div>

                        <div class="bg-gray-50 rounded-lg p-4 space-y-2 text-sm">
                          <div class="flex items-center justify-between">
                            <span class="text-gray-600">Linked to:</span>
                            <span class="font-semibold text-gray-800">
                              {user()?.username}
                            </span>
                          </div>
                          <div class="flex items-center justify-between">
                            <span class="text-gray-600">Valid until:</span>
                            <span class="font-semibold text-gray-800">
                              {formatDate(token().expiresAt)}
                            </span>
                          </div>
                          <div class="flex items-center justify-between">
                            <span class="text-gray-600">Duration:</span>
                            <span class="font-semibold text-green-600">
                              1 Year
                            </span>
                          </div>
                        </div>

                        <div class="pt-4 space-y-3">
                          <A
                            href="/"
                            class="block w-full px-4 py-3 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 transition-colors font-medium text-center"
                          >
                            Return to Home
                          </A>
                          <A
                            href="/tokens"
                            class="block w-full px-4 py-3 border border-indigo-600 text-indigo-600 rounded-lg hover:bg-indigo-50 transition-colors font-medium text-center"
                          >
                            Manage All Tokens
                          </A>
                        </div>
                      </div>
                    )}
                  </Show>
                </div>
              }
            >
              <div class="text-center mb-8">
                <div class="inline-flex items-center justify-center w-20 h-20 bg-indigo-100 rounded-full mb-4">
                  <svg
                    class="w-10 h-10 text-indigo-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z"
                    />
                  </svg>
                </div>
                <h2 class="text-2xl font-bold text-gray-800 mb-3">
                  Link This Device
                </h2>
                <p class="text-gray-600 mb-2">
                  You are about to link a device to your account:
                </p>
                <p class="text-lg font-semibold text-indigo-600">
                  {user()?.username}
                </p>
              </div>

              <div class="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
                <div class="flex items-start gap-3">
                  <svg
                    class="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fill-rule="evenodd"
                      d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                      clip-rule="evenodd"
                    />
                  </svg>
                  <div class="text-sm text-blue-800">
                    <p class="font-semibold mb-1">What happens next?</p>
                    <ul class="list-disc list-inside space-y-1 text-blue-700">
                      <li>A new authentication token will be created</li>
                      <li>This token will be valid for 1 year</li>
                      <li>You can use it to authenticate API requests</li>
                      <li>You can revoke it anytime from the Tokens page</li>
                    </ul>
                  </div>
                </div>
              </div>

              <form action={linkDevice} method="post" class="space-y-4">
                <button
                  type="submit"
                  disabled={linking.pending}
                  class="w-full px-6 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg font-semibold text-lg hover:from-indigo-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-all transform hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Show when={!linking.pending} fallback="Linking Device...">
                    ✓ Yes, Link This Device
                  </Show>
                </button>

                <A
                  href="/"
                  class="block w-full px-6 py-4 bg-gray-200 text-gray-700 rounded-lg font-semibold text-lg hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 transition-all text-center"
                >
                  ✗ Cancel
                </A>
              </form>
            </Show>
          </div>
        </div>

        {/* Additional Info */}
        <Show when={!showToken()}>
          <div class="mt-6 bg-white rounded-xl shadow-lg p-6">
            <h3 class="text-sm font-semibold text-gray-800 mb-2">
              🔒 Security Note
            </h3>
            <p class="text-xs text-gray-600">
              Keep your token secure and don't share it with anyone. If you
              suspect your token has been compromised, revoke it immediately
              from the{" "}
              <A href="/tokens" class="text-indigo-600 hover:underline">
                Tokens page
              </A>
              .
            </p>
          </div>
        </Show>
      </div>
    </main>
  );
}
