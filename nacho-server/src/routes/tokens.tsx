import { createSignal, Show, For } from "solid-js";
import { useSubmission, createAsync } from "@solidjs/router";
import { A } from "@solidjs/router";
import { getAuthTokens, createToken, deleteToken, getUser } from "~/lib";

export default function Tokens() {
  const user = createAsync(() => getUser());
  const tokens = createAsync(() => getAuthTokens());
  const creatingToken = useSubmission(createToken);
  const deletingToken = useSubmission(deleteToken);

  const [expiresInDays, setExpiresInDays] = createSignal(30);
  const [showNewToken, setShowNewToken] = createSignal(false);
  const [newTokenValue, setNewTokenValue] = createSignal("");
  const [copiedTokenId, setCopiedTokenId] = createSignal<string | null>(null);

  const handleCreateToken = () => {
    setShowNewToken(false);
    setNewTokenValue("");
  };

  const copyToClipboard = (text: string, tokenId?: string) => {
    navigator.clipboard.writeText(text);
    if (tokenId) {
      setCopiedTokenId(tokenId);
      setTimeout(() => setCopiedTokenId(null), 2000);
    }
  };

  const formatDate = (date: string | Date) => {
    return new Date(date).toLocaleString();
  };

  const isExpired = (date: string | Date) => {
    return new Date(date) < new Date();
  };

  // Show new token after creation
  const newToken = () => {
    const result = creatingToken.result;
    if (result && "success" in result && result.success && result.token) {
      if (!showNewToken() && result.token.token !== newTokenValue()) {
        setShowNewToken(true);
        setNewTokenValue(result.token.token);
      }
      return result.token;
    }
    return null;
  };

  return (
    <main class="min-h-screen bg-gradient-to-br from-indigo-100 via-purple-50 to-pink-100 px-4 py-12">
      <div class="max-w-4xl mx-auto">
        {/* Header */}
        <div class="bg-white rounded-2xl shadow-2xl overflow-hidden mb-6">
          <div class="bg-gradient-to-r from-indigo-600 to-purple-600 px-8 py-6">
            <div class="flex items-center justify-between">
              <div>
                <h1 class="text-3xl font-bold text-white">🔑 API Tokens</h1>
                <p class="text-indigo-100 mt-2">
                  Manage your bearer authentication tokens
                </p>
                <p class="text-indigo-200 text-sm mt-1">
                  Logged in as:{" "}
                  <span class="font-semibold">{user()?.username}</span>
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

          {/* Create Token Form */}
          <div class="px-8 py-6 border-b border-gray-200">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">
              Create New Token
            </h2>
            <form
              action={createToken}
              method="post"
              class="flex gap-4 items-end"
            >
              <div class="flex-1">
                <label
                  for="expires"
                  class="block text-sm font-medium text-gray-700 mb-2"
                >
                  Expires In (Days)
                </label>
                <select
                  id="expires"
                  name="expiresInDays"
                  value={expiresInDays()}
                  onChange={(e) =>
                    setExpiresInDays(Number(e.currentTarget.value))
                  }
                  class="block w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                >
                  <option value="7">7 days</option>
                  <option value="30">30 days</option>
                  <option value="90">90 days</option>
                  <option value="180">180 days</option>
                  <option value="365">1 year</option>
                </select>
              </div>
              <button
                type="submit"
                onClick={handleCreateToken}
                disabled={creatingToken.pending}
                class="px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg font-semibold hover:from-indigo-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-all transform hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Show when={!creatingToken.pending} fallback="Creating...">
                  Create Token
                </Show>
              </button>
            </form>

            {/* Show newly created token */}
            <Show when={showNewToken() && newToken()}>
              <div class="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                <div class="flex items-start justify-between">
                  <div class="flex-1">
                    <p class="text-sm font-semibold text-green-800 mb-2">
                      ✅ Token Created Successfully!
                    </p>
                    <p class="text-xs text-green-700 mb-3">
                      Copy this token now. You won't be able to see it again!
                    </p>
                    <div class="flex items-center gap-2">
                      <code class="flex-1 px-3 py-2 bg-white border border-green-300 rounded text-xs font-mono break-all">
                        {newTokenValue()}
                      </code>
                      <button
                        onClick={() => copyToClipboard(newTokenValue())}
                        class="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 text-sm font-medium whitespace-nowrap"
                      >
                        Copy
                      </button>
                    </div>
                    <p class="text-xs text-green-600 mt-2">
                      Use as:{" "}
                      <code class="bg-white px-2 py-1 rounded">
                        Authorization: Bearer {newTokenValue()}
                      </code>
                    </p>
                  </div>
                  <button
                    onClick={() => setShowNewToken(false)}
                    class="ml-4 text-green-600 hover:text-green-800"
                  >
                    <svg
                      class="w-5 h-5"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fill-rule="evenodd"
                        d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                        clip-rule="evenodd"
                      />
                    </svg>
                  </button>
                </div>
              </div>
            </Show>
          </div>

          {/* Token List */}
          <div class="px-8 py-6">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">
              Your Tokens
            </h2>

            <Show
              when={tokens() && tokens()!.length > 0}
              fallback={
                <div class="text-center py-12 text-gray-500">
                  <svg
                    class="mx-auto h-12 w-12 text-gray-400 mb-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"
                    />
                  </svg>
                  <p class="text-lg font-medium">No tokens yet</p>
                  <p class="text-sm mt-1">
                    Create your first API token to get started
                  </p>
                </div>
              }
            >
              <div class="space-y-4">
                <For each={tokens()}>
                  {(token) => (
                    <div
                      class={`border rounded-lg p-4 ${
                        isExpired(token.expiresAt)
                          ? "border-red-300 bg-red-50"
                          : "border-gray-300 bg-gray-50"
                      }`}
                    >
                      <div class="flex items-start justify-between">
                        <div class="flex-1">
                          <div class="flex items-center gap-2 mb-2">
                            <span
                              class={`inline-block px-2 py-1 text-xs font-semibold rounded ${
                                isExpired(token.expiresAt)
                                  ? "bg-red-200 text-red-800"
                                  : "bg-green-200 text-green-800"
                              }`}
                            >
                              {isExpired(token.expiresAt)
                                ? "❌ Expired"
                                : "✅ Active"}
                            </span>
                            <span class="text-xs text-gray-500">
                              Token ID: {token.id.substring(0, 8)}...
                            </span>
                          </div>
                          <div class="flex items-center gap-2 mb-2">
                            <code class="text-xs font-mono text-gray-600 bg-white px-2 py-1 rounded border border-gray-200">
                              {token.token.substring(0, 20)}...
                            </code>
                            <button
                              onClick={() =>
                                copyToClipboard(token.token, token.id)
                              }
                              class="text-indigo-600 hover:text-indigo-800 text-sm"
                              title="Copy full token"
                            >
                              {copiedTokenId() === token.id ? (
                                <span class="text-green-600">✓ Copied!</span>
                              ) : (
                                <svg
                                  class="w-4 h-4"
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
                              )}
                            </button>
                          </div>
                          <div class="text-xs text-gray-600 space-y-1">
                            <p>Created: {formatDate(token.createdAt)}</p>
                            <p
                              class={
                                isExpired(token.expiresAt)
                                  ? "text-red-600 font-semibold"
                                  : ""
                              }
                            >
                              Expires: {formatDate(token.expiresAt)}
                            </p>
                          </div>
                        </div>
                        <form action={deleteToken} method="post">
                          <input
                            type="hidden"
                            name="tokenId"
                            value={token.id}
                          />
                          <button
                            type="submit"
                            disabled={deletingToken.pending}
                            class="ml-4 px-3 py-2 bg-red-600 text-white rounded hover:bg-red-700 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            Revoke
                          </button>
                        </form>
                      </div>
                    </div>
                  )}
                </For>
              </div>
            </Show>
          </div>
        </div>

        {/* Usage Instructions */}
        <div class="bg-white rounded-xl shadow-lg p-6">
          <h2 class="text-lg font-semibold text-gray-800 mb-3">
            📖 How to Use
          </h2>
          <div class="space-y-3 text-sm text-gray-700">
            <p>
              Include your token in the{" "}
              <code class="bg-gray-100 px-2 py-1 rounded text-xs">
                Authorization
              </code>{" "}
              header of your API requests:
            </p>
            <pre class="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-xs">
              {`curl -H "Authorization: Bearer YOUR_TOKEN_HERE" \\
  https://api.example.com/endpoint`}
            </pre>
            <p class="text-xs text-gray-600 mt-2">
              ⚠️ Keep your tokens secure! They provide access to your account.
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
