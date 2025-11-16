import { createAsync, type RouteDefinition } from "@solidjs/router";
import { A } from "@solidjs/router";
import { Show } from "solid-js";
import { getUser, logout } from "~/lib";
import AdminPanel from "~/components/AdminPanel";

export const route = {
  preload() {
    getUser();
  },
} satisfies RouteDefinition;

export default function Home() {
  const user = createAsync(() => getUser(), { deferStream: true });
  return (
    <main class="min-h-screen bg-gradient-to-br from-indigo-100 via-purple-50 to-pink-100 p-8">
      <div class="max-w-4xl mx-auto space-y-6">
        <div class="bg-white rounded-2xl shadow-2xl overflow-hidden">
          <div class="bg-gradient-to-r from-indigo-600 to-purple-600 px-8 py-6">
            <h2 class="text-3xl font-bold text-white">
              Hello {user()?.username}! 👋
            </h2>
            <p class="text-indigo-100 mt-2">Welcome to Nacho Time</p>
          </div>

          <div class="px-8 py-6">
            <h3 class="text-xl font-bold text-gray-800 mb-4">Quick Actions</h3>
            <div class="grid gap-4 md:grid-cols-2">
              <A
                href="/dashboard"
                class="block p-6 bg-gradient-to-br from-blue-50 to-indigo-50 border-2 border-blue-200 rounded-xl hover:border-blue-400 transition-all transform hover:scale-105"
              >
                <div class="flex items-center gap-3 mb-2">
                  <span class="text-3xl">�</span>
                  <h4 class="text-lg font-semibold text-gray-800">Dashboard</h4>
                </div>
                <p class="text-sm text-gray-600">
                  View your watch history with posters and stats
                </p>
              </A>

              <A
                href="/tokens"
                class="block p-6 bg-gradient-to-br from-indigo-50 to-purple-50 border-2 border-indigo-200 rounded-xl hover:border-indigo-400 transition-all transform hover:scale-105"
              >
                <div class="flex items-center gap-3 mb-2">
                  <span class="text-3xl">�</span>
                  <h4 class="text-lg font-semibold text-gray-800">
                    API Tokens
                  </h4>
                </div>
                <p class="text-sm text-gray-600">
                  Manage your authentication tokens for API access
                </p>
              </A>
            </div>

            <div class="mt-8 pt-6 border-t border-gray-200">
              <form action={logout} method="post">
                <button
                  name="logout"
                  type="submit"
                  class="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                >
                  Logout
                </button>
              </form>
            </div>
          </div>
        </div>

        {/* Admin Panel - Only visible to admins */}
        <Show when={user()?.isAdmin}>
          <AdminPanel />
        </Show>
      </div>
    </main>
  );
}
