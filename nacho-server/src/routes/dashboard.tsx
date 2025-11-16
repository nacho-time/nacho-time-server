import { createAsync, type RouteDefinition } from "@solidjs/router";
import { For, Show } from "solid-js";
import { A } from "@solidjs/router";
import { getUser, getWatchHistory } from "~/lib";

// Icon components
const ChartBarIcon = (props: any) => (
  <svg {...props} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      stroke-linecap="round"
      stroke-linejoin="round"
      stroke-width="2"
      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
    />
  </svg>
);

const MovieIcon = (props: any) => (
  <svg {...props} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      stroke-linecap="round"
      stroke-linejoin="round"
      stroke-width="2"
      d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z"
    />
  </svg>
);

const TvIcon = (props: any) => (
  <svg {...props} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      stroke-linecap="round"
      stroke-linejoin="round"
      stroke-width="2"
      d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
    />
  </svg>
);

const ClockIcon = (props: any) => (
  <svg {...props} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      stroke-linecap="round"
      stroke-linejoin="round"
      stroke-width="2"
      d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
    />
  </svg>
);

export const route = {
  preload() {
    getUser();
    getWatchHistory();
  },
} satisfies RouteDefinition;

export default function Dashboard() {
  const user = createAsync(() => getUser());
  const history = createAsync(() => getWatchHistory());

  const formatDate = (date: string | Date) => {
    return new Date(date).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getDisplayTitle = (item: any) => {
    const mediaInfo = item.mediaInfo;
    if (!mediaInfo) return item.tmdbID;

    // For TV shows, use 'name', for movies use 'title'
    const title = mediaInfo.title || mediaInfo.name || item.tmdbID;

    if (item.type === "episode") {
      return `${title} - S${item.season}E${item.episode}`;
    }
    return title;
  };

  const getYear = (item: any) => {
    const mediaInfo = item.mediaInfo;
    if (!mediaInfo) return "N/A";

    // For movies: release_date, for TV shows: first_air_date
    const date = mediaInfo.release_date || mediaInfo.first_air_date;
    if (!date) return "N/A";

    return new Date(date).getFullYear();
  };

  const getPoster = (item: any) => {
    const posterUrl = item.mediaInfo?.poster_url;
    return posterUrl || null;
  };

  return (
    <main class="min-h-screen bg-gradient-to-br from-indigo-100 via-purple-50 to-pink-100 p-4 md:p-8">
      <div class="max-w-7xl mx-auto">
        {/* Header */}
        <div class="bg-white rounded-2xl shadow-2xl overflow-hidden mb-6">
          <div class="bg-gradient-to-r from-indigo-600 to-purple-600 px-6 md:px-8 py-6">
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-3">
                <ChartBarIcon class="w-8 h-8 text-white" />
                <div>
                  <h1 class="text-2xl md:text-3xl font-bold text-white">
                    Watch History Dashboard
                  </h1>
                  <p class="text-indigo-100 mt-2">
                    Welcome back,{" "}
                    <span class="font-semibold">{user()?.username}</span>
                  </p>
                </div>
              </div>
              <A
                href="/"
                class="px-4 py-2 bg-white/20 hover:bg-white/30 text-white rounded-lg transition-all text-sm font-medium"
              >
                ← Back
              </A>
            </div>
          </div>
        </div>

        {/* Content */}
        <Show
          when={history() && history()!.length > 0}
          fallback={
            <div class="bg-white rounded-xl shadow-lg p-12 text-center">
              <svg
                class="mx-auto h-16 w-16 text-gray-400 mb-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z"
                />
              </svg>
              <h3 class="text-xl font-semibold text-gray-700 mb-2">
                No Watch History Yet
              </h3>
              <p class="text-gray-500">
                Start watching movies and TV shows to see them here!
              </p>
            </div>
          }
        >
          <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
            <For each={history()}>
              {(item) => (
                <div class="bg-white rounded-lg shadow-lg overflow-hidden transform transition-all hover:scale-105 hover:shadow-xl">
                  {/* Poster */}
                  <div class="relative aspect-[2/3] bg-gradient-to-br from-gray-300 to-gray-400">
                    <Show
                      when={getPoster(item)}
                      fallback={
                        <div class="w-full h-full flex items-center justify-center">
                          <svg
                            class="w-20 h-20 text-gray-500"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              stroke-linecap="round"
                              stroke-linejoin="round"
                              stroke-width="1.5"
                              d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z"
                            />
                          </svg>
                        </div>
                      }
                    >
                      <img
                        src={getPoster(item)!}
                        alt={getDisplayTitle(item)}
                        class="w-full h-full object-cover"
                        loading="lazy"
                        onError={(e) => {
                          // Hide the image and show fallback by removing src
                          e.currentTarget.style.display = "none";
                        }}
                      />
                    </Show>
                    {/* Type Badge */}
                    <div class="absolute top-2 right-2">
                      <span
                        class={`px-2 py-1 flex items-center gap-1 text-xs font-bold rounded ${
                          item.type === "movie"
                            ? "bg-blue-500 text-white"
                            : "bg-purple-500 text-white"
                        }`}
                      >
                        {item.type === "movie" ? (
                          <MovieIcon class="w-3.5 h-3.5" />
                        ) : (
                          <TvIcon class="w-3.5 h-3.5" />
                        )}
                      </span>
                    </div>
                  </div>

                  {/* Info */}
                  <div class="p-3">
                    <h3
                      class="font-semibold text-sm text-gray-800 line-clamp-2 mb-1"
                      title={getDisplayTitle(item)}
                    >
                      {getDisplayTitle(item)}
                    </h3>
                    <p class="text-xs text-gray-500 mb-1">{getYear(item)}</p>
                    <div class="flex items-center gap-1 text-xs text-gray-400">
                      <ClockIcon class="w-3 h-3" />
                      <span class="truncate">
                        {formatDate(item.timestampWatched)}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </For>
          </div>

          {/* Stats Footer */}
          <div class="mt-8 bg-white rounded-xl shadow-lg p-6">
            <div class="flex items-center gap-2 mb-4">
              <ChartBarIcon class="w-5 h-5 text-gray-700" />
              <h2 class="text-lg font-semibold text-gray-800">Your Stats</h2>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div class="text-center p-4 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg">
                <div class="text-3xl font-bold text-indigo-600">
                  {history()?.filter((item) => item.type === "movie").length ||
                    0}
                </div>
                <div class="text-sm text-gray-600 mt-1">Movies Watched</div>
              </div>
              <div class="text-center p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg">
                <div class="text-3xl font-bold text-purple-600">
                  {history()?.filter((item) => item.type === "episode")
                    .length || 0}
                </div>
                <div class="text-sm text-gray-600 mt-1">Episodes Watched</div>
              </div>
              <div class="text-center p-4 bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg">
                <div class="text-3xl font-bold text-green-600">
                  {history()?.length || 0}
                </div>
                <div class="text-sm text-gray-600 mt-1">Total Items</div>
              </div>
              <div class="text-center p-4 bg-gradient-to-br from-orange-50 to-amber-50 rounded-lg">
                <div class="text-3xl font-bold text-orange-600">
                  {new Set(history()?.map((item) => item.tmdbID)).size || 0}
                </div>
                <div class="text-sm text-gray-600 mt-1">Unique Titles</div>
              </div>
            </div>
          </div>
        </Show>
      </div>
    </main>
  );
}
