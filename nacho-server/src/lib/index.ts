import { action, query, redirect } from "@solidjs/router";
import { db } from "./db";
import {
  getSession,
  login,
  logout as logoutSession,
  register,
  validatePassword,
  validateUsername,
  createAuthToken,
  getUserAuthTokens,
  revokeAuthToken,
  getUserWatchHistory,
  fetchMediaInfo,
  getAllUsers,
  deleteUser,
  resetUserPassword,
  toggleUserAdmin,
  getSystemSettings,
  updateSystemSettings,
} from "./server";

export const getUser = query(async () => {
  "use server";
  try {
    const session = await getSession();
    const userId = session.data.userId;
    if (userId === undefined) throw new Error("User not found");
    const user = await db.user.findUnique({ where: { id: userId } });
    if (!user) throw new Error("User not found");
    return { id: user.id, username: user.username, isAdmin: user.isAdmin };
  } catch {
    await logoutSession();
    throw redirect("/login");
  }
}, "user");

export const loginUser = action(async (formData: FormData) => {
  "use server";
  const username = String(formData.get("username"));
  const password = String(formData.get("password"));

  const usernameError = await validateUsername(username);
  const passwordError = await validatePassword(password);
  const error = usernameError || passwordError;
  if (error) return error;

  try {
    const user = await login(username, password);
    const session = await getSession();
    await session.update((d) => {
      d.userId = user.id;
    });
  } catch (err) {
    return err as Error;
  }
  return redirect("/");
});

export const registerUser = action(async (formData: FormData) => {
  "use server";
  const username = String(formData.get("username"));
  const password = String(formData.get("password"));

  const usernameError = await validateUsername(username);
  const passwordError = await validatePassword(password);
  const error = usernameError || passwordError;
  if (error) return error;

  try {
    const user = await register(username, password);
    const session = await getSession();
    await session.update((d) => {
      d.userId = user.id;
    });
  } catch (err) {
    return err as Error;
  }
  return redirect("/");
});

export const logout = action(async () => {
  "use server";
  await logoutSession();
  return redirect("/login");
});

// Auth Token Actions
export const getAuthTokens = query(async () => {
  "use server";
  try {
    const session = await getSession();
    const userId = session.data.userId;
    if (userId === undefined) throw new Error("User not found");
    const tokens = await getUserAuthTokens(userId);
    return tokens;
  } catch (err) {
    throw redirect("/login");
  }
}, "authTokens");

export const createToken = action(async (formData: FormData) => {
  "use server";
  try {
    const session = await getSession();
    const userId = session.data.userId;
    if (userId === undefined) throw new Error("Not authenticated");

    const expiresInDays = Number(formData.get("expiresInDays")) || 30;

    const token = await createAuthToken(userId, expiresInDays);
    return { success: true, token };
  } catch (err) {
    return { success: false, error: err as Error };
  }
});

export const deleteToken = action(async (formData: FormData) => {
  "use server";
  try {
    const session = await getSession();
    const userId = session.data.userId;
    if (userId === undefined) throw new Error("Not authenticated");

    const tokenId = String(formData.get("tokenId"));
    await revokeAuthToken(tokenId, userId);
    return { success: true };
  } catch (err) {
    return { success: false, error: err as Error };
  }
});

// Watch History Query
export const getWatchHistory = query(async () => {
  "use server";
  try {
    const session = await getSession();
    const userId = session.data.userId;
    if (userId === undefined) throw new Error("Not authenticated");

    const history = await getUserWatchHistory(userId);

    // Fetch media info for all unique TMDB IDs
    const uniqueTmdbIds = new Set<string>();
    history.movies.forEach((m) => uniqueTmdbIds.add(m.tmdbID));
    history.episodes.forEach((e) => uniqueTmdbIds.add(e.tmdbID));

    const mediaInfoPromises = Array.from(uniqueTmdbIds).map(async (tmdbID) => {
      // For movies, we need to determine the type. Since we're mixing movies and episodes,
      // we'll try to fetch as both and see which succeeds
      // For now, we'll need to track which IDs are movies vs TV shows
      const isMovie = history.movies.some((m) => m.tmdbID === tmdbID);
      const info = await fetchMediaInfo(tmdbID, isMovie ? "movie" : "tv");
      return { tmdbID, info };
    });

    const mediaInfos = await Promise.all(mediaInfoPromises);
    const mediaInfoMap = new Map(
      mediaInfos.map(({ tmdbID, info }) => [tmdbID, info])
    );

    // Combine movies and episodes with their media info
    const moviesWithInfo = history.movies.map((movie) => ({
      ...movie,
      mediaInfo: mediaInfoMap.get(movie.tmdbID),
      type: "movie" as const,
    }));

    const episodesWithInfo = history.episodes.map((episode) => ({
      ...episode,
      mediaInfo: mediaInfoMap.get(episode.tmdbID),
      type: "episode" as const,
    }));

    // Combine and sort by timestamp watched
    const combined = [...moviesWithInfo, ...episodesWithInfo].sort(
      (a, b) =>
        new Date(b.timestampWatched).getTime() -
        new Date(a.timestampWatched).getTime()
    );

    return combined;
  } catch (err) {
    throw redirect("/login");
  }
}, "watchHistory");

// Admin Actions
export const getUsers = query(async () => {
  "use server";
  try {
    const session = await getSession();
    const userId = session.data.userId;
    if (userId === undefined) throw new Error("Not authenticated");

    const users = await getAllUsers(userId);
    return users;
  } catch (err) {
    throw new Error((err as Error).message);
  }
}, "users");

export const removeUser = action(async (formData: FormData) => {
  "use server";
  try {
    const session = await getSession();
    const userId = session.data.userId;
    if (userId === undefined) throw new Error("Not authenticated");

    const targetUserId = String(formData.get("userId"));
    await deleteUser(userId, targetUserId);
    return { success: true };
  } catch (err) {
    return { success: false, error: (err as Error).message };
  }
});

export const resetPassword = action(async (formData: FormData) => {
  "use server";
  try {
    const session = await getSession();
    const userId = session.data.userId;
    if (userId === undefined) throw new Error("Not authenticated");

    const targetUserId = String(formData.get("userId"));
    const newPassword = String(formData.get("newPassword"));

    await resetUserPassword(userId, targetUserId, newPassword);
    return { success: true };
  } catch (err) {
    return { success: false, error: (err as Error).message };
  }
});

export const toggleAdmin = action(async (formData: FormData) => {
  "use server";
  try {
    const session = await getSession();
    const userId = session.data.userId;
    if (userId === undefined) throw new Error("Not authenticated");

    const targetUserId = String(formData.get("userId"));
    await toggleUserAdmin(userId, targetUserId);
    return { success: true };
  } catch (err) {
    return { success: false, error: (err as Error).message };
  }
});

export const getSettings = query(async () => {
  "use server";
  const settings = await getSystemSettings();
  return settings;
}, "settings");

export const updateSettings = action(async (formData: FormData) => {
  "use server";
  try {
    const session = await getSession();
    const userId = session.data.userId;
    if (userId === undefined) throw new Error("Not authenticated");

    const allowNewUserRegistration =
      formData.get("allowNewUserRegistration") === "true";

    await updateSystemSettings(userId, allowNewUserRegistration);
    return { success: true };
  } catch (err) {
    return { success: false, error: (err as Error).message };
  }
});
