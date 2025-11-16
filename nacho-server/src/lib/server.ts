"use server";

import { useSession } from "vinxi/http";
import { db } from "./db";
import * as argon2 from "argon2";
import { randomBytes } from "crypto";

// Cache for TMDB data to avoid repeated requests
const tmdbCache = new Map<string, { data: any; timestamp: number }>();
const CACHE_DURATION = 24 * 60 * 60 * 1000; // 24 hours

// TMDB Configuration
const TMDB_API_KEY = process.env.API_TMDB_API_KEY;
const TMDB_BASE_URL = process.env.API_TMDB_BASEURL || "https://api.tmdb.org";
const TMDB_IMAGES_BASE_URL =
  process.env.API_TMDB_IMAGES_BASEURL || "https://image.tmdb.org/t/p";

// Brute force protection constants
const MAX_LOGIN_ATTEMPTS = 5;
const LOCKOUT_DURATION_MS = 15 * 60 * 1000; // 15 minutes
const ATTEMPT_RESET_TIME_MS = 60 * 60 * 1000; // 1 hour

// In-memory rate limiting for IP addresses (for additional protection)
const ipAttempts = new Map<
  string,
  { count: number; firstAttempt: number; lockedUntil?: number }
>();

export async function validateUsername(username: unknown) {
  if (typeof username !== "string" || username.length < 3) {
    return new Error(`Usernames must be at least 3 characters long`);
  }

  // ensure username is not used
  db.user.findUnique({ where: { username } }).then((user) => {
    if (user) {
      return new Error(`Username "${username}" is already taken`);
    }
  });

  return null;
}

export async function validatePassword(password: unknown) {
  if (typeof password !== "string" || password.length < 6) {
    return new Error(`Passwords must be at least 6 characters long`);
  }
  return null;
}

function getClientIp(): string {
  // Try to get the real IP from various headers
  if (typeof window === "undefined") {
    try {
      const headers = new Headers();
      // Common headers that contain the client IP
      const ipHeaders = [
        "x-forwarded-for",
        "x-real-ip",
        "x-client-ip",
        "cf-connecting-ip",
      ];

      for (const header of ipHeaders) {
        const value = headers.get(header);
        if (value) {
          // x-forwarded-for can contain multiple IPs, take the first one
          return value.split(",")[0].trim();
        }
      }
    } catch (e) {
      // Fallback if headers are not available
    }
  }
  return "unknown";
}

function checkIpRateLimit(ip: string): {
  allowed: boolean;
  lockoutRemaining?: number;
} {
  const now = Date.now();
  const ipData = ipAttempts.get(ip);

  if (!ipData) {
    return { allowed: true };
  }

  // Check if IP is currently locked out
  if (ipData.lockedUntil && ipData.lockedUntil > now) {
    const remaining = Math.ceil((ipData.lockedUntil - now) / 1000);
    return { allowed: false, lockoutRemaining: remaining };
  }

  // Reset attempts if enough time has passed
  if (now - ipData.firstAttempt > ATTEMPT_RESET_TIME_MS) {
    ipAttempts.delete(ip);
    return { allowed: true };
  }

  // Check if too many attempts
  if (ipData.count >= MAX_LOGIN_ATTEMPTS) {
    ipData.lockedUntil = now + LOCKOUT_DURATION_MS;
    const remaining = Math.ceil(LOCKOUT_DURATION_MS / 1000);
    return { allowed: false, lockoutRemaining: remaining };
  }

  return { allowed: true };
}

function recordIpAttempt(ip: string) {
  const now = Date.now();
  const ipData = ipAttempts.get(ip);

  if (!ipData) {
    ipAttempts.set(ip, { count: 1, firstAttempt: now });
  } else {
    ipData.count++;
  }
}

function clearIpAttempts(ip: string) {
  ipAttempts.delete(ip);
}

export async function login(username: string, password: string) {
  console.log("Attempting login for user:", username);

  // Check IP rate limit
  const clientIp = getClientIp();
  const ipCheck = checkIpRateLimit(clientIp);
  if (!ipCheck.allowed) {
    throw new Error(
      `Too many login attempts. Please try again in ${ipCheck.lockoutRemaining} seconds.`
    );
  }

  // Find user
  const user = await db.user.findUnique({ where: { username } });

  if (!user) {
    // Record failed attempt even for non-existent users to prevent user enumeration timing attacks
    recordIpAttempt(clientIp);
    // Add a small delay to make timing attacks harder
    await new Promise((resolve) => setTimeout(resolve, 100));
    throw new Error("Invalid username or password");
  }

  // Check if account is locked
  if (user.lockedUntil && user.lockedUntil > new Date()) {
    const remainingSeconds = Math.ceil(
      (user.lockedUntil.getTime() - Date.now()) / 1000
    );
    throw new Error(
      `Account is locked due to too many failed attempts. Try again in ${remainingSeconds} seconds.`
    );
  }

  // Reset failed attempts if lockout period has passed
  if (user.lockedUntil && user.lockedUntil <= new Date()) {
    await db.user.update({
      where: { id: user.id },
      data: {
        failedLoginAttempts: 0,
        lockedUntil: null,
        lastFailedLogin: null,
      },
    });
  }

  // Verify password using argon2
  const isValid = await argon2.verify(user.passwordHash, password);

  if (!isValid) {
    // Record failed attempt
    recordIpAttempt(clientIp);

    const newFailedAttempts = user.failedLoginAttempts + 1;
    const updateData: any = {
      failedLoginAttempts: newFailedAttempts,
      lastFailedLogin: new Date(),
    };

    // Lock account if max attempts reached
    if (newFailedAttempts >= MAX_LOGIN_ATTEMPTS) {
      updateData.lockedUntil = new Date(Date.now() + LOCKOUT_DURATION_MS);
    }

    await db.user.update({
      where: { id: user.id },
      data: updateData,
    });

    if (newFailedAttempts >= MAX_LOGIN_ATTEMPTS) {
      throw new Error(
        `Account locked due to too many failed attempts. Try again in ${Math.ceil(
          LOCKOUT_DURATION_MS / 1000
        )} seconds.`
      );
    } else {
      const attemptsRemaining = MAX_LOGIN_ATTEMPTS - newFailedAttempts;
      throw new Error(
        `Invalid username or password. ${attemptsRemaining} attempt(s) remaining before account lockout.`
      );
    }
  }

  // Successful login - reset failed attempts
  if (
    user.failedLoginAttempts > 0 ||
    user.lockedUntil ||
    user.lastFailedLogin
  ) {
    await db.user.update({
      where: { id: user.id },
      data: {
        failedLoginAttempts: 0,
        lockedUntil: null,
        lastFailedLogin: null,
      },
    });
  }

  // Clear IP attempts on successful login
  clearIpAttempts(clientIp);

  return user;
}

export async function logout() {
  const session = await getSession();
  await session.update((d) => {
    d.userId = undefined;
  });
}

// System settings functions (defined before register to be used there)
export async function getSystemSettings() {
  let settings = await db.systemSettings.findUnique({
    where: { id: "system" },
  });

  // Create default settings if they don't exist
  if (!settings) {
    settings = await db.systemSettings.create({
      data: { id: "system", allowNewUserRegistration: true },
    });
  }

  return settings;
}

export async function register(username: string, password: string) {
  // Check if registration is allowed
  const settings = await getSystemSettings();
  if (!settings.allowNewUserRegistration) {
    throw new Error("New user registration is currently disabled");
  }

  const existingUser = await db.user.findUnique({ where: { username } });
  if (existingUser) throw new Error("User already exists");

  // Generate a random salt
  const salt = randomBytes(32).toString("hex");

  // Hash the password with Argon2id
  const passwordHash = await argon2.hash(password, {
    type: argon2.argon2id,
    salt: Buffer.from(salt, "hex"),
  });

  return db.user.create({
    data: {
      username: username,
      salt: salt,
      passwordHash: passwordHash,
      isAdmin: false,
      failedLoginAttempts: 0,
      lockedUntil: null,
      lastFailedLogin: null,
    },
  });
}

export function getSession() {
  return useSession({
    password:
      process.env.SESSION_SECRET ?? "areallylongsecretthatyoushouldreplace",
  });
}

// Auth Token Management
export async function createAuthToken(
  userId: string,
  expiresInDays: number = 30
) {
  // Generate a secure random token
  const tokenBytes = randomBytes(32);
  const token = tokenBytes.toString("base64url"); // URL-safe base64

  const expiresAt = new Date();
  expiresAt.setDate(expiresAt.getDate() + expiresInDays);

  const authToken = await db.authToken.create({
    data: {
      userId,
      token,
      expiresAt,
    },
  });

  return authToken;
}

export async function getUserAuthTokens(userId: string) {
  const tokens = await db.authToken.findMany({
    where: { userId },
    orderBy: { createdAt: "desc" },
  });
  return tokens;
}

export async function revokeAuthToken(tokenId: string, userId: string) {
  // Ensure the token belongs to the user
  const token = await db.authToken.findFirst({
    where: { id: tokenId, userId },
  });

  if (!token) {
    throw new Error("Token not found or does not belong to you");
  }

  await db.authToken.delete({
    where: { id: tokenId },
  });

  return true;
}

export async function validateAuthToken(token: string) {
  const authToken = await db.authToken.findUnique({
    where: { token },
    include: { user: true },
  });

  if (!authToken) {
    return null;
  }

  // Check if token is expired
  if (authToken.expiresAt < new Date()) {
    return null;
  }

  return authToken.user;
}

// Fetch movie/show data from TMDB API using TMDB ID
export async function fetchMediaInfo(
  tmdbID: string,
  type: "movie" | "tv" = "movie"
) {
  // Check cache first
  const cacheKey = `${type}:${tmdbID}`;
  const cached = tmdbCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
    return cached.data;
  }

  try {
    // Fetch detailed info from TMDB using TMDB ID directly
    const endpoint =
      type === "movie"
        ? `${TMDB_BASE_URL}/3/movie/${tmdbID}?api_key=${TMDB_API_KEY}`
        : `${TMDB_BASE_URL}/3/tv/${tmdbID}?api_key=${TMDB_API_KEY}`;

    const response = await fetch(endpoint);

    if (!response.ok) {
      throw new Error(`Failed to fetch data for TMDB ID ${tmdbID}`);
    }

    const data = await response.json();

    // Normalize the data structure
    const normalizedData = {
      ...data,
      type,
      tmdbID,
      // Add full poster URL
      poster_url: data.poster_path
        ? `${TMDB_IMAGES_BASE_URL}/w500${data.poster_path}`
        : null,
    };

    // Cache the result
    tmdbCache.set(cacheKey, { data: normalizedData, timestamp: Date.now() });

    return normalizedData;
  } catch (error) {
    console.error(`Error fetching media info for TMDB ID ${tmdbID}:`, error);
    // Return placeholder data on error
    return {
      title: tmdbID,
      name: tmdbID,
      release_date: null,
      first_air_date: null,
      poster_path: null,
      poster_url: null,
      type,
      tmdbID,
    };
  }
}

export async function getUserWatchHistory(userId: string) {
  const [movies, episodes] = await Promise.all([
    db.movieEntry.findMany({
      where: { userId },
      orderBy: { timestampWatched: "desc" },
    }),
    db.episodeEntry.findMany({
      where: { userId },
      orderBy: { timestampWatched: "desc" },
    }),
  ]);

  return { movies, episodes };
}

// Admin functions
async function requireAdmin(userId: string) {
  const user = await db.user.findUnique({ where: { id: userId } });
  if (!user || !user.isAdmin) {
    throw new Error("Unauthorized: Admin access required");
  }
  return user;
}

export async function getAllUsers(adminUserId: string) {
  await requireAdmin(adminUserId);

  const users = await db.user.findMany({
    select: {
      id: true,
      username: true,
      isAdmin: true,
      failedLoginAttempts: true,
      lockedUntil: true,
      lastFailedLogin: true,
      _count: {
        select: {
          movieEntries: true,
          episodeEntries: true,
          authTokens: true,
        },
      },
    },
    orderBy: { username: "asc" },
  });

  return users;
}

export async function deleteUser(adminUserId: string, targetUserId: string) {
  await requireAdmin(adminUserId);

  // Prevent deleting yourself
  if (adminUserId === targetUserId) {
    throw new Error("Cannot delete your own account");
  }

  // Delete user and all related data (cascade)
  await db.user.delete({
    where: { id: targetUserId },
  });

  return true;
}

export async function resetUserPassword(
  adminUserId: string,
  targetUserId: string,
  newPassword: string
) {
  await requireAdmin(adminUserId);

  // Validate new password
  const passwordError = await validatePassword(newPassword);
  if (passwordError) {
    throw passwordError;
  }

  // Generate a random salt
  const salt = randomBytes(32).toString("hex");

  // Hash the password with Argon2id
  const passwordHash = await argon2.hash(newPassword, {
    type: argon2.argon2id,
    salt: Buffer.from(salt, "hex"),
  });

  // Update user password and reset lockout
  await db.user.update({
    where: { id: targetUserId },
    data: {
      salt,
      passwordHash,
      failedLoginAttempts: 0,
      lockedUntil: null,
      lastFailedLogin: null,
    },
  });

  return true;
}

export async function toggleUserAdmin(
  adminUserId: string,
  targetUserId: string
) {
  await requireAdmin(adminUserId);

  // Prevent removing your own admin status
  if (adminUserId === targetUserId) {
    throw new Error("Cannot modify your own admin status");
  }

  const targetUser = await db.user.findUnique({
    where: { id: targetUserId },
  });

  if (!targetUser) {
    throw new Error("User not found");
  }

  await db.user.update({
    where: { id: targetUserId },
    data: { isAdmin: !targetUser.isAdmin },
  });

  return true;
}

export async function updateSystemSettings(
  adminUserId: string,
  allowNewUserRegistration: boolean
) {
  await requireAdmin(adminUserId);

  const settings = await db.systemSettings.upsert({
    where: { id: "system" },
    update: { allowNewUserRegistration },
    create: { id: "system", allowNewUserRegistration },
  });

  return settings;
}
