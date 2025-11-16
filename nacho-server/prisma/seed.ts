import { PrismaClient } from "@prisma/client";
import * as argon2 from "argon2";
import { randomBytes } from "crypto";

const prisma = new PrismaClient();

async function main() {
  console.log("Starting database seed...");

  // Create admin user with username "admin" and password "migrateduser"
  const username = "admin";
  const password = "migrateduser";

  // Generate a random salt
  const salt = randomBytes(32).toString("hex");

  // Hash the password with Argon2id
  const passwordHash = await argon2.hash(password, {
    type: argon2.argon2id,
    salt: Buffer.from(salt, "hex"),
  });

  const admin = await prisma.user.create({
    data: {
      username,
      salt,
      passwordHash,
      isAdmin: true,
    },
  });

  console.log(`✅ Admin user created with ID: ${admin.id}`);
  console.log(`   Username: ${username}`);
  console.log(`   Password: ${password}`);
  console.log("⚠️  Please change the admin password after first login!");

  // Add sample movie entries
  console.log("\n📽️  Adding sample movie entries...");
  const sampleMovies = [
    {
      tmdbID: "278",
      title: "The Shawshank Redemption",
      timestampWatched: new Date("2024-01-15T20:30:00Z"),
      timestampAdded: new Date("2024-01-15T22:45:00Z"),
    },
    {
      tmdbID: "155",
      title: "The Dark Knight",
      timestampWatched: new Date("2024-02-20T19:00:00Z"),
      timestampAdded: new Date("2024-02-20T21:30:00Z"),
    },
    {
      tmdbID: "550",
      title: "Fight Club",
      timestampWatched: new Date("2024-03-10T21:00:00Z"),
      timestampAdded: new Date("2024-03-10T23:15:00Z"),
    },
    {
      tmdbID: "157336",
      title: "Interstellar",
      timestampWatched: new Date("2024-04-05T18:30:00Z"),
      timestampAdded: new Date("2024-04-05T21:45:00Z"),
    },
    {
      tmdbID: "27205",
      title: "Inception",
      timestampWatched: new Date("2024-05-12T20:00:00Z"),
      timestampAdded: new Date("2024-05-12T22:30:00Z"),
    },
  ];

  for (const movie of sampleMovies) {
    await prisma.movieEntry.create({
      data: {
        userId: admin.id,
        tmdbID: movie.tmdbID,
        timestampWatched: movie.timestampWatched,
        timestampAdded: movie.timestampAdded,
      },
    });
    console.log(`   ✓ Added movie: ${movie.title} (${movie.tmdbID})`);
  }

  // Add sample episode entries
  console.log("\n📺 Adding sample TV show episodes...");
  const sampleEpisodes = [
    // Breaking Bad - Season 1
    {
      tmdbID: "1396",
      title: "Breaking Bad S01E01",
      season: 1,
      episode: 1,
      timestampWatched: new Date("2024-01-20T20:00:00Z"),
      timestampAdded: new Date("2024-01-20T20:50:00Z"),
    },
    {
      tmdbID: "1396",
      title: "Breaking Bad S01E02",
      season: 1,
      episode: 2,
      timestampWatched: new Date("2024-01-20T21:00:00Z"),
      timestampAdded: new Date("2024-01-20T21:50:00Z"),
    },
    {
      tmdbID: "1396",
      title: "Breaking Bad S01E03",
      season: 1,
      episode: 3,
      timestampWatched: new Date("2024-01-21T20:00:00Z"),
      timestampAdded: new Date("2024-01-21T20:50:00Z"),
    },
    // The Office
    {
      tmdbID: "2316",
      title: "The Office S02E01",
      season: 2,
      episode: 1,
      timestampWatched: new Date("2024-02-15T19:30:00Z"),
      timestampAdded: new Date("2024-02-15T20:00:00Z"),
    },
    {
      tmdbID: "2316",
      title: "The Office S02E02",
      season: 2,
      episode: 2,
      timestampWatched: new Date("2024-02-15T20:00:00Z"),
      timestampAdded: new Date("2024-02-15T20:30:00Z"),
    },
    // Stranger Things
    {
      tmdbID: "66732",
      title: "Stranger Things S01E01",
      season: 1,
      episode: 1,
      timestampWatched: new Date("2024-03-05T21:00:00Z"),
      timestampAdded: new Date("2024-03-05T22:00:00Z"),
    },
    {
      tmdbID: "66732",
      title: "Stranger Things S01E02",
      season: 1,
      episode: 2,
      timestampWatched: new Date("2024-03-05T22:00:00Z"),
      timestampAdded: new Date("2024-03-05T23:00:00Z"),
    },
    // Game of Thrones
    {
      tmdbID: "1399",
      title: "Game of Thrones S01E01",
      season: 1,
      episode: 1,
      timestampWatched: new Date("2024-04-10T20:00:00Z"),
      timestampAdded: new Date("2024-04-10T21:00:00Z"),
    },
  ];

  for (const episode of sampleEpisodes) {
    await prisma.episodeEntry.create({
      data: {
        userId: admin.id,
        tmdbID: episode.tmdbID,
        season: episode.season,
        episode: episode.episode,
        timestampWatched: episode.timestampWatched,
        timestampAdded: episode.timestampAdded,
      },
    });
    console.log(`   ✓ Added episode: ${episode.title} (${episode.tmdbID})`);
  }

  console.log("\n✅ Sample data added successfully!");
  console.log(`   Movies: ${sampleMovies.length}`);
  console.log(`   Episodes: ${sampleEpisodes.length}`);
}

main()
  .catch((e) => {
    console.error("Error seeding database:", e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
