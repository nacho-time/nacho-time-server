#!/bin/sh
set -e

echo "Waiting for database to be ready..."
sleep 5

echo "Pushing database schema..."
npx prisma db push --accept-data-loss

echo "Seeding database..."
npx prisma db seed || echo "Seeding skipped (data already exists)"

echo "Starting application..."
exec "$@"
