1. Sync schemas:
   npx prisma db push
2. Create initial migration:
   npx prisma migrate dev --name init
3. Reseed the database:
   npx prisma migrate reset
