# Use Node.js 22 slim (Debian-based) for better Prisma compatibility
FROM node:22-slim

# Set working directory
WORKDIR /app

# Install OpenSSL and build dependencies for native modules (argon2, prisma)
RUN apt-get update -y && \
    apt-get install -y openssl libssl-dev python3 make g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci

# Copy prisma schema
COPY prisma ./prisma

# Generate Prisma Client
RUN npx prisma generate

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Expose the port the app runs on
EXPOSE 3000

# Set environment to production
ENV NODE_ENV=production

# Use entrypoint script to run migrations before starting the app
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["npm", "start"]
