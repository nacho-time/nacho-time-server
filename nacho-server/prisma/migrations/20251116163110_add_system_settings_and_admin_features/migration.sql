-- CreateTable
CREATE TABLE "SystemSettings" (
    "id" TEXT NOT NULL PRIMARY KEY DEFAULT 'system',
    "allowNewUserRegistration" BOOLEAN NOT NULL DEFAULT true
);
