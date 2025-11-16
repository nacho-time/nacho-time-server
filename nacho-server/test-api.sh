#!/bin/bash

# Test script for adding watch history via API
# Usage: ./test-api.sh [YOUR_API_TOKEN]

# Configuration
API_URL="http://localhost:3000/api/history"
TOKEN="${1:-c3O79Ulmj0CHHJ3aI6sHFMRo7qSv-D_e-5O6BJyXTZQ}"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Testing Nacho Time API"
echo "=========================================="
echo ""

if [ "$TOKEN" = "YOUR_TOKEN_HERE" ]; then
    echo -e "${RED}ERROR: No API token provided!${NC}"
    echo "Usage: $0 YOUR_API_TOKEN"
    echo ""
    echo "Get your token from: http://localhost:3000/tokens"
    exit 1
fi

echo -e "${YELLOW}Using token:${NC} ${TOKEN:0:20}..."
echo -e "${YELLOW}API endpoint:${NC} $API_URL"
echo ""

# Test 1: Add a TV show episode (Breaking Bad)
echo -e "${YELLOW}Test 1: Adding a TV show episode (Breaking Bad S01E01)${NC}"
response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X POST "$API_URL" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "episodes": [
      {
        "imdbID": "tt0959621",
        "season": 1,
        "episode": 1,
        "timestampWatched": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"
      }
    ]
  }')

http_status=$(echo "$response" | grep "HTTP_STATUS" | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_STATUS/d')

if [ "$http_status" = "201" ]; then
    echo -e "${GREEN}✓ Success!${NC} Status: $http_status"
    echo "$body" | jq '.'
else
    echo -e "${RED}✗ Failed!${NC} Status: $http_status"
    echo "$body" | jq '.' 2>/dev/null || echo "$body"
fi
echo ""

# Test 2: Add multiple episodes from a show
echo -e "${YELLOW}Test 2: Adding multiple episodes (The Office S01E01-E03)${NC}"
response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X POST "$API_URL" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "episodes": [
      {
        "imdbID": "tt0386676",
        "season": 1,
        "episode": 1,
        "timestampWatched": "'$(date -u -v-2d +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date -u -d "2 days ago" +"%Y-%m-%dT%H:%M:%SZ")'"
      },
      {
        "imdbID": "tt0386676",
        "season": 1,
        "episode": 2,
        "timestampWatched": "'$(date -u -v-1d +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date -u -d "1 day ago" +"%Y-%m-%dT%H:%M:%SZ")'"
      },
      {
        "imdbID": "tt0386676",
        "season": 1,
        "episode": 3,
        "timestampWatched": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"
      }
    ]
  }')

http_status=$(echo "$response" | grep "HTTP_STATUS" | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_STATUS/d')

if [ "$http_status" = "201" ]; then
    echo -e "${GREEN}✓ Success!${NC} Status: $http_status"
    echo "$body" | jq '.'
else
    echo -e "${RED}✗ Failed!${NC} Status: $http_status"
    echo "$body" | jq '.' 2>/dev/null || echo "$body"
fi
echo ""

# Test 3: Add a movie (for comparison)
echo -e "${YELLOW}Test 3: Adding a movie (The Matrix)${NC}"
response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X POST "$API_URL" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "movies": [
      {
        "imdbID": "tt0133093",
        "timestampWatched": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"
      }
    ]
  }')

http_status=$(echo "$response" | grep "HTTP_STATUS" | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_STATUS/d')

if [ "$http_status" = "201" ]; then
    echo -e "${GREEN}✓ Success!${NC} Status: $http_status"
    echo "$body" | jq '.'
else
    echo -e "${RED}✗ Failed!${NC} Status: $http_status"
    echo "$body" | jq '.' 2>/dev/null || echo "$body"
fi
echo ""

# Test 4: Add both movies and episodes in one request
echo -e "${YELLOW}Test 4: Adding both movies and episodes together${NC}"
response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X POST "$API_URL" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "movies": [
      {
        "imdbID": "tt0468569",
        "timestampWatched": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"
      }
    ],
    "episodes": [
      {
        "imdbID": "tt0944947",
        "season": 1,
        "episode": 1,
        "timestampWatched": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"
      }
    ]
  }')

http_status=$(echo "$response" | grep "HTTP_STATUS" | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_STATUS/d')

if [ "$http_status" = "201" ]; then
    echo -e "${GREEN}✓ Success!${NC} Status: $http_status"
    echo "$body" | jq '.'
else
    echo -e "${RED}✗ Failed!${NC} Status: $http_status"
    echo "$body" | jq '.' 2>/dev/null || echo "$body"
fi
echo ""

# Test 5: Retrieve history
echo -e "${YELLOW}Test 5: Retrieving watch history (last 5 items)${NC}"
response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X GET "$API_URL?limit=5" \
  -H "Authorization: Bearer $TOKEN")

http_status=$(echo "$response" | grep "HTTP_STATUS" | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_STATUS/d')

if [ "$http_status" = "200" ]; then
    echo -e "${GREEN}✓ Success!${NC} Status: $http_status"
    echo "$body" | jq '.'
else
    echo -e "${RED}✗ Failed!${NC} Status: $http_status"
    echo "$body" | jq '.' 2>/dev/null || echo "$body"
fi
echo ""

echo "=========================================="
echo "Tests completed!"
echo "=========================================="
echo ""
echo "Example curl commands for manual testing:"
echo ""
echo "# Add a single episode:"
echo "curl -X POST http://localhost:3000/api/history \\"
echo "  -H \"Authorization: Bearer YOUR_TOKEN\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"episodes\": [{\"imdbID\": \"tt0959621\", \"season\": 1, \"episode\": 1}]}'"
echo ""
echo "# Get history:"
echo "curl -X GET http://localhost:3000/api/history?limit=10 \\"
echo "  -H \"Authorization: Bearer YOUR_TOKEN\""
