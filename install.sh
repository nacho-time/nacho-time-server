#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Nacho Time Server Installation ===${NC}\n"

# Prompt for domain name
read -p "Enter your domain name (e.g., example.com): " DOMAIN_NAME

if [ -z "$DOMAIN_NAME" ]; then
    echo -e "${RED}Error: Domain name cannot be empty${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Domain name: ${DOMAIN_NAME}${NC}\n"

# Check if Caddy is installed
echo "Checking if Caddy is installed..."
if ! command -v caddy &> /dev/null; then
    echo -e "${RED}Caddy is not installed. Installing Caddy...${NC}"
    
    # Detect OS and install Caddy
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
        curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
        curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
        sudo apt update
        sudo apt install -y caddy
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install caddy
        else
            echo -e "${RED}Homebrew not found. Please install Homebrew first: https://brew.sh${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Unsupported OS. Please install Caddy manually: https://caddyserver.com/docs/install${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Caddy installed successfully${NC}"
else
    echo -e "${GREEN}Caddy is already installed${NC}"
fi

# Check if nginx is installed and running
echo -e "\nChecking nginx status..."
if command -v nginx &> /dev/null; then
    echo -e "${YELLOW}nginx is installed${NC}"
    
    # Check if nginx is running
    if systemctl is-active --quiet nginx 2>/dev/null || pgrep nginx > /dev/null 2>&1; then
        echo -e "${YELLOW}nginx is currently running. Stopping and disabling nginx...${NC}"
        
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo systemctl stop nginx
            sudo systemctl disable nginx
            echo -e "${GREEN}nginx has been stopped and disabled${NC}"
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            sudo nginx -s stop 2>/dev/null || true
            # On macOS, disable nginx launch agent if it exists
            if [ -f ~/Library/LaunchAgents/homebrew.mxcl.nginx.plist ]; then
                launchctl unload ~/Library/LaunchAgents/homebrew.mxcl.nginx.plist 2>/dev/null || true
            fi
            echo -e "${GREEN}nginx has been stopped${NC}"
        fi
    else
        echo -e "${GREEN}nginx is not running${NC}"
    fi
    
    # Check if nginx is using port 80 or 443
    if lsof -Pi :80 -sTCP:LISTEN -t >/dev/null 2>&1 || lsof -Pi :443 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}Warning: Port 80 or 443 is still in use. Please ensure no other web server is running.${NC}"
    fi
else
    echo -e "${GREEN}nginx is not installed${NC}"
fi

# Modify Caddyfile with the provided domain
echo -e "\nUpdating Caddyfile with domain: ${DOMAIN_NAME}..."
cat > Caddyfile << EOF
nacho.${DOMAIN_NAME} {
    reverse_proxy localhost:8123
}

prowlarr.${DOMAIN_NAME} {
    reverse_proxy localhost:9696
}
EOF

echo -e "${GREEN}Caddyfile updated successfully${NC}"

# Check if Caddy is running and reload, otherwise start it
echo -e "\nConfiguring Caddy..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Enable and start Caddy service on Linux
    sudo systemctl enable caddy
    
    # Copy Caddyfile to /etc/caddy/ if it doesn't exist or is different
    sudo mkdir -p /etc/caddy
    sudo cp Caddyfile /etc/caddy/Caddyfile
    
    if systemctl is-active --quiet caddy; then
        echo "Reloading Caddy configuration..."
        sudo systemctl reload caddy
    else
        echo "Starting Caddy..."
        sudo systemctl start caddy
    fi
    
    echo -e "${GREEN}Caddy is running${NC}"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # On macOS, run Caddy with the local Caddyfile
    if pgrep caddy > /dev/null 2>&1; then
        echo "Reloading Caddy configuration..."
        caddy reload --config Caddyfile --adapter caddyfile
    else
        echo "Starting Caddy in the background..."
        caddy start --config Caddyfile --adapter caddyfile
    fi
    
    echo -e "${GREEN}Caddy is running${NC}"
fi

# Check if Docker is installed
echo -e "\nChecking if Docker is installed..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is available
if ! docker compose version &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Start Docker Compose services
echo -e "\nStarting Docker Compose services..."
docker compose up -d

echo -e "\n${GREEN}=== Installation Complete ===${NC}"
echo -e "\nYour services are now running:"
echo -e "  - Nacho Server: ${GREEN}https://nacho.${DOMAIN_NAME}${NC}"
echo -e "  - Prowlarr: ${GREEN}https://prowlarr.${DOMAIN_NAME}${NC}"
echo -e "  - FlareSolverr (internal): ${GREEN}http://flaresolverr:8191${NC}"
echo -e "\nMake sure your DNS records point to this server:"
echo -e "  - nacho.${DOMAIN_NAME} → $(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_SERVER_IP')"
echo -e "  - prowlarr.${DOMAIN_NAME} → $(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_SERVER_IP')"
echo -e "\nCaddy will automatically provision SSL certificates."
