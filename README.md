# Nacho Server (Self-Hostable)

## Steps to run on a fresh VPS / Bare metal server

aa

1. Update all
   ```bash
   sudo apt update && sudo apt upgrade -y && sudo apt full-upgrade -y
   ```
2. Install Docker

   ```bash
   sudo apt install ca-certificates curl
   sudo install -m 0755 -d /etc/apt/keyrings
   sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
   sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the repository to Apt sources:
    sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
    Types: deb
    URIs: https://download.docker.com/linux/ubuntu
    Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
    Components: stable
    Signed-By: /etc/apt/keyrings/docker.asc
    EOF

    sudo apt update
    sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

3. Install Caddy

   ```bash
   sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
   chmod o+r /usr/share/keyrings/caddy-stable-archive-keyring.gpg
   chmod o+r /etc/apt/sources.list.d/caddy-stable.list
   sudo apt update
   sudo apt install caddy
   ```

   Because it is installed via apt, Caddy will run as a system service and is enabled on boot by default.

4. Clone the repo

   ```bash
   git clone https://github.com/nacho-time/nacho-time-server.git
    cd nacho-time-server
   ```

5. Create a `.env` file

   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file to set your desired configuration, including `DOMAIN_NAME`, `TMDB_API_KEY`, and `PROWLARR_API_KEY`.

   Make sure `DOMAIN_NAME` points to your server's domain or IP address.

   Example:

   ```
   DOMAIN_NAME=http://yourdomain.com
   TMDB_API_KEY=your_tmdb_api_key
   PROWLARR_API_KEY=your_prowlarr_api_key
   POSTGRES_USER=nacho
   POSTGRES_PASSWORD=nacho_password
   POSTGRES_DB=nacho_db
   DATABASE_URL=postgresql://nacho:nacho_password@postgres:5432/nacho_db?schema=public
   ```

6. Edit the Caddyfile

   ```bash
   sudo nano Caddyfile
   ```

   Replace the contents with:

   ```
   yourdomain.com {
       reverse_proxy localhost:8123
   }
   ```

   Replace `yourdomain.com` with your actual domain or server IP.
   Don't forget to validate the Caddyfile syntax:

   ```bash
   sudo caddy validate --config Caddyfile
   ```

7. Move Caddyfile to /etc/caddy/Caddyfile (or manually append if you are using other sites)
   ```bash
   sudo cp Caddyfile /etc/caddy/Caddyfile
   ```
8. Restart Caddy to apply changes

   ```bash
   sudo systemctl restart caddy
   ```

9. Start Nacho Server with Docker Compose

   ```bash
   sudo docker compose up
   ```

10. Connect to your prowlarr instance via `http://prowlarr.yourdomain.com/` (or your server IP) and set up your indexers as needed. You will also need to head into Settings -> General and retrieve your prowlarr API key to add to your Nacho Server `.env` file under `PROWLARR_API_KEY`. This unfortunately cannot be automated due to prowlarr security restrictions.

11. Restart Nacho Server to apply the new API key

```bash
sudo docker compose up
```

12. Verify everything is working by accessing Nacho Server at `http://nacho.yourdomain.com/` (or your server IP). You can then run the following to enable the docker containers to start on boot:

```bash
 SERVICE_NAME="nacho-server"
 WORKDIR="$(pwd)"

 sudo bash -c "cat > /etc/systemd/system/${SERVICE_NAME}.service" <<EOF
 [Unit]
 Description=Docker Compose Service: ${SERVICE_NAME}
 Requires=docker.service
 After=docker.service

 [Service]
 Type=oneshot
 WorkingDirectory=${WORKDIR}
 ExecStart=/usr/bin/docker compose up -d
 ExecStop=/usr/bin/docker compose down
 RemainAfterExit=yes

 [Install]
 WantedBy=multi-user.target
 EOF

 sudo systemctl daemon-reload
 sudo systemctl enable ${SERVICE_NAME}
 sudo systemctl start ${SERVICE_NAME}
```
