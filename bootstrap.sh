#!/bin/bash
# Codelexin bootstrap — run directly on the EC2 instance.
# Safe to re-run (idempotent). Logs to /var/log/codelexin-bootstrap.log
set -euo pipefail
exec > >(tee /var/log/codelexin-bootstrap.log | logger -t codelexin-bootstrap) 2>&1

API_DOMAIN="api.lexinflow.com"
VOICE_DOMAIN="voice.lexinflow.com"
APP_DIR="/opt/codelexin"

echo "=== Codelexin bootstrap started ==="

# ── System packages ────────────────────────────────────────────────────────────
dnf install -y python3.12 python3.12-pip python3.12-devel git nginx certbot python3-certbot-nginx

# ── EBS data volume ────────────────────────────────────────────────────────────
if [ -b /dev/xvdf ]; then
  if ! blkid /dev/xvdf; then
    echo "Formatting /dev/xvdf as ext4..."
    mkfs -t ext4 /dev/xvdf
  fi
  mkdir -p /data
  if ! mountpoint -q /data; then
    mount /dev/xvdf /data
    grep -q '/dev/xvdf' /etc/fstab || \
      echo "/dev/xvdf  /data  ext4  defaults,nofail  0  2" >> /etc/fstab
  fi
else
  echo "WARNING: /dev/xvdf not found — using local /data (no persistent EBS)"
  mkdir -p /data
fi

mkdir -p /data/control-plane/db /data/control-plane/storage
mkdir -p /data/data-plane/db /data/data-plane/storage/transcripts /data/data-plane/storage/recordings

# ── App user ───────────────────────────────────────────────────────────────────
useradd -m -s /bin/bash codelexin 2>/dev/null || true
chown -R codelexin:codelexin /data

# ── Repo ───────────────────────────────────────────────────────────────────────
if [ -d "$APP_DIR/.git" ]; then
  echo "Repo exists — pulling latest..."
  git -C "$APP_DIR" pull
else
  git clone https://github.com/UllasSankhla/lexin.git "$APP_DIR"
fi
chown -R codelexin:codelexin "$APP_DIR"

# ── Python virtual environments ────────────────────────────────────────────────
cd "$APP_DIR/control-plane"
python3.12 -m venv .venv
.venv/bin/pip install --quiet -r requirements.txt

cd "$APP_DIR/data-plane"
python3.12 -m venv .venv
.venv/bin/pip install --quiet -r requirements.txt

# ── Secrets directory ──────────────────────────────────────────────────────────
mkdir -p /etc/codelexin
chmod 750 /etc/codelexin

# Only write env templates if they don't already exist (preserve filled-in secrets)
if [ ! -f /etc/codelexin/control-plane.env ]; then
  cat > /etc/codelexin/control-plane.env <<ENVEOF
APP_ENV=production
APP_HOST=127.0.0.1
APP_PORT=8000
DATABASE_URL=sqlite:////data/control-plane/db/control_plane.db
STORAGE_BASE_PATH=/data/control-plane/storage
LOG_LEVEL=INFO
CORS_ORIGINS=https://${API_DOMAIN},https://${VOICE_DOMAIN}

# ── Fill these in ──────────────────────────────────────────────────────────────
CONTROL_PLANE_API_KEY=CHANGE_ME_STRONG_RANDOM_SECRET
ENVEOF
fi

if [ ! -f /etc/codelexin/data-plane.env ]; then
  cat > /etc/codelexin/data-plane.env <<ENVEOF
APP_ENV=production
APP_HOST=127.0.0.1
APP_PORT=8001
DATABASE_URL=sqlite:////data/data-plane/db/data_plane.db
STORAGE_BASE_PATH=/data/data-plane/storage
TRANSCRIPTS_PATH=/data/data-plane/storage/transcripts
RECORDINGS_PATH=/data/data-plane/storage/recordings
CONTROL_PLANE_URL=http://127.0.0.1:8000
LOG_LEVEL=INFO
CORS_ORIGINS=https://${API_DOMAIN},https://${VOICE_DOMAIN}
CUSTOMER_KEYS_PATH=/etc/codelexin/customer_keys.json
CALENDLY_TIMEZONE=America/Los_Angeles
MAX_CONCURRENT_CALLS=10

# ── Fill these in ──────────────────────────────────────────────────────────────
CONTROL_PLANE_API_KEY=CHANGE_ME_STRONG_RANDOM_SECRET
DEEPGRAM_API_KEY=
CEREBRAS_API_KEY=
CALENDLY_API_KEY=
CALENDLY_SCHEDULING_LINK=
ENVEOF
fi

if [ ! -f /etc/codelexin/customer_keys.json ]; then
  cat > /etc/codelexin/customer_keys.json <<'KEYSEOF'
{
  "ck_live_REPLACE_WITH_RANDOM_32_CHARS": { "name": "Default Customer", "enabled": true }
}
KEYSEOF
fi

chmod 640 /etc/codelexin/*.env /etc/codelexin/customer_keys.json
chown root:codelexin /etc/codelexin/*.env /etc/codelexin/customer_keys.json

# ── systemd service: control-plane ────────────────────────────────────────────
cat > /etc/systemd/system/control-plane.service <<'SVCEOF'
[Unit]
Description=Codelexin Control Plane
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=codelexin
WorkingDirectory=/opt/codelexin/control-plane
EnvironmentFile=/etc/codelexin/control-plane.env
ExecStart=/opt/codelexin/control-plane/.venv/bin/uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1 \
  --loop uvloop \
  --no-access-log
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=control-plane

[Install]
WantedBy=multi-user.target
SVCEOF

# ── systemd service: data-plane ───────────────────────────────────────────────
cat > /etc/systemd/system/data-plane.service <<'SVCEOF'
[Unit]
Description=Codelexin Data Plane
After=network.target control-plane.service
Wants=control-plane.service

[Service]
Type=simple
User=codelexin
WorkingDirectory=/opt/codelexin/data-plane
EnvironmentFile=/etc/codelexin/data-plane.env
ExecStart=/opt/codelexin/data-plane/.venv/bin/uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8001 \
  --workers 1 \
  --loop uvloop \
  --no-access-log
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=data-plane

[Install]
WantedBy=multi-user.target
SVCEOF

# ── nginx configuration ────────────────────────────────────────────────────────
cat > /etc/nginx/conf.d/control-plane.conf <<NGINXEOF
server {
    listen 80;
    server_name ${API_DOMAIN};

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }
    location / {
        return 301 https://\$host\$request_uri;
    }
}

server {
    listen 443 ssl;
    server_name ${API_DOMAIN};

    ssl_certificate     /etc/letsencrypt/live/${API_DOMAIN}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/${API_DOMAIN}/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options SAMEORIGIN always;

    proxy_set_header Host              \$host;
    proxy_set_header X-Real-IP         \$remote_addr;
    proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto \$scheme;
    proxy_set_header X-Forwarded-Host  \$host;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_read_timeout 30s;
    }
    location /static/ {
        alias /opt/codelexin/frontend/;
        expires 1h;
    }
}
NGINXEOF

cat > /etc/nginx/conf.d/data-plane.conf <<NGINXEOF
server {
    listen 80;
    server_name ${VOICE_DOMAIN};

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }
    location / {
        return 301 https://\$host\$request_uri;
    }
}

server {
    listen 443 ssl;
    server_name ${VOICE_DOMAIN};

    ssl_certificate     /etc/letsencrypt/live/${VOICE_DOMAIN}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/${VOICE_DOMAIN}/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff always;

    proxy_set_header Host              \$host;
    proxy_set_header X-Real-IP         \$remote_addr;
    proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto \$scheme;
    proxy_set_header X-Forwarded-Host  \$host;

    location /api/ {
        if (\$request_method = OPTIONS) {
            add_header Access-Control-Allow-Origin * always;
            add_header Access-Control-Allow-Methods "GET, POST, OPTIONS" always;
            add_header Access-Control-Allow-Headers "Content-Type, X-Customer-Key" always;
            add_header Access-Control-Max-Age 86400 always;
            add_header Content-Length 0;
            return 204;
        }
        proxy_pass http://127.0.0.1:8001;
        proxy_read_timeout 30s;
        proxy_hide_header Access-Control-Allow-Origin;
        add_header Access-Control-Allow-Origin * always;
        add_header Access-Control-Allow-Headers "Content-Type, X-Customer-Key" always;
    }
    location /ws/ {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version      1.1;
        proxy_set_header Upgrade    \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout      660s;
        proxy_send_timeout      660s;
    }
}
NGINXEOF

mkdir -p /var/www/certbot
rm -f /etc/nginx/conf.d/default.conf

# Self-signed placeholder certs so nginx starts before certbot runs
for domain in "$API_DOMAIN" "$VOICE_DOMAIN"; do
  mkdir -p /etc/letsencrypt/live/"$domain"
  if [ ! -f /etc/letsencrypt/live/"$domain"/fullchain.pem ]; then
    openssl req -x509 -nodes -newkey rsa:2048 -days 1 \
      -keyout /etc/letsencrypt/live/"$domain"/privkey.pem \
      -out    /etc/letsencrypt/live/"$domain"/fullchain.pem \
      -subj   "/CN=$domain" 2>/dev/null
  fi
done

# ── Enable and start nginx ─────────────────────────────────────────────────────
systemctl daemon-reload
systemctl enable --now nginx
systemctl enable control-plane data-plane

echo ""
echo "=== Bootstrap complete ==="
echo ""
echo "Next steps:"
echo "  1. Fill in secrets:  sudo nano /etc/codelexin/control-plane.env"
echo "                       sudo nano /etc/codelexin/data-plane.env"
echo "  2. Add customer key: sudo nano /etc/codelexin/customer_keys.json"
echo "  3. Issue TLS cert:   sudo certbot --nginx -d $API_DOMAIN -d $VOICE_DOMAIN --non-interactive --agree-tos -m ops@lexinflow.com"
echo "  4. Start services:   sudo systemctl start control-plane data-plane"
echo "  5. Check health:     curl https://$API_DOMAIN/api/v1/health"
