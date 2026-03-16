# Codelexin — AWS Terraform Deployment

## Prerequisites

- Terraform >= 1.6
- AWS CLI configured (`aws configure`)
- An existing EC2 key pair in the target region
- Two DNS subdomains ready to point at the Elastic IP

## Quick start

```bash
cd terraform/

# 1. Copy and edit variables
cp terraform.tfvars.example terraform.tfvars
nano terraform.tfvars

# 2. Deploy infrastructure (~3 min)
terraform init
terraform plan
terraform apply

# 3. Point DNS A records to the Elastic IP shown in outputs
#    api.yourdomain.com   → <public_ip>
#    voice.yourdomain.com → <public_ip>

# 4. SSH in and populate secrets
ssh -i ~/.ssh/<key>.pem ec2-user@<public_ip>
sudo nano /etc/codelexin/control-plane.env   # fill in CONTROL_PLANE_API_KEY
sudo nano /etc/codelexin/data-plane.env      # fill in all API keys
sudo nano /etc/codelexin/customer_keys.json  # add customer keys

# 5. Issue TLS certificates (DNS must be propagated first)
sudo certbot --nginx \
  -d api.yourdomain.com \
  -d voice.yourdomain.com \
  --non-interactive --agree-tos -m you@yourdomain.com
sudo systemctl reload nginx

# 6. Start application services
sudo systemctl start control-plane data-plane
sudo systemctl status control-plane data-plane

# 7. Verify
curl https://api.yourdomain.com/api/v1/health
curl https://voice.yourdomain.com/api/v1/health
```

## terraform.tfvars example

```hcl
aws_region       = "us-west-2"
instance_type    = "t3.small"
key_pair_name    = "my-key-pair"
ssh_allowed_cidr = "203.0.113.10/32"   # your office IP
api_domain       = "api.yourdomain.com"
voice_domain     = "voice.yourdomain.com"
admin_email      = "ops@yourdomain.com"
```

## Customer key management

Keys live in `/etc/codelexin/customer_keys.json`:

```json
{
  "ck_live_abc123...": { "name": "Nexus Law",  "enabled": true },
  "ck_live_def456...": { "name": "Acme Corp",  "enabled": true },
  "ck_test_xyz789...": { "name": "Test Client","enabled": false }
}
```

Generate a key:
```bash
python3 -c "import secrets; print('ck_live_' + secrets.token_hex(20))"
```

Changes take effect immediately — no service restart needed.
Pass the key in the `X-Customer-Key` header when calling `POST /api/v1/calls/initiate`.

## Day-2 operations

| Task | Command |
|---|---|
| View logs | `sudo journalctl -fu control-plane` / `data-plane` |
| Restart a service | `sudo systemctl restart control-plane` |
| Update app code | `cd /opt/lexin && git pull && sudo systemctl restart control-plane data-plane` |
| Renew TLS certs (auto) | `sudo certbot renew --dry-run` |
| Check disk usage | `df -h /data` |
| Manual EBS snapshot | AWS Console → EC2 → Volumes → Create Snapshot |

## Architecture

```
Internet
  │
  ▼ 443/80
nginx (TLS termination, Let's Encrypt)
  ├── api.yourdomain.com   → :8000 control-plane (REST)
  └── voice.yourdomain.com → :8001 data-plane (REST + WebSocket)

Both services run as systemd units under the `codelexin` user.
SQLite databases and transcripts live on /data (persistent EBS volume).
Daily snapshots retained for 7 days via AWS DLM.
```
