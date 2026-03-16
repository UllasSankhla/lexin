output "public_ip" {
  description = "Elastic IP — point your DNS A records here"
  value       = aws_eip.codelexin.public_ip
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.codelexin.id
}

output "data_volume_id" {
  description = "EBS data volume ID (back this up before any destructive operations)"
  value       = aws_ebs_volume.data.id
}

output "ssh_command" {
  description = "SSH into the instance"
  value       = "ssh -i ~/.ssh/${var.key_pair_name}.pem ec2-user@${aws_eip.codelexin.public_ip}"
}

output "next_steps" {
  description = "Post-deploy checklist"
  value = <<-EOT
    ── Post-deploy checklist ──────────────────────────────────────────────
    1. Point DNS:
         ${var.api_domain}   → ${aws_eip.codelexin.public_ip}
         ${var.voice_domain} → ${aws_eip.codelexin.public_ip}

    2. Wait for DNS to propagate, then issue TLS certificates:
         ssh ec2-user@${aws_eip.codelexin.public_ip}
         sudo certbot --nginx -d ${var.api_domain} -d ${var.voice_domain} \
           --non-interactive --agree-tos -m ${var.admin_email}
         sudo systemctl reload nginx

    3. Populate secrets (never committed to git):
         sudo nano /etc/codelexin/control-plane.env
         sudo nano /etc/codelexin/data-plane.env

    4. Add customer keys:
         sudo nano /etc/codelexin/customer_keys.json

    5. Restart services:
         sudo systemctl restart control-plane data-plane

    6. Verify health:
         curl https://${var.api_domain}/api/v1/health
         curl https://${var.voice_domain}/api/v1/health
    ───────────────────────────────────────────────────────────────────────
  EOT
}
