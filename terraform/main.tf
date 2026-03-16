terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Data sources ──────────────────────────────────────────────────────────────

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Latest Amazon Linux 2023 AMI
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ── Security group ────────────────────────────────────────────────────────────

resource "aws_security_group" "codelexin" {
  name        = "codelexin-${var.app_env}"
  description = "Codelexin voice booking platform"
  vpc_id      = data.aws_vpc.default.id

  # SSH — restricted to ops IP
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_allowed_cidr]
  }

  # HTTP — required only for Let's Encrypt ACME challenge
  ingress {
    description = "HTTP (ACME challenge)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTPS — all traffic
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "codelexin-${var.app_env}"
    Env  = var.app_env
  }
}

# ── EC2 instance ──────────────────────────────────────────────────────────────

resource "aws_instance" "codelexin" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.codelexin.id]
  subnet_id              = tolist(data.aws_subnets.default.ids)[0]

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_size_gb
    delete_on_termination = true
    encrypted             = true
  }

  user_data = templatefile("${path.module}/templates/user_data.sh.tpl", {
    api_domain   = var.api_domain
    voice_domain = var.voice_domain
    admin_email  = var.admin_email
    app_env      = var.app_env
  })

  tags = {
    Name = "codelexin-${var.app_env}"
    Env  = var.app_env
  }

  lifecycle {
    ignore_changes = [ami, user_data]  # prevent replacement on AMI updates
  }
}

# ── Elastic IP ────────────────────────────────────────────────────────────────

resource "aws_eip" "codelexin" {
  domain = "vpc"
  tags = {
    Name = "codelexin-${var.app_env}"
    Env  = var.app_env
  }
}

resource "aws_eip_association" "codelexin" {
  instance_id   = aws_instance.codelexin.id
  allocation_id = aws_eip.codelexin.id
}

# ── Persistent data EBS volume ────────────────────────────────────────────────
# Databases, transcripts, recordings — survives instance replacement.

resource "aws_ebs_volume" "data" {
  availability_zone = aws_instance.codelexin.availability_zone
  size              = var.data_volume_size_gb
  type              = "gp3"
  encrypted         = true

  tags = {
    Name = "codelexin-data-${var.app_env}"
    Env  = var.app_env
  }
}

resource "aws_volume_attachment" "data" {
  device_name  = "/dev/xvdf"
  volume_id    = aws_ebs_volume.data.id
  instance_id  = aws_instance.codelexin.id
  force_detach = false
}

# ── Automated EBS snapshots (daily, 7-day retention) ─────────────────────────

resource "aws_iam_role" "dlm" {
  name = "codelexin-dlm-${var.app_env}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "dlm.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "dlm" {
  role       = aws_iam_role.dlm.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSDataLifecycleManagerServiceRole"
}

resource "aws_dlm_lifecycle_policy" "data_snapshots" {
  description        = "Daily snapshots of Codelexin data volume"
  execution_role_arn = aws_iam_role.dlm.arn
  state              = "ENABLED"

  policy_details {
    resource_types = ["VOLUME"]

    target_tags = {
      Name = "codelexin-data-${var.app_env}"
    }

    schedule {
      name = "daily-7-day-retention"

      create_rule {
        interval      = 24
        interval_unit = "HOURS"
        times         = ["03:00"]
      }

      retain_rule {
        count = 7
      }

      copy_tags = true
    }
  }
}
