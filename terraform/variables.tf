variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-west-2"
}

variable "instance_type" {
  description = "EC2 instance type (t3.small = 2vCPU/2GB, t3.medium = 2vCPU/4GB)"
  type        = string
  default     = "t3.small"
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair for SSH access"
  type        = string
}

variable "ssh_allowed_cidr" {
  description = "CIDR block allowed SSH access (restrict to your office/VPN IP)"
  type        = string
  default     = "0.0.0.0/0"  # narrow this down in production
}

variable "api_domain" {
  description = "Domain for the control plane (e.g. api.yourdomain.com)"
  type        = string
}

variable "voice_domain" {
  description = "Domain for the data plane (e.g. voice.yourdomain.com)"
  type        = string
}

variable "admin_email" {
  description = "Email for Let's Encrypt certificate notifications"
  type        = string
}

variable "data_volume_size_gb" {
  description = "Size of the persistent EBS data volume in GB"
  type        = number
  default     = 20
}

variable "root_volume_size_gb" {
  description = "Size of the root EBS volume in GB"
  type        = number
  default     = 20
}

variable "app_env" {
  description = "Application environment tag"
  type        = string
  default     = "production"
}
