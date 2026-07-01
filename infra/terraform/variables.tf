variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Project name; used as a prefix for resource names and tags."
  type        = string
  default     = "stofs-gnn"
}

variable "environment" {
  description = "Deployment environment (e.g. dev, staging, prod)."
  type        = string
  default     = "dev"
}

variable "artifacts_bucket_name" {
  description = "S3 bucket for model artifacts (must be globally unique)."
  type        = string
}

variable "data_bucket_name" {
  description = "S3 bucket for preprocessed training/validation data (must be globally unique)."
  type        = string
}

variable "inference_instance_type" {
  description = "EC2 instance type for inference. Default is a small CPU instance; use e.g. g4dn.xlarge for GPU."
  type        = string
  default     = "t3.large"
}

variable "service_ingress_cidr" {
  description = "CIDR allowed to reach the service port. Restrict this; 0.0.0.0/0 opens it to the world."
  type        = string
  default     = "0.0.0.0/0"
}

variable "service_port" {
  description = "Port the FastAPI serving container listens on."
  type        = number
  default     = 8000
}

variable "log_retention_days" {
  description = "CloudWatch log retention (days)."
  type        = number
  default     = 14
}

variable "tags" {
  description = "Extra tags applied to all resources."
  type        = map(string)
  default     = {}
}
