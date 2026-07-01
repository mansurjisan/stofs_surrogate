# STOFS-GNN cloud deployment skeleton (AWS).
#
# WARNING: applying this provisions billable AWS resources. This repo is intended to be
# validated/planned only -- run `terraform validate` and `terraform plan` and review the
# plan before ever running `terraform apply`. See infra/README.md for the cost estimate.
#
# For GCP the equivalent mapping is: S3 -> GCS, ECR -> Artifact Registry, EC2 -> Compute
# Engine, CloudWatch -> Cloud Logging.

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = merge({
      Project     = var.project
      Environment = var.environment
      ManagedBy   = "terraform"
    }, var.tags)
  }
}

locals {
  name_prefix = "${var.project}-${var.environment}"
}

# --- Object storage: model artifacts + data --------------------------------
resource "aws_s3_bucket" "artifacts" {
  bucket = var.artifacts_bucket_name
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "data" {
  bucket = var.data_bucket_name
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- Container registry for the serving image ------------------------------
resource "aws_ecr_repository" "serving" {
  name                 = "${local.name_prefix}-serving"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# --- IAM: inference host role with read access to the artifacts bucket -----
data "aws_iam_policy_document" "assume_ec2" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "inference" {
  name               = "${local.name_prefix}-inference"
  assume_role_policy = data.aws_iam_policy_document.assume_ec2.json
}

data "aws_iam_policy_document" "artifacts_read" {
  statement {
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
  }
}

resource "aws_iam_role_policy" "artifacts_read" {
  name   = "${local.name_prefix}-artifacts-read"
  role   = aws_iam_role.inference.id
  policy = data.aws_iam_policy_document.artifacts_read.json
}

resource "aws_iam_instance_profile" "inference" {
  name = "${local.name_prefix}-inference"
  role = aws_iam_role.inference.name
}

# --- Networking: security group for the service ----------------------------
data "aws_vpc" "default" {
  default = true
}

resource "aws_security_group" "inference" {
  name        = "${local.name_prefix}-inference"
  description = "Inbound access to the STOFS-GNN serving port."
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "Serving port"
    from_port   = var.service_port
    to_port     = var.service_port
    protocol    = "tcp"
    cidr_blocks = [var.service_ingress_cidr]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- Compute: inference host (runs the serving container) ------------------
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

resource "aws_instance" "inference" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.inference_instance_type
  iam_instance_profile   = aws_iam_instance_profile.inference.name
  vpc_security_group_ids = [aws_security_group.inference.id]

  metadata_options {
    http_tokens = "required" # enforce IMDSv2
  }

  tags = {
    Name = "${local.name_prefix}-inference"
  }
}

# --- Monitoring: CloudWatch log group for the serving app ------------------
resource "aws_cloudwatch_log_group" "service" {
  name              = "/${var.project}/${var.environment}/serving"
  retention_in_days = var.log_retention_days
}
