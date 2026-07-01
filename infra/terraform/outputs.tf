output "artifacts_bucket" {
  description = "S3 bucket for model artifacts."
  value       = aws_s3_bucket.artifacts.bucket
}

output "data_bucket" {
  description = "S3 bucket for preprocessed data."
  value       = aws_s3_bucket.data.bucket
}

output "ecr_repository_url" {
  description = "ECR repository URL for the serving image."
  value       = aws_ecr_repository.serving.repository_url
}

output "inference_public_dns" {
  description = "Public DNS of the inference host (serving app on the configured port)."
  value       = aws_instance.inference.public_dns
}

output "log_group" {
  description = "CloudWatch log group for the serving app."
  value       = aws_cloudwatch_log_group.service.name
}
