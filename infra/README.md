# Infrastructure (Terraform)

Terraform skeleton to provision the STOFS-GNN serving stack on AWS. **It is intended to be
validated and planned, not applied as-is.**

> ## Cost warning
> Running `terraform apply` provisions **billable AWS resources** (an EC2 instance, S3
> buckets, an ECR repository, a CloudWatch log group). Always run `terraform plan` and
> review it before `apply`, and `terraform destroy` when finished. Order-of-magnitude for
> the defaults (one `t3.large`, two S3 buckets, one ECR repo, one log group) is a few US$
> per day if left running — dominated by the EC2 instance; a GPU instance (`g4dn.xlarge`)
> is several times that. Confirm current prices for your region.

## What it provisions
- **S3** — an artifacts bucket (versioned) for model checkpoints and a data bucket, both
  with public access blocked.
- **ECR** — a repository for the serving container image (scan-on-push).
- **EC2** — a single inference host (CPU by default; set `inference_instance_type` to a
  `g4dn.*` for GPU) with an IAM instance profile granting read access to the artifacts
  bucket, behind a security group exposing only the serving port (IMDSv2 enforced).
- **CloudWatch** — a log group for the serving app.

GCP equivalents: S3 → GCS, ECR → Artifact Registry, EC2 → Compute Engine, CloudWatch →
Cloud Logging.

## Usage (validate / plan only)
```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # then edit (bucket names must be unique)

terraform init
terraform fmt -check
terraform validate
terraform plan                                 # review carefully
# terraform apply    # <- ONLY after reviewing the plan and accepting the cost
```

CI runs `fmt -check` + `init -backend=false` + `validate` (no credentials, no apply).

Never commit `terraform.tfvars`, `*.tfstate`, or credentials (see `.gitignore`). For real
deployments use a remote backend with locking (e.g. S3 + DynamoDB) rather than local state.
