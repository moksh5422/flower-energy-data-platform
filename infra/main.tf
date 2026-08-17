terraform {
  required_version = ">= 1.5.0"
}

variable "environment" {
  default = "dev"
}

output "environment" {
  value = var.environment
}

# Production mapping:
# object storage -> ADLS/S3
# Spark -> Databricks
# secrets -> Key Vault/Secrets Manager
# monitoring -> Azure Monitor/CloudWatch
