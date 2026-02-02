variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run"
  type        = string
  default     = "asia-northeast1"
}

variable "image" {
  description = "Container image to deploy"
  type        = string
}

variable "gemini_api_key" {
  description = "Gemini API key for summarization and embeddings"
  type        = string
  sensitive   = true
}

variable "qdrant_url" {
  description = "Qdrant Cloud URL"
  type        = string
}

variable "qdrant_api_key" {
  description = "Qdrant Cloud API key"
  type        = string
  sensitive   = true
}
