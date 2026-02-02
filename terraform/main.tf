terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_cloud_run_service" "api" {
  name     = "atcoder-similarity-search-api"
  location = var.region

  template {
    spec {
      containers {
        image = var.image

        env {
          name  = "GEMINI_API_KEY"
          value = var.gemini_api_key
        }

        env {
          name  = "OPENAI_API_KEY"
          value = var.openai_api_key
        }

        env {
          name  = "QDRANT_URL"
          value = var.qdrant_url
        }

        env {
          name  = "QDRANT_API_KEY"
          value = var.qdrant_api_key
        }

        resources {
          limits = {
            cpu    = "1000m"
            memory = "512Mi"
          }
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

resource "google_cloud_run_service_iam_member" "public" {
  service  = google_cloud_run_service.api.name
  location = google_cloud_run_service.api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
