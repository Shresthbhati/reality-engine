"""Tests for deployment configuration (Docker / docker-compose) and OpenAPI documentation (Tasks 9.10 & 9.11)."""

import json
from pathlib import Path
import pytest
import yaml

from apps.api.export_openapi import export_openapi_schema, generate_markdown_docs, write_openapi_json


def test_dockerfile_backend_configuration():
    dockerfile_path = Path("Dockerfile")
    assert dockerfile_path.exists(), "Dockerfile must exist at repository root"
    content = dockerfile_path.read_text(encoding="utf-8")

    assert "FROM python:" in content
    assert "AS runtime-base" in content
    assert "AS builder" in content
    assert "AS runner" in content
    assert "USER appuser" in content, "Backend Dockerfile must run as non-root user"
    assert "EXPOSE 8100" in content, "Backend Dockerfile must expose port 8100"
    assert "HEALTHCHECK" in content, "Backend Dockerfile must declare HEALTHCHECK"
    assert "/healthz" in content


def test_dockerfile_frontend_configuration():
    dockerfile_path = Path("Dockerfile.frontend")
    assert dockerfile_path.exists(), "Dockerfile.frontend must exist at repository root"
    content = dockerfile_path.read_text(encoding="utf-8")

    assert "FROM node:20-alpine" in content
    assert "AS deps" in content
    assert "AS builder" in content
    assert "AS runner" in content
    assert "USER nextjs" in content, "Frontend Dockerfile must run as non-root user"
    assert "EXPOSE 3000" in content, "Frontend Dockerfile must expose port 3000"
    assert "HEALTHCHECK" in content, "Frontend Dockerfile must declare HEALTHCHECK"


def test_docker_compose_configuration():
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists(), "docker-compose.yml must exist at repository root"
    content = compose_path.read_text(encoding="utf-8")

    data = yaml.safe_load(content)
    assert "services" in data, "docker-compose must declare services"
    assert "api" in data["services"], "docker-compose must declare 'api' service"
    assert "studio" in data["services"], "docker-compose must declare 'studio' service"
    assert "volumes" in data, "docker-compose must declare volumes"
    assert "reality_data" in data["volumes"], "docker-compose must declare persistent reality_data volume"

    # API configuration
    api_service = data["services"]["api"]
    assert "8100:8100" in api_service.get("ports", [])
    assert "healthcheck" in api_service

    # Studio configuration
    studio_service = data["services"]["studio"]
    assert "3000:3000" in studio_service.get("ports", [])
    assert "api" in studio_service.get("depends_on", {})


def test_dockerignore_configuration():
    ignore_path = Path(".dockerignore")
    assert ignore_path.exists(), ".dockerignore must exist at repository root"
    content = ignore_path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

    assert ".git" in lines
    assert "node_modules" in lines
    assert ".next" in lines
    assert "__pycache__" in lines


def test_openapi_schema_generation_and_export():
    schema = export_openapi_schema()
    assert isinstance(schema, dict)
    assert schema.get("openapi", "").startswith("3.")
    assert "info" in schema
    assert schema["info"].get("title") == "Reality Engine API"
    assert "paths" in schema

    paths = schema["paths"]
    assert "/api/reconstructions" in paths
    assert "/api/worlds" in paths
    assert "/api/worlds/{world_id}/export" in paths
    assert "/healthz" in paths

    # Test file writing
    json_path = write_openapi_json("docs/openapi.json")
    assert json_path.exists()
    assert json_path.stat().st_size > 50_000

    md_path = generate_markdown_docs("docs/API_DOCUMENTATION.md")
    assert md_path.exists()
    assert md_path.stat().st_size > 2_000
    md_content = md_path.read_text(encoding="utf-8")
    assert "# Reality Engine API" in md_content
    assert "Route Catalog" in md_content
