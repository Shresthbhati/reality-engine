"""Utility to export the OpenAPI JSON specification and generate developer API documentation.

Phase 9 Task 9.11:
  - Extracts canonical OpenAPI 3.1 schema from the live FastAPI app.
  - Generates docs/openapi.json.
  - Generates docs/API_DOCUMENTATION.md with route catalog, parameters, responses, and examples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def export_openapi_schema() -> Dict[str, Any]:
    """Generate and return the OpenAPI schema from the FastAPI application."""
    from apps.api.main import app
    return app.openapi()


def write_openapi_json(dest_path: Path | str = "docs/openapi.json") -> Path:
    """Write the OpenAPI schema as formatted JSON."""
    path = Path(dest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = export_openapi_schema()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    return path


def generate_markdown_docs(dest_path: Path | str = "docs/API_DOCUMENTATION.md") -> Path:
    """Generate Markdown developer documentation from the OpenAPI schema."""
    schema = export_openapi_schema()
    path = Path(dest_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        f"# {schema.get('info', {}).get('title', 'Reality Engine API')}",
        "",
        f"> **Version**: {schema.get('info', {}).get('version', '0.1.0')} | **OpenAPI**: {schema.get('openapi', '3.1.0')}",
        "",
        "The Reality Engine API is an asynchronous HTTP/REST service for spatial reconstruction, canonical WorldIR management, multi-modal evidence ingestion, versioned persistence, and multi-format 3D export.",
        "",
        "## Base URL & Interactive Docs",
        "- **Base URL**: `http://localhost:8100`",
        "- **Interactive Swagger UI**: `/docs`",
        "- **ReDoc Specification**: `/redoc`",
        "- **OpenAPI Specification**: [`docs/openapi.json`](./openapi.json)",
        "",
        "## Authentication & Security",
        "- **Header**: `Authorization: Bearer <token>` (or `REALITY_AUTH_DISABLED=1` during offline development).",
        "- **CORS**: Configurable via `CORS_ORIGINS` environment variable.",
        "- **Rate & Size Limits**: Strict bounded streaming (`_read_bounded_upload`), path traversal validation in storage, and maximum file count enforcement.",
        "",
        "## Route Catalog",
        "",
    ]

    paths = schema.get("paths", {})
    by_tag: dict[str, list[tuple[str, str, dict]]] = {}

    for route_path, methods in paths.items():
        for method, details in methods.items():
            if method.lower() not in ("get", "post", "put", "delete", "patch"):
                continue
            tags = details.get("tags", ["General"])
            tag = tags[0] if tags else "General"
            by_tag.setdefault(tag, []).append((method.upper(), route_path, details))

    for tag in sorted(by_tag):
        lines.append(f"### {tag}")
        lines.append("")
        for method, route_path, details in sorted(by_tag[tag], key=lambda x: (x[1], x[0])):
            summary = details.get("summary") or details.get("description", "No description")
            summary_first_line = summary.split("\n")[0] if summary else ""
            lines.append(f"- **`{method} {route_path}`**: {summary_first_line}")
        lines.append("")

    lines.extend([
        "## Key Workflows",
        "",
        "### 1. Ingestion & Reconstruction",
        "```http",
        "POST /api/reconstructions HTTP/1.1",
        "Content-Type: multipart/form-data; boundary=----FormBoundary",
        "",
        "------FormBoundary",
        'Content-Disposition: form-data; name="files"; filename="photo_001.jpg"',
        "Content-Type: image/jpeg",
        "",
        "<bytes>",
        "------FormBoundary--",
        "```",
        "",
        "### 2. WorldIR Inspection & Versions",
        "- `GET /api/worlds/{id}/worldir`: Retrieve active canonical WorldIR JSON representation.",
        "- `GET /api/worlds/{id}/versions`: Version lineage and immutable commit history.",
        "- `POST /api/worlds/{id}/commit`: Commit human corrections or automated reconciliations.",
        "",
        "### 3. Multi-Format 3D & BIM Export",
        "- `POST /api/worlds/{id}/export`: Export current world to `gltf`, `glb`, `usda`, `obj`, `ply`, `stl`, `ifc`, `blender`, `cityjson`, `floorplan_svg`, `floorplan_dxf`, or `room_report_pdf`.",
        "",
        "## Error Semantics",
        "| Status Code | Description | Meaning |",
        "|---|---|---|",
        "| `400` | Bad Request | Malformed JSON or invalid parameter syntax |",
        "| `401` | Unauthorized | Missing or invalid Bearer token |",
        "| `404` | Not Found | Target world, session, evidence, or artifact does not exist |",
        "| `409` | Conflict | Optimistic concurrency conflict on WorldStore commit |",
        "| `413` | Payload Too Large | File upload exceeds maximum size or count ceiling |",
        "| `415` | Unsupported Media Type | Uploaded byte signature rejected by evidence sniffers |",
        "| `422` | Unprocessable Entity | Schema or geometry validation gate failure |",
        "",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


if __name__ == "__main__":
    json_path = write_openapi_json()
    print(f"Exported OpenAPI JSON to {json_path}")
    md_path = generate_markdown_docs()
    print(f"Generated API documentation to {md_path}")
