# Reality Engine API

> **Version**: 0.1.0 | **OpenAPI**: 3.1.0

The Reality Engine API is an asynchronous HTTP/REST service for spatial reconstruction, canonical WorldIR management, multi-modal evidence ingestion, versioned persistence, and multi-format 3D export.

## Base URL & Interactive Docs
- **Base URL**: `http://localhost:8100`
- **Interactive Swagger UI**: `/docs`
- **ReDoc Specification**: `/redoc`
- **OpenAPI Specification**: [`docs/openapi.json`](./openapi.json)

## Authentication & Security
- **Header**: `Authorization: Bearer <token>` (or `REALITY_AUTH_DISABLED=1` during offline development).
- **CORS**: Configurable via `CORS_ORIGINS` environment variable.
- **Rate & Size Limits**: Strict bounded streaming (`_read_bounded_upload`), path traversal validation in storage, and maximum file count enforcement.

## Route Catalog

### activity

- **`GET /api/activity`**: List Activity

### assistant

- **`GET /api/assistant/tools`**: List Tools
- **`POST /api/worlds/{world_id}/assistant/call`**: Call
- **`POST /api/worlds/{world_id}/assistant/confirm`**: Confirm

### completion

- **`GET /api/worlds/{world_id}/completion/proposals`**: List Proposals
- **`POST /api/worlds/{world_id}/completion/proposals`**: Create Proposals
- **`POST /api/worlds/{world_id}/completion/proposals/{pid}/accept`**: Accept Proposal
- **`GET /api/worlds/{world_id}/completion/proposals/{pid}/diff`**: Proposal Diff
- **`POST /api/worlds/{world_id}/completion/proposals/{pid}/reject`**: Reject Proposal

### evidence

- **`GET /api/evidence`**: List Evidence
- **`DELETE /api/evidence/{evidence_id}`**: Delete Evidence
- **`GET /api/evidence/{evidence_id}`**: Get Evidence
- **`GET /api/evidence/{evidence_id}/artifact`**: Evidence Artifact

### evidence-quality

- **`GET /api/evidence/quality`**: List Quality
- **`PUT /api/evidence/{evidence_id}/decision`**: Put Decision
- **`GET /api/evidence/{evidence_id}/quality`**: Get Quality

### export

- **`POST /api/worlds/{world_id}/export`**: Export World
- **`GET /api/worlds/{world_id}/export/formats`**: Export Formats
- **`GET /api/worlds/{world_id}/export/{content_hash}/download`**: Download Export

### geo

- **`POST /api/geo/context`**: Import Context

### health

- **`GET /api/health`**: Health Check
- **`GET /api/status`**: Status
- **`GET /healthz`**: Healthz Check

### intent

- **`POST /api/intent/estimate`**: Estimate
- **`GET /api/worlds/{world_id}/intent`**: Get Intent
- **`PUT /api/worlds/{world_id}/intent`**: Put Intent

### jobs

- **`GET /api/jobs`**: List Jobs
- **`GET /api/jobs/{job_id}`**: Get Job
- **`POST /api/jobs/{job_id}/cancel`**: Cancel Job
- **`GET /api/jobs/{job_id}/events`**: Job Events

### measurements

- **`GET /api/worlds/{world_id}/measurements`**: List Measurements
- **`POST /api/worlds/{world_id}/measurements`**: Create Measurement
- **`DELETE /api/worlds/{world_id}/measurements/{measurement_id}`**: Delete Measurement

### notifications

- **`GET /api/notifications`**: List Notifications

### procedural

- **`GET /api/procedural/templates`**: List Room Templates
- **`POST /api/worlds/{world_id}/apartments`**: Create Apartment
- **`POST /api/worlds/{world_id}/apartments/solve`**: Solve Apartment
- **`DELETE /api/worlds/{world_id}/apartments/{name}`**: Delete Apartment
- **`PUT /api/worlds/{world_id}/apartments/{name}`**: Edit Apartment
- **`POST /api/worlds/{world_id}/buildings`**: Create Building
- **`DELETE /api/worlds/{world_id}/buildings/{name}`**: Delete Building
- **`PUT /api/worlds/{world_id}/buildings/{name}`**: Edit Building
- **`POST /api/worlds/{world_id}/cities`**: Create City
- **`DELETE /api/worlds/{world_id}/cities/{name}`**: Delete City
- **`PUT /api/worlds/{world_id}/cities/{name}`**: Edit City
- **`POST /api/worlds/{world_id}/corridors`**: Create Corridor
- **`DELETE /api/worlds/{world_id}/corridors/{name}`**: Delete Corridor
- **`PUT /api/worlds/{world_id}/corridors/{name}`**: Edit Corridor
- **`POST /api/worlds/{world_id}/districts`**: Create District
- **`DELETE /api/worlds/{world_id}/districts/{name}`**: Delete District
- **`PUT /api/worlds/{world_id}/districts/{name}`**: Edit District
- **`GET /api/worlds/{world_id}/generated`**: List Generated
- **`POST /api/worlds/{world_id}/rooms`**: Create Room
- **`DELETE /api/worlds/{world_id}/rooms/{name}`**: Delete Room
- **`PUT /api/worlds/{world_id}/rooms/{name}`**: Edit Room

### proposals

- **`GET /api/worlds/{world_id}/proposals`**: List World Proposals
- **`POST /api/worlds/{world_id}/proposals`**: Create Proposal
- **`GET /api/worlds/{world_id}/proposals/{proposal_id}`**: Read Proposal
- **`POST /api/worlds/{world_id}/proposals/{proposal_id}/accept`**: Accept
- **`POST /api/worlds/{world_id}/proposals/{proposal_id}/reject`**: Reject

### query

- **`GET /api/worlds/{world_id}/query/container/{entity_id}`**: Query Container
- **`GET /api/worlds/{world_id}/query/contents/{entity_id}`**: Query Contents
- **`GET /api/worlds/{world_id}/query/nearest`**: Query Nearest
- **`GET /api/worlds/{world_id}/query/within_radius`**: Query Within Radius

### reconstructions

- **`POST /api/reconstructions`**: Create Reconstruction
- **`GET /api/worlds/{world_id}/capture-coverage`**: Get Capture Coverage
- **`GET /api/worlds/{world_id}/status`**: Get World Status

### registration

- **`POST /api/registration/pose-graph`**: Solve Pose Graph
- **`POST /api/registration/session-alignments`**: Refine Alignments

### sessions

- **`GET /api/sessions`**: List Sessions
- **`POST /api/sessions`**: Create Session
- **`GET /api/sessions/{session_id}`**: Get Session
- **`GET /api/sessions/{session_id}/location`**: Session Location
- **`POST /api/sessions/{session_id}/reconstruct`**: Reconstruct Session
- **`GET /api/sessions/{session_id}/trajectory`**: Session Trajectory

### sites

- **`GET /api/sites`**: List Sites
- **`POST /api/sites`**: Create Site
- **`DELETE /api/sites/{site_id}`**: Delete Site
- **`GET /api/sites/{site_id}`**: Get Site
- **`GET /api/sites/{site_id}/anchor`**: Get Anchor
- **`PUT /api/sites/{site_id}/anchor`**: Put Anchor
- **`POST /api/sites/{site_id}/compose`**: Compose Site
- **`POST /api/sites/{site_id}/placements`**: Add Placement
- **`DELETE /api/sites/{site_id}/placements/{placement_id}`**: Delete Placement
- **`PATCH /api/sites/{site_id}/placements/{placement_id}`**: Patch Placement

### system

- **`GET /api/system/verification`**: Get Verification

### uploads

- **`POST /api/uploads`**: Create Upload
- **`POST /api/uploads/check`**: Check Uploads

### worlds

- **`GET /api/worlds`**: List Worlds
- **`POST /api/worlds`**: Create World
- **`DELETE /api/worlds/{world_id}`**: Delete World
- **`GET /api/worlds/{world_id}`**: Get World
- **`PATCH /api/worlds/{world_id}`**: Patch World
- **`POST /api/worlds/{world_id}/archive`**: Archive World
- **`GET /api/worlds/{world_id}/artifacts/{sha256}`**: World Artifact
- **`POST /api/worlds/{world_id}/attach/{session_id}`**: Attach Session
- **`GET /api/worlds/{world_id}/cameras`**: World Cameras
- **`POST /api/worlds/{world_id}/commit`**: Commit World Correction
- **`GET /api/worlds/{world_id}/coverage`**: World Coverage
- **`GET /api/worlds/{world_id}/diff`**: Get World Diff
- **`POST /api/worlds/{world_id}/duplicate`**: Duplicate World
- **`GET /api/worlds/{world_id}/entities/{entity_id}/provenance`**: Entity Provenance
- **`GET /api/worlds/{world_id}/points`**: World Points
- **`GET /api/worlds/{world_id}/quality`**: World Quality
- **`GET /api/worlds/{world_id}/query/spaces`**: Query World Spaces
- **`GET /api/worlds/{world_id}/report`**: World Report
- **`POST /api/worlds/{world_id}/restore`**: Restore World
- **`POST /api/worlds/{world_id}/rollback`**: Rollback World
- **`POST /api/worlds/{world_id}/scale`**: Set World Scale
- **`GET /api/worlds/{world_id}/scale/suggestion`**: Get Scale Suggestion
- **`POST /api/worlds/{world_id}/scale/suggestion/reject`**: Reject Scale Suggestion
- **`GET /api/worlds/{world_id}/space-graph`**: Get World Space Graph
- **`GET /api/worlds/{world_id}/thumbnail`**: World Thumbnail
- **`GET /api/worlds/{world_id}/versions`**: List World Versions
- **`GET /api/worlds/{world_id}/worldir`**: World Worldir

## Key Workflows

### 1. Ingestion & Reconstruction
```http
POST /api/reconstructions HTTP/1.1
Content-Type: multipart/form-data; boundary=----FormBoundary

------FormBoundary
Content-Disposition: form-data; name="files"; filename="photo_001.jpg"
Content-Type: image/jpeg

<bytes>
------FormBoundary--
```

### 2. WorldIR Inspection & Versions
- `GET /api/worlds/{id}/worldir`: Retrieve active canonical WorldIR JSON representation.
- `GET /api/worlds/{id}/versions`: Version lineage and immutable commit history.
- `POST /api/worlds/{id}/commit`: Commit human corrections or automated reconciliations.

### 3. Multi-Format 3D & BIM Export
- `POST /api/worlds/{id}/export`: Export current world to `gltf`, `glb`, `usda`, `obj`, `ply`, `stl`, `ifc`, `blender`, `cityjson`, `floorplan_svg`, `floorplan_dxf`, or `room_report_pdf`.

## Error Semantics
| Status Code | Description | Meaning |
|---|---|---|
| `400` | Bad Request | Malformed JSON or invalid parameter syntax |
| `401` | Unauthorized | Missing or invalid Bearer token |
| `404` | Not Found | Target world, session, evidence, or artifact does not exist |
| `409` | Conflict | Optimistic concurrency conflict on WorldStore commit |
| `413` | Payload Too Large | File upload exceeds maximum size or count ceiling |
| `415` | Unsupported Media Type | Uploaded byte signature rejected by evidence sniffers |
| `422` | Unprocessable Entity | Schema or geometry validation gate failure |
