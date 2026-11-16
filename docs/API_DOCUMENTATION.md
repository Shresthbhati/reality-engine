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

### account

- **`DELETE /api/account/data`**: Delete My Data
- **`POST /api/account/delete-my-data`**: Delete My Data
- **`GET /api/account/invites`**: List Invites
- **`POST /api/account/invites`**: Create Invite
- **`DELETE /api/account/invites/{invite_id}`**: Revoke Invite
- **`GET /api/account/me`**: Get Current Account

### activity

- **`GET /api/activity`**: List Activity

### adoptions

- **`POST /api/worlds/{world_id}/adoptions`**: Propose Adoption
- **`GET /api/worlds/{world_id}/adoptions/{adoption_id}`**: Get Adoption Proposal
- **`POST /api/worlds/{world_id}/adoptions/{adoption_id}/commit`**: Commit Adoption
- **`POST /api/worlds/{world_id}/adoptions/{adoption_id}/reject`**: Reject Adoption

### assets

- **`GET /api/assets`**: List Assets
- **`GET /api/assets/{asset_id}`**: Get Asset
- **`GET /api/assets/{asset_id}/download`**: Download Asset
- **`POST /api/worlds/{world_id}/extract-asset`**: Extract World Asset

### assistant

- **`GET /api/assistant/tools`**: List Tools
- **`POST /api/worlds/{world_id}/assistant/ask`**: Ask Copilot
- **`POST /api/worlds/{world_id}/assistant/call`**: Call
- **`POST /api/worlds/{world_id}/assistant/capture-plan`**: Plan Capture
- **`POST /api/worlds/{world_id}/assistant/confirm`**: Confirm

### auth

- **`GET /api/auth/invites/{token}`**: Inspect Invite
- **`POST /api/auth/invites/{token}/accept`**: Accept Invite
- **`POST /api/auth/login`**: Login
- **`POST /api/auth/signup`**: Signup

### billing

- **`POST /api/billing/checkout`**: Create Subscription Checkout
- **`POST /api/billing/portal`**: Create Billing Portal Session
- **`GET /api/billing/status`**: Get Current Tenant Billing Status
- **`GET /api/billing/tiers`**: List Pricing Tiers
- **`POST /api/billing/webhook`**: Handle Stripe Webhook

### clash_bcf

- **`GET /api/worlds/{world_id}/bcf/export`**: Export Bcf Package
- **`POST /api/worlds/{world_id}/bcf/import`**: Import Bcf Package
- **`GET /api/worlds/{world_id}/clashes`**: Get World Clashes
- **`POST /api/worlds/{world_id}/clashes/detect`**: Run Clash Detection

### collab

- **`GET /api/worlds/{world_id}/collab/events`**: Collab Events Stream
- **`GET /api/worlds/{world_id}/collab/export/bcf`**: Export Collab Bcf
- **`POST /api/worlds/{world_id}/collab/heartbeat`**: Post Heartbeat
- **`GET /api/worlds/{world_id}/collab/markups`**: List Markups
- **`POST /api/worlds/{world_id}/collab/markups`**: Create Markup
- **`DELETE /api/worlds/{world_id}/collab/markups/{markup_id}`**: Delete Markup
- **`GET /api/worlds/{world_id}/collab/presence`**: Get Presence

### completion

- **`GET /api/worlds/{world_id}/completion/proposals`**: List Proposals
- **`POST /api/worlds/{world_id}/completion/proposals`**: Create Proposals
- **`POST /api/worlds/{world_id}/completion/proposals/{pid}/accept`**: Accept Proposal
- **`GET /api/worlds/{world_id}/completion/proposals/{pid}/diff`**: Proposal Diff
- **`POST /api/worlds/{world_id}/completion/proposals/{pid}/reject`**: Reject Proposal

### construction

- **`POST /api/construction/report/pdf`**: Generate Pdf Report
- **`POST /api/construction/timeline`**: Timeline Progression
- **`POST /api/construction/verify`**: Verify Plan

### drawings

- **`GET /api/worlds/{world_id}/drawings/export/dxf`**: Export Drawing Dxf
- **`POST /api/worlds/{world_id}/drawings/generate`**: Generate Drawing Endpoint
- **`GET /api/worlds/{world_id}/drawings/presets`**: List Drawing Presets
- **`GET /api/worlds/{world_id}/drawings/render/svg`**: Render Drawing Svg

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
- **`POST /api/worlds/{world_id}/export/compare`**: Export Options Compare
- **`GET /api/worlds/{world_id}/export/formats`**: Export Formats
- **`GET /api/worlds/{world_id}/export/package`**: Export Package
- **`GET /api/worlds/{world_id}/export/{content_hash}/download`**: Download Export

### furniture

- **`GET /api/furniture/catalog`**: Get Furniture Catalog
- **`GET /api/furniture/catalog/{catalog_id}`**: Get Furniture Catalog Item
- **`GET /api/furniture/catalog/{catalog_id}/glb`**: Download Furniture Catalog Glb

### geo

- **`POST /api/geo/context`**: Import Context

### health

- **`GET /api/health`**: Health Check
- **`GET /api/status`**: Status
- **`GET /health`**: Healthz Check
- **`GET /healthz`**: Healthz Check

### intent

- **`POST /api/intent/estimate`**: Estimate
- **`GET /api/worlds/{world_id}/intent`**: Get Intent
- **`PUT /api/worlds/{world_id}/intent`**: Put Intent

### jobs

- **`GET /api/jobs`**: List Jobs
- **`GET /api/jobs/{job_id}`**: Get Job
- **`POST /api/jobs/{job_id}/cancel`**: Cancel Job
- **`GET /api/jobs/{job_id}/diagnostics`**: Get Job Diagnostic Report
- **`GET /api/jobs/{job_id}/events`**: Job Events
- **`GET /api/jobs/{job_id}/logs`**: Get Job Log History
- **`GET /api/jobs/{job_id}/stages`**: Get Job Stages
- **`GET /api/jobs/{job_id}/timings`**: Get Job Timing Dashboard

### life-safety

- **`GET /api/worlds/{world_id}/life-safety`**: Get Life Safety Report
- **`POST /api/worlds/{world_id}/life-safety/audit`**: Run Custom Life Safety Audit
- **`GET /api/worlds/{world_id}/life-safety/certificate`**: Download Life Safety Certificate
- **`GET /api/worlds/{world_id}/life-safety/egress-path`**: Get Egress Routes

### measurements

- **`GET /api/worlds/{world_id}/measurements`**: List Measurements
- **`POST /api/worlds/{world_id}/measurements`**: Create Measurement
- **`DELETE /api/worlds/{world_id}/measurements/{measurement_id}`**: Delete Measurement

### mep

- **`POST /api/worlds/{world_id}/mep/detect`**: Detect Mep Topology Endpoint
- **`GET /api/worlds/{world_id}/mep/export/geojson`**: Export Mep Geojson Endpoint
- **`GET /api/worlds/{world_id}/mep/export/ifc`**: Export Mep Ifc Endpoint
- **`POST /api/worlds/{world_id}/mep/route`**: Compute Mep Custom Route
- **`GET /api/worlds/{world_id}/mep/topology`**: Get Mep Topology

### navigation

- **`GET /api/worlds/{world_id}/bookmarks`**: List World Bookmarks
- **`POST /api/worlds/{world_id}/bookmarks`**: Create World Bookmark
- **`GET /api/worlds/{world_id}/navigation`**: Get World Navigation
- **`POST /api/worlds/{world_id}/navigation/path`**: Calculate Navigation Path

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

### proposals_pipeline

- **`GET /proposals`**: List Pipeline Proposals
- **`POST /proposals`**: Submit Pipeline Proposal
- **`GET /proposals/{proposal_id}`**: Get Pipeline Proposal
- **`POST /proposals/{proposal_id}/approve`**: Approve Pipeline Proposal
- **`POST /proposals/{proposal_id}/reject`**: Reject Pipeline Proposal

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
- **`POST /api/sessions/group`**: Group Sessions Endpoint
- **`GET /api/sessions/{session_id}`**: Get Session
- **`POST /api/sessions/{session_id}/deduplicate`**: Deduplicate Session
- **`GET /api/sessions/{session_id}/location`**: Session Location
- **`POST /api/sessions/{session_id}/locations`**: Add Session Locations
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

### splats

- **`GET /api/worlds/{world_id}/splats`**: Get Splat Metadata
- **`POST /api/worlds/{world_id}/splats/generate`**: Generate Splats
- **`GET /api/worlds/{world_id}/splats/stream`**: Stream Splats
- **`POST /api/worlds/{world_id}/splats/upload`**: Upload Splats

### surface_quality

- **`GET /api/worlds/{world_id}/surface-quality`**: Get World Surface Quality
- **`POST /api/worlds/{world_id}/surface-quality/analyze`**: Analyze Surface Quality Endpoint
- **`GET /api/worlds/{world_id}/surface-quality/certificate`**: Export Surface Quality Certificate
- **`GET /api/worlds/{world_id}/surface-quality/heatmap`**: Get Surface Heatmap

### system

- **`GET /api/system/verification`**: Get Verification

### takeoff

- **`GET /api/worlds/{world_id}/takeoff`**: Get Material Takeoff
- **`POST /api/worlds/{world_id}/takeoff/estimate`**: Calculate Custom Estimate
- **`GET /api/worlds/{world_id}/takeoff/export/csv`**: Export Takeoff Csv
- **`GET /api/worlds/{world_id}/takeoff/export/pdf`**: Export Takeoff Pdf

### technologies

- **`GET /api/technologies`**: List Technologies
- **`POST /api/technologies/benchmark`**: Run Benchmark
- **`GET /api/technologies/{tech_id}`**: Get Technology

### temporal

- **`GET /api/worlds/{world_id}/epochs`**: List World Epochs
- **`GET /api/worlds/{world_id}/temporal/compare`**: Compare World Epochs

### thermal-envelope

- **`GET /api/worlds/{world_id}/thermal-envelope`**: Get Thermal Envelope Report
- **`GET /api/worlds/{world_id}/thermal-envelope/certificate`**: Download Thermal Certificate
- **`POST /api/worlds/{world_id}/thermal-envelope/evaluate`**: Evaluate Custom Thermal Scenario

### uploads

- **`POST /api/uploads`**: Create Upload
- **`POST /api/uploads/check`**: Check Uploads
- **`POST /api/uploads/resumable/chunk`**: Resumable Chunk
- **`POST /api/uploads/resumable/start`**: Resumable Start
- **`POST /api/uploads/resumable/{upload_id}/finalize`**: Resumable Finalize

### usage

- **`GET /api/tenants/{tenant_id}/usage`**: Get Specific Tenant Usage
- **`GET /api/usage`**: Get Current Tenant Usage

### worlds

- **`GET /api/worlds`**: List Worlds
- **`POST /api/worlds`**: Create World
- **`POST /api/worlds/import/package`**: Import World Package
- **`GET /api/worlds/shared/access/{token}`**: Access Shared World
- **`DELETE /api/worlds/{world_id}`**: Delete World
- **`GET /api/worlds/{world_id}`**: Get World
- **`PATCH /api/worlds/{world_id}`**: Patch World
- **`POST /api/worlds/{world_id}/archive`**: Archive World
- **`GET /api/worlds/{world_id}/artifacts/{sha256}`**: World Artifact
- **`POST /api/worlds/{world_id}/attach/{session_id}`**: Attach Session
- **`GET /api/worlds/{world_id}/camera`**: Get World Camera
- **`POST /api/worlds/{world_id}/camera`**: Set World Camera
- **`GET /api/worlds/{world_id}/cameras`**: World Cameras
- **`GET /api/worlds/{world_id}/capture-requests`**: List Capture Requests
- **`PATCH /api/worlds/{world_id}/capture-requests/{request_id}`**: Update Capture Request Status
- **`GET /api/worlds/{world_id}/commands`**: Get World Commands
- **`POST /api/worlds/{world_id}/commands`**: Execute World Command
- **`POST /api/worlds/{world_id}/commit`**: Commit World Correction
- **`POST /api/worlds/{world_id}/corrections/merge`**: Api Merge Entities
- **`POST /api/worlds/{world_id}/corrections/reclassify`**: Api Reclassify Entity
- **`POST /api/worlds/{world_id}/corrections/split`**: Api Split Entity
- **`GET /api/worlds/{world_id}/coverage`**: World Coverage
- **`GET /api/worlds/{world_id}/diff`**: Get World Diff
- **`POST /api/worlds/{world_id}/duplicate`**: Duplicate World
- **`GET /api/worlds/{world_id}/entities/{entity_id}/confidence`**: Entity Confidence
- **`GET /api/worlds/{world_id}/entities/{entity_id}/provenance`**: Entity Provenance
- **`GET /api/worlds/{world_id}/options`**: List Design Options
- **`POST /api/worlds/{world_id}/options`**: Create Design Option
- **`GET /api/worlds/{world_id}/options/compare`**: Compare Design Options
- **`GET /api/worlds/{world_id}/points`**: World Points
- **`GET /api/worlds/{world_id}/quality`**: World Quality
- **`GET /api/worlds/{world_id}/query/spaces`**: Query World Spaces
- **`POST /api/worlds/{world_id}/redo`**: Redo World Command
- **`GET /api/worlds/{world_id}/render`**: Render World View
- **`GET /api/worlds/{world_id}/report`**: World Report
- **`POST /api/worlds/{world_id}/restore`**: Restore World
- **`POST /api/worlds/{world_id}/rollback`**: Rollback World
- **`POST /api/worlds/{world_id}/scale`**: Set World Scale
- **`GET /api/worlds/{world_id}/scale/suggestion`**: Get Scale Suggestion
- **`POST /api/worlds/{world_id}/scale/suggestion/reject`**: Reject Scale Suggestion
- **`GET /api/worlds/{world_id}/shares`**: List World Share Links
- **`POST /api/worlds/{world_id}/shares`**: Create World Share Link
- **`POST /api/worlds/{world_id}/shares/{share_id}/revoke`**: Revoke World Share Link
- **`GET /api/worlds/{world_id}/space-graph`**: Get World Space Graph
- **`GET /api/worlds/{world_id}/spatial-intelligence`**: Get World Spatial Intelligence
- **`GET /api/worlds/{world_id}/thumbnail`**: World Thumbnail
- **`POST /api/worlds/{world_id}/undo`**: Undo World Command
- **`GET /api/worlds/{world_id}/versions`**: List World Versions
- **`PATCH /api/worlds/{world_id}/versions/{version_id}/label`**: Set Version Label
- **`GET /api/worlds/{world_id}/views`**: List Camera Views
- **`POST /api/worlds/{world_id}/views`**: Create Camera View
- **`DELETE /api/worlds/{world_id}/views/{view_id}`**: Delete Camera View
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
- `POST /api/worlds/{id}/export`: Export current world to `gltf`, `glb`, `usda`, `obj`, `ply`, `stl`, `dae`, `ifc`, `blender`, `cityjson`, `floorplan_svg`, `floorplan_dxf`, `room_report_pdf`, or `room_schedule_csv`.

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
