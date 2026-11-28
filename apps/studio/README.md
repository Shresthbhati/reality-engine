# Reality Studio (`@reality-engine/studio`)

Interactive TypeScript + Three.js reconstruction inspection, evidence transparency, correction, and version comparison frontend for Reality Engine.

## Workflow
`Project → Upload → Reconstruct → Inspect → Correct → Compare → Export`

## Backend Integration (`apps/api`)
Consumes real `WorldIR`, `WorldStore`, and `Evidence` endpoints without fabricated client data:
- `GET /worlds`, `POST /worlds`
- `POST /reconstructions`, `GET /worlds/{world_id}/status`
- `GET /worlds/{world_id}/worldir`, `/points`, `/cameras`, `/coverage`, `/quality`, `/versions`
- `GET /worlds/{world_id}/entities/{entity_id}/provenance`, `/confidence`
- `POST /worlds/{world_id}/corrections/reclassify`, `/split`, `/merge`, `POST /worlds/{world_id}/rollback`
- `GET /worlds/{world_id}/diff`
- `GET /api/worlds/{world_id}/export/formats`, `POST /api/worlds/{world_id}/export`
