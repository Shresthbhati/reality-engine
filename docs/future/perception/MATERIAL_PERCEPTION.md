# Material Perception

Status: MISSING (2026-09-14)

## Purpose

Attach honest, evidence-backed material properties to entities —
category, roughness, reflectance, transparency, mass-relevant density
class — so simulation (P7) consumes physics-plausible worlds instead
of default-material geometry.

## Current state

Entities carry geometry + semantic label (from SAM/SigLIP detection)
and measurement-derived properties (planes from measurements). No
material model exists.

## Approach

- **Class-based priors first.** Semantic category → material prior
  table (wood floor: roughness 0.6, density 700 kg/m3, …). This is a
  recorded inference, not a measurement; provenance
  `class_prior:v1`, confidence carried through.
- **Appearance-based refinement (later).** Multi-view photometric
  cues (shading, specularity) refine per-entity material when views
  allow; explicit UNVERIFIED otherwise.
- **Measured overrides.** User-supplied material via CLI/Studio beats
  any inference; provenance `user_supplied`.

## Canonical representation

`MaterialProperties` on the entity (WorldIR extension):
`category`, `roughness`, `metallic`, `transparency`, `density_class`,
`perception_method`, `confidence` — never a silent default material
when nothing is known: fields are None/UNKNOWN.

## Rules

- No LLM in the runtime path (Decision 019): material inference = CV +
  lookup tables + photometric methods.
- Physics coupling: density feeds rigid-body mass (P7 physics);
  uncertainty propagates (P3) into mass uncertainty.

## Failure modes

- Misclassification → wrong material prior; confidence + provenance
  let Studio/physics treat it as low-trust; review workflow corrects.
- Transparent/glassy objects defeat photometric refinement; honest
  UNKNOWN, never fabricated roughness.

## Acceptance criteria

- Deterministic: synthetic labeled renders → material fields match
  the prior table, provenance recorded.
- Unknown category → UNKNOWN fields, not defaults.

## Priority

P2-P3 (after multi-view identity; before physics is useful).
