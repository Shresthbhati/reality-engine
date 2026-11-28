# Persistence Integrity Report — Reality Engine V1

> Authoritative report on the integrity of persisted reconstruction data: atomic commits, concurrent modifications, version lineage, evidence persistence, geometry serialization, restart recovery, duplicate commits, corrupted records, partial writes, rollback, referential integrity, asset accessibility, and world reconstruction from persisted records.
>
> Part of the KILO integrity mandate (§6: "WorldStore and persistence correctness"). Owner: World Reconstruction Integrity / OpenCode (storage).
> Basis: `worldstore/store.py`, `worldstore/lifecycle.py`, `apps/api/jobs.py`, `world_ir/artifact_store.py`.

## 0. Governing rule

> A successful job must not conceal a failed persistence operation. Every persistence result is verified (`verify_version` clean) before the version is adopted as HEAD.

## 1. WorldStore contract — immutable, content-addressed, versioned storage

**Schema location:** `worldstore/store.py` `WorldStore`, `StoredVersion`; `world_ir/artifact_store.py` `FileArtifactStore`.

### 1.1 Storage model

| Concept | Contract |
|---|---|
| Version record | JSON at `<root>/versions/<vid>.json` |
| World payload | content-addressed blob at `<root>/artifacts/<h[:2]>/<h[2:4]>/<h>`, referenced as `sha256://<h>` |
| Version id | `v-<hex12>` or explicit; validated against `^[A-Za-z0-9_-]{1,128}$` |
| Content hash | sha256 of serialized WorldIR dict (`artifact_hash`) |
| Lineage | `parent` field per version; DAG V1→V2→V3 |

### 1.2 Write path (`save_version`)

| Step | Contract |
|---|---|
| 1. Validate id | `version_id` matches regex; refused before touching filesystem |
| 2. Immutability check | `path.exists()` → refused (versions immutable; save a new version) |
| 3. Diff from parent | `diff_worlds(parent_world, world)` → `changed_entity_ids`, `changed_geometry_ids` |
| 4. Serialize | `json.dumps(world.to_dict(), sort_keys=True).encode()` |
| 5. Store artifact | `FileArtifactStore.put(payload)` → `(uri, digest)` |
| 6. Write record | `_atomic_write_text(path, json.dumps(record))` — temp file + fsync + os.replace |
| 7. Sequence index | append `vid` to `sequence.json` under the same sequence lock |
| 8. Return | `StoredVersion` (immutable record) |

**Atomicity guarantees:**
- `_atomic_write_text`: writes to sibling temp file, fsyncs, then `os.replace()` — atomic on POSIX and Windows NTFS. Readers always see complete previous or new content, never a partial write.
- A crash mid-write leaves no truncated record at the final path.
- Cross-process sequence lock (`_SequenceLock` via `os.mkdir()` atomic primitive) guards the read-modify-write of `sequence.json` — prevents lost updates and WinError 5 under concurrency.
- The immutability check is re-verified inside the lock (TOCTOU protection); exactly one writer wins, the loser gets an explicit `WorldStoreError`.

### 1.3 Read path (`load_version`)

| Step | Contract |
|---|---|
| 1. Read record | `_record(version_id)` — raises `WorldStoreError` if unknown or corrupted |
| 2. Fetch artifact | `FileArtifactStore.get(artifact_uri)` |
| 3. Verify hash | `sha256(payload) == artifact_hash`; mismatch → `WorldStoreError` (tampered/corrupted) |
| 4. Deserialize | `WorldIR.from_dict(json.loads(payload))` |

**Contract rule:** `load_version()` re-verifies the hash **before** returning a WorldIR. Corrupted or tampered bytes raise an explicit error — never silently returned.

### 1.4 Integrity verification (`verify_version`)

| Check | Result |
|---|---|
| Artifact present | missing → failure record |
| Hash matches | mismatch → failure record ("tampered or corrupted") |
| Both pass | empty failure list (version verifies) |

Returns a list of failure records (empty when valid). Callers must check this before trusting a version.

## 2. Version lifecycle & adoption

**Schema location:** `worldstore/lifecycle.py`; `apps/api/jobs.py` seg_changeset_commit.

### 2.1 Adoption protocol (ordered, atomic)

| Step | Contract |
|---|---|
| 1. Intent file | record adoption intent |
| 2. WorldStore version | `save_version()` (immutable, content-addressed) |
| 3. Read-back verify | `verify_version()` clean **before** pointer moves |
| 4. DB HEAD + mirror row | ONE transaction with the version record |
| 5. Clear intent | mark intent complete |

**Failure handling:** failed adoption undoes its own pending HEAD move. A cancelled job never erases an adopted version (`adopted_before_cancel` recorded; durable work is not un-written).

### 2.2 Rollback

| Aspect | Contract |
|---|---|
| What moves | only the HEAD pointer |
| What stays | all versions immutable; COLMAP base set aside (renamed) |
| Next version | chains onto rolled-back HEAD |

## 3. Version lineage

| Operation | Contract |
|---|---|
| `parents(version_id)` | returns `[parent]` or `[]` |
| `ancestors(version_id)` | root-first chain; cycle-safe walk |
| `list_versions()` | save-order listing; raises `WorldStoreError` naming the specific version if any record is corrupted — history never silently truncated |

**Contract rule:** History is never silently truncated or dropped to hide corruption. A corrupted record is an explicit, actionable failure.

## 4. Concurrency & concurrent modification

| Concern | Contract |
|---|---|
| Sequence lock | cross-process, cross-thread mutex; bounded poll; acquisition times out and raises `WorldStoreError` |
| Lost update | prevented by atomic read-modify-write under lock |
| `ConcurrentModificationError` | retryable (next attempt reloads HEAD) |
| `LeaseLost` | worker reaped/requeued |
| Stale lock | never silently overridden; operator must inspect/remove manually |

**Contract rule:** Lock staleness (holder crashed without releasing) is never silently overridden — that risks two processes believing they hold the same lock.

## 5. Quarantine & retention

| Operation | Contract |
|---|---|
| `quarantine_version` | moves never-adopted version record to `quarantine/`; content-addressed artifact kept; idempotent; unknown ids raise |
| `quarantined_versions()` | lists quarantined version ids |
| `prune_quarantine` | removes old unreferenced blobs only after: age floor + keep-last + not a live parent + no pending adoption intent; dry_run default |

**Retention safety:**
- Live versions (HEAD and its whole lineage) are never touched.
- A blob `put()` just before its record is written is protected by the age floor.
- Blobs no referenced by any retained record (live or quarantined) and older than max_age are deleted.

## 6. Path traversal & path safety

| Concern | Contract |
|---|---|
| `version_id` in file path | validated against regex before `self._versions_dir / f"{version_id}.json"` |
| `artifact://` digest | `FileArtifactStore.digest_of()` validates path-building input against the same pattern |
| Unvalidated `../../outside/secret` | resolved outside the store root exactly like an unvalidated digest — refused |

## 7. Restart recovery

**Contract rule:** A fresh `WorldStore` instance over the same root reads everything back — no in-memory state is required. This is exactly the "survives process restart" property. All state is on disk: version records, content-addressed artifacts, sequence index.

## 8. Duplicate commits

**Contract rule:** Re-saving an existing `version_id` is refused. Idempotent re-persist of an identical world produces the same digest (content-addressed), so re-persisting the same world is safe as a no-op at the content level, but a duplicate explicit `version_id` raises `WorldStoreError`.

## 9. Referential integrity

| Aspect | Contract |
|---|---|
| `artifact_uri` → artifact | verified by hash on every `load_version()` |
| Geometry `data_uri`/`data_hash` | content-addressed; exporters resolve via `artifact_store.get()` |
| `changed_entity_ids` / `changed_geometry_ids` | computed via `diff_worlds` at save; survive even if parent deleted/corrupted |
| `source_session_ids` | traceability link to evidence-acquisition sessions |

## 10. What this report forbids (mandate §6)

- A successful job concealing a failed persistence operation — every result verified before HEAD adoption.
- Silent overwrite of an observed version — versions immutable.
- Corrupted records silently skipped — explicit, actionable `WorldStoreError`.
- Lost history from concurrent writes — sequence lock + atomic writes.
- Unverified bytes returned as a WorldIR — `load_version()` re-verifies hash.

## 11. Verification checklist

| Check | Method |
|---|---|
| Atomic commit | crash mid-write leaves no truncated record (temp + fsync + replace) |
| Concurrent modification | sequence lock; `ConcurrentModificationError` retryable |
| Version lineage | `ancestors()` root-first chain; cycle-safe |
| Evidence persistence | `source_session_ids` + `artifact_uri` content-addressed |
| Geometry serialization | `data_uri`/`data_hash`; exporters resolve via artifact store |
| Restart recovery | fresh `WorldStore` over same root reads all state |
| Duplicate commits | duplicate explicit `version_id` → `WorldStoreError` |
| Corrupted records | `_record()` / `load_version()` / `verify_version()` → explicit error |
| Partial writes | `_atomic_write_text()` — no truncated file at final path |
| Rollback | HEAD pointer moves only; versions immutable |
| Referential integrity | hash verified on every load; `changed_*` computed at save |
| Asset accessibility | `FileArtifactStore.get()` resolves `sha256://<h>` |
