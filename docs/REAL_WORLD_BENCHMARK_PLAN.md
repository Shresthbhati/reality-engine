# Reality Engine — Real-World Benchmark Plan
## Datasets, Ground Truth Protocols, Quantitative Metrics, and Evaluation Methodology

> **Document ID**: `RE-BENCH-003`  
> **Status**: APPROVED BENCHMARK SPECIFICATION  
> **Baseline**: `origin/main` @ `6db0f86`  

---

## 1. Objective & Scientific Discipline

The benchmark programme evaluates Reality Engine against real, physically measured environments.
**Visual convincingness is not geometric accuracy.** A photorealistic mesh or neural render that misrepresents a room dimension by 10 centimeters or hallucinates an opening is an unacceptable failure for an intelligent spatial engine.

Every capability must be evaluated against independent, verifiable ground truth across the 6-tier maturity scale:
- **L0** (Designed) $\to$ **L1** (Implemented) $\to$ **L2** (Unit Verified) $\to$ **L3** (Integration Verified) $\to$ **L4** (Real-World Verified) $\to$ **L5** (Production Validated).

---

## 2. The 12 Real-World Benchmark Domains

The benchmark programme defines 12 standard physical domains representing the full spectrum of indoor, outdoor, architectural, and detail challenges:

| # | Benchmark Domain | Primary Test Purpose | Physical Complexity & Ground Truth Target |
|---|---|---|---|
| **D01** | **Small Room (Bedroom / Study)** | Baseline room boundary solving and metric scale. | Rectangular 4-wall room ($3\text{m}\times 4\text{m}$), door, window, bed, desk. Terrestrial Laser Scan (TLS) ground truth. |
| **D02** | **Connected Multi-Room Apartment** | Inter-room topology and shared wall consistency. | 4-room layout (hall, living, bed, bath) with doorways. Total area $65\text{m}^2$. Laser distometer network. |
| **D03** | **Single-Family Multi-Storey House** | Vertical floor-to-floor registration and stairs. | 2 floors, internal staircase, exterior facades. Total height $6.5\text{m}$. Surveyed control points. |
| **D04** | **Commercial Office & Repetitive Corridors** | Visual ambiguity and repetitive feature matching. | $40\text{m}$ corridor with identical doors, recessed lighting, and glass partitions. LightGlue feature discrimination. |
| **D05** | **Complex Kitchen** | Dense cabinetry, countertop appliances, plumbing. | Base/upper cabinets, sink, faucet, refrigerator, microwave, gas hob. Dense occlusions. |
| **D06** | **Bathroom & Sanitary Suite** | Reflective surfaces, mirrors, sanitary fixtures. | Wall-hung toilet, washbasin, mirror, glass shower enclosure, glazed tiles. Specular challenge. |
| **D07** | **Multi-Storey Commercial Facade** | Exterior building envelope, window schedules. | 4-storey urban building facade ($15\text{m}$ height), repetitive fenestration, entrance canopy. |
| **D08** | **Fine Architectural Details & Fixtures** | Millimeter-scale detection and fixture mounting. | Switchboards, electrical sockets, light switches, door hardware, baseboards, crown mouldings. |
| **D09** | **Urban Street & Infrastructure** | Civil infrastructure, road boundaries, utilities. | $80\text{m}$ street segment: asphalt carriageway, concrete curbs, paved sidewalks, road markings, manholes. |
| **D10** | **Landscape & Outdoor Site** | Natural terrain, vegetation, outdoor fixtures. | Sloped terrain ($15\%$ grade), deciduous trees, shrubbery, lawn, street lampposts, park benches. |
| **D11** | **Occluded & Cluttered Environment** | Robustness to moving clutter and visual barriers. | Living room with dense furniture, boxes, hanging clothes, partially visible floor/wall corners. |
| **D12** | **Specular & Transparent Surfaces** | Handling transparent glass partitions and mirrors. | Modern office with floor-to-ceiling glass meeting rooms and large reflective mirrors. |

---

## 3. Ground Truth Acquisition Protocols

To prevent circular verification (evaluating algorithms against other uncalibrated algorithms), ground truth data is captured using independent, survey-grade physical instruments:

1. **Terrestrial Laser Scanning (TLS)**:
   - High-precision survey scanners (Faro Focus / Leica BLK360) generating millimeter-accurate point clouds ($\le 2\text{mm}$ error at $10\text{m}$).
   - Scans are registered using survey spheres and target checkerboards to establish the absolute coordinate frame.
2. **Calibrated Laser Distometer Measurements**:
   - Class II laser distance meters ($\pm 1.0\text{mm}$ accuracy) measuring:
     - Clear wall-to-wall spans across room diagonals.
     - Finished floor-to-ceiling clear heights.
     - Door/window opening widths, heights, and sill elevations.
     - Socket/switchboard center-point elevations above finished floor.
3. **Manual Semantic Annotation**:
   - Independent human verification identifying true entity boundaries, classes, and bilateral spatial relationships (`ATTACHED_TO`, `RESTS_ON`, `CONTAINS`).

---

## 4. Quantitative Evaluation Metrics

Every benchmark run produces a machine-readable JSON report evaluated against standardized mathematical criteria:

### 4.1 Geometric Reconstruction Accuracy
- **Chamfer Distance (CD)**:
  $$\text{CD}(S_1, S_2) = \frac{1}{|S_1|}\sum_{x \in S_1} \min_{y \in S_2} \|x - y\|_2 + \frac{1}{|S_2|}\sum_{y \in S_2} \min_{x \in S_1} \|x - y\|_2$$
  *Target*: $\text{CD} \le 15\text{mm}$ for indoor architectural surfaces.
- **F-Score at Distance Threshold $\tau$ ($5\text{mm}, 10\text{mm}, 50\text{mm}$)**:
  Harmonic mean of precision and recall for surface points reconstructed within distance $\tau$ of ground truth.
  *Target*: $F\text{-Score}@10\text{mm} \ge 90\%$.
- **Surface Normal Consistency**:
  Mean angular error between reconstructed surface normals and ground truth normals.
  *Target*: Mean angular error $\le 8.0^\circ$.

### 4.2 Dimensional & Metric Error
- **Absolute Dimensional Error**:
  $$e_{\text{dim}} = |d_{\text{reconstructed}} - d_{\text{ground\_truth}}|$$
  *Targets*:
  - Room wall spans ($3\text{m} - 8\text{m}$): $e_{\text{dim}} \le 20\text{mm}$ ($< 0.5\%$).
  - Ceiling height: $e_{\text{dim}} \le 15\text{mm}$.
  - Door/window openings: $e_{\text{dim}} \le 10\text{mm}$.
  - Socket/switch elevations: $e_{\text{dim}} \le 10\text{mm}$.
- **Scale Drift Factor**:
  Cumulative scale dilation/contraction across large buildings.
  *Target*: $\le 0.3\%$ over $50\text{m}$ span.

### 4.3 Semantic & Instance Perception Accuracy
- **Classification Precision & Recall**:
  Evaluated across all 114 taxonomy classes:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  *Target*: Macro-averaged Precision $\ge 92\%$, Recall $\ge 88\%$.
- **3D Bounding Box IoU (Intersection over Union)**:
  Oriented 3D bounding box overlap with ground truth.
  *Target*: $\text{IoU}_{3\text{D}} \ge 0.70$ for large furniture; $\ge 0.60$ for small fixtures.
- **Multi-View Instance Identity Consistency**:
  Percentage of objects correctly associated across viewpoints without splitting into duplicates or merging distinct physical entities.
  *Target*: Identity accuracy $\ge 98\%$.

### 4.4 Architectural Topology & Graph Validity
- **Room Boundary Completeness**:
  Every interior space must resolve into a closed, non-self-intersecting 2D polygon with area matching ground truth within $\pm 2\%$.
- **Bilateral Relationship Accuracy**:
  Precision of inferred relationships (`ATTACHED_TO`, `RESTS_ON`, `CONTAINS`):
  *Target*: Precision $\ge 96\%$, Recall $\ge 90\%$. Zero impossible physical relationships (e.g. wall floating in mid-air).

### 4.5 System Performance & Resource Budgets
- **100-Image Reconstruction Wall Time**: $\le 30.0\text{ seconds}$ on standard desktop GPU (NVIDIA RTX 4070 / Apple M3 Max equivalent).
- **Peak Memory Overhead**: $\le 4.0\text{ GB}$ system RAM and $\le 6.0\text{ GB}$ GPU VRAM for scenes up to 500 images.
- **Spatial Index Throughput**: $\ge 1,000\text{ queries/second}$ for 5-nearest neighbor queries on 50,000-entity scenes.

---

## 5. Automated Benchmark Suite Architecture

The benchmark programme is fully automated in the repository under `benchmarks/`:

```
benchmarks/
├── golden_datasets.py            # 10 permanent benchmark suites (single_room, connected_rooms, etc.)
├── spatial_index_bench.py        # Spatial indexing query throughput & build scaling
├── pipeline_measurement.py       # Full pipeline tracemalloc memory and stage latency profiling
├── real_data_city_scale_bench.py # Large-scale multi-session registration and district scaling
└── results/                      # Content-addressed JSON benchmark run records
```

### 5.1 Continuous Regression Policy
1. Every Pull Request must pass the automated benchmark suite in CI (`tests/test_golden_datasets.py` and `tests/test_performance_benchmarks.py`).
2. Any commit that causes a metric regression $> 5\%$ in dimensional accuracy or $> 10\%$ in execution runtime is automatically blocked by the release gate.
3. Benchmark run records are committed as immutable, timestamped JSON artifacts in `benchmarks/results/` to guarantee historical performance transparency.
