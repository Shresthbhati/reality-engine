# Reality Studio — Quickstart Guide

> **"I opened Reality Studio. What do I do?"**  
> *A 5-minute practical orientation for new users, architects, and surveyors.*

---

## 1. What a World Is

A **World** in Reality Engine is a persistent digital twin of a real physical space (such as an office floor, lab, or building). It contains the metric 3D point cloud, rooms, walls, doors, and furniture extracted from field scans.

---

## 2. Opening and Selecting a World

### WHAT DO I CLICK?
1. Open your browser and go to `http://localhost:3000/worlds` (or `http://localhost:3005/worlds` in production).
2. You will see the **Worlds** card catalog. Click on any active world card (for example, **Browser E2E Tower**).

### WHAT SHOULD I SEE?
The screen transitions into the 3-column **Reality Studio Workstation**:
* **Left:** Navigation sidebar with tabs (`Worlds`, `Sessions`, `Evidence`, `Hierarchy`, etc.).
* **Center:** 3D Spatial Viewport with dark background, coordinate compass at bottom right, and a layer toolbar at top.
* **Right:** Adaptive Inspector showing entity metadata.
* **Bottom:** Telemetry bar displaying coordinates, storey elevation, scale status, and commit versions.

### WHAT DOES IT MEAN?
You are now connected to the canonical spatial intermediate representation (WorldIR) for that physical site.

### WHAT DO I DO NEXT?
If the 3D viewport displays a point cloud or building model, proceed to Step 3.  
If the center displays **"No 3D Reconstructed World Available"**, see Step 8 to trigger reconstruction.

---

## 3. Navigating the 3D Scene

### WHAT DO I CLICK?
* **Orbit (Tumble):** Hold `Left-Click` and drag the mouse across the viewport (must move more than 4 pixels).
* **Pan (Translate):** Hold `Right-Click` and drag the mouse up, down, left, or right.
* **Zoom (Dolly):** Spin the `Mouse Wheel` forward to zoom in, backward to zoom out.
* **Reset to True North:** Click the **`N`** icon on the circular compass rose at the bottom-right corner.

### WHAT SHOULD I SEE?
The perspective camera smoothly rotates or shifts. If an animation was running, moving the wheel stops it instantly.

### WHAT DOES IT MEAN?
Reality Studio uses metric **East-North-Up (ENU)** coordinates:
* $+X$ axis = East
* $+Z$ axis = North
* $+Y$ axis = Elevation / Height

### WHAT DO I DO NEXT?
Press **`F`** on your keyboard to center and frame the full building within your screen.

---

## 4. Selecting and Inspecting an Object

### WHAT DO I CLICK?
Move your mouse cursor over any visible room, wall, door, or piece of furniture in the 3D scene and perform a stationary `Left-Click` (without dragging).

### WHAT SHOULD I SEE?
* The clicked object is highlighted with an outline in the 3D scene.
* The **Right Inspector** panel opens automatically displaying **ENTITY INSPECTOR**.

### WHAT DOES IT MEAN?
The inspector presents algorithmic and physical truths derived directly from the reconstruction engine:
1. **Semantic Class:** Architectural category (`room`, `wall`, `door`, `window`, `column`, `furniture`).
2. **Metric Dimensions:** Real-world size in meters ($L \times W \times H$).
3. **Centroid Position:** Center coordinate $(X, Y, Z)$ in meters relative to origin.
4. **Confidence Score:** Algorithmic certainty percentage. If unmeasured by the pipeline, it honestly states: *"Confidence unrecorded"*.
5. **Level Attribution:** Which storey the object belongs to.

### WHAT DO I DO NEXT?
Press **`Escape`** to clear the selection, or click **Trace Evidence** to see the original photos that observed this object.

---

## 5. Changing Storeys & Isolating Levels

### WHAT DO I CLICK?
1. In the Left Navigation Panel, click the **Hierarchy** tab.
2. In the tree, locate the storeys (e.g., `Storey 0`, `Storey 1`).
3. Click on **`Storey 1`**.

### WHAT SHOULD I SEE?
* All objects belonging to other storeys become dimmed and semi-transparent.
* Inactive storeys cannot be accidentally clicked in the 3D scene.
* The bottom telemetry bar updates to indicate the active storey elevation.

### WHAT DOES IT MEAN?
You are focusing exclusively on Level 1. Any measurements or queries you execute will isolate this floor.

### WHAT DO I DO NEXT?
* Press **`W`** on your keyboard to hide walls and see interior furniture and room partitions.
* Press **`Escape`** or click **Show All Storeys** in the tree header to restore the entire building.

---

## 6. Measuring Real-World Distances

### WHAT DO I CLICK?
1. Press **`M`** on your keyboard or click the **`Ruler`** icon in the viewport top toolbar.
2. The mouse cursor turns into a precision crosshair.
3. Click any point on a wall, floor, or point cloud surface. A cyan anchor marker appears.
4. Move your mouse and click a second point across the space.

### WHAT SHOULD I SEE?
A floating measurement badge appears showing:
* **Distance:** Direct Euclidean line distance in meters (e.g., `3.412 m`).
* **$\Delta X, \Delta Y, \Delta Z$:** Axis-aligned displacements ($\Delta X$ = East-West, $\Delta Y$ = Height, $\Delta Z$ = North-South).

### WHAT DOES IT MEAN?
These measurements are calculated directly from physical metric coordinates. If the world header says *"Scale: Calibrated Metric"*, these represent true physical distances.

### WHAT DO I DO NEXT?
Press **`Escape`** or click **Clear** to finish measuring.

---

## 7. How to Tell if Reconstruction Succeeded

Reality Studio never synthesizes fake geometry to mask errors. Look at the top bar and center viewport to determine the exact state:

| Visual Indicator | Exact Meaning | Required Action |
| :--- | :--- | :--- |
| **Normal 3D Scene + Green Badges** | Reconstruction succeeded with full metric calibration. | You can inspect, measure, and export. |
| **"No 3D Reconstructed World Available"** | No 3D reconstruction has been run yet for this world. | Click **Construct Space / Compile IR** to run reconstruction. |
| **"Partial Reconstruction: M/N cameras" (Amber Badge)** | Some photos could not be geometrically aligned due to low visual overlap. | Check the Evidence tab to see unaligned camera cones. |
| **"Spatial Reconstruction Failed" (Red Banner)** | Algorithmic failure (e.g., insufficient features, corrupt images, timeout). | Review the diagnostic error message in the banner. |
| **"Scale: Nominal (Uncalibrated)" (Yellow Pill)** | Relative geometry was reconstructed, but physical scale is unreferenced. | Distances are relative units rather than true meters. |

---

## 8. Where Reconstruction and Evidence Live

* **Evidence Tab (Left Panel):** Contains all raw field photographs, depth scans, and IMU pose trajectories uploaded by the capture app.
* **Sessions Tab (Left Panel):** Lists field capture visits (`s-1`, `s-2`).
* **Construction Button (Top Header):** Opens the **Room Construction Pipeline** modal. From here, select a session and click **`▶ Start Reconstruction`** to run the 7-stage reconstruction pipeline (SfM $\to$ Depth Fusion $\to$ Segmentation $\to$ Room Assembly).

---

## 9. Exporting Your Work

### WHAT DO I CLICK?
1. Click the **Export** button with the download icon in the top header.
2. The **Export Panel** modal opens.

### WHAT SHOULD I SEE?
Four export cards:
* **WorldIR (JSON):** Complete scene graph, hierarchy, bounding boxes, and semantics.
* **Point Cloud (PLY):** Binary 3D point cloud file for MeshLab, CloudCompare, or Revit.
* **Cameras (JSON):** Extrinsic matrices and focal calibrations for all registered photos.
* **Audit Report (JSON):** Full quality report with camera registration ratio and scale verification.

### WHAT DOES IT MEAN?
These are the canonical files generated by Reality Engine. They are not downsampled representations.

### WHAT DO I DO NEXT?
Click **Download** next to any format to save the file to your computer.

---

## 10. Summary of Essential Keyboard Shortcuts

| Shortcut | Action |
| :---: | :--- |
| **`F`** | Frame active selection (or frame entire world) |
| **`Escape`** | Deselect, exit measurement mode, or exit storey isolation |
| **`M`** | Toggle two-point 3D distance ruler |
| **`W`** | Toggle wall visibility (peel back walls) |
| **`T`** | Toggle 3D spatial topology connection vectors |
| **`G`** | Toggle 20m ground coordinate grid |
| **`1` / `2` / `3`** | Toggle Points (`1`), Entities (`2`), Cameras (`3`) |
| **`[` / `]` / `\`** | Toggle Left Nav (`[`), Right Inspector (`]`), Bottom Telemetry (`\`) |
| **`⌘K` / `Ctrl+K`** | Open Command Palette search |
