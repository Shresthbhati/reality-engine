# Reality Studio — Keyboard & Mouse Interaction Reference

> **Complete desk reference for Reality Studio (`frontend/src`).**  
> *Memorization is never required: all primary shortcuts have visible mouse/UI alternatives.*

---

## 1. Viewport Navigation & Camera Controls

| Shortcut | Action | When It Works | Mouse / UI Alternative | Notes & Edge Cases |
| :--- | :--- | :--- | :--- | :--- |
| **`Left-Click + Drag`** | **Orbit Camera** | Over 3D Viewport when not measuring | None (Standard orbital drag) | Threshold of $> 4\text{px}$ movement prevents accidental selection. |
| **`Right-Click + Drag`** | **Pan Camera** | Over 3D Viewport | None (Standard pan drag) | Translates view plane along horizontal and vertical screen axes. |
| **`Mouse Wheel`** | **Zoom In / Out** | Over 3D Viewport | None (Standard scroll dolly) | Dollying immediately interrupts and cancels active camera animations. |
| **`Stationary Left-Click`** | **Select Entity** | Over 3D Viewport ($\le 4\text{px}$ movement) | Click entity in **Hierarchy** tab in Left Navigation | Raycasts to nearest surface. Dimmed entities on inactive storeys cannot be selected. |
| **`F`** | **Frame Selection / All** | In 3D Viewport; ignores text inputs | Click **`[F]`** button in top-right viewport toolbar | If an entity is selected, flies to center it; otherwise frames full bounding extent. |
| **`Escape`** | **Deselect / Cancel** | Globally in Workstation; ignores text inputs | Click **`Clear Selection`** in bottom bar or **`Clear`** on ruler | Exits measurement first; exits space isolation second; deselects entity third. |
| **Compass Click** | **Reset to True North** | When Compass Widget is visible (bottom right) | Click **`N`** on the bottom-right circular compass | Rotates camera smoothly to top-down view facing North ($+Z\text{ North}$, $+X\text{ East}$). |

---

## 2. Layer & Display Toggles

| Shortcut | Layer Name | Default | When It Works | Mouse / UI Alternative |
| :---: | :--- | :---: | :--- | :--- |
| **`1`** | **Point Cloud** | ON | 3D Viewport active; outside text fields | Click **`Pts`** icon button in viewport top toolbar |
| **`2`** | **Semantic Entities** | ON | 3D Viewport active; outside text fields | Click **`Ent`** icon button in viewport top toolbar |
| **`3`** | **Camera Cones** | ON | 3D Viewport active; outside text fields | Click **`Cam`** icon button in viewport top toolbar |
| **`W`** | **Wall Visibility** | ON | 3D Viewport active; outside text fields | Click **`Walls`** button in viewport top toolbar |
| **`T`** | **Spatial Topology** | ON | 3D Viewport active; outside text fields | Click **`Topo`** button in viewport top toolbar |
| **`G`** | **20m Ground Grid** | ON | 3D Viewport active; outside text fields | Click **`Grid`** button in viewport top toolbar |
| **`O`** | **Noise Suppression** | OFF | 3D Viewport active; outside text fields | Click **`Noise`** button in viewport top toolbar |
| **`U`** | **Uncertainty Heatmap**| OFF | 3D Viewport active; outside text fields | Click **`Uncert`** button in viewport top toolbar |
| **`M`** | **Distance Ruler** | OFF | 3D Viewport active; outside text fields | Click **`Ruler`** icon in viewport top toolbar |

---

## 3. Workspace Layout & Panel Toggles

| Shortcut | Panel Target | Default | Behavior | Mouse / UI Alternative |
| :---: | :--- | :---: | :--- | :--- |
| **`[`** | **Left Navigation Panel** | OPEN | Toggles 9-tab sidebar (280px) | Click **`[ [ ]`** button in top header bar |
| **`]`** | **Right Adaptive Inspector**| OPEN | Toggles entity inspector (340px) | Click **`[ ] ]`** button in top header bar |
| **`\`** | **Bottom Telemetry Console**| OPEN | Toggles coordinate/version bar | Click **`Terminal`** icon button in top header bar |
| **`⌘K` or `Ctrl+K`** | **Command Palette** | CLOSED | Opens global search modal | Click **`Commands ⌘K`** button in top header bar |

---

## 4. Measurement Tool Workflow (`M`)

When activated via **`M`** or the **`Ruler`** toolbar icon:
1. **Crosshair Active:** Mouse cursor switches to `cursor-crosshair`.
2. **First Click:** Click any surface or point. Cyan dot anchors at $(X_1, Y_1, Z_1)$.
3. **Second Click:** Click the target surface. A dashed measurement line connects $(X_1, Y_1, Z_1)$ to $(X_2, Y_2, Z_2)$.
4. **Readout HUD:** Floating badge displays:
   - **Distance:** Euclidean length in meters ($d = \sqrt{\Delta X^2 + \Delta Y^2 + \Delta Z^2}$).
   - **$\Delta X$:** East-West delta in meters.
   - **$\Delta Y$:** Elevation/Height delta in meters.
   - **$\Delta Z$:** North-South delta in meters.
5. **Exit:** Press **`Escape`** or click **Clear** to dismiss.
