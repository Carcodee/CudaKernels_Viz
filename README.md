# CodeCuda Visualization

<img width="1532" height="980" alt="image" src="https://github.com/user-attachments/assets/9564261a-10d1-420c-bd0a-066eeb1b8c6a" />

Interactive browser visualization for CUDA-style launch geometry.

This project renders an explicit `grid -> block -> thread` launch model in 3D and keeps the math visible in the UI. Grid size is the number of dispatched blocks. Block size is the number of threads inside each block. Nothing is inferred by implicit division.

## What It Shows

- Combined launch view with resident blocks, queued blocks, thread cubes, SM grouping, and shared-memory indicators
- Separate 3D viewport with camera controls and animated mesh spawning
- Grid and block configuration controls with live validation and thread-budget limits
- GPU occupancy and residency estimates for built-in GPU profiles
- Hierarchy view for a simpler structural summary of the same launch
- Thread mode and warp mode color visualization

## Project Layout

- `index.html`: app structure and panels
- `styles.css`: page styling and viewport motion
- `app.js`: UI logic, Three.js scene construction, camera behavior, spawning animations, hover/selection behavior
- `src/launchMath.js`: launch math, occupancy math, residency math, and GPU profiles
- `vendor/three.min.js`: vendored Three.js build
- `tests/launchMath.test.js`: regression tests for math and UI/layout contracts

## Run Locally

The app is static and only needs a local HTTP server.

### Windows

```bat
run_visualization.bat
```

Then open:

```text
http://127.0.0.1:8000/visualization/
```

### Manual server

From the project parent directory:

```bash
py -3 -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000/visualization/
```

## Test

```bash
npm test
```

Equivalent direct command:

```bash
node --test tests/launchMath.test.js
```

## Interaction Model

- Left drag: orbit camera
- Right drag: pan camera
- Mouse wheel: zoom
- Hover: inspect blocks and threads
- Click resident block in dense launches: scope thread hover to that block
- Reset View: fit camera back to the current structure

The camera also auto-fits when layout-driving configuration changes, so the full structure stays visible after grid, block, GPU, or residency-affecting updates.

## Built-In GPU Profiles

Current presets in `src/launchMath.js`:

- `H100`
- `A100`
- `RTX4090`

These are used for occupancy, active blocks per SM, shared-memory limits, and queued/resident block estimation.

## Rendering Notes

- Uses a vendored local Three.js build, not a CDN
- Heavy geometry is rendered with `InstancedMesh` to keep large launches practical
- Spawn animations are applied at the scene-object batch level so they stay cheap even for dense launches
- Hover overlays and labels are rebuilt separately from the main instanced geometry

## Current Scope

This is a visualization and teaching tool for launch structure and GPU residency behavior. It is not a CUDA runtime, profiler, or simulator of exact hardware scheduling.
