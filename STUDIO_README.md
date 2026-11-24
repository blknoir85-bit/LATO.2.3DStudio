# LATO.2 3D Studio (Windows + AMD ROCm Edition)

An extended local 3D generation & retopology studio built on top of **[LoHhhha/LATO.2](https://github.com/LoHhhha/LATO.2)**, adding:
1. **Native Windows + AMD ROCm 7.2.1 support** (zero CUDA toolkit / `nvcc` required).
2. **Model, sampler, and geometry-pipeline upgrades** (2nd-order Heun ODE sampling, multi-engine dense scaffolds, voxel island pruning, deterministic manifold cleanup, surface-snap projection, hero-part generation, and Blender 5.2 normal baking & hard-surface tightening).
3. **A 4-tab local Gradio Web UI (`app.py`)** with isolated-subprocess VRAM management for 16 GB consumer GPUs.

---

## 1. Windows + AMD ROCm Port (No CUDA / `nvcc` Required)

Upstream LATO.2 targets Linux + CUDA 12.4 and builds several C++/CUDA extensions (`spconv`, `flash-attn`, `torch-scatter`, `nvdiffrast`, `CuMesh`, `FlexGEMM`, `o_voxel`). To run natively on **Windows + AMD RDNA3 (tested on Radeon RX 7800 XT 16 GB, ROCm 7.2.1, PyTorch 2.9.1)**, we added automatic pure-PyTorch/Open3D fallbacks that activate whenever CUDA extensions are absent (without breaking standard CUDA installs):

| Upstream CUDA Dependency | Pure-PyTorch / ROCm Replacement | File(s) |
|---|---|---|
| `spconv` (`SubMConv3d`) | Vectorized GPU `SubMConv3d` using 64-bit packed coordinate keys + `torch.searchsorted`, matching `spconv`'s exact `(out_c, kD, kH, kW, in_c)` weight layout | [`modules/sparse/conv/conv_torch.py`](modules/sparse/conv/conv_torch.py), [`modules/sparse/basic.py`](modules/sparse/basic.py), [`modules/sparse/__init__.py`](modules/sparse/__init__.py) |
| `flash-attn` / `xformers` | PyTorch native `F.scaled_dot_product_attention` (SDPA) with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` for AOTriton Flash Attention on RDNA3 | [`modules/sparse/attention/full_attn.py`](modules/sparse/attention/full_attn.py), [`serialized_attn.py`](modules/sparse/attention/serialized_attn.py), [`windowed_attn.py`](modules/sparse/attention/windowed_attn.py) |
| `torch_scatter.scatter_mean` | Pure-PyTorch `scatter_add_` mean reduction | [`modules/pointnet.py`](modules/pointnet.py) |
| `o_voxel` (`mesh_to_flexible_dual_grid`) | Open3D `VoxelGrid.create_from_triangle_mesh_within_bounds` | [`dataset/utils.py`](dataset/utils.py) |

### Critical ROCm Performance Setting
Without AOTriton experimental attention enabled, PyTorch ROCm falls back to the unfused O(N²) math SDPA kernel on RDNA3, causing multi-minute stalls on long sparse-voxel sequences. Setting:
```powershell
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1"
```
enables AOTriton Flash Attention automatically (also set by default in `modules/sparse/__init__.py`, `app.py`, and `launch_studio.bat`).

### Environment Setup (Windows + AMD ROCm 7.2.1)
Run [`setup_rocm_windows.ps1`](setup_rocm_windows.ps1) (requires [`uv`](https://docs.astral.sh/uv/) and a recent AMD Adrenalin driver matching ROCm 7.2.1):
```powershell
.\setup_rocm_windows.ps1
```
Then download the LATO.2 checkpoints:
```powershell
.venv\Scripts\python.exe scripts/ckpt_download.py
```

---

## 2. Model, Sampler & Geometry Pipeline Improvements

### A. 2nd-Order Heun ODE Sampler (`models/flow_sampler.py`)
Added a 2nd-order predictor-corrector **Heun ODE solver** (`--solver heun`) to both `VertFlowEulerCfgSampler` (V-Flow) and `TopoFlowEulerSampler` (T-Flow), exposed via `--solver {euler,heun}` in [`scripts/e2e_inference.py`](scripts/e2e_inference.py).

Measured head-to-head on the same 150k hard-surface mecha mesh (`megatron_hunyuan21.glb`, `--edge_threshold 0.0 --max_ring_size 18`, RX 7800 XT):

| Metric | Euler (`40`/`50` steps, 226s) | **Heun 2nd-Order (`24`/`30` steps, 218s)** | Improvement |
|---|---:|---:|---:|
| **Raw NMER** (non-manifold edges) | 0.95% | **0.64%** | **−33%** |
| **Raw BER** (boundary/open edges) | 12.86% | **8.88%** | **−31%** |
| **Raw Disconnected Components** | 167 | **86** | **−49%** |
| **Cleaned Faces / Vert** | 1.74 F/V | **1.83 F/V** | **+0.09 F/V** |
| **Cleaned BER** | 11.73% | **7.60%** | **−35%** |
| **Cleaned Open Loops (holes)** | 9 | **7** | fewer holes |
| **Cleaned Slivers (< 5°)** | 4.36% | **3.64%** | cleaner triangles |

> **Note on `--edge_threshold`**: LATO.2's `edge_threshold` is a **logit** threshold (default `0.0` = probability `0.50`). Keep it at `0.0` (or slightly negative like `-0.15`) rather than `+0.4` to avoid dropping valid edges.

### B. 64³ Voxel Island Pruning & Deterministic Manifold Cleanup
- **Pre-V-Flow Voxel Island Pruning** ([`scripts/image_e2e_inference.py`](scripts/image_e2e_inference.py)): Prunes small disconnected 26-connected floating voxel islands (`--prune_voxel_islands`, `--min_voxel_component_ratio 0.05`) from the 64³ TRELLIS occupancy grid before V-Flow runs, preventing wasted vertex budget and floating artifacts.
- **Topology Quality Gate & Manifold Cleanup** ([`utils/mesh_cleanup.py`](utils/mesh_cleanup.py), wired into [`utils/export.py`](utils/export.py)):
  - Automatically computes and logs **NMER**, **BER**, open boundary loops, connected components, sliver-triangle ratio (`< 5°`), watertightness, and winding consistency on every export.
  - Deterministically removes duplicate/degenerate/sliver triangles, prunes non-manifold fin faces (`> 2` faces per edge), drops tiny floating face islands, fills closed boundary loops up to 24 edges via minimum-area DP triangulation, and repairs triangle winding + outward normals (upstream `edges_to_faces` emits unoriented `[u, v, w]` triples).

### C. Stronger Dense Front-End Scaffolds + Surface-Snap Projection
LATO.2's built-in `image_e2e_inference.py` relies on TRELLIS's coarse 64³ single-image voxel stage. Replacing that front-end with a high-resolution dense image-to-3D generator and feeding the resulting watertight mesh into LATO.2's Mesh-to-Mesh pipeline (`scripts/e2e_inference.py`) dramatically improves silhouette and topology quality:
- **Hunyuan3D-2.1 (shape-only, 3.3B MoE DiT)** — [`scripts/hunyuan3d_scaffold.py`](scripts/hunyuan3d_scaffold.py): Sharpest hard-surface mechanical detail, `watertight=True`, pure PyTorch (`mc_algo="mc"`), stray-shard pruning, and automatic 150k-face QEM cap (prevents Open3D Vulkan buffer overflow during DINOv2 rendering).
- **TripoSG (1.5B Rectified Flow + SDF VAE, MIT)** — [`scripts/triposg_scaffold.py`](scripts/triposg_scaffold.py): Fast (~2.5s for 50 diffusion steps on RX 7800 XT), commercial-clear MIT weights, with finite-corner masked `skimage.measure.marching_cubes` (eliminating sparse-band NaN-cliff ripples) and stray-shard pruning.
- **Surface-Snap Projection** (`snap_mesh_to_reference` in [`utils/mesh_cleanup.py`](utils/mesh_cleanup.py)): Uses Open3D's Embree `RaycastingScene` closest-point queries to project LATO.2's low-poly vertices directly onto the high-poly dense scaffold surface (reducing mean surface drift to `~0.004–0.005` in unit-cube space with zero topology distortion).

Measured progression on the same single-image hard-surface character (`megatron.jpg`):

| Pipeline Stage | Cleaned V / F | NMER | BER | Open Loops | Components |
|---|---:|---:|---:|---:|---:|
| Upstream TRELLIS → LATO.2 (raw) | 7,458 / 9,397 (1.26 F/V) | 2.08% | 34.00% | 117 | 646 |
| TRELLIS + Voxel Pruning + Cleanup | 6,804 / 9,793 (1.44 F/V) | 0.15% | 25.10% | 10 | 61 |
| TripoSG → LATO.2 + Cleanup | 6,029 / 9,944 (1.65 F/V) | 0.06% | 16.69% | 1 | 14 |
| **Hunyuan3D-2.1 → LATO.2 (Heun) + Cleanup** | **7,718 / 14,127 (1.83 F/V)** | **0.10%** | **7.60%** | **7** | **17** |

### D. Hero-Part Cropping & Headless Blender 5.2 Automation
- **Hero-Part Generation**: Because DINOv2 encodes every input at `518×518`, small focal features (like a character's head or weapon) only receive ~30 ViT patches in a full-body shot. Cropping individual hero parts to fill the `518×518` frame gives them ~45× more visual tokens and ~5× higher effective 3D resolution.
- **2K/4K Tangent-Space Normal Baking** ([`scripts/bake_normals_blender.py`](scripts/bake_normals_blender.py)): Headless Blender Cycles script that Smart-UV unwraps the low-poly LATO.2 mesh, bakes a tangent-space normal map from the 150k high-poly master, wires it into Principled BSDF, and exports a ready-to-use `.glb` + `.png`.
- **Hard-Surface Tightening** ([`scripts/hardsurface_tighten_blender.py`](scripts/hardsurface_tighten_blender.py)): Headless Blender script applying optional bisect/mirror symmetry, planar-preserving dissolve, dihedral sharp-edge marking (`> 32°`), and face-area Weighted Normals to turn soft marching-cubes bevels into crisp machined planes.

---

## 3. LATO.2 3D Studio Web UI (`app.py`)

Launch the local Gradio Studio UI:
```powershell
.\launch_studio.bat
# or:
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1"
.venv\Scripts\python.exe app.py
```
Open **http://127.0.0.1:7860** (automatically falls back to the next open port if `7860` is busy).

### Studio Tabs
1. **Image → 3D Studio**: Full 1-click flow — choose front-end (`Hunyuan3D-2.1`, `TripoSG`, or `TRELLIS`), configure LATO.2 (`heun`/`euler`, target vertex count, steps, edge logit threshold), enable Surface-Snap + Blender 2K/4K Normal Bake, and inspect the result in an interactive 3D viewer with live Topology Quality Gate metrics.
2. **Mesh → LATO.2 Retopology**: Upload any `.glb`/`.obj`/`.ply` mesh to retopologize with LATO.2, surface-snap, and bake normals.
3. **Hero-Part Cropper & Generator**: Interactive percentage crop box + `518×518` preview → generate high-res hero parts with Hunyuan3D-2.1 or TripoSG → optional Blender hard-surface tightening.
4. **Hard-Surface Tighten & Output Browser**: Browse and inspect all generated `.glb`/`.obj` outputs in `outputs/` with live topology metrics, or run standalone Blender hard-surface tightening.
5. **Batch Queue (Multi-Image)**: Drop multiple images or point to a folder to run the entire pipeline unattended with automatic resume and CSV/JSON summary export.

---

## 4. Agent-Friendly Batch Pipeline (`scripts/batch_pipeline.py`)

For AI agents or unattended overnight runs, [`scripts/batch_pipeline.py`](scripts/batch_pipeline.py) processes either a folder of images (`--image_dir`) or a structured JSON job manifest (`--manifest`):

```powershell
# Process an entire folder (resumable, VRAM-isolated per item)
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1"
.venv\Scripts\python.exe scripts/batch_pipeline.py `
  --image_dir assets/batch_in `
  --out_dir outputs/batch_out `
  --scaffold hunyuan3d --solver heun --vert_num 4200 --snap --bake

# Or feed a JSON manifest with per-asset overrides
.venv\Scripts\python.exe scripts/batch_pipeline.py --manifest jobs.json --out_dir outputs/batch_out
```

Example `jobs.json` manifest:
```json
[
  {"id": "mecha_01", "image": "assets/mecha.jpg", "scaffold": "hunyuan3d", "vert_num": 4200, "bake": true},
  {"id": "prop_02",  "image": "assets/crate.png", "scaffold": "triposg",   "vert_num": 2500, "lato2": false, "tighten": true}
]
```

### Agent-Ready Outputs
- `<out_dir>/<id>/result.json`: Per-asset status (`ok`/`failed`), elapsed time, exact paths (`dense_scaffold`, `lato2_mesh`, `snapped_mesh`, `baked_glb`, `normal_map`, `final_mesh`), and full Topology Quality Gate metrics (`nmer_pct`, `ber_pct`, `num_open_loops`, `num_components`, `faces_per_vert`).
- `<out_dir>/batch_summary.json` & `<out_dir>/batch_summary.csv`: Updated live after every item; `--resume` (default) skips already-completed items cleanly.

> **VRAM Management**: Every pipeline stage runs in an isolated subprocess so VRAM is 100% released between Hunyuan3D-2.1/TripoSG, LATO.2, and Blender — running comfortably on a 16 GB GPU.

---

## License & Provenance Notes
- **LATO.2** code: see [`LICENSE`](LICENSE) (upstream [LoHhhha/LATO.2](https://github.com/LoHhhha/LATO.2)).
- **TripoSG** weights (`VAST-AI/TripoSG`): MIT License.
- **Hunyuan3D-2.1** weights (`tencent/Hunyuan3D-2.1`): Tencent Hunyuan Community License (check regional/commercial terms before redistribution).
