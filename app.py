"""
LATO.2 3D Studio — Local Gradio Web UI for the full Image -> Dense Scaffold ->
LATO.2 Retopology (Heun/Euler) -> Surface-Snap -> Blender 2K Normal Bake &
Hard-Surface Tightening pipeline on AMD ROCm / CUDA.

Run:
    .venv\\Scripts\\python.exe app.py
"""

import glob
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

import gradio as gr
import numpy as np
import trimesh
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
sys.path.insert(0, ROOT)

from utils.mesh_cleanup import (
    compute_topology_metrics,
    format_topology_report,
    snap_mesh_to_reference,
)


def _find_blender() -> str:
    candidates = [
        r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    found = shutil.which("blender")
    return found or ""


BLENDER_EXE = _find_blender()


def _base_env() -> dict:
    env = os.environ.copy()
    env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_subprocess(cmd: list[str], log_lines: list[str]):
    """Run a command, streaming stdout/stderr lines into log_lines."""
    log_lines.append(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=_base_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log_lines.append(line)
            yield "\n".join(log_lines[-80:])
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (exit {rc}): {cmd[0]}")


def _make_run_dir(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "outputs", "studio_runs", f"{prefix}_{ts}")
    os.makedirs(d, exist_ok=True)
    return d


def _list_existing_models() -> list[str]:
    patterns = [
        os.path.join(ROOT, "outputs", "**", "*.glb"),
        os.path.join(ROOT, "outputs", "**", "*.obj"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files), key=lambda p: os.path.getmtime(p), reverse=True)
    return [os.path.relpath(f, ROOT) for f in files]


# ---------------------------------------------------------------------------
# Tab 1: Full Image -> Clean 3D Pipeline
# ---------------------------------------------------------------------------
def run_full_pipeline(
    image_path,
    scaffold_engine,
    octree_res,
    scaffold_steps,
    scaffold_cfg,
    run_lato2,
    vert_num,
    solver,
    vflow_steps,
    tflow_steps,
    edge_threshold,
    max_ring_size,
    do_snap,
    do_bake,
    bake_res,
    do_tighten,
    smooth_iters,
    planar_angle,
    sharp_angle,
    seed,
):
    if not image_path:
        yield None, None, None, "Please upload an input image first.", ""
        return

    run_dir = _make_run_dir("full")
    stem = os.path.splitext(os.path.basename(image_path))[0]
    in_img = os.path.join(run_dir, f"{stem}.png")
    Image.open(image_path).convert("RGB").save(in_img)

    logs = [f"Run directory: {run_dir}"]
    dense_mesh_path = None
    final_display = None
    normal_map_path = None
    metrics_md = "Running..."

    try:
        # ---- Stage 1: Scaffold ----
        if scaffold_engine == "TRELLIS (Built-in LATO.2 64³)":
            img_dir = os.path.join(run_dir, "input_img")
            os.makedirs(img_dir, exist_ok=True)
            shutil.copy2(in_img, os.path.join(img_dir, f"{stem}.png"))
            out_lato = os.path.join(run_dir, "lato2")
            cmd = [
                PYTHON, "scripts/image_e2e_inference.py",
                "--image_dir", img_dir,
                "--out_dir", out_lato,
                "--vert_num", str(int(vert_num)),
                "--ss_steps", str(int(scaffold_steps)),
                "--ss_cfg_strength", str(float(scaffold_cfg)),
                "--vflow_steps", str(int(vflow_steps)),
                "--tflow_steps", str(int(tflow_steps)),
                "--edge_threshold", str(float(edge_threshold)),
                "--max_ring_size", str(int(max_ring_size)),
                "--seed", str(int(seed)),
            ]
            for log_txt in _run_subprocess(cmd, logs):
                yield gr.skip(), gr.skip(), gr.skip(), "Stage 1/1: Running TRELLIS -> LATO.2...", log_txt
            pred_obj = os.path.join(out_lato, f"{stem}_pred.obj")
            if os.path.isfile(pred_obj):
                final_display = pred_obj
                m = trimesh.load(pred_obj, process=False)
                metrics_md = f"```\n{format_topology_report(stem, compute_topology_metrics(m))}\n```"
            yield final_display, gr.skip(), gr.skip(), metrics_md, "\n".join(logs[-80:])
            return

        scaffold_dir = os.path.join(run_dir, "scaffold")
        os.makedirs(scaffold_dir, exist_ok=True)
        dense_mesh_path = os.path.join(scaffold_dir, f"{stem}_scaffold.glb")

        if scaffold_engine == "Hunyuan3D-2.1 (Sharpest Hard-Surface)":
            cmd = [
                PYTHON, "scripts/hunyuan3d_scaffold.py",
                "--image", in_img,
                "--out_mesh", dense_mesh_path,
                "--steps", str(int(scaffold_steps)),
                "--guidance_scale", str(float(scaffold_cfg)),
                "--octree_resolution", str(int(octree_res)),
                "--seed", str(int(seed)),
                "--target_faces", "150000",
            ]
        else:  # TripoSG
            cmd = [
                PYTHON, "scripts/triposg_scaffold.py",
                "--image", in_img,
                "--out_mesh", dense_mesh_path,
                "--steps", str(int(scaffold_steps)),
                "--guidance_scale", str(float(scaffold_cfg)),
                "--seed", str(int(seed)),
                "--target_faces", "150000",
            ]

        for log_txt in _run_subprocess(cmd, logs):
            yield gr.skip(), gr.skip(), gr.skip(), f"Stage 1: Generating dense scaffold with {scaffold_engine}...", log_txt

        final_display = dense_mesh_path
        dm = trimesh.load(dense_mesh_path, force="mesh", process=False)
        metrics_md = f"**Dense Scaffold**: V={len(dm.vertices):,}, F={len(dm.faces):,}, watertight={dm.is_watertight}"
        # Push the dense scaffold once so the 3D viewer renders it cleanly while Stage 2 runs
        yield dense_mesh_path, dense_mesh_path, gr.skip(), metrics_md, "\n".join(logs[-80:])

        # Optional Hard-Surface Tighten on dense scaffold when skipping LATO.2
        if do_tighten and not run_lato2 and BLENDER_EXE:
            tight_glb = os.path.join(run_dir, f"{stem}_tight.glb")
            cmd = [
                BLENDER_EXE, "-b", "--python", "scripts/hardsurface_tighten_blender.py", "--",
                "--in_mesh", dense_mesh_path,
                "--out_mesh", tight_glb,
                "--mirror_axis", "NONE",
                "--smooth_iters", str(int(smooth_iters)),
                "--planar_angle", str(float(planar_angle)),
                "--sharp_angle", str(float(sharp_angle)),
            ]
            for log_txt in _run_subprocess(cmd, logs):
                yield gr.skip(), gr.skip(), gr.skip(), "Applying Blender hard-surface tightening...", log_txt
            final_display = tight_glb

        if not run_lato2:
            yield final_display, dense_mesh_path, None, metrics_md, "\n".join(logs[-80:])
            return

        # ---- Stage 2: LATO.2 Retopology ----
        out_lato = os.path.join(run_dir, "lato2")
        cmd = [
            PYTHON, "scripts/e2e_inference.py",
            "--mesh_dir", scaffold_dir,
            "--out_dir", out_lato,
            "--vert_num", str(int(vert_num)),
            "--solver", str(solver),
            "--vflow_steps", str(int(vflow_steps)),
            "--tflow_steps", str(int(tflow_steps)),
            "--edge_threshold", str(float(edge_threshold)),
            "--max_ring_size", str(int(max_ring_size)),
            "--seed", str(int(seed)),
            "--num_workers", "0",
        ]
        for log_txt in _run_subprocess(cmd, logs):
            yield gr.skip(), gr.skip(), gr.skip(), f"Stage 2: Running LATO.2 ({solver})...", log_txt

        pred_obj = os.path.join(out_lato, f"{stem}_scaffold_pred.obj")
        if not os.path.isfile(pred_obj):
            raise RuntimeError("LATO.2 skipped topology. Try lowering Target Vertices (e.g. 4200).")

        low_mesh_path = pred_obj
        low_mesh = trimesh.load(pred_obj, process=False)
        topo_report = format_topology_report("LATO.2 cleaned", compute_topology_metrics(low_mesh))

        # ---- Stage 3: Surface-Snap ----
        if do_snap:
            logs.append("Stage 3: Surface-snapping LATO.2 vertices onto dense scaffold...")
            yield gr.skip(), gr.skip(), gr.skip(), "Stage 3: Surface-snapping...", "\n".join(logs[-80:])
            snapped, snap_stats = snap_mesh_to_reference(
                low_mesh, dm, normalize_ref=True, max_snap_dist=0.04, blend=1.0
            )
            snapped_obj = os.path.join(out_lato, f"{stem}_snapped.obj")
            snapped.export(snapped_obj)
            low_mesh_path = snapped_obj
            logs.append(f"Surface-snap done: mean_dist={snap_stats['mean_dist']:.4f}, max={snap_stats['max_dist']:.4f}")

        final_display = low_mesh_path

        # ---- Stage 4: Blender 2K Normal Bake ----
        if do_bake and BLENDER_EXE:
            baked_glb = os.path.join(run_dir, f"{stem}_lato2_baked.glb")
            normal_map_path = os.path.join(run_dir, f"{stem}_normal.png")
            cmd = [
                BLENDER_EXE, "-b", "--python", "scripts/bake_normals_blender.py", "--",
                "--high", dense_mesh_path,
                "--low", low_mesh_path,
                "--out_glb", baked_glb,
                "--out_normal", normal_map_path,
                "--res", str(int(bake_res)),
            ]
            for log_txt in _run_subprocess(cmd, logs):
                yield gr.skip(), gr.skip(), gr.skip(), f"Stage 4: Baking {int(bake_res)}x{int(bake_res)} normal map in Blender...", log_txt
            final_display = baked_glb

        metrics_md = (
            f"### Completed!\n"
            f"- **Final Mesh**: `{os.path.relpath(final_display, ROOT)}`\n"
            f"- **Dense Scaffold**: `{os.path.relpath(dense_mesh_path, ROOT)}`\n\n"
            f"```\n{topo_report}\n```"
        )
        yield final_display, dense_mesh_path, normal_map_path, metrics_md, "\n".join(logs[-80:])

    except Exception as e:
        logs.append(f"ERROR: {e}")
        yield gr.skip(), gr.skip(), gr.skip(), f"**Error:** `{e}`", "\n".join(logs[-80:])


# ---------------------------------------------------------------------------
# Tab 2: Mesh -> LATO.2 Retopology
# ---------------------------------------------------------------------------
def run_mesh_retopo(
    mesh_file,
    vert_num,
    solver,
    vflow_steps,
    tflow_steps,
    edge_threshold,
    max_ring_size,
    do_snap,
    do_bake,
    bake_res,
    seed,
):
    if not mesh_file:
        yield None, None, "Please upload a 3D mesh (.glb, .obj, .ply).", ""
        return

    run_dir = _make_run_dir("retopo")
    in_dir = os.path.join(run_dir, "in_mesh")
    os.makedirs(in_dir, exist_ok=True)
    base = os.path.basename(mesh_file)
    stem = os.path.splitext(base)[0]
    in_mesh = os.path.join(in_dir, base)
    shutil.copy2(mesh_file, in_mesh)
    out_lato = os.path.join(run_dir, "lato2")

    logs = [f"Retopologizing {base} -> {run_dir}"]
    final_mesh = None
    normal_png = None

    try:
        cmd = [
            PYTHON, "scripts/e2e_inference.py",
            "--mesh_dir", in_dir,
            "--out_dir", out_lato,
            "--vert_num", str(int(vert_num)),
            "--solver", str(solver),
            "--vflow_steps", str(int(vflow_steps)),
            "--tflow_steps", str(int(tflow_steps)),
            "--edge_threshold", str(float(edge_threshold)),
            "--max_ring_size", str(int(max_ring_size)),
            "--seed", str(int(seed)),
            "--num_workers", "0",
        ]
        for log_txt in _run_subprocess(cmd, logs):
            yield gr.skip(), gr.skip(), f"Running LATO.2 ({solver})...", log_txt

        pred_obj = os.path.join(out_lato, f"{stem}_pred.obj")
        if not os.path.isfile(pred_obj):
            raise RuntimeError("LATO.2 skipped topology (likely > 10,000 generated vertices). Try lowering Target Vertices.")

        low = trimesh.load(pred_obj, process=False)
        topo_report = format_topology_report(stem, compute_topology_metrics(low))
        low_path = pred_obj

        if do_snap:
            ref = trimesh.load(in_mesh, force="mesh", process=False)
            snapped, stats = snap_mesh_to_reference(low, ref, normalize_ref=True, max_snap_dist=0.04, blend=1.0)
            low_path = os.path.join(out_lato, f"{stem}_snapped.obj")
            snapped.export(low_path)
            logs.append(f"Snapped: mean_dist={stats['mean_dist']:.4f}")

        final_mesh = low_path
        if do_bake and BLENDER_EXE:
            baked_glb = os.path.join(run_dir, f"{stem}_baked.glb")
            normal_png = os.path.join(run_dir, f"{stem}_normal.png")
            cmd = [
                BLENDER_EXE, "-b", "--python", "scripts/bake_normals_blender.py", "--",
                "--high", in_mesh,
                "--low", low_path,
                "--out_glb", baked_glb,
                "--out_normal", normal_png,
                "--res", str(int(bake_res)),
            ]
            for log_txt in _run_subprocess(cmd, logs):
                yield gr.skip(), gr.skip(), "Baking normal map in Blender...", log_txt
            final_mesh = baked_glb

        yield final_mesh, normal_png, f"```\n{topo_report}\n```", "\n".join(logs[-80:])
    except Exception as e:
        logs.append(f"ERROR: {e}")
        yield final_mesh, normal_png, f"**Error:** `{e}`", "\n".join(logs[-80:])


# ---------------------------------------------------------------------------
# Tab 3: Hero-Part Cropper & Generator
# ---------------------------------------------------------------------------
def preview_crop(image_path, x0_pct, y0_pct, x1_pct, y1_pct, pad_ratio):
    if not image_path:
        return None
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    x0 = int(W * min(x0_pct, x1_pct) / 100.0)
    x1 = max(x0 + 4, int(W * max(x0_pct, x1_pct) / 100.0))
    y0 = int(H * min(y0_pct, y1_pct) / 100.0)
    y1 = max(y0 + 4, int(H * max(y0_pct, y1_pct) / 100.0))
    crop = im.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    side = int(max(cw, ch) * float(pad_ratio))
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(crop, ((side - cw) // 2, (side - ch) // 2))
    return canvas.resize((518, 518), Image.Resampling.LANCZOS)


def run_hero_part(
    image_path,
    x0_pct, y0_pct, x1_pct, y1_pct, pad_ratio,
    part_name,
    engine,
    octree_res,
    steps,
    cfg,
    do_tighten,
    mirror_axis,
    planar_angle,
    sharp_angle,
    seed,
):
    if not image_path:
        yield None, None, "Please upload an image first.", ""
        return

    run_dir = _make_run_dir(f"hero_{part_name or 'part'}")
    crop_img = preview_crop(image_path, x0_pct, y0_pct, x1_pct, y1_pct, pad_ratio)
    crop_path = os.path.join(run_dir, "crop_518.png")
    crop_img.save(crop_path)

    raw_glb = os.path.join(run_dir, f"{part_name or 'part'}_raw.glb")
    logs = [f"Hero part run: {run_dir}"]
    final_glb = None

    try:
        if engine == "Hunyuan3D-2.1":
            cmd = [
                PYTHON, "scripts/hunyuan3d_scaffold.py",
                "--image", crop_path,
                "--out_mesh", raw_glb,
                "--steps", str(int(steps)),
                "--guidance_scale", str(float(cfg)),
                "--octree_resolution", str(int(octree_res)),
                "--seed", str(int(seed)),
                "--target_faces", "120000",
            ]
        else:
            cmd = [
                PYTHON, "scripts/triposg_scaffold.py",
                "--image", crop_path,
                "--out_mesh", raw_glb,
                "--steps", str(int(steps)),
                "--guidance_scale", str(float(cfg)),
                "--seed", str(int(seed)),
                "--target_faces", "120000",
            ]

        yield crop_img, gr.skip(), f"Generating hero part with {engine}...", "\n".join(logs[-80:])
        for log_txt in _run_subprocess(cmd, logs):
            yield gr.skip(), gr.skip(), f"Generating hero part with {engine}...", log_txt

        final_glb = raw_glb
        if do_tighten and BLENDER_EXE:
            tight_glb = os.path.join(run_dir, f"{part_name or 'part'}_tight.glb")
            cmd = [
                BLENDER_EXE, "-b", "--python", "scripts/hardsurface_tighten_blender.py", "--",
                "--in_mesh", raw_glb,
                "--out_mesh", tight_glb,
                "--mirror_axis", str(mirror_axis),
                "--planar_angle", str(float(planar_angle)),
                "--sharp_angle", str(float(sharp_angle)),
            ]
            for log_txt in _run_subprocess(cmd, logs):
                yield gr.skip(), gr.skip(), "Tightening hard surfaces in Blender...", log_txt
            final_glb = tight_glb

        yield crop_img, final_glb, f"Saved: `{os.path.relpath(final_glb, ROOT)}`", "\n".join(logs[-80:])
    except Exception as e:
        logs.append(f"ERROR: {e}")
        yield crop_img, final_glb, f"**Error:** `{e}`", "\n".join(logs[-80:])


# ---------------------------------------------------------------------------
# Tab 4: Blender Tighten / Bake / Output Gallery
# ---------------------------------------------------------------------------
def run_standalone_tighten(mesh_file, mirror_axis, smooth_iters, planar_angle, sharp_angle):
    if not mesh_file or not BLENDER_EXE:
        yield None, "Provide a mesh and ensure Blender is installed.", ""
        return
    run_dir = _make_run_dir("tighten")
    stem = os.path.splitext(os.path.basename(mesh_file))[0]
    out_glb = os.path.join(run_dir, f"{stem}_tight.glb")
    logs = []
    cmd = [
        BLENDER_EXE, "-b", "--python", "scripts/hardsurface_tighten_blender.py", "--",
        "--in_mesh", mesh_file,
        "--out_mesh", out_glb,
        "--mirror_axis", str(mirror_axis),
        "--smooth_iters", str(int(smooth_iters)),
        "--planar_angle", str(float(planar_angle)),
        "--sharp_angle", str(float(sharp_angle)),
    ]
    try:
        for log_txt in _run_subprocess(cmd, logs):
            yield None, "Running Blender tightening...", log_txt
        yield out_glb, f"Saved: `{os.path.relpath(out_glb, ROOT)}`", "\n".join(logs[-80:])
    except Exception as e:
        yield None, f"**Error:** `{e}`", "\n".join(logs[-80:])


def inspect_existing_model(rel_path):
    if not rel_path:
        return None, ""
    full = os.path.join(ROOT, rel_path)
    if not os.path.isfile(full):
        return None, "File not found."
    m = trimesh.load(full, force="mesh", process=False)
    rep = format_topology_report(os.path.basename(full), compute_topology_metrics(m))
    return full, f"```\n{rep}\n```"


# ---------------------------------------------------------------------------
# Tab 5: Batch Queue (Multi-Image / Agent Mode)
# ---------------------------------------------------------------------------
def run_batch_ui(
    files,
    folder_path,
    out_dir_rel,
    scaffold,
    octree_res,
    run_lato2,
    vert_num,
    solver,
    do_snap,
    do_bake,
    bake_res,
    resume,
):
    in_dir = None
    if files:
        batch_dir = _make_run_dir("batch_uploads")
        in_dir = os.path.join(batch_dir, "images")
        os.makedirs(in_dir, exist_ok=True)
        for f in files:
            src = f if isinstance(f, str) else getattr(f, "name", str(f))
            shutil.copy2(src, os.path.join(in_dir, os.path.basename(src)))
    elif folder_path and os.path.isdir(folder_path):
        in_dir = folder_path
    else:
        yield "Upload one or more images or specify a valid folder path.", "", None
        return

    out_dir = out_dir_rel.strip() if out_dir_rel and out_dir_rel.strip() else os.path.join("outputs", "batch_runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    cmd = [
        PYTHON, "scripts/batch_pipeline.py",
        "--image_dir", in_dir,
        "--out_dir", out_dir,
        "--scaffold", scaffold,
        "--octree_resolution", str(int(octree_res)),
        "--vert_num", str(int(vert_num)),
        "--solver", solver,
        "--bake_res", str(int(bake_res)),
        "--lato2" if run_lato2 else "--no-lato2",
        "--snap" if do_snap else "--no-snap",
        "--bake" if do_bake else "--no-bake",
        "--resume" if resume else "--no-resume",
    ]
    logs = []
    summary_csv = os.path.join(ROOT, out_dir, "batch_summary.csv") if not os.path.isabs(out_dir) else os.path.join(out_dir, "batch_summary.csv")
    try:
        for log_txt in _run_subprocess(cmd, logs):
            yield f"Running batch -> `{out_dir}`...", log_txt, gr.skip()
        yield f"### Batch Finished!\n- Output folder: `{out_dir}`\n- Summary JSON: `{os.path.join(out_dir, 'batch_summary.json')}`", "\n".join(logs[-100:]), summary_csv if os.path.isfile(summary_csv) else None
    except Exception as e:
        yield f"**Error:** `{e}`", "\n".join(logs[-100:]), summary_csv if os.path.isfile(summary_csv) else None


# ---------------------------------------------------------------------------
# Quality Presets for Tab 1
# ---------------------------------------------------------------------------
QUALITY_PRESETS = {
    "Fast (~30s)": dict(
        engine="TripoSG (MIT, Fast)",
        octree=380, s_steps=30, s_cfg=7.0,
        run_lato=False, vnum=3000, solver="heun", vsteps=14, tsteps=18,
        snap=True, bake=False, bake_res=1024, tighten=True, smooth=3,
    ),
    "Medium (~2m)": dict(
        engine="Hunyuan3D-2.1 (Sharpest Hard-Surface)",
        octree=380, s_steps=25, s_cfg=5.0,
        run_lato=True, vnum=3600, solver="heun", vsteps=14, tsteps=18,
        snap=True, bake=bool(BLENDER_EXE), bake_res=1024, tighten=False, smooth=3,
    ),
    "High Quality (~4m)": dict(
        engine="Hunyuan3D-2.1 (Sharpest Hard-Surface)",
        octree=380, s_steps=40, s_cfg=5.0,
        run_lato=True, vnum=4200, solver="heun", vsteps=20, tsteps=26,
        snap=True, bake=bool(BLENDER_EXE), bake_res=2048, tighten=False, smooth=3,
    ),
    "Ultra Quality (~7m)": dict(
        engine="Hunyuan3D-2.1 (Sharpest Hard-Surface)",
        octree=512, s_steps=50, s_cfg=5.5,
        run_lato=True, vnum=5000, solver="heun", vsteps=24, tsteps=30,
        snap=True, bake=bool(BLENDER_EXE), bake_res=4096, tighten=False, smooth=3,
    ),
}


def apply_quality_preset(preset_name: str):
    p = QUALITY_PRESETS.get(preset_name, QUALITY_PRESETS["High Quality (~4m)"])
    return (
        p["engine"],
        p["octree"],
        p["s_steps"],
        p["s_cfg"],
        p["run_lato"],
        p["vnum"],
        p["solver"],
        p["vsteps"],
        p["tsteps"],
        p["snap"],
        p["bake"],
        p["bake_res"],
        p["tighten"],
        p["smooth"],
    )


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="LATO.2 3D Studio") as demo:
    gr.Markdown(
        f"# LATO.2 3D Studio\n"
        f"**AMD ROCm Native Pipeline** · Hunyuan3D-2.1 / TripoSG / TRELLIS → LATO.2 (Heun/Euler) → Surface-Snap → Blender 2K Normal Bake & Hard-Surface Tightening  \n"
        f"Blender CLI: `{BLENDER_EXE or 'Not found'}`"
    )

    # ===== TAB 1: FULL PIPELINE =====
    with gr.Tab("1. Image → 3D Studio"):
        with gr.Row():
            with gr.Column(scale=1):
                t1_img = gr.Image(type="filepath", label="Input Reference Image")
                t1_preset = gr.Radio(
                    list(QUALITY_PRESETS.keys()),
                    value="High Quality (~4m)",
                    label="Quality Preset (auto-fills all settings below)",
                )
                t1_engine = gr.Radio(
                    [
                        "Hunyuan3D-2.1 (Sharpest Hard-Surface)",
                        "TripoSG (MIT, Fast)",
                        "TRELLIS (Built-in LATO.2 64³)",
                    ],
                    value="Hunyuan3D-2.1 (Sharpest Hard-Surface)",
                    label="Front-End Scaffold Engine",
                )
                with gr.Accordion("Scaffold Settings", open=False):
                    t1_octree = gr.Slider(256, 512, value=380, step=16, label="Hunyuan3D Octree Resolution")
                    t1_s_steps = gr.Slider(10, 60, value=50, step=5, label="Scaffold Steps")
                    t1_s_cfg = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Scaffold CFG")
                    t1_seed = gr.Number(value=42, precision=0, label="Seed")

                t1_run_lato = gr.Checkbox(value=True, label="Run LATO.2 Retopology (V-Flow → T-Flow)")
                with gr.Accordion("LATO.2 Retopology Settings", open=True):
                    t1_vnum = gr.Slider(1000, 7000, value=4200, step=200, label="Target Vertex Count (keep ≤5000 for Heun)")
                    t1_solver = gr.Radio(["heun", "euler"], value="heun", label="ODE Solver (Heun = 2nd-order, lower BER/NMER)")
                    with gr.Row():
                        t1_vsteps = gr.Slider(12, 60, value=24, step=2, label="V-Flow Steps")
                        t1_tsteps = gr.Slider(15, 60, value=30, step=5, label="T-Flow Steps")
                    with gr.Row():
                        t1_ethresh = gr.Slider(-0.5, 0.5, value=0.0, step=0.05, label="T-Flow Edge Logit Threshold")
                        t1_ring = gr.Slider(6, 24, value=18, step=2, label="Max Open Ring Fill Size")

                with gr.Accordion("Post-Processing & Baking", open=True):
                    t1_snap = gr.Checkbox(value=True, label="Surface-Snap LATO.2 onto Dense Scaffold")
                    t1_bake = gr.Checkbox(value=bool(BLENDER_EXE), label="Bake Tangent-Space Normal Map in Blender")
                    t1_bake_res = gr.Radio([1024, 2048, 4096], value=2048, label="Normal Map Resolution")
                    t1_tighten = gr.Checkbox(value=False, label="Hard-Surface Tighten (when LATO.2 is off)")
                    with gr.Row():
                        t1_smooth = gr.Slider(0, 12, value=3, step=1, label="Volume-Preserving Smooth Iters")
                        t1_planar = gr.Slider(0.0, 8.0, value=3.5, step=0.5, label="Planar Dissolve Angle (°)")
                        t1_sharp = gr.Slider(15.0, 60.0, value=32.0, step=1.0, label="Sharp Edge Angle (°)")

                t1_btn = gr.Button("Generate 3D Asset", variant="primary")

            with gr.Column(scale=2):
                t1_model = gr.Model3D(label="Final 3D Output (Interactive)", height=480)
                with gr.Row():
                    t1_dense = gr.File(label="Dense Scaffold (.glb)")
                    t1_normal = gr.Image(label="Baked Normal Map", height=220)
                t1_metrics = gr.Markdown(label="Topology Quality Gate")
                t1_logs = gr.Textbox(label="Live Pipeline Log", lines=10, max_lines=16)

        t1_preset.change(
            apply_quality_preset,
            inputs=[t1_preset],
            outputs=[
                t1_engine, t1_octree, t1_s_steps, t1_s_cfg,
                t1_run_lato, t1_vnum, t1_solver, t1_vsteps, t1_tsteps,
                t1_snap, t1_bake, t1_bake_res, t1_tighten, t1_smooth,
            ],
        )
        t1_btn.click(
            run_full_pipeline,
            inputs=[
                t1_img, t1_engine, t1_octree, t1_s_steps, t1_s_cfg,
                t1_run_lato, t1_vnum, t1_solver, t1_vsteps, t1_tsteps,
                t1_ethresh, t1_ring, t1_snap, t1_bake, t1_bake_res,
                t1_tighten, t1_smooth, t1_planar, t1_sharp, t1_seed,
            ],
            outputs=[t1_model, t1_dense, t1_normal, t1_metrics, t1_logs],
        )

    # ===== TAB 2: MESH -> LATO.2 RETOPOLOGY =====
    with gr.Tab("2. Mesh → LATO.2 Retopology"):
        with gr.Row():
            with gr.Column(scale=1):
                t2_mesh = gr.File(label="Input Dense Mesh (.glb / .obj / .ply)")
                t2_vnum = gr.Slider(1000, 7000, value=4200, step=200, label="Target Vertex Count")
                t2_solver = gr.Radio(["heun", "euler"], value="heun", label="ODE Solver")
                with gr.Row():
                    t2_vsteps = gr.Slider(12, 60, value=24, step=2, label="V-Flow Steps")
                    t2_tsteps = gr.Slider(15, 60, value=30, step=5, label="T-Flow Steps")
                with gr.Row():
                    t2_ethresh = gr.Slider(-0.5, 0.5, value=0.0, step=0.05, label="Edge Logit Threshold")
                    t2_ring = gr.Slider(6, 24, value=18, step=2, label="Max Ring Size")
                t2_snap = gr.Checkbox(value=True, label="Surface-Snap to Input Mesh")
                t2_bake = gr.Checkbox(value=bool(BLENDER_EXE), label="Bake Normal Map in Blender")
                t2_bake_res = gr.Radio([1024, 2048, 4096], value=2048, label="Normal Map Resolution")
                t2_seed = gr.Number(value=42, precision=0, label="Seed")
                t2_btn = gr.Button("Retopologize Mesh", variant="primary")
            with gr.Column(scale=2):
                t2_out_model = gr.Model3D(label="Retopologized Output", height=460)
                t2_out_nrm = gr.Image(label="Baked Normal Map", height=200)
                t2_metrics = gr.Markdown()
                t2_logs = gr.Textbox(label="Live Log", lines=10)

        t2_btn.click(
            run_mesh_retopo,
            inputs=[
                t2_mesh, t2_vnum, t2_solver, t2_vsteps, t2_tsteps,
                t2_ethresh, t2_ring, t2_snap, t2_bake, t2_bake_res, t2_seed,
            ],
            outputs=[t2_out_model, t2_out_nrm, t2_metrics, t2_logs],
        )

    # ===== TAB 3: HERO-PART CROPPER & GENERATOR =====
    with gr.Tab("3. Hero-Part Cropper & Generator"):
        with gr.Row():
            with gr.Column(scale=1):
                t3_img = gr.Image(type="filepath", label="Source Image")
                t3_name = gr.Textbox(value="hero_head", label="Part Name")
                with gr.Row():
                    t3_x0 = gr.Slider(0, 100, value=50, step=1, label="Left X (%)")
                    t3_x1 = gr.Slider(0, 100, value=70, step=1, label="Right X (%)")
                with gr.Row():
                    t3_y0 = gr.Slider(0, 100, value=0, step=1, label="Top Y (%)")
                    t3_y1 = gr.Slider(0, 100, value=16, step=1, label="Bottom Y (%)")
                t3_pad = gr.Slider(1.05, 1.6, value=1.35, step=0.05, label="Padding Ratio")
                t3_preview_btn = gr.Button("Preview 518×518 Crop")
                t3_engine = gr.Radio(["Hunyuan3D-2.1", "TripoSG"], value="Hunyuan3D-2.1", label="Generator")
                with gr.Row():
                    t3_octree = gr.Slider(256, 512, value=380, step=16, label="Octree Res")
                    t3_steps = gr.Slider(20, 60, value=50, step=5, label="Steps")
                    t3_cfg = gr.Slider(1.0, 10.0, value=5.5, step=0.5, label="CFG")
                t3_tight = gr.Checkbox(value=bool(BLENDER_EXE), label="Apply Blender Hard-Surface Tightening")
                with gr.Row():
                    t3_mirror = gr.Radio(["NONE", "X", "Y"], value="NONE", label="Mirror Axis")
                    t3_planar = gr.Slider(0.0, 8.0, value=3.0, step=0.5, label="Planar (°)")
                    t3_sharp = gr.Slider(15.0, 60.0, value=32.0, step=1.0, label="Sharp (°)")
                t3_seed = gr.Number(value=42, precision=0, label="Seed")
                t3_btn = gr.Button("Generate Hero Part", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    t3_crop_preview = gr.Image(label="518×518 Crop Preview", height=320)
                    t3_model = gr.Model3D(label="Generated Hero Part", height=420)
                t3_status = gr.Markdown()
                t3_logs = gr.Textbox(label="Live Log", lines=10)

        for ctrl in [t3_preview_btn]:
            ctrl.click(preview_crop, [t3_img, t3_x0, t3_y0, t3_x1, t3_y1, t3_pad], [t3_crop_preview])
        t3_btn.click(
            run_hero_part,
            inputs=[
                t3_img, t3_x0, t3_y0, t3_x1, t3_y1, t3_pad,
                t3_name, t3_engine, t3_octree, t3_steps, t3_cfg,
                t3_tight, t3_mirror, t3_planar, t3_sharp, t3_seed,
            ],
            outputs=[t3_crop_preview, t3_model, t3_status, t3_logs],
        )

    # ===== TAB 4: BLENDER TIGHTEN & OUTPUT BROWSER =====
    with gr.Tab("4. Hard-Surface Tighten & Output Browser"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Browse Existing Outputs")
                t4_dd = gr.Dropdown(choices=_list_existing_models(), label="Generated Meshes in outputs/")
                t4_refresh = gr.Button("Refresh List")
                gr.Markdown("### Standalone Blender Hard-Surface Tighten")
                t4_in_mesh = gr.File(label="Upload Mesh (.glb / .obj)")
                t4_mirror = gr.Radio(["NONE", "X", "Y"], value="NONE", label="Mirror Axis")
                with gr.Row():
                    t4_smooth = gr.Slider(0, 12, value=3, step=1, label="Volume-Preserving Smooth Iters")
                    t4_planar = gr.Slider(0.0, 8.0, value=3.5, step=0.5, label="Planar Dissolve (°)")
                    t4_sharp = gr.Slider(15.0, 60.0, value=32.0, step=1.0, label="Sharp Edge (°)")
                t4_btn = gr.Button("Tighten Mesh in Blender", variant="primary")
            with gr.Column(scale=2):
                t4_model = gr.Model3D(label="3D Preview", height=480)
                t4_info = gr.Markdown()
                t4_logs = gr.Textbox(label="Log", lines=8)

        t4_refresh.click(lambda: gr.update(choices=_list_existing_models()), outputs=[t4_dd])
        t4_dd.change(inspect_existing_model, inputs=[t4_dd], outputs=[t4_model, t4_info])
        t4_btn.click(
            run_standalone_tighten,
            inputs=[t4_in_mesh, t4_mirror, t4_smooth, t4_planar, t4_sharp],
            outputs=[t4_model, t4_info, t4_logs],
        )

    # ===== TAB 5: BATCH QUEUE (MULTI-IMAGE / AGENT MODE) =====
    with gr.Tab("5. Batch Queue (Multi-Image)"):
        with gr.Row():
            with gr.Column(scale=1):
                t5_files = gr.File(file_count="multiple", file_types=["image"], label="Upload Multiple Images")
                t5_folder = gr.Textbox(placeholder="Or paste a local image folder path (e.g. assets/example_image)", label="Local Image Folder (optional)")
                t5_out = gr.Textbox(value="outputs/batch_run", label="Output Directory")
                t5_scaffold = gr.Radio(["hunyuan3d", "triposg", "trellis"], value="hunyuan3d", label="Front-End Scaffold")
                t5_octree = gr.Slider(256, 512, value=380, step=16, label="Hunyuan3D Octree Resolution")
                t5_lato = gr.Checkbox(value=True, label="Run LATO.2 Retopology")
                with gr.Row():
                    t5_vnum = gr.Slider(1000, 7000, value=4200, step=200, label="Target Vertices")
                    t5_solver = gr.Radio(["heun", "euler"], value="heun", label="ODE Solver")
                with gr.Row():
                    t5_snap = gr.Checkbox(value=True, label="Surface-Snap")
                    t5_bake = gr.Checkbox(value=bool(BLENDER_EXE), label="Blender Normal Bake")
                    t5_resume = gr.Checkbox(value=True, label="Resume (Skip Done)")
                t5_bake_res = gr.Radio([1024, 2048, 4096], value=2048, label="Normal Map Resolution")
                t5_btn = gr.Button("Run Batch Pipeline", variant="primary")
            with gr.Column(scale=2):
                t5_status = gr.Markdown("Drop multiple images or point to a folder to process them unattended.")
                t5_csv = gr.File(label="Batch Summary CSV")
                t5_logs = gr.Textbox(label="Live Batch Log", lines=18, max_lines=24)

        t5_btn.click(
            run_batch_ui,
            inputs=[
                t5_files, t5_folder, t5_out, t5_scaffold, t5_octree,
                t5_lato, t5_vnum, t5_solver, t5_snap, t5_bake, t5_bake_res, t5_resume,
            ],
            outputs=[t5_status, t5_logs, t5_csv],
        )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", allowed_paths=[ROOT])
