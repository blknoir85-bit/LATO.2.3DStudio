"""
Agent-friendly, resumable, VRAM-safe batch 3D pipeline for LATO.2 3D Studio.

Processes a directory of images (--image_dir) or a structured JSON job manifest
(--manifest) through:
    Image -> Dense Scaffold (Hunyuan3D-2.1 / TripoSG / TRELLIS)
          -> LATO.2 Retopology (Heun / Euler + Topology Quality Gate)
          -> Surface-Snap -> Optional Blender 2K/4K Normal Bake / Hard-Surface Tighten

Every item runs in isolated subprocesses (100% VRAM release between stages and
items), writes a machine-readable <out_dir>/<id>/result.json, and updates
<out_dir>/batch_summary.json + batch_summary.csv after every item.

Usage (directory):
    .venv\\Scripts\\python.exe scripts/batch_pipeline.py \\
        --image_dir assets/batch_in --out_dir outputs/batch_out \\
        --scaffold hunyuan3d --solver heun --vert_num 4200 --snap --bake

Usage (JSON manifest for AI agents):
    .venv\\Scripts\\python.exe scripts/batch_pipeline.py \\
        --manifest jobs.json --out_dir outputs/batch_out

Manifest format (jobs.json):
    [
      {"id": "robot_01", "image": "assets/robot.jpg", "scaffold": "hunyuan3d", "vert_num": 4200, "bake": true},
      {"id": "prop_02",  "image": "assets/crate.png", "scaffold": "triposg",   "vert_num": 2500, "tighten": true}
    ]
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable
sys.path.insert(0, ROOT)

import trimesh
from PIL import Image

from utils.image import IMAGE_EXTENSIONS
from utils.mesh_cleanup import compute_topology_metrics, snap_mesh_to_reference


def _find_blender() -> str:
    for c in (
        r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    ):
        if os.path.isfile(c):
            return c
    return shutil.which("blender") or ""


def _base_env() -> dict:
    env = os.environ.copy()
    env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_cmd(cmd: list[str], log_file: str):
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"\n$ {' '.join(cmd)}\n")
        lf.flush()
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
            lf.write(line)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Subprocess failed (exit {rc}): {' '.join(cmd[:3])}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Agent-friendly resumable batch pipeline for LATO.2 3D Studio"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image_dir", help="directory of input images")
    src.add_argument("--manifest", help="JSON file containing a list of job dicts")

    p.add_argument("--out_dir", required=True, help="output root directory")
    p.add_argument(
        "--scaffold",
        choices=["hunyuan3d", "triposg", "trellis"],
        default="hunyuan3d",
        help="front-end 3D scaffold generator",
    )
    p.add_argument("--octree_resolution", type=int, default=380)
    p.add_argument("--scaffold_steps", type=int, default=50)
    p.add_argument("--scaffold_cfg", type=float, default=5.0)
    p.add_argument("--target_faces", type=int, default=150000)

    p.add_argument(
        "--lato2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run LATO.2 V-Flow -> T-Flow retopology",
    )
    p.add_argument("--vert_num", type=int, default=4200)
    p.add_argument("--solver", choices=["heun", "euler"], default="heun")
    p.add_argument("--vflow_steps", type=int, default=24)
    p.add_argument("--tflow_steps", type=int, default=30)
    p.add_argument("--edge_threshold", type=float, default=0.0)
    p.add_argument("--max_ring_size", type=int, default=18)

    p.add_argument(
        "--snap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="surface-snap LATO.2 mesh onto dense scaffold",
    )
    p.add_argument(
        "--bake",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="bake tangent-space normal map in headless Blender",
    )
    p.add_argument("--bake_res", type=int, default=2048)
    p.add_argument(
        "--tighten",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply Blender hard-surface tightening (when --no-lato2)",
    )
    p.add_argument("--planar_angle", type=float, default=3.5)
    p.add_argument("--sharp_angle", type=float, default=32.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip items that already have result.json with status == 'ok'",
    )
    p.add_argument("--force", action="store_true", help="re-run even if already completed")
    p.add_argument("--dry_run", action="store_true", help="list resolved jobs and exit without running")
    return p.parse_args()


def load_jobs(args) -> list[dict]:
    defaults = {
        "scaffold": args.scaffold,
        "octree_resolution": args.octree_resolution,
        "scaffold_steps": args.scaffold_steps,
        "scaffold_cfg": args.scaffold_cfg,
        "target_faces": args.target_faces,
        "lato2": args.lato2,
        "vert_num": args.vert_num,
        "solver": args.solver,
        "vflow_steps": args.vflow_steps,
        "tflow_steps": args.tflow_steps,
        "edge_threshold": args.edge_threshold,
        "max_ring_size": args.max_ring_size,
        "snap": args.snap,
        "bake": args.bake,
        "bake_res": args.bake_res,
        "tighten": args.tighten,
        "planar_angle": args.planar_angle,
        "sharp_angle": args.sharp_angle,
        "seed": args.seed,
    }

    jobs = []
    if args.image_dir:
        for fname in sorted(os.listdir(args.image_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                stem = os.path.splitext(fname)[0]
                job = dict(defaults)
                job["id"] = stem
                job["image"] = os.path.abspath(os.path.join(args.image_dir, fname))
                jobs.append(job)
    else:
        with open(args.manifest, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError("Manifest JSON must be a list of job objects.")
        for idx, item in enumerate(raw):
            job = dict(defaults)
            job.update(item)
            if "image" not in job:
                raise ValueError(f"Job #{idx} is missing required 'image' field.")
            img_path = job["image"]
            if not os.path.isabs(img_path):
                img_path = os.path.abspath(os.path.join(ROOT, img_path))
            job["image"] = img_path
            job["id"] = str(job.get("id") or os.path.splitext(os.path.basename(img_path))[0])
            jobs.append(job)
    return jobs


def process_job(job: dict, out_root: str, blender_exe: str) -> dict:
    job_id = job["id"]
    item_dir = os.path.join(out_root, job_id)
    os.makedirs(item_dir, exist_ok=True)
    log_file = os.path.join(item_dir, "run.log")
    t0 = time.time()

    result = {
        "id": job_id,
        "image": job["image"],
        "status": "failed",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": 0.0,
        "settings": {k: v for k, v in job.items() if k not in ("image",)},
        "outputs": {},
        "topology_metrics": {},
        "error": None,
    }

    try:
        if not os.path.isfile(job["image"]):
            raise FileNotFoundError(f"Input image not found: {job['image']}")

        in_png = os.path.join(item_dir, f"{job_id}_input.png")
        Image.open(job["image"]).convert("RGB").save(in_png)

        scaffold = job["scaffold"]
        if scaffold == "trellis":
            in_dir = os.path.join(item_dir, "in_img")
            os.makedirs(in_dir, exist_ok=True)
            shutil.copy2(in_png, os.path.join(in_dir, f"{job_id}.png"))
            out_lato = os.path.join(item_dir, "lato2")
            _run_cmd(
                [
                    PYTHON, "scripts/image_e2e_inference.py",
                    "--image_dir", in_dir,
                    "--out_dir", out_lato,
                    "--vert_num", str(int(job["vert_num"])),
                    "--ss_steps", str(int(job["scaffold_steps"])),
                    "--ss_cfg_strength", str(float(job["scaffold_cfg"])),
                    "--vflow_steps", str(int(job["vflow_steps"])),
                    "--tflow_steps", str(int(job["tflow_steps"])),
                    "--edge_threshold", str(float(job["edge_threshold"])),
                    "--max_ring_size", str(int(job["max_ring_size"])),
                    "--seed", str(int(job["seed"])),
                ],
                log_file,
            )
            pred_obj = os.path.join(out_lato, f"{job_id}_pred.obj")
            if not os.path.isfile(pred_obj):
                raise RuntimeError("TRELLIS -> LATO.2 produced no mesh.")
            result["outputs"]["final_mesh"] = pred_obj
            m = trimesh.load(pred_obj, process=False)
            result["topology_metrics"] = compute_topology_metrics(m)
            result["status"] = "ok"
            return result

        # Stage 1: Dense Scaffold (Hunyuan3D-2.1 or TripoSG)
        scaffold_dir = os.path.join(item_dir, "scaffold")
        os.makedirs(scaffold_dir, exist_ok=True)
        dense_glb = os.path.join(scaffold_dir, f"{job_id}_scaffold.glb")

        if scaffold == "hunyuan3d":
            _run_cmd(
                [
                    PYTHON, "scripts/hunyuan3d_scaffold.py",
                    "--image", in_png,
                    "--out_mesh", dense_glb,
                    "--steps", str(int(job["scaffold_steps"])),
                    "--guidance_scale", str(float(job["scaffold_cfg"])),
                    "--octree_resolution", str(int(job["octree_resolution"])),
                    "--seed", str(int(job["seed"])),
                    "--target_faces", str(int(job["target_faces"])),
                ],
                log_file,
            )
        else:
            _run_cmd(
                [
                    PYTHON, "scripts/triposg_scaffold.py",
                    "--image", in_png,
                    "--out_mesh", dense_glb,
                    "--steps", str(int(job["scaffold_steps"])),
                    "--guidance_scale", str(float(job["scaffold_cfg"])),
                    "--seed", str(int(job["seed"])),
                    "--target_faces", str(int(job["target_faces"])),
                ],
                log_file,
            )

        result["outputs"]["dense_scaffold"] = dense_glb
        final_mesh = dense_glb

        if not job["lato2"]:
            if job["tighten"] and blender_exe:
                tight_glb = os.path.join(item_dir, f"{job_id}_tight.glb")
                _run_cmd(
                    [
                        blender_exe, "-b", "--python", "scripts/hardsurface_tighten_blender.py", "--",
                        "--in_mesh", dense_glb,
                        "--out_mesh", tight_glb,
                        "--mirror_axis", "NONE",
                        "--planar_angle", str(float(job["planar_angle"])),
                        "--sharp_angle", str(float(job["sharp_angle"])),
                    ],
                    log_file,
                )
                result["outputs"]["tightened_mesh"] = tight_glb
                final_mesh = tight_glb
            result["outputs"]["final_mesh"] = final_mesh
            dm = trimesh.load(final_mesh, force="mesh", process=False)
            result["topology_metrics"] = compute_topology_metrics(dm)
            result["status"] = "ok"
            return result

        # Stage 2: LATO.2 Retopology
        out_lato = os.path.join(item_dir, "lato2")
        _run_cmd(
            [
                PYTHON, "scripts/e2e_inference.py",
                "--mesh_dir", scaffold_dir,
                "--out_dir", out_lato,
                "--vert_num", str(int(job["vert_num"])),
                "--solver", str(job["solver"]),
                "--vflow_steps", str(int(job["vflow_steps"])),
                "--tflow_steps", str(int(job["tflow_steps"])),
                "--edge_threshold", str(float(job["edge_threshold"])),
                "--max_ring_size", str(int(job["max_ring_size"])),
                "--seed", str(int(job["seed"])),
                "--num_workers", "0",
            ],
            log_file,
        )
        pred_obj = os.path.join(out_lato, f"{job_id}_scaffold_pred.obj")
        if not os.path.isfile(pred_obj):
            raise RuntimeError("LATO.2 produced no mesh.")
        result["outputs"]["lato2_mesh"] = pred_obj

        low_mesh = trimesh.load(pred_obj, process=False)
        result["topology_metrics"] = compute_topology_metrics(low_mesh)
        low_path = pred_obj

        # Stage 3: Surface-Snap
        if job["snap"]:
            dm = trimesh.load(dense_glb, force="mesh", process=False)
            snapped, snap_stats = snap_mesh_to_reference(
                low_mesh, dm, normalize_ref=True, max_snap_dist=0.04, blend=1.0
            )
            snapped_obj = os.path.join(out_lato, f"{job_id}_snapped.obj")
            snapped.export(snapped_obj)
            result["outputs"]["snapped_mesh"] = snapped_obj
            result["snap_stats"] = snap_stats
            low_path = snapped_obj

        final_mesh = low_path

        # Stage 4: Blender Normal Bake
        if job["bake"] and blender_exe:
            baked_glb = os.path.join(item_dir, f"{job_id}_baked.glb")
            normal_png = os.path.join(item_dir, f"{job_id}_normal.png")
            _run_cmd(
                [
                    blender_exe, "-b", "--python", "scripts/bake_normals_blender.py", "--",
                    "--high", dense_glb,
                    "--low", low_path,
                    "--out_glb", baked_glb,
                    "--out_normal", normal_png,
                    "--res", str(int(job["bake_res"])),
                ],
                log_file,
            )
            result["outputs"]["baked_glb"] = baked_glb
            result["outputs"]["normal_map"] = normal_png
            final_mesh = baked_glb

        result["outputs"]["final_mesh"] = final_mesh
        result["status"] = "ok"
    except Exception as e:
        result["error"] = str(e)
    finally:
        result["elapsed_sec"] = round(time.time() - t0, 2)
        with open(os.path.join(item_dir, "result.json"), "w", encoding="utf-8") as rf:
            json.dump(result, rf, indent=2)
    return result


def write_summary(out_root: str, results: list[dict]):
    summary = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(results),
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "failed": sum(1 for r in results if r["status"] != "ok"),
        "items": results,
    }
    with open(os.path.join(out_root, "batch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(out_root, "batch_summary.csv")
    fields = [
        "id", "status", "elapsed_sec", "num_verts", "num_faces",
        "faces_per_vert", "nmer_pct", "ber_pct", "num_open_loops",
        "num_components", "final_mesh", "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            tm = r.get("topology_metrics") or {}
            w.writerow({
                "id": r["id"],
                "status": r["status"],
                "elapsed_sec": r.get("elapsed_sec", 0.0),
                "num_verts": tm.get("num_verts", ""),
                "num_faces": tm.get("num_faces", ""),
                "faces_per_vert": tm.get("faces_per_vert", ""),
                "nmer_pct": tm.get("nmer_pct", ""),
                "ber_pct": tm.get("ber_pct", ""),
                "num_open_loops": tm.get("num_open_loops", ""),
                "num_components": tm.get("num_components", ""),
                "final_mesh": (r.get("outputs") or {}).get("final_mesh", ""),
                "error": r.get("error") or "",
            })


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    blender_exe = _find_blender()
    jobs = load_jobs(args)

    print(f"[BATCH] Discovered {len(jobs)} jobs -> {os.path.abspath(args.out_dir)}")
    if args.dry_run:
        for i, j in enumerate(jobs, 1):
            print(f"  [{i}/{len(jobs)}] {j['id']} <- {j['image']} (scaffold={j['scaffold']}, lato2={j['lato2']}, solver={j['solver']}, vert_num={j['vert_num']})")
        return

    results = []
    for i, job in enumerate(jobs, 1):
        res_file = os.path.join(args.out_dir, job["id"], "result.json")
        if args.resume and not args.force and os.path.isfile(res_file):
            try:
                with open(res_file, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                if prev.get("status") == "ok":
                    print(f"[BATCH] [{i}/{len(jobs)}] SKIP (already ok): {job['id']}")
                    results.append(prev)
                    write_summary(args.out_dir, results)
                    continue
            except Exception:
                pass

        print(f"[BATCH] [{i}/{len(jobs)}] START: {job['id']} ({job['scaffold']} -> lato2={job['lato2']})")
        res = process_job(job, args.out_dir, blender_exe)
        tm = res.get("topology_metrics") or {}
        if res["status"] == "ok":
            print(
                f"[BATCH] [{i}/{len(jobs)}] OK in {res['elapsed_sec']}s: {job['id']} "
                f"(V={tm.get('num_verts')}, F={tm.get('num_faces')}, "
                f"NMER={tm.get('nmer_pct')}%, BER={tm.get('ber_pct')}%) "
                f"-> {res['outputs'].get('final_mesh')}"
            )
        else:
            print(f"[BATCH] [{i}/{len(jobs)}] FAILED in {res['elapsed_sec']}s: {job['id']} — {res['error']}")
        results.append(res)
        write_summary(args.out_dir, results)

    ok_cnt = sum(1 for r in results if r["status"] == "ok")
    fail_cnt = len(results) - ok_cnt
    print(f"[BATCH] Done: {ok_cnt} ok, {fail_cnt} failed. Summary: {os.path.join(args.out_dir, 'batch_summary.json')}")


if __name__ == "__main__":
    main()
