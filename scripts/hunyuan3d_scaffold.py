"""
Generate a high-resolution dense 3D scaffold mesh from an input image using
Tencent Hunyuan3D-2.1 (shape-only, pure PyTorch, reusing LATO.2's BiRefNet),
ready to compare against TripoSG or feed into LATO.2's Mesh-to-Mesh pipeline.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HY3D_SHAPE_ROOT = os.path.join(ROOT, "third_party", "Hunyuan3D-2.1", "hy3dshape")
sys.path.insert(0, ROOT)
sys.path.insert(0, HY3D_SHAPE_ROOT)
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import numpy as np
import torch
import trimesh
from PIL import Image

from models import BackgroundRemover
from utils.image import load_rgba
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline


def main():
    p = argparse.ArgumentParser(description="Hunyuan3D-2.1 shape-only scaffold generator for LATO.2")
    p.add_argument("--image", required=True, help="path to input image")
    p.add_argument("--out_mesh", required=True, help="output mesh path (.glb or .obj)")
    p.add_argument("--model_id", default="tencent/Hunyuan3D-2.1")
    p.add_argument("--rembg_dir", default=os.path.join(ROOT, "ckpt", "rembg"))
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--octree_resolution", type=int, default=380)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--target_faces",
        type=int,
        default=150000,
        help="QEM-decimate output mesh to this many faces (<=0 disables)",
    )
    args = p.parse_args()

    device = "cuda"
    os.makedirs(os.path.dirname(os.path.abspath(args.out_mesh)), exist_ok=True)

    print("Running BiRefNet foreground matting...")
    rembg = BackgroundRemover(args.rembg_dir, torch.device(device))
    rgba = load_rgba(args.image, rembg)
    del rembg
    torch.cuda.empty_cache()

    print(f"Loading Hunyuan3DDiTFlowMatchingPipeline from {args.model_id}...")
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model_id)

    print(f"Running Hunyuan3D-2.1 ({args.steps} steps, cfg={args.guidance_scale}, octree={args.octree_resolution}, seed={args.seed})...")
    with torch.no_grad():
        mesh = pipe(
            image=rgba,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            octree_resolution=args.octree_resolution,
            mc_algo="mc",
            generator=torch.Generator(device=device).manual_seed(args.seed),
        )[0]

    # Prune tiny disconnected shards
    if len(mesh.faces) > 0:
        comps = trimesh.graph.connected_components(
            mesh.face_adjacency, nodes=np.arange(len(mesh.faces))
        )
        if len(comps) > 1:
            largest = max(len(c) for c in comps)
            thresh = max(50, int(largest * 0.01))
            keep_faces = np.concatenate([c for c in comps if len(c) >= thresh])
            mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=np.asarray(mesh.faces)[np.sort(keep_faces)],
                process=False,
            )
            mesh.remove_unreferenced_vertices()
            print(f"Pruned {len(comps) - sum(1 for c in comps if len(c) >= thresh)}/{len(comps)} stray fragments")

    if args.target_faces > 0 and len(mesh.faces) > args.target_faces:
        import open3d as o3d
        o3m = o3d.geometry.TriangleMesh()
        o3m.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
        o3m.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
        dec = o3m.simplify_quadric_decimation(target_number_of_triangles=args.target_faces)
        mesh = trimesh.Trimesh(
            vertices=np.asarray(dec.vertices, dtype=np.float32),
            faces=np.asarray(dec.triangles, dtype=np.int32),
            process=False,
        )

    trimesh.repair.fix_normals(mesh)
    mesh.export(args.out_mesh)
    print(f"Saved Hunyuan3D-2.1 scaffold: {args.out_mesh} (V={len(mesh.vertices)}, F={len(mesh.faces)}, watertight={mesh.is_watertight})")


if __name__ == "__main__":
    main()
