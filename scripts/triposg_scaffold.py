"""
Generate a high-resolution dense 3D scaffold mesh from an input image using
VAST-AI/TripoSG (reusing LATO.2's BiRefNet background remover), ready to be
retopologized by LATO.2's Mesh-to-Mesh pipeline (scripts/e2e_inference.py).
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPOSG_ROOT = os.path.join(ROOT, "third_party", "TripoSG")
sys.path.insert(0, ROOT)
sys.path.insert(0, TRIPOSG_ROOT)
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image

from models import BackgroundRemover
from utils.image import crop_foreground, load_rgba
import triposg.schedulers.scheduling_rectified_flow  # noqa: F401
import triposg.models.autoencoders.autoencoder_kl_triposg  # noqa: F401
import triposg.models.transformers.triposg_transformer  # noqa: F401
from triposg.pipelines.pipeline_triposg import TripoSGPipeline


def prepare_white_bg_image(image_path: str, rembg: BackgroundRemover, padding: float = 1.15) -> Image.Image:
    rgba = load_rgba(image_path, rembg)
    alpha = np.array(rgba)[:, :, 3]
    ys, xs = np.nonzero(alpha > 0.8 * 255)
    if xs.size > 0:
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        side = max(2, int(max(x1 - x0, y1 - y0) * padding))
        box = (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)
        rgba = rgba.crop(box).resize((518, 518), Image.Resampling.LANCZOS)
    arr = np.asarray(rgba, dtype=np.float32) / 255.0
    rgb = arr[..., :3]
    a = arr[..., 3:4]
    white_comp = rgb * a + 1.0 * (1.0 - a)
    return Image.fromarray((np.clip(white_comp, 0.0, 1.0) * 255.0).astype(np.uint8))


def main():
    p = argparse.ArgumentParser(description="TripoSG dense mesh scaffold generator for LATO.2")
    p.add_argument("--image", required=True, help="path to input image")
    p.add_argument("--out_mesh", required=True, help="output mesh path (.glb or .obj)")
    p.add_argument("--weights_dir", default=os.path.join(ROOT, "ckpt", "triposg"))
    p.add_argument("--rembg_dir", default=os.path.join(ROOT, "ckpt", "rembg"))
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--target_faces",
        type=int,
        default=150000,
        help="QEM-decimate output mesh to this many faces (<=0 disables; prevents Open3D Vulkan buffer overflow)",
    )
    args = p.parse_args()

    device = "cuda"
    dtype = torch.float16

    os.makedirs(os.path.dirname(os.path.abspath(args.out_mesh)), exist_ok=True)
    if not os.path.exists(os.path.join(args.weights_dir, "model_index.json")):
        print(f"Downloading VAST-AI/TripoSG -> {args.weights_dir}")
        snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=args.weights_dir)

    print("Running BiRefNet foreground matting...")
    rembg = BackgroundRemover(args.rembg_dir, torch.device(device))
    img_pil = prepare_white_bg_image(args.image, rembg)
    del rembg
    torch.cuda.empty_cache()

    print(f"Loading TripoSGPipeline from {args.weights_dir}...")
    pipe = TripoSGPipeline.from_pretrained(args.weights_dir).to(device, dtype)

    print(f"Running TripoSG ({args.steps} steps, cfg={args.guidance_scale}, seed={args.seed})...")
    with torch.no_grad():
        outputs = pipe(
            image=img_pil,
            generator=torch.Generator(device=device).manual_seed(args.seed),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        ).samples[0]

    verts = outputs[0].astype(np.float32)
    faces = np.ascontiguousarray(outputs[1])

    # Remove tiny disconnected marching-cubes boundary shards before decimation
    raw_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if len(raw_mesh.faces) > 0:
        comps = trimesh.graph.connected_components(
            raw_mesh.face_adjacency, nodes=np.arange(len(raw_mesh.faces))
        )
        if len(comps) > 1:
            largest = max(len(c) for c in comps)
            thresh = max(50, int(largest * 0.01))
            keep_faces = np.concatenate([c for c in comps if len(c) >= thresh])
            raw_mesh = trimesh.Trimesh(
                vertices=raw_mesh.vertices,
                faces=np.asarray(raw_mesh.faces)[np.sort(keep_faces)],
                process=False,
            )
            raw_mesh.remove_unreferenced_vertices()
            print(f"Pruned {len(comps) - sum(1 for c in comps if len(c) >= thresh)}/{len(comps)} stray marching-cubes fragments")
        verts = np.asarray(raw_mesh.vertices, dtype=np.float32)
        faces = np.asarray(raw_mesh.faces, dtype=np.int32)

    if args.target_faces > 0 and len(faces) > args.target_faces:
        import open3d as o3d
        o3m = o3d.geometry.TriangleMesh()
        o3m.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        o3m.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        dec = o3m.simplify_quadric_decimation(target_number_of_triangles=args.target_faces)
        verts = np.asarray(dec.vertices, dtype=np.float32)
        faces = np.asarray(dec.triangles, dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    trimesh.repair.fix_normals(mesh)
    mesh.export(args.out_mesh)
    print(f"Saved TripoSG dense scaffold: {args.out_mesh} (V={len(mesh.vertices)}, F={len(mesh.faces)}, watertight={mesh.is_watertight})")


if __name__ == "__main__":
    main()
