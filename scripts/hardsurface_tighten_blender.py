"""
Headless Blender 5.2 hard-surface tightening & symmetry pass for AI-generated
mechanical meshes (Hunyuan3D-2.1 / TripoSG / LATO.2):
  1. Optional exact symmetry (Bisect + Mirror across the symmetry axis)
  2. Planar-preserving decimation (flattens wobbly planar armor plates)
  3. Sharp-edge marking by dihedral angle (> --sharp_angle deg)
  4. Weighted Normals (keeps flat faces razor-flat and bevels crisp)

Usage:
    blender -b --python scripts/hardsurface_tighten_blender.py -- \
        --in_mesh <input.glb> --out_mesh <output.glb> \
        [--mirror_axis X|Y|NONE] [--planar_angle 3.5] [--sharp_angle 35.0]
"""

import argparse
import math
import os
import sys
import bpy
import bmesh


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--in_mesh", required=True)
    p.add_argument("--out_mesh", required=True)
    p.add_argument("--mirror_axis", choices=["X", "Y", "Z", "NONE"], default="NONE")
    p.add_argument("--planar_angle", type=float, default=3.5, help="planar decimate angle limit in degrees (0 disables)")
    p.add_argument("--sharp_angle", type=float, default=35.0, help="mark edges sharper than this angle (deg) as sharp")
    p.add_argument("--triangulate", action="store_true", default=False)
    return p.parse_args(argv)


def import_mesh(path: str):
    ext = os.path.splitext(path)[1].lower()
    before = set(bpy.data.objects)
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise ValueError(f"Unsupported format: {ext}")
    added = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in added:
        o.select_set(True)
    bpy.context.view_layer.objects.active = added[0]
    if len(added) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # Center origin to bounding-box center so mirror plane passes cleanly through center
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0.0, 0.0, 0.0)
    bpy.ops.object.transform_apply(location=True)
    return obj


def main():
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)

    obj = import_mesh(args.in_mesh)
    v_before = len(obj.data.vertices)
    f_before = len(obj.data.polygons)

    # 1. Optional exact symmetry via Mirror modifier with bisect
    if args.mirror_axis != "NONE":
        axis_idx = {"X": 0, "Y": 1, "Z": 2}[args.mirror_axis]
        mod = obj.modifiers.new("SymmetryMirror", "MIRROR")
        mod.use_axis = [i == axis_idx for i in range(3)]
        mod.use_bisect_axis = [i == axis_idx for i in range(3)]
        mod.use_clip = True
        mod.merge_threshold = 0.002
        bpy.ops.object.modifier_apply(modifier=mod.name)

    # 2. Planar-preserving decimation to tighten wobbly flat armor faces
    if args.planar_angle > 0:
        dec = obj.modifiers.new("PlanarTighten", "DECIMATE")
        dec.decimate_type = "DISSOLVE"
        dec.angle_limit = math.radians(args.planar_angle)
        dec.use_dissolve_boundaries = False
        bpy.ops.object.modifier_apply(modifier=dec.name)

    # 3. Shade smooth + mark sharp edges by dihedral angle
    bpy.ops.object.shade_smooth()
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    sharp_rad = math.radians(args.sharp_angle)
    sharp_count = 0
    for e in bm.edges:
        if len(e.link_faces) == 2:
            ang = e.calc_face_angle(0.0)
            if ang >= sharp_rad:
                e.smooth = False
                sharp_count += 1
            else:
                e.smooth = True
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    # 4. Weighted Normal modifier (face-area weighted, keeps sharp edges crisp)
    wn = obj.modifiers.new("WeightedNormal", "WEIGHTED_NORMAL")
    wn.keep_sharp = True
    wn.weight = 50
    bpy.ops.object.modifier_apply(modifier=wn.name)

    if args.triangulate:
        tri = obj.modifiers.new("Triangulate", "TRIANGULATE")
        tri.keep_custom_normals = True
        bpy.ops.object.modifier_apply(modifier=tri.name)

    v_after = len(obj.data.vertices)
    f_after = len(obj.data.polygons)
    print(f"Tightened {os.path.basename(args.in_mesh)}: V {v_before}->{v_after}, F {f_before}->{f_after}, sharp_edges={sharp_count}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_mesh)), exist_ok=True)
    ext = os.path.splitext(args.out_mesh)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.export_scene.gltf(
            filepath=os.path.abspath(args.out_mesh),
            export_format="GLB",
            use_selection=True,
            export_normals=True,
        )
    else:
        bpy.ops.wm.obj_export(filepath=os.path.abspath(args.out_mesh), export_selected_objects=True)
    print(f"Saved tightened mesh: {args.out_mesh}")


if __name__ == "__main__":
    main()
