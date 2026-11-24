"""
Headless Blender script to UV-unwrap a low-poly LATO.2 mesh and bake a
tangent-space normal map from a high-poly master mesh (e.g. 150k Hunyuan3D-2.1),
exporting both the normal map PNG and a textured GLB with the normal map packed.

Usage:
    blender -b --python scripts/bake_normals_blender.py -- \
        --high <high_poly.glb> --low <low_poly.obj> \
        --out_glb <out_baked.glb> --out_normal <out_normal.png> [--res 2048]
"""

import argparse
import os
import sys
import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--high", required=True)
    p.add_argument("--low", required=True)
    p.add_argument("--out_glb", required=True)
    p.add_argument("--out_normal", required=True)
    p.add_argument("--res", type=int, default=2048)
    p.add_argument("--cage_extrusion", type=float, default=0.03)
    return p.parse_args(argv)


def import_mesh(path: str, name: str):
    ext = os.path.splitext(path)[1].lower()
    before = set(bpy.data.objects)
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")
    added = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not added:
        raise RuntimeError(f"No mesh objects imported from {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for o in added:
        o.select_set(True)
    bpy.context.view_layer.objects.active = added[0]
    if len(added) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def normalize_object_to_unit_cube(obj):
    """Match LATO.2's [-0.5, 0.5]^3 bbox normalization."""
    coords = [v.co for v in obj.data.vertices]
    xs = [c.x for c in coords]
    ys = [c.y for c in coords]
    zs = [c.z for c in coords]
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    cz = 0.5 * (min(zs) + max(zs))
    extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1e-7)
    for v in obj.data.vertices:
        v.co.x = (v.co.x - cx) / extent
        v.co.y = (v.co.y - cy) / extent
        v.co.z = (v.co.z - cz) / extent
    obj.data.update()


def main():
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Note: OBJ import in Blender applies Y-up -> Z-up rotation automatically,
    # matching glTF import's Y-up -> Z-up convention.
    high = import_mesh(args.high, "HighPolyMaster")
    low = import_mesh(args.low, "LowPolyLATO2")

    # Align high-poly master into the exact same [-0.5, 0.5]^3 space as LATO.2 output
    normalize_object_to_unit_cube(high)

    # Smooth-shade both for clean tangent-space normal baking
    for obj in (high, low):
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()

    # Smart UV unwrap the low-poly mesh
    bpy.ops.object.select_all(action="DESELECT")
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.1519, island_margin=0.01)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Create material + normal map image target
    img = bpy.data.images.new("BakedNormal", width=args.res, height=args.res, alpha=False)
    img.colorspace_settings.name = "Non-Color"

    mat = bpy.data.materials.new("LATO2_BakedMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.80, 0.80, 0.82, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.65

    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = img
    tex_node.location = (-600, -200)

    nrm_node = nodes.new("ShaderNodeNormalMap")
    nrm_node.location = (-250, -200)
    links.new(tex_node.outputs["Color"], nrm_node.inputs["Color"])

    # Keep tex_node active (unconnected to BSDF during bake to avoid feedback loop)
    nodes.active = tex_node
    low.data.materials.clear()
    low.data.materials.append(mat)

    # Configure Cycles CPU bake
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.cage_extrusion = args.cage_extrusion
    scene.render.bake.max_ray_distance = args.cage_extrusion * 2.0
    scene.render.bake.margin = 16
    scene.render.bake.normal_space = "TANGENT"

    bpy.ops.object.select_all(action="DESELECT")
    high.select_set(True)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low

    print(f"Baking {args.res}x{args.res} tangent-space normal map...")
    bpy.ops.object.bake(type="NORMAL")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_normal)), exist_ok=True)
    img.filepath_raw = os.path.abspath(args.out_normal)
    img.file_format = "PNG"
    img.save()
    print(f"Saved normal map: {args.out_normal}")

    # Now wire the baked normal map into Principled BSDF and export low-poly GLB
    links.new(nrm_node.outputs["Normal"], bsdf.inputs["Normal"])
    bpy.ops.object.select_all(action="DESELECT")
    low.select_set(True)
    bpy.context.view_layer.objects.active = low

    os.makedirs(os.path.dirname(os.path.abspath(args.out_glb)), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=os.path.abspath(args.out_glb),
        export_format="GLB",
        use_selection=True,
    )
    print(f"Exported baked GLB: {args.out_glb}")


if __name__ == "__main__":
    main()
