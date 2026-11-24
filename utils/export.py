import os
import numpy as np
import trimesh

import utils.logging as logging
from utils.mesh_cleanup import clean_mesh, compute_topology_metrics, format_topology_report


def export_vertex(out_dir, base_name, type_name, vert_int, vert_offsets, resolution):
    res = float(resolution)
    vert_with_offset = (
        vert_int.astype(np.float64) / res
        - 0.5
        + vert_offsets.astype(np.float64) / (res * 2.0)
    )

    trimesh.PointCloud(vert_with_offset).export(
        os.path.join(out_dir, f"{base_name}_{type_name}.ply")
    )
    trimesh.PointCloud(vert_int).export(
        os.path.join(out_dir, f"{base_name}_{type_name}_coords.ply")
    )


def export_mesh(out_dir, base_name, vert_int, vert_offsets, faces, resolution):
    """OBJ pair matching export_vertex's PLY conventions (offset verts / int coords)."""
    res = float(resolution)
    vert_with_offset = (
        vert_int.astype(np.float64) / res
        - 0.5
        + vert_offsets.astype(np.float64) / (res * 2.0)
    )
    raw_mesh = trimesh.Trimesh(vertices=vert_with_offset, faces=faces, process=False)
    raw_metrics = compute_topology_metrics(raw_mesh)
    logging.info(format_topology_report(f"{base_name} (raw)", raw_metrics))

    cleaned = clean_mesh(vert_with_offset, faces)
    clean_metrics = compute_topology_metrics(cleaned)
    logging.info(format_topology_report(f"{base_name} (cleaned)", clean_metrics))
    cleaned.export(os.path.join(out_dir, f"{base_name}_pred.obj"))

    cleaned_coords = clean_mesh(vert_int.astype(np.float64), faces)
    cleaned_coords.export(os.path.join(out_dir, f"{base_name}_pred_coords.obj"))
