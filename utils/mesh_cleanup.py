"""
Deterministic mesh topology cleanup and automated Geometry & Topology Quality Gate.
"""

from collections import defaultdict
import numpy as np
import trimesh


def _edge_face_counts(faces: np.ndarray):
    """Return dict[(u, v)] -> list of face indices for undirected edges."""
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces.tolist()):
        for x, y in ((a, b), (b, c), (c, a)):
            key = (x, y) if x < y else (y, x)
            edge_faces[key].append(fi)
    return edge_faces


def _trace_boundary_loops(boundary_edges: list[tuple[int, int]]) -> list[list[int]]:
    """Trace simple closed loops from boundary edges (edges with 1 incident face)."""
    adj: dict[int, set[int]] = defaultdict(set)
    for u, v in boundary_edges:
        adj[u].add(v)
        adj[v].add(u)

    visited_edges: set[tuple[int, int]] = set()
    loops: list[list[int]] = []

    for u0, v0 in boundary_edges:
        key0 = (u0, v0) if u0 < v0 else (v0, u0)
        if key0 in visited_edges:
            continue
        # Only trace simple 2-regular chains cleanly
        if len(adj[u0]) != 2 and len(adj[v0]) != 2:
            continue
        loop = [u0, v0]
        visited_edges.add(key0)
        prev, curr = u0, v0
        closed = False
        while True:
            nbrs = [n for n in adj[curr] if n != prev]
            if len(nbrs) != 1:
                break
            nxt = nbrs[0]
            ekey = (curr, nxt) if curr < nxt else (nxt, curr)
            if nxt == u0:
                visited_edges.add(ekey)
                closed = True
                break
            if ekey in visited_edges or nxt in loop:
                break
            visited_edges.add(ekey)
            loop.append(nxt)
            prev, curr = curr, nxt
        if closed and len(loop) >= 3:
            loops.append(loop)
    return loops


def _min_area_fill_loop(loop: list[int], vertices: np.ndarray) -> list[list[int]]:
    """Minimum-area triangulation of a 3D boundary loop via O(n^3) DP."""
    n = len(loop)
    if n < 3:
        return []
    if n == 3:
        return [[loop[0], loop[1], loop[2]]]
    p = np.asarray(vertices, dtype=np.float64)[loop]
    d = p[None, :, :] - p[:, None, :]
    area = 0.5 * np.linalg.norm(
        np.cross(d[:, :, None, :], d[:, None, :, :]), axis=-1
    )
    eps = 1e-10 * max(float(np.sum(np.ptp(p, axis=0) ** 2)), 1e-12)
    area_list = area.tolist()
    cost = [[(0, 0.0)] * n for _ in range(n)]
    split = [[-1] * n for _ in range(n)]
    for span in range(2, n):
        for i in range(n - span):
            j = i + span
            best = None
            for k in range(i + 1, j):
                a = area_list[i][k][j]
                c = (
                    cost[i][k][0] + cost[k][j][0] + (a <= eps),
                    cost[i][k][1] + cost[k][j][1] + a,
                )
                if best is None or c < best:
                    best = c
                    split[i][j] = k
            cost[i][j] = best
    tris: list[list[int]] = []
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        k = split[i][j]
        tris.append([loop[i], loop[k], loop[j]])
        stack.append((i, k))
        stack.append((k, j))
    return tris


def _triangle_min_angles_deg(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute the minimum interior angle (in degrees) for each triangle."""
    if len(faces) == 0:
        return np.empty((0,), dtype=np.float64)
    v = vertices[faces]  # (F, 3, 3)
    e0 = v[:, 1] - v[:, 0]
    e1 = v[:, 2] - v[:, 1]
    e2 = v[:, 0] - v[:, 2]
    l0 = np.linalg.norm(e0, axis=1).clip(min=1e-12)
    l1 = np.linalg.norm(e1, axis=1).clip(min=1e-12)
    l2 = np.linalg.norm(e2, axis=1).clip(min=1e-12)
    c0 = np.clip(np.sum(e0 * (-e2), axis=1) / (l0 * l2), -1.0, 1.0)
    c1 = np.clip(np.sum(e1 * (-e0), axis=1) / (l1 * l0), -1.0, 1.0)
    c2 = np.clip(np.sum(e2 * (-e1), axis=1) / (l2 * l1), -1.0, 1.0)
    ang = np.degrees(np.arccos(np.stack([c0, c1, c2], axis=1)))
    return ang.min(axis=1)


def compute_topology_metrics(mesh: trimesh.Trimesh, sliver_angle_deg: float = 5.0) -> dict:
    """Compute automated geometry & topology gate metrics for a mesh."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    num_verts = int(len(verts))
    num_faces = int(len(faces))

    if num_faces == 0:
        return {
            "num_verts": num_verts,
            "num_faces": 0,
            "faces_per_vert": 0.0,
            "nmer_pct": 0.0,
            "ber_pct": 0.0,
            "num_components": 0,
            "num_open_loops": 0,
            "sliver_pct": 0.0,
            "watertight": False,
            "winding_consistent": False,
            "gate_pass": False,
        }

    edge_faces = _edge_face_counts(faces)
    num_edges = len(edge_faces)
    non_manifold_edges = sum(1 for fl in edge_faces.values() if len(fl) > 2)
    boundary_edges = [e for e, fl in edge_faces.items() if len(fl) == 1]

    nmer_pct = 100.0 * non_manifold_edges / max(num_edges, 1)
    ber_pct = 100.0 * len(boundary_edges) / max(num_edges, 1)
    open_loops = _trace_boundary_loops(boundary_edges)

    min_angles = _triangle_min_angles_deg(verts, faces)
    sliver_pct = 100.0 * float(np.sum(min_angles < sliver_angle_deg)) / num_faces

    try:
        comps = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(num_faces))
        num_components = len(comps)
    except Exception:
        num_components = 1

    watertight = bool(mesh.is_watertight)
    winding_consistent = bool(mesh.is_winding_consistent)
    gate_pass = (
        nmer_pct <= 1.0
        and ber_pct <= 5.0
        and num_components == 1
        and sliver_pct <= 5.0
    )

    return {
        "num_verts": num_verts,
        "num_faces": num_faces,
        "faces_per_vert": round(num_faces / max(num_verts, 1), 2),
        "nmer_pct": round(nmer_pct, 2),
        "ber_pct": round(ber_pct, 2),
        "num_components": num_components,
        "num_open_loops": len(open_loops),
        "sliver_pct": round(sliver_pct, 2),
        "watertight": watertight,
        "winding_consistent": winding_consistent,
        "gate_pass": gate_pass,
    }


def format_topology_report(name: str, metrics: dict) -> str:
    status = "PASS" if metrics["gate_pass"] else "WARN"
    return (
        f"[TOPO-GATE:{status}] {name}: "
        f"V={metrics['num_verts']}, F={metrics['num_faces']} ({metrics['faces_per_vert']:.2f} F/V) | "
        f"NMER={metrics['nmer_pct']:.2f}%, BER={metrics['ber_pct']:.2f}% | "
        f"loops={metrics['num_open_loops']}, comps={metrics['num_components']}, "
        f"slivers={metrics['sliver_pct']:.2f}% | "
        f"watertight={metrics['watertight']}, winding={metrics['winding_consistent']}"
    )


def _prune_non_manifold_fins(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Iteratively remove non-manifold fin faces (edges shared by >2 faces)."""
    if len(faces) == 0:
        return faces
    alive = np.ones(len(faces), dtype=bool)
    min_angles = _triangle_min_angles_deg(vertices, faces)

    edge_faces = _edge_face_counts(faces)
    face_edges = []
    for a, b, c in faces.tolist():
        face_edges.append([
            (a, b) if a < b else (b, a),
            (b, c) if b < c else (c, b),
            (c, a) if c < a else (a, c),
        ])

    for _ in range(32):
        nm_edges = [e for e, fl in edge_faces.items() if len(fl) > 2]
        if not nm_edges:
            break
        removed_any = False
        for e in nm_edges:
            fl = [fi for fi in edge_faces[e] if alive[fi]]
            if len(fl) <= 2:
                edge_faces[e] = fl
                continue
            # Score each incident face: prefer dropping fins (faces with boundary edges)
            # or sliver faces (smallest min interior angle).
            scored = []
            for fi in fl:
                boundary_cnt = sum(1 for fe in face_edges[fi] if len(edge_faces[fe]) == 1)
                nm_cnt = sum(1 for fe in face_edges[fi] if len(edge_faces[fe]) > 2)
                # Higher score = more likely to be a spurious fin
                scored.append((boundary_cnt, nm_cnt, -min_angles[fi], fi))
            scored.sort(reverse=True)
            drop_fi = scored[0][3]
            alive[drop_fi] = False
            for fe in face_edges[drop_fi]:
                if drop_fi in edge_faces[fe]:
                    edge_faces[fe].remove(drop_fi)
            removed_any = True
        if not removed_any:
            break

    return faces[alive]


def clean_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    prune_non_manifold: bool = True,
    min_component_ratio: float = 0.02,
    max_hole_edges: int = 24,
    drop_sliver_deg: float = 1.0,
) -> trimesh.Trimesh:
    """Deterministic silhouette-preserving topology cleanup."""
    verts = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    if len(f) == 0:
        return trimesh.Trimesh(vertices=verts, faces=f, process=False)

    # 1. Remove degenerate & duplicate faces (unordered vertex triples)
    valid = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])
    f = f[valid]
    if len(f) > 0:
        sorted_f = np.sort(f, axis=1)
        _, uniq_idx = np.unique(sorted_f, axis=0, return_index=True)
        f = f[np.sort(uniq_idx)]

    # Remove extreme degenerate slivers (< drop_sliver_deg)
    if len(f) > 0 and drop_sliver_deg > 0:
        ang = _triangle_min_angles_deg(verts, f)
        f = f[ang >= drop_sliver_deg]

    # 2. Prune non-manifold fin faces (>2 faces sharing an edge)
    if prune_non_manifold and len(f) > 0:
        f = _prune_non_manifold_fins(verts, f)

    # 3. Drop tiny disconnected floating islands
    mesh = trimesh.Trimesh(vertices=verts, faces=f, process=False)
    if min_component_ratio > 0 and len(mesh.faces) > 0:
        comps = trimesh.graph.connected_components(
            mesh.face_adjacency, nodes=np.arange(len(mesh.faces))
        )
        if len(comps) > 1:
            largest = max(len(c) for c in comps)
            thresh = max(3, int(largest * min_component_ratio))
            keep_faces = np.concatenate([c for c in comps if len(c) >= thresh])
            f = np.asarray(mesh.faces, dtype=np.int64)[np.sort(keep_faces)]

    # 4. Fill residual closed boundary loops (up to max_hole_edges)
    if max_hole_edges >= 3 and len(f) > 0:
        edge_faces = _edge_face_counts(f)
        boundary_edges = [e for e, fl in edge_faces.items() if len(fl) == 1]
        loops = _trace_boundary_loops(boundary_edges)
        existing_tri_keys = {tuple(sorted(tri)) for tri in f.tolist()}
        added = []
        for loop in loops:
            if len(loop) <= max_hole_edges:
                for tri in _min_area_fill_loop(loop, verts):
                    tkey = tuple(sorted(tri))
                    if tkey not in existing_tri_keys and len(set(tri)) == 3:
                        existing_tri_keys.add(tkey)
                        added.append(tri)
        if added:
            f = np.concatenate([f, np.asarray(added, dtype=np.int64)], axis=0)

    # 5. Build final mesh, fix winding/normals, and drop unreferenced verts
    mesh = trimesh.Trimesh(vertices=verts, faces=f, process=False)
    if len(mesh.faces) > 0:
        mesh.remove_unreferenced_vertices()
        try:
            trimesh.repair.fix_winding(mesh)
            trimesh.repair.fix_normals(mesh)
        except Exception:
            pass
    return mesh


def normalize_mesh_to_unit_cube(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalize a mesh into [-0.5, 0.5]^3 using LATO.2's exact bbox convention."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    bbox_min, bbox_max = v.min(axis=0), v.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    max_extent = max(float((bbox_max - bbox_min).max()), 1e-7)
    v_norm = (v - center) / max_extent
    return trimesh.Trimesh(vertices=v_norm, faces=np.asarray(mesh.faces, dtype=np.int64), process=False)


def snap_mesh_to_reference(
    mesh: trimesh.Trimesh,
    ref_mesh: trimesh.Trimesh,
    normalize_ref: bool = True,
    max_snap_dist: float = 0.04,
    blend: float = 1.0,
) -> tuple[trimesh.Trimesh, dict]:
    """Project mesh vertices onto the closest points on a dense reference mesh surface."""
    import open3d as o3d

    ref = normalize_mesh_to_unit_cube(ref_mesh) if normalize_ref else ref_mesh
    t_ref = o3d.t.geometry.TriangleMesh()
    t_ref.vertex.positions = o3d.core.Tensor(
        np.asarray(ref.vertices, dtype=np.float32)
    )
    t_ref.triangle.indices = o3d.core.Tensor(
        np.asarray(ref.faces, dtype=np.uint32)
    )

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_ref)

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    query = o3d.core.Tensor(verts, dtype=o3d.core.Dtype.Float32)
    ans = scene.compute_closest_points(query)
    closest = ans["points"].numpy()

    deltas = closest - verts
    dists = np.linalg.norm(deltas, axis=1)
    # Clamp large displacements so vertices never jump across gaps
    scale = np.where(dists > max_snap_dist, max_snap_dist / np.maximum(dists, 1e-8), 1.0)
    snapped_verts = verts + (deltas * scale[:, None] * float(blend))

    out = trimesh.Trimesh(
        vertices=snapped_verts.astype(np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )
    try:
        trimesh.repair.fix_winding(out)
        trimesh.repair.fix_normals(out)
    except Exception:
        pass

    stats = {
        "mean_dist": float(dists.mean()),
        "p95_dist": float(np.percentile(dists, 95)),
        "max_dist": float(dists.max()),
        "clamped_verts": int(np.sum(dists > max_snap_dist)),
    }
    return out, stats
