import numpy as np
import open3d as o3d
import json


def get_open_air_direction(ray_scene, point, samples=1000):
    phi = np.pi * (3. - np.sqrt(5.))
    y = 1 - (np.arange(samples) / float(samples - 1)) * 2
    radius = np.sqrt(1 - y * y)
    theta = phi * np.arange(samples)
    directions = np.vstack((np.cos(theta) * radius, y, np.sin(theta) * radius)).T

    origins = np.tile(point, (samples, 1)) + (directions * 1.0)
    ray_tensor = o3d.core.Tensor(np.hstack((origins, directions)).astype(np.float32))
    ans = ray_scene.cast_rays(ray_tensor)

    open_air_directions = directions[ans['t_hit'].numpy() > 90.0]

    if len(open_air_directions) == 0:
        return np.array([0.0, 0.0, 1.0])

    avg = np.mean(open_air_directions, axis=0)
    return avg / np.linalg.norm(avg)


def get_raw_normals(mesh, point, radius=20.0):
    kd_tree = o3d.geometry.KDTreeFlann(mesh)
    [k, idx, _] = kd_tree.search_radius_vector_3d(point, radius)

    if k == 0:
        return []

    mesh.compute_triangle_normals()
    triangles = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.triangle_normals)

    nearby_faces = set()
    for v_idx in idx:
        for f in np.where(triangles == v_idx)[0]:
            nearby_faces.add(f)

    unique_n = []
    for f in nearby_faces:
        n = normals[f]
        if np.linalg.norm(n) < 0.5:
            continue
        if not any(abs(np.dot(n, un)) > 0.95 for un in unique_n):
            unique_n.append(n)

    return unique_n[:3]


def run_analysis(obj_path, json_path):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    bbox_center = mesh.get_axis_aligned_bounding_box().get_center()
    mesh.translate(-bbox_center)

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(mesh_t)

    with open(json_path, 'r') as f:
        json_edges = json.load(f).get("welding_data", {}).get("edges", [])

    lines = []
    for edge in json_edges:
        seg_name = edge.get("segment_name")
        A = np.array([edge['start']['x'], edge['start']['y'], edge['start']['z']])
        B = np.array([edge['end']['x'],   edge['end']['y'],   edge['end']['z']])
        travel_dir = (B - A) / np.linalg.norm(B - A)

        for pt_name, pt in [("START", A), ("END", B)]:
            air_dir = get_open_air_direction(ray_scene, pt)
            plates = get_raw_normals(mesh, pt)

            lines.append(f"EDGE: {seg_name} | {pt_name}")
            lines.append(f"  Travel_Dir: {np.round(travel_dir, 4).tolist()}")
            lines.append(f"  Guiding_Air_Vector: {np.round(air_dir, 4).tolist()}")
            lines.append(f"  Raw_Plates: {[np.round(n, 4).tolist() for n in plates]}")

    return "\n".join(lines)
