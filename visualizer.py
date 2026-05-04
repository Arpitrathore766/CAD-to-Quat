"""
Standalone script launched as a subprocess by app.py.
Usage: python visualizer.py <obj_path> <json_path> <vectors_json_path>
"""
import sys
import json
import numpy as np
import open3d as o3d


def create_cylinder_line(start, end, radius=0.8, color=[0, 0, 1]):
    dist = np.linalg.norm(end - start)
    if dist < 1e-6:
        return None
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=dist)
    cylinder.paint_uniform_color(color)
    direction = (end - start) / dist
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, direction)
    c = np.dot(z_axis, direction)
    s = np.linalg.norm(v)
    if s > 1e-6:
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        cylinder.rotate(np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2)), center=(0, 0, 0))
    elif c < -0.99:
        cylinder.rotate(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), center=(0, 0, 0))
    cylinder.translate((start + end) / 2)
    return cylinder


def create_approach_arrow(target_point, approach_vector, standoff_dist=15.0, color=[0, 1, 0]):
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=standoff_dist * 0.05, cone_radius=standoff_dist * 0.1,
        cylinder_height=standoff_dist * 0.7, cone_height=standoff_dist * 0.3,
    )
    arrow.paint_uniform_color(color)
    direction = np.array(approach_vector) / np.linalg.norm(approach_vector)
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, direction)
    c = np.dot(z_axis, direction)
    s = np.linalg.norm(v)
    if s > 1e-6:
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        arrow.rotate(np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2)), center=(0, 0, 0))
    elif c < -0.99:
        arrow.rotate(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), center=(0, 0, 0))
    arrow.translate(target_point - direction * standoff_dist)
    return arrow


if __name__ == "__main__":
    obj_path, json_path, vectors_path = sys.argv[1], sys.argv[2], sys.argv[3]

    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_axis_aligned_bounding_box().get_center())
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    geometries = [mesh]

    with open(json_path) as f:
        json_edges = json.load(f).get("welding_data", {}).get("edges", [])

    with open(vectors_path) as f:
        vectors = json.load(f)

    for edge in json_edges:
        seg_name = edge.get("segment_name")
        if seg_name not in vectors:
            continue

        A = np.array([edge['start']['x'], edge['start']['y'], edge['start']['z']])
        B = np.array([edge['end']['x'],   edge['end']['y'],   edge['end']['z']])

        seam = create_cylinder_line(A, B, radius=0.8, color=[0, 0, 1])
        if seam:
            geometries.append(seam)

        geometries.append(create_approach_arrow(A, vectors[seg_name]["start"], color=[0, 1, 0]))
        geometries.append(create_approach_arrow(B, vectors[seg_name]["end"],   color=[1, 0, 0]))

    o3d.visualization.draw_geometries(geometries, window_name="Approach Vectors", mesh_show_back_face=True)
