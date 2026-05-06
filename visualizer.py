"""
Standalone interactive visualizer launched as a subprocess by app.py.
Usage: python visualizer.py <obj_path> <json_path> <vectors_json_path>
Press W to step forward along the robot toolpath.
"""
import sys
import json
import numpy as np
import open3d as o3d


# =========================================================
# DRAWING FUNCTIONS
# =========================================================

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


def is_direct_path_safe(ray_scene, start_pt, end_pt):
    """Returns True if the direct hover-to-hover path has no collisions and no window/overhang above it."""
    direction = end_pt - start_pt
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return True

    dir_norm = direction / dist

    ray = o3d.core.Tensor(
        [[start_pt[0], start_pt[1], start_pt[2], dir_norm[0], dir_norm[1], dir_norm[2]]],
        dtype=o3d.core.Dtype.Float32,
    )
    if ray_scene.cast_rays(ray)['t_hit'].numpy()[0] < dist:
        return False

    num_samples = max(int(dist / 10.0), 3)
    for t in np.linspace(0.1, 0.9, num_samples):
        sample = start_pt + t * direction
        ray_up = o3d.core.Tensor(
            [[sample[0], sample[1], sample[2], 0.0, 0.0, 1.0]],
            dtype=o3d.core.Dtype.Float32,
        )
        if not np.isinf(ray_scene.cast_rays(ray_up)['t_hit'].numpy()[0]):
            return False

    return True


def compute_interim_path(pt_A, vec_A, pt_B, vec_B, max_z, mesh_scale, ray_scene, retract_dist=15.0):
    """Direct hop if safe, otherwise strict vertical retract → traverse → drop."""
    dir_A = np.array(vec_A) / np.linalg.norm(vec_A)
    dir_B = np.array(vec_B) / np.linalg.norm(vec_B)

    hover_A = pt_A - dir_A * retract_dist
    hover_B = pt_B - dir_B * retract_dist

    if is_direct_path_safe(ray_scene, hover_A, hover_B):
        return [hover_A, hover_B]

    clearance_margin = max(mesh_scale * 0.1, 15.0)
    safe_z = max_z + clearance_margin
    safe_z = max(safe_z, hover_A[2] + clearance_margin, hover_B[2] + clearance_margin)

    clear_A = hover_A.copy(); clear_A[2] = safe_z
    clear_B = hover_B.copy(); clear_B[2] = safe_z

    return [hover_A, clear_A, clear_B, hover_B]


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    obj_path, json_path, vectors_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # Load mesh
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_axis_aligned_bounding_box().get_center())
    mesh.paint_uniform_color([0.8, 0.8, 0.8])

    bbox       = mesh.get_axis_aligned_bounding_box()
    max_z      = bbox.get_max_bound()[2]
    mesh_scale = np.linalg.norm(bbox.get_extent())

    print("[*] Building raycasting scene for collision checks...")
    mesh_t     = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    ray_scene  = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(mesh_t)

    geometries = [mesh]

    # Load computed vectors and weld JSON
    with open(vectors_path) as f:
        vectors = json.load(f)
    with open(json_path) as f:
        json_edges = json.load(f).get("welding_data", {}).get("edges", [])

    # =========================================================
    # BUILD EDGE LIST FROM JSON + VECTORS
    # =========================================================
    standoff_dist  = 15.0
    CONNECTION_TOL = 5.0
    unordered_edges = []

    for edge in json_edges:
        seg_name = edge.get("segment_name")
        if seg_name not in vectors:
            continue

        A = np.array([edge['start']['x'], edge['start']['y'], edge['start']['z']])
        B = np.array([edge['end']['x'],   edge['end']['y'],   edge['end']['z']])

        v_start = vectors[seg_name]["start"]
        v_end   = vectors[seg_name]["end"]

        unordered_edges.append({'A': A, 'B': B, 'vA': v_start, 'vB': v_end, 'name': seg_name})

        geometries.append(create_approach_arrow(A, v_start, standoff_dist, color=[0, 1, 0]))
        geometries.append(create_approach_arrow(B, v_end,   standoff_dist, color=[1, 0, 0]))

    # =========================================================
    # CHAIN BUILDING & ORDERING
    # =========================================================
    chains = []
    unprocessed = list(unordered_edges)

    while unprocessed:
        current_chain = [unprocessed.pop(0)]

        while True:
            last_pt = current_chain[-1]['B']
            found = False
            for i, e in enumerate(unprocessed):
                if np.linalg.norm(e['A'] - last_pt) < CONNECTION_TOL:
                    current_chain.append(unprocessed.pop(i)); found = True; break
                elif np.linalg.norm(e['B'] - last_pt) < CONNECTION_TOL:
                    e['A'], e['B'] = e['B'], e['A']
                    e['vA'], e['vB'] = e['vB'], e['vA']
                    current_chain.append(unprocessed.pop(i)); found = True; break
            if not found:
                break

        while True:
            first_pt = current_chain[0]['A']
            found = False
            for i, e in enumerate(unprocessed):
                if np.linalg.norm(e['B'] - first_pt) < CONNECTION_TOL:
                    current_chain.insert(0, unprocessed.pop(i)); found = True; break
                elif np.linalg.norm(e['A'] - first_pt) < CONNECTION_TOL:
                    e['A'], e['B'] = e['B'], e['A']
                    e['vA'], e['vB'] = e['vB'], e['vA']
                    current_chain.insert(0, unprocessed.pop(i)); found = True; break
            if not found:
                break

        chains.append(current_chain)

    # Greedy nearest-neighbour chain ordering
    ordered_edges = []
    if chains:
        ordered_chains = [chains.pop(0)]
        while chains:
            last_pt = ordered_chains[-1][-1]['B']
            best_idx, best_dist, flip = -1, float('inf'), False
            for i, chain in enumerate(chains):
                d_start = np.linalg.norm(chain[0]['A']  - last_pt)
                d_end   = np.linalg.norm(chain[-1]['B'] - last_pt)
                if d_start < best_dist: best_dist = d_start; best_idx = i; flip = False
                if d_end   < best_dist: best_dist = d_end;   best_idx = i; flip = True

            next_chain = chains.pop(best_idx)
            if flip:
                next_chain.reverse()
                for e in next_chain:
                    e['A'], e['B'] = e['B'], e['A']
                    e['vA'], e['vB'] = e['vB'], e['vA']
            ordered_chains.append(next_chain)

        for chain in ordered_chains:
            ordered_edges.extend(chain)

    # =========================================================
    # BUILD SEGMENTS (WELD=BLUE, AIR=YELLOW)
    # =========================================================
    segments = []

    for i, curr_e in enumerate(ordered_edges):
        segments.append((curr_e['A'], curr_e['B'], [0, 0, 1]))

        if i < len(ordered_edges) - 1:
            next_e = ordered_edges[i + 1]
            end_A, start_B = curr_e['B'], next_e['A']

            if np.linalg.norm(start_B - end_A) < CONNECTION_TOL:
                segments.append((end_A, start_B, [1, 1, 0]))
            else:
                waypoints = compute_interim_path(
                    end_A, curr_e['vB'], start_B, next_e['vA'], max_z, mesh_scale, ray_scene, standoff_dist
                )
                full_path = [end_A] + waypoints + [start_B]
                for j in range(len(full_path) - 1):
                    segments.append((full_path[j], full_path[j + 1], [1, 1, 0]))

    # =========================================================
    # INTERACTIVE VISUALIZER
    # =========================================================
    print(f"\n[*] {len(ordered_edges)} edges | {len(segments)} segments")
    print(">>> PRESS 'W' TO STEP FORWARD ALONG THE ROBOT PATH <<<")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Interactive Toolpath  —  Press W to step", width=1280, height=720)

    for geo in geometries:
        vis.add_geometry(geo)

    cursor = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
    cursor.paint_uniform_color([0, 1, 0])
    if segments:
        cursor.translate(segments[0][0])
    vis.add_geometry(cursor)

    state = {'step': 0}

    def advance_path(vis):
        idx = state['step']
        if idx < len(segments):
            p1, p2, color = segments[idx]
            line = create_cylinder_line(p1, p2, radius=0.8, color=color)
            if line:
                vis.add_geometry(line, reset_bounding_box=False)
            cursor.translate(p2 - p1)
            vis.update_geometry(cursor)
            state['step'] += 1
            print(f"  Segment {state['step']}/{len(segments)}")
        else:
            print("  Path complete.")
        return False

    vis.register_key_callback(87, advance_path)  # 87 = 'W'
    vis.run()
    vis.destroy_window()
