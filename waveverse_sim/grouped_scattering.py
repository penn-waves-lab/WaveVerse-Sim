"""Stable human-vertex groups and power-normalized first-hit expansion.

The split conserves the parent ray's *sampling weight*: sum |w_child|^2 = 1
per parent and scattering-prefix depth, after visibility checks. It does not
force equal received power after changing geometry or coherent interference.
"""

import pickle

import numpy as np
import tensorflow as tf
import trimesh
from scipy.spatial import cKDTree


def expand_other_depths(tensor, repeat_lengths_tensor):
    """
    For each subsequent depth slice in the tensor, repeat the values according to repeat_lengths_tensor.
    Works for both indices (shape: (N,)) and points (shape: (N, 3)), as tf.repeat repeats along axis 0.
    """
    expanded_list = []
    for depth in range(1, tensor.shape[0]):
        # For indices: tensor[depth, 0, :] has shape (N,)
        # For points: tensor[depth, 0, :, :] has shape (N, 3)
        slice_values = tensor[depth, 0, ...]
        repeated = tf.repeat(slice_values, repeat_lengths_tensor, axis=0)
        expanded_list.append(repeated)
    return expanded_list


def expand_hit_mesh_index(
    hit_mesh_index, hit_mesh_points, mapping_table_indices, mapping_table_points
):
    np_indices = hit_mesh_index[0, 0, :].numpy()  # shape: (hitSurfaceNumber,)

    np_points = hit_mesh_points[0, 0, :, :].numpy()  # shape: (hitSurfaceNumber, 3)

    # Allocate each parent ray a contiguous segment for its body-part vertices.
    n = np_indices.shape[0]
    lengths = np.ones(n, dtype=np.int64)
    expanded_pos = np.fromiter(
        mapping_table_indices.keys(), dtype=np.int64, count=len(mapping_table_indices)
    )
    expanded_pos.sort()
    if expanded_pos.size:
        lengths[expanded_pos] = [
            len(mapping_table_indices[int(p)]) for p in expanded_pos
        ]
    offsets = np.concatenate(([0], np.cumsum(lengths)))

    # Start from the traced hit repeated in place, then overwrite the segments
    # that expand to a whole body part.
    flat_idx = np.repeat(np_indices.astype(np.int64), lengths)
    flat_pts = np.repeat(np_points, lengths, axis=0)
    for p_ in expanded_pos:
        p_ = int(p_)
        a, b = int(offsets[p_]), int(offsets[p_ + 1])
        flat_idx[a:b] = np.asarray(mapping_table_indices[p_], dtype=np.int64)
        flat_pts[a:b] = np.asarray(mapping_table_points[p_])

    flat_expanded_indices = tf.constant(flat_idx, dtype=hit_mesh_index.dtype)
    repeat_lengths_indices_tensor = tf.constant(lengths, dtype=tf.int32)
    repeat_lengths_points_tensor = repeat_lengths_indices_tensor
    flat_expanded_points = tf.constant(flat_pts, dtype=hit_mesh_points.dtype)

    expanded_indices_other = expand_other_depths(
        hit_mesh_index, repeat_lengths_indices_tensor
    )
    expanded_points_other = expand_other_depths(
        hit_mesh_points, repeat_lengths_points_tensor
    )

    # Stack the expanded first depth with the repeated values from other depths.

    new_hit_mesh_index = tf.stack(
        [flat_expanded_indices] + expanded_indices_other, axis=0
    )
    new_hit_mesh_index = tf.expand_dims(
        new_hit_mesh_index, axis=1
    )  # final shape: [new_depth, 1, N_expanded]

    new_hit_points = tf.stack([flat_expanded_points] + expanded_points_other, axis=0)
    new_hit_points = tf.expand_dims(
        new_hit_points, axis=1
    )  # final shape: [new_depth, 1, N_expanded, 3]

    return new_hit_mesh_index, new_hit_points


class HumanVertexGroups:
    def __init__(self, vertices, faces, face_group, vertex_face, names):
        self.vertices = np.asarray(vertices, np.float32)
        self.faces = np.asarray(faces, np.int32)
        self.face_group = np.asarray(face_group, np.int32)
        self.vertex_face = np.asarray(vertex_face, np.int32)
        self.names = list(names)
        # Boundary vertices in the correspondence tables belong to multiple
        # groups. Their stored representative face supplies deterministic,
        # disjoint ownership, invariant under motion and vertex coordinates.
        self.vertex_group = self.face_group[self.vertex_face]
        self.groups = {
            g: np.flatnonzero(self.vertex_group == g) for g in range(len(self.names))
        }
        if len(self.face_group) != len(self.faces):
            raise ValueError("Group topology does not match the human mesh")
        if any(len(v) == 0 for v in self.groups.values()):
            raise ValueError("Empty vertex group")
        if not np.all(
            np.any(
                self.faces[self.vertex_face] == np.arange(len(self.vertices))[:, None],
                axis=1,
            )
        ):
            raise ValueError("Representative face must contain its vertex")

    @classmethod
    def from_correspondence(cls, pose, correspondence, placed_pose):
        """Read a trusted local correspondence file, preserving IDs."""
        mesh = trimesh.load(pose, process=False)
        placed = trimesh.load(placed_pose, process=False)
        if not np.array_equal(mesh.faces, placed.faces):
            raise ValueError("Placed pose must preserve original mesh topology")
        with open(correspondence, "rb") as stream:
            corr = pickle.load(stream)
        names = list(corr["group2vertices"])
        name_id = {name: i for i, name in enumerate(names)}
        face_group = [
            name_id[corr["meshindex2group"][i]] for i in range(len(mesh.faces))
        ]
        coordinates = np.asarray(list(corr["vertices2meshindex"]))
        distance, indices = cKDTree(mesh.vertices).query(coordinates)
        if distance.max() > 1e-6 or len(np.unique(indices)) != len(mesh.vertices):
            raise ValueError("Correspondence coordinates do not match this pose")
        vertex_face = np.empty(len(mesh.vertices), np.int32)
        vertex_face[indices] = list(corr["vertices2meshindex"].values())
        return cls(placed.vertices, placed.faces, face_group, vertex_face, names)

    def save(self, path):
        np.savez(
            path,
            vertices=self.vertices,
            faces=self.faces,
            face_group=self.face_group,
            vertex_face=self.vertex_face,
            names=np.asarray(self.names),
        )


def power_split_weights(parent, depth):
    """Return field weights and an audit for surviving parent/prefix families."""
    keys = np.column_stack((parent, depth))
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    weights = (1 / np.sqrt(counts[inverse])).astype(np.float32)
    budget = np.bincount(
        inverse, weights=weights.astype(np.float64) ** 2, minlength=len(counts)
    )
    return weights, {
        "valid_parent_prefix_families": len(counts),
        "max_split_power_error": float(np.max(np.abs(budget - 1), initial=0)),
    }


def _blocked(solver, start, end):
    delta = end - start
    distance = np.linalg.norm(delta, axis=-1)
    direction = delta / np.maximum(distance[:, None], 1e-12)
    return np.asarray(
        solver._test_obstruction(
            tf.constant(start, solver._rdtype),
            tf.constant(direction, solver._rdtype),
            tf.constant(distance, solver._rdtype),
        )
    ) | (distance < 1e-7)


def grouped_path_batches(
    scene, traced, moving_object, groups, displacement, batch_candidates=100000
):
    """Yield regenerated scattered paths, tmp data, field weights and audits.

    One synthetic TX/RX link; antenna elements are handled by compute_fields.
    Whole parent families stay in a batch so occlusion normalization is exact.
    Reflection prefixes are retained; only the first human hit is expanded,
    as specified in the paper. No global sensor-to-all-vertices connection.
    """
    solver = scene._solver_paths
    original, tmp = traced[2], traced[6]
    candidates = np.asarray(tmp.candidates_scat)
    points = np.asarray(tmp.hit_points)
    if candidates.shape[1] != 1 or len(original.targets) != 1:
        raise ValueError("Grouped bursts require one synthetic TX/RX link")
    if batch_candidates < 1:
        raise ValueError("batch_candidates must be positive")
    primitives = np.flatnonzero(
        np.asarray(solver._primitives_2_objects) == scene.get(moving_object).object_id
    )
    vertices = groups.vertices + np.asarray(displacement, np.float32)
    if len(primitives) != len(groups.faces) or not np.allclose(
        np.asarray(solver._primitives)[primitives],
        vertices[groups.faces],
        atol=2e-5,
        rtol=0,
    ):
        raise ValueError("Human primitive IDs/positions do not match group topology")
    primitive_to_face = np.full(solver._primitives_2_objects.shape[0] + 1, -1, np.int32)
    primitive_to_face[primitives] = np.arange(len(primitives))
    first_face = primitive_to_face[candidates[0, 0]]
    group_ids = np.full(len(first_face), -1, np.int32)
    human = first_face >= 0
    group_ids[human] = groups.face_group[first_face[human]]
    lengths = np.ones(len(first_face), np.int64)
    for group, vertex_ids in groups.groups.items():
        lengths[group_ids == group] = len(vertex_ids)
    offsets = np.r_[0, np.cumsum(lengths)]
    first = 0
    while first < len(lengths):
        last = max(
            first + 1,
            int(
                np.searchsorted(
                    offsets, offsets[first] + batch_candidates, side="right"
                )
                - 1
            ),
        )
        last = min(last, len(lengths))
        local_groups = group_ids[first:last]
        target_faces, target_vertices = {}, {}
        for position in np.flatnonzero(local_groups >= 0):
            vi = groups.groups[local_groups[position]]
            target_faces[int(position)] = primitives[groups.vertex_face[vi]]
            target_vertices[int(position)] = vertices[vi]
        indices, hits = expand_hit_mesh_index(
            tf.constant(candidates[:, :, first:last]),
            tf.constant(points[:, :, first:last]),
            target_faces,
            target_vertices,
        )
        idx, xyz = np.asarray(indices).copy(), np.asarray(hits)
        parents = np.repeat(np.arange(last - first), lengths[first:last])
        expanded = np.repeat(local_groups >= 0, lengths[first:last])
        changed = np.flatnonzero(expanded)
        tx_blocked = next_blocked = 0
        if len(changed):
            # Recheck visibility for the changed TX->vertex and vertex->next-bounce segments.
            blocked = _blocked(
                solver,
                np.broadcast_to(
                    np.asarray(original.sources)[0], (len(changed), 3)
                ).copy(),
                xyz[0, 0, changed],
            )
            idx[:, 0, changed[blocked]] = -1
            tx_blocked = int(blocked.sum())
            if len(idx) > 1:
                onward = changed[idx[1, 0, changed] >= 0]
                blocked = (
                    _blocked(solver, xyz[0, 0, onward], xyz[1, 0, onward])
                    if len(onward)
                    else np.zeros(0, bool)
                )
                idx[1:, 0, onward[blocked]] = -1
                next_blocked = int(blocked.sum())
        paths, data = solver._scat_test_rx_blockage(
            original.targets, original.sources, tf.constant(idx), hits, None
        )
        surviving = np.flatnonzero(np.asarray(data.prefix_mask).reshape(-1))
        paths, data = solver._compute_directions_distances_delays_angles(
            paths, data, True
        )
        paths, data = solver._scat_discard_crossing_paths(
            paths, data, tmp.scat_keep_prob
        )
        surviving = surviving[np.asarray(data.discard_mask)]
        prefix = np.asarray(data.scat_prefix_mask)
        depth, _, _, position = np.nonzero(prefix)
        child = surviving[position]
        weights, audit = power_split_weights(parents[child], depth)
        paths, data = solver._scat_prefixes_2_paths(paths, data)
        data.num_samples = tmp.num_samples
        data.scat_keep_prob = tmp.scat_keep_prob
        audit.update(
            input_candidates=last - first,
            expanded_candidates=len(parents),
            expanded_human_parents=int((local_groups >= 0).sum()),
            tx_blocked_children=tx_blocked,
            next_segment_blocked_children=next_blocked,
            output_paths=len(weights),
            human_first_paths_by_depth=[
                int(((depth == d) & expanded[child]).sum()) for d in range(len(idx))
            ],
        )
        yield paths, data, weights, audit
        first = last
