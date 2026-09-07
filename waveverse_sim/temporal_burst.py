"""Phase-coherent Doppler bursts with rigidly moving scattering points.

Each burst traces at its first chirp, keeps those material points attached to
the moving object, and updates geometric delay, spreading, directive gain and
array phase across chirps.
Visibility, intermediate Fresnel/polarization terms and antenna gain are frozen
within the short burst. A fresh trace is needed for each frame.
"""

import time

import numpy as np
import tensorflow as tf
from sionna.rt import Paths
from sionna.rt.solver_paths import PathsTmpData
from sionna.rt.scattering_pattern import ScatteringPattern

from sionna.rt.utils import rotation_matrix
from . import cuda_mixer
from .grouped_scattering import grouped_path_batches

C0 = 299792458.0
_CORRECTORS = {}


def _shared_corrector(burst):
    """One graph per radar/depth, with dynamic path and chirp dimensions.

    Capturing only the immutable radar equations avoids retaining an old scene
    or its path tensors in the graph cache. New frame path counts do not retrace.
    """
    key = (
        burst.G["D"],
        float(burst.spec.fc),
        float(burst.spec.wavelength),
        burst.tx.tobytes(),
        burst.rx.tobytes(),
        burst.look_at.tobytes(),
        np.asarray(burst.pos_rx).tobytes(),
        np.asarray(burst.pos_tx).tobytes(),
    )
    if key not in _CORRECTORS:
        equations = TemporalBurst.__new__(TemporalBurst)
        for name in ("spec", "tx", "rx", "look_at", "pos_rx", "pos_tx"):
            setattr(equations, name, getattr(burst, name))
        equations.spec = burst.spec.copy()
        depth = burst.G["D"]
        signature = {}
        for name, value in burst.G.items():
            if isinstance(value, tf.Tensor):
                shape = value.shape.as_list()
                shape[1 if name in ("pts0", "vidx") else 0] = None
                signature[name] = tf.TensorSpec(shape, value.dtype)

        @tf.function(
            input_signature=[signature, tf.TensorSpec([None, 2, 3], tf.float32)]
        )
        def correction(geometry, displacement):
            geometry = dict(geometry, D=depth, P=tf.shape(geometry["depth"])[0])
            return equations._correct_impl(geometry, displacement)

        if len(_CORRECTORS) >= 4:
            _CORRECTORS.clear()
        _CORRECTORS[key] = correction
    return _CORRECTORS[key]


class TemporalBurst:
    def _lengths(self, pts, depth, D):
        """Total path length TX -> v1 -> ... -> v_depth -> RX; pts (K, D, P, 3)."""
        tx = tf.constant(self.tx)
        L = tf.norm(pts[:, 0] - tx, axis=-1)
        for d in range(D - 1):
            seg = tf.norm(pts[:, d + 1] - pts[:, d], axis=-1)
            L += tf.where((depth > d + 1)[None], seg, 0.0)
        last = tf.reduce_sum(
            tf.one_hot(depth - 1, D, dtype=tf.float32)[None, :, :, None]
            * tf.transpose(pts, [0, 2, 1, 3]),
            2,
        )
        prev = tf.reduce_sum(
            tf.one_hot(depth - 2, D, dtype=tf.float32)[None, :, :, None]
            * tf.transpose(pts, [0, 2, 1, 3]),
            2,
        )
        prev = tf.where(
            (depth == 1)[None, :, None],
            tf.broadcast_to(tx[None, None], tf.shape(prev)),
            prev,
        )
        r_last = tf.norm(tf.constant(self.rx)[None, None] - last, axis=-1)
        return L + r_last, last, prev, r_last

    def _correct_impl(self, G, disp):
        """disp: (K, n_verts+1, 3) vertex displacements per chirp (last row zero)."""
        K = tf.shape(disp)[0]
        D, P = G["D"], G["P"]
        move = tf.gather(disp, G["vidx"], axis=1)  # (K, D, P, 3)
        pts = G["pts0"][None] + move
        L, last, prev, r_last = self._lengths(pts, G["depth"], D)
        L0, _, _, r_last0 = self._lengths(G["pts0"][None], G["depth"], D)
        L0, r_last0 = L0[0], r_last0[0]  # (P,)
        dtau = (L - L0[None]) / C0  # (K, P)
        tx = tf.constant(self.tx)
        k_s = (tf.constant(self.rx)[None, None] - last) / r_last[..., None]
        k_i = last - prev
        k_i = k_i / tf.norm(k_i, axis=-1, keepdims=True)
        nn = tf.broadcast_to(G["n"][None], tf.shape(k_s))
        rep = lambda x: tf.reshape(tf.broadcast_to(x[None], [K, P]), [-1])
        fsk = tf.reshape(
            ScatteringPattern.pattern(
                tf.reshape(k_i, [-1, 3]),
                tf.reshape(k_s, [-1, 3]),
                tf.reshape(nn, [-1, 3]),
                rep(G["alpha_r"]),
                rep(G["alpha_i"]),
                rep(G["lambda_"]),
            ),
            [K, P],
        )
        amp = (r_last0[None] / r_last) * tf.sqrt(
            fsk / tf.maximum(G["fs0"][None], 1e-30)
        )
        same_side = tf.reduce_sum(nn * k_s, -1) > 1e-6
        ph = -2 * np.pi * self.spec.fc * dtau
        scale = tf.complex(amp * tf.cos(ph), amp * tf.sin(ph))
        # synthetic-array phase change from the moved directions (radar static)
        yaw = np.arctan2(self.look_at[1] - self.tx[1], self.look_at[0] - self.tx[0])
        R = rotation_matrix(tf.constant([yaw, 0.0, 0.0], tf.float32))
        k_rx = -k_s
        k_tx = pts[:, 0] - tx
        k_tx = k_tx / tf.norm(k_tx, axis=-1, keepdims=True)
        k_rx0 = -G["k_s0"][None]
        k_tx0 = G["pts0"][0][None] - tx
        k_tx0 = k_tx0 / tf.norm(k_tx0, axis=-1, keepdims=True)
        two_pi_l = 2 * np.pi / self.spec.wavelength
        d_rx = two_pi_l * tf.einsum(
            "ei,kpi->kep", tf.linalg.matvec(R[None], self.pos_rx), k_rx - k_rx0
        )
        d_tx = two_pi_l * tf.einsum(
            "ei,kpi->kep", tf.linalg.matvec(R[None], self.pos_tx), k_tx - k_tx0
        )
        return scale, dtau, same_side, last, k_s, r_last, d_rx, d_tx

    def __init__(
        self,
        scene,
        traced,
        spec,
        tx_position,
        rx_position,
        look_at,
        moving_object,
        mixer_split=64,
        groups=None,
        group_displacement=None,
        group_batch_candidates=100000,
    ):
        self.spec = spec
        self.tx = np.asarray(tx_position, np.float32)
        self.rx = np.asarray(rx_position, np.float32)
        self.look_at = np.asarray(look_at, np.float32)
        self.pos_rx = tf.constant(np.asarray(spec.rx_pos), tf.float32)
        self.pos_tx = tf.constant(np.asarray(spec.tx_pos), tf.float32)
        self.mixer_split = mixer_split
        self.group_audit = {"enabled": False}
        if groups is not None:
            parts, audits = [], []
            offset = np.zeros(3) if group_displacement is None else group_displacement
            for paths, data, weights, audit in grouped_path_batches(
                scene, traced, moving_object, groups, offset, group_batch_candidates
            ):
                audits.append(audit)
                if not len(weights):
                    continue
                chunk_trace = list(traced)
                chunk_trace[2], chunk_trace[6] = paths, data
                part = TemporalBurst(
                    scene,
                    chunk_trace,
                    spec,
                    tx_position,
                    rx_position,
                    look_at,
                    moving_object,
                    mixer_split,
                )
                if part.G["P"] != len(weights):
                    raise AssertionError("Field/path normalization alignment changed")
                part.a0 *= tf.cast(weights[None, None], part.a0.dtype)
                parts.append(part)
            if not parts:
                raise ValueError(
                    "No scattered paths survived grouped visibility checks"
                )
            self.G = dict(parts[0].G)
            self.G["P"] = sum(p.G["P"] for p in parts)
            for key, value in self.G.items():
                if isinstance(value, tf.Tensor):
                    self.G[key] = tf.concat(
                        [p.G[key] for p in parts],
                        axis=1 if key in ("pts0", "vidx") else 0,
                    )
            self.a0 = tf.concat([p.a0 for p in parts], axis=2)
            self.tau0 = tf.concat([p.tau0 for p in parts], axis=0)
            self.moving = np.concatenate([p.moving for p in parts])
            self._correct = _shared_corrector(self)
            self.group_audit = dict(
                enabled=True,
                normalization="field: 1/sqrt(N_valid), per parent/prefix",
                groups=len(groups.groups),
                vertices=len(groups.vertices),
                batches=len(audits),
                max_split_power_error=max(a["max_split_power_error"] for a in audits),
            )
            for key in (
                "input_candidates",
                "expanded_candidates",
                "expanded_human_parents",
                "tx_blocked_children",
                "next_segment_blocked_children",
                "output_paths",
                "valid_parent_prefix_families",
            ):
                self.group_audit[key] = sum(a[key] for a in audits)
            self.group_audit["human_first_paths_by_depth"] = np.sum(
                [a["human_first_paths_by_depth"] for a in audits], axis=0
            ).tolist()
            return
        source, target = traced[2].sources, traced[2].targets

        def empty(kind):
            return (
                Paths(sources=source, targets=target, scene=scene, types=kind),
                PathsTmpData(source, target, scene._solver_paths._dtype),
            )

        sp, st = empty(Paths.SPECULAR)
        dp, dt = empty(Paths.DIFFRACTED)
        rp, rt = empty(Paths.RIS)
        paths = scene.compute_fields(
            spec_paths=sp,
            spec_paths_tmp=st,
            diff_paths=dp,
            diff_paths_tmp=dt,
            scat_paths=traced[2],
            scat_paths_tmp=traced[6],
            ris_paths=rp,
            ris_paths_tmp=rt,
            check_scene=False,
            scat_random_phases=False,
            testing=True,
        )
        paths.normalize_delays = False
        a, tau = paths.cir(los=False)
        self.a0 = tf.constant(np.asarray(a)[0, 0, :, 0, :, :, 0])
        self.tau0 = tf.constant(np.asarray(tau).reshape(-1))
        objects = np.asarray(paths.objects)[:, 0, 0]
        points = np.asarray(paths.vertices)[:, 0, 0]
        depth = (objects != -1).sum(0).astype(np.int32)
        D, P = objects.shape
        if not P:
            raise ValueError("No scattered paths in this scene")
        moving = objects == scene.get(moving_object).object_id
        self.moving = moving.any(0)
        f32 = lambda x: tf.constant(x, tf.float32)
        s = paths.scat_tmp
        ki = np.asarray(s.scat_last_k_i).reshape(-1, 3)
        normals = np.asarray(s.scat_last_normals).reshape(-1, 3)
        normals = normals * -np.sign((ki * normals).sum(-1, keepdims=True))
        obj = np.maximum(np.asarray(s.scat_last_objects).reshape(-1), 0)
        _, _, _, ar, ai, lam, _ = (
            scene._solver_paths._build_scene_object_properties_tensors()
        )
        ar, ai, lam = (tf.gather(x, obj) for x in (ar, ai, lam))
        ks = f32(np.asarray(s.scat_k_s).reshape(-1, 3))
        self.G = dict(
            P=P,
            D=D,
            pts0=f32(points),
            vidx=tf.constant(np.where(moving, 0, 1), tf.int32),
            depth=tf.constant(depth),
            n=f32(normals),
            k_s0=ks,
            alpha_r=ar,
            alpha_i=ai,
            lambda_=lam,
            fs0=ScatteringPattern.pattern(f32(ki), ks, f32(normals), ar, ai, lam),
        )
        self._correct = _shared_corrector(self)

    def _signals(self, G, a0, tau0, displacement, phases=None, chunk=16):
        outputs = []
        nrx, ntx = a0.shape[:2]
        for first in range(0, len(displacement), chunk):
            xyz = displacement[first : first + chunk]
            disp = tf.constant(np.stack((xyz, np.zeros_like(xyz)), axis=1), tf.float32)
            geometry = {k: v for k, v in G.items() if isinstance(v, tf.Tensor)}
            scale, dtau, valid, _, _, _, pr, pt = self._correct(geometry, disp)
            phase = pr[:, :, None] + pt[:, None, :]
            a = a0[None] * (scale * tf.cast(valid, tf.complex64))[:, None, None]
            a *= tf.complex(tf.cos(phase), tf.sin(phase))
            if phases is not None:
                a *= tf.constant(
                    phases[first : first + len(xyz), None, None], tf.complex64
                )
            signal = cuda_mixer.mix(
                tf.reshape(a, [len(xyz), nrx * ntx, G["P"]]),
                tau0[None] + dtau,
                self.spec,
                split=self.mixer_split,
            )
            outputs.append(np.asarray(signal))
        return np.concatenate(outputs, axis=0)

    def signals(self, displacement, mode="coherent", seed=0, chunk=16):
        """Evaluate all paths for every chirp; return (chirps, RX*TX, ADC).

        incoherent randomizes moving-path phase without randomizing the room.
        """
        if mode not in ("coherent", "incoherent"):
            raise ValueError("Unknown temporal control")
        displacement = np.asarray(displacement, np.float32)
        if (
            displacement.ndim != 2
            or displacement.shape[1] != 3
            or not len(displacement)
        ):
            raise ValueError(
                "displacement must have shape (chirps, 3) with at least one chirp"
            )
        if chunk < 1 or not np.all(np.isfinite(displacement)):
            raise ValueError("chunk must be positive and displacements must be finite")
        phases = None
        if mode == "incoherent":
            rng = np.random.default_rng(seed)
            phase = rng.uniform(-np.pi, np.pi, (len(displacement), self.moving.sum()))
            phases = np.ones((len(displacement), self.G["P"]), np.complex64)
            phases[:, self.moving] = np.exp(1j * phase).astype(np.complex64)
        start = time.perf_counter()
        out = self._signals(self.G, self.a0, self.tau0, displacement, phases, chunk)
        self.last_seconds = time.perf_counter() - start
        return out
