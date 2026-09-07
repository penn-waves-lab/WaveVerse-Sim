"""Trace-once synthetic aperture with per-(path, position) corrections.

Reprojecting traced paths to every aperture position can reuse most of the
field computation: the hit points are frozen by the
trace-once premise, so intermediate Fresnel coefficients, material lookups and
the polarization chain are identical at every position. What actually changes
is the last leg (hit -> RX), the first leg (TX -> hit), and the antenna
orientation.

This module runs Sionna once at the array centre with an isotropic single
element and applies, per (path, position) on the GPU:

  * delay and spreading of the first and last legs
  * the scattering pattern at the new incidence/exit directions
  * surface-side and last-leg visibility checks, plus first-leg visibility for moving TX
  * the antenna pattern with the per-position yaw, and the synthetic-array
    phase for every element pair

The polarization basis at the last hit and the intermediate Fresnel
coefficients remain fixed at their centre values. Paths blocked at the
centre but visible from another aperture position are not recovered.
"""

import numpy as np
import tensorflow as tf
import sionna.rt as rt
from sionna.rt import Paths
from sionna.rt.solver_paths import PathsTmpData
from sionna.rt.scattering_pattern import ScatteringPattern
from sionna.rt.utils import rotation_matrix, theta_phi_from_unit_vec

from . import cuda_mixer

C0 = 299792458.0


class SpatialAperture:
    """Beat signals for a cylindrical aperture from one centre field solve.

    Args:
        scene: Sionna scene with the transmitter ``tx`` and receiver ``rx``
            already added at the array centre.
        traced: the tuple returned by ``scene.trace_paths``.
        spec: radar spec (``chirp_per_frame`` must be 1).
        tx_position, rx_position: array-centre device positions.
        look_at: where the devices point at the centre.
        radius: aperture radius in metres.
        obstruction_batch_size: maximum rays per visibility query. Larger
            batches reduce launch overhead but require more GPU memory.
        mixer_split: number of partial path sums in the CUDA FMCW mixer.
    """

    def __init__(
        self,
        scene,
        traced,
        spec,
        tx_position,
        rx_position,
        look_at,
        radius,
        xla=True,
        cuda=True,
        obstruction_batch_size=3_000_000,
        mixer_split=16,
    ):
        """
        cuda: use the CUDA mixer when CuPy/NVRTC are available, otherwise TensorFlow.
        xla: compile per-position corrections with XLA.
        """
        assert spec.chirp_per_frame == 1, "one chirp per aperture position"
        if obstruction_batch_size < 1 or mixer_split < 1:
            raise ValueError("obstruction_batch_size and mixer_split must be positive")
        self.obstruction_batch_size = int(obstruction_batch_size)
        self.mixer_split = int(mixer_split)
        self.cuda = bool(cuda) and cuda_mixer.available()
        self._scat = tf.function(self._scat_impl, jit_compile=xla)
        self.scene, self.spec, self.radius = (
            scene,
            spec,
            radius,
        )
        self.solver = scene._solver_paths
        self.lam = spec.wavelength
        self.tx0 = np.asarray(tx_position, np.float32)
        self.rx0 = np.asarray(rx_position, np.float32)
        tx, rx = scene.get("tx"), scene.get("rx")
        self.pattern = scene.rx_array.antenna.patterns[0]
        self.pos_rx = tf.constant(np.asarray(spec.rx_pos), tf.float32)
        self.pos_tx = tf.constant(np.asarray(spec.tx_pos), tf.float32)

        # Sionna's Scene is a singleton, so swap the arrays in place and put
        # them back; the traced paths do not depend on the array.
        saved = (scene.tx_array, scene.rx_array)
        iso = rt.AntennaArray(
            antenna=rt.Antenna("iso", "V"), positions=[[0.0, 0.0, 0.0]]
        )
        scene.tx_array, scene.rx_array = iso, iso
        tx.look_at(look_at)
        rx.look_at(look_at)
        try:
            # Reuse the centre trace without repeating zero-displacement visibility queries.
            self.S = self._centre((traced[2], traced[6]))
            if self.S is None:
                raise ValueError("No scattered paths were found in this scene")
        finally:
            scene.tx_array, scene.rx_array = saved

        # per-object scattering-pattern parameters, gathered per path exactly
        # as Sionna's field computation does
        _, _, _, alpha_r, alpha_i, lambda_, _ = (
            self.solver._build_scene_object_properties_tensors()
        )
        obj = tf.where(self.S["objects"] < 0, 0, self.S["objects"])
        self.S["alpha_r"] = tf.gather(alpha_r, obj)
        self.S["alpha_i"] = tf.gather(alpha_i, obj)
        self.S["lambda_"] = tf.gather(lambda_, obj)
        self.S["fs0"] = ScatteringPattern.pattern(
            self.S["k_i0"],
            self.S["k_s0"],
            self.S["n"],
            self.S["alpha_r"],
            self.S["alpha_i"],
            self.S["lambda_"],
        )

    def _fields(self, scat_):
        ref = scat_[0]
        src, tgt = ref.sources, ref.targets

        def empty(ty):
            return (
                Paths(sources=src, targets=tgt, scene=self.scene, types=ty),
                PathsTmpData(src, tgt, self.solver._dtype),
            )

        sp, st = empty(Paths.SPECULAR)
        sc, ct = scat_
        dp, dt = empty(Paths.DIFFRACTED)
        rp, rtt = empty(Paths.RIS)
        p = self.scene.compute_fields(
            spec_paths=sp,
            diff_paths=dp,
            scat_paths=sc,
            ris_paths=rp,
            spec_paths_tmp=st,
            diff_paths_tmp=dt,
            scat_paths_tmp=ct,
            ris_paths_tmp=rtt,
            check_scene=False,
            scat_random_phases=False,
            testing=True,
        )
        p.normalize_delays = False
        return p

    def _centre(self, comp):
        """Field and geometry for every path at the array centre."""
        _, tmp = comp
        p = self._fields(comp)
        # cir() is where the propagation phase exp(-j 2 pi f tau) is applied;
        # Paths.a alone does not carry it.
        a0, tau0 = (np.asarray(x).reshape(-1) for x in p.cir(los=False))
        ob = np.asarray(p.objects)
        ob = ob.reshape(ob.shape[0], -1)
        vt = np.asarray(p.vertices)
        vt = vt.reshape(vt.shape[0], -1, 3)
        depth = (ob != -1).sum(0)
        P = depth.size
        if P == 0:
            return None
        first, last = vt[0], vt[depth - 1, np.arange(P)]
        f32 = lambda x: tf.constant(x, tf.float32)
        G = dict(
            P=P,
            a0=tf.constant(a0.astype(np.complex64)),
            tau0=f32(tau0),
            first=f32(first),
            last=f32(last),
            r_last0=f32(np.linalg.norm(self.rx0 - last, axis=-1)),
            r1_0=f32(np.linalg.norm(first - self.tx0, axis=-1)),
            d1=tf.constant(depth == 1),
        )
        k_i0 = np.asarray(tmp.scat_last_k_i).reshape(-1, 3)
        n = np.asarray(tmp.scat_last_normals).reshape(-1, 3)
        n = n * (-np.sign((k_i0 * n).sum(-1, keepdims=True)))  # face the incoming ray
        G.update(
            k_i0=f32(k_i0),
            n=f32(n),
            k_s0=f32(np.asarray(tmp.scat_k_s).reshape(-1, 3)),
            objects=tf.constant(
                np.asarray(tmp.scat_last_objects).reshape(-1), tf.int32
            ),
        )
        return G

    def _legs(self, G, dirs):
        off = self.radius * dirs
        tx_k = tf.constant(self.tx0)[None, :] + off
        rx_k = tf.constant(self.rx0)[None, :] + off
        v = rx_k[:, None, :] - G["last"][None]
        r_k = tf.norm(v, axis=-1)
        k_s = v / r_k[..., None]
        u = G["first"][None] - tx_k[:, None, :]
        r1_k = tf.norm(u, axis=-1)
        k_tx = u / r1_k[..., None]
        return r_k, k_s, r1_k, k_tx

    def _antenna(self, dirs, k_rx, k_tx):
        K, P = tf.shape(k_rx)[0], tf.shape(k_rx)[1]
        yaw = tf.atan2(dirs[:, 1], dirs[:, 0])
        R = rotation_matrix(
            tf.stack([yaw, 0 * yaw, 0 * yaw], -1)
        )  # pure yaw: look_at in-plane

        def gain(k):
            kp = tf.einsum("kji,kpj->kpi", R, k)  # device frame
            th, ph = theta_phi_from_unit_vec(tf.reshape(kp, [-1, 3]))
            return tf.reshape(self.pattern(th, ph)[0], [K, P])

        two_pi_l = 2 * np.pi / self.lam
        ph_rx = two_pi_l * tf.einsum(
            "kei,kpi->kep", tf.einsum("kij,ej->kei", R, self.pos_rx), k_rx
        )
        ph_tx = two_pi_l * tf.einsum(
            "kei,kpi->kep", tf.einsum("kij,ej->kei", R, self.pos_tx), k_tx
        )
        return gain(k_rx) * gain(k_tx), ph_rx, ph_tx

    def _scat_impl(self, dirs):
        G = self.S
        K = tf.shape(dirs)[0]
        P = G["P"]
        r_k, k_s, r1_k, k_tx = self._legs(G, dirs)
        k_i = tf.where(
            G["d1"][None, :, None],
            k_tx,
            tf.broadcast_to(G["k_i0"][None], tf.shape(k_tx)),
        )
        dtau = ((r_k - G["r_last0"][None]) + (r1_k - G["r1_0"][None])) / C0
        nn = tf.broadcast_to(G["n"][None], tf.shape(k_s))
        rep = lambda x: tf.reshape(tf.broadcast_to(x[None], [K, P]), [-1])
        fsk = ScatteringPattern.pattern(
            tf.reshape(k_i, [-1, 3]),
            tf.reshape(k_s, [-1, 3]),
            tf.reshape(nn, [-1, 3]),
            rep(G["alpha_r"]),
            rep(G["alpha_i"]),
            rep(G["lambda_"]),
        )
        fsk = tf.reshape(fsk, [K, P])
        same_side = tf.reduce_sum(nn * k_s, -1) > 1e-6
        # For a direct scattered path, TX must remain on the incident side.
        incoming_side = tf.reduce_sum(nn * k_tx, -1) < -1e-6
        same_side &= tf.logical_or(~G["d1"][None], incoming_side)
        ant, ph_rx, ph_tx = self._antenna(dirs, -k_s, k_tx)
        # Sionna's ray-tube area cancels incoming-distance spreading; retain its
        # sampling normalization while updating receiver spreading and directivity.
        amp = (G["r_last0"][None] / r_k) * tf.sqrt(
            fsk / tf.maximum(G["fs0"][None], 1e-30)
        )
        ph = -2 * np.pi * self.spec.fc * dtau
        core = G["a0"][None] * tf.complex(amp * tf.cos(ph), amp * tf.sin(ph)) * ant
        return core, k_s, r_k, G["tau0"][None] + dtau, same_side, ph_rx, ph_tx

    def _blocked(self, k_s, r_k, K, P):
        o = tf.reshape(tf.broadcast_to(self.S["last"][None], [K, P, 3]), [-1, 3])
        d = tf.reshape(k_s, [-1, 3])
        m = tf.reshape(r_k, [-1])
        out, step = [], self.obstruction_batch_size
        for a in range(0, K * P, step):
            out.append(
                self.solver._test_obstruction(
                    o[a : a + step], d[a : a + step], m[a : a + step]
                )
            )
        return tf.reshape(tf.concat(out, 0), [K, P])

    def _tx_blocked(self, dirs):
        """Visibility from each moving TX position to the retained first hit."""
        K, P = int(dirs.shape[0]), self.S["P"]
        tx = tf.constant(self.tx0)[None] + self.radius * dirs
        delta = self.S["first"][None] - tx[:, None]
        distance = tf.norm(delta, axis=-1)
        direction = delta / tf.maximum(distance[..., None], 1e-12)
        origins = tf.reshape(tf.broadcast_to(tx[:, None], [K, P, 3]), [-1, 3])
        direction = tf.reshape(direction, [-1, 3])
        lengths = tf.reshape(distance, [-1])
        blocked, step = [], self.obstruction_batch_size
        for start in range(0, K * P, step):
            blocked.append(
                self.solver._test_obstruction(
                    origins[start : start + step],
                    direction[start : start + step],
                    lengths[start : start + step],
                )
            )
        return tf.reshape(tf.concat(blocked, 0), [K, P]) | (distance < 1e-7)

    def _mix(self, simulator, core, tau, ph_rx, ph_tx, path_batch_size):
        ph = ph_rx[:, :, None, :] + ph_tx[:, None, :, :]  # (K, n_rx, n_tx, P)
        a = core[:, None, None, :] * tf.complex(tf.cos(ph), tf.sin(ph))
        K, n_rx, n_tx = (int(v) for v in a.shape[:3])
        if self.cuda and n_rx * n_tx <= cuda_mixer.MAX_CHANNELS:
            s = cuda_mixer.mix(
                tf.reshape(a, [K, n_rx * n_tx, -1]),
                tau,
                self.spec,
                split=self.mixer_split,
            )
            return np.asarray(s).reshape(K, 1, n_rx, 1, n_tx, 1, -1)
        return np.asarray(
            simulator.simulate_propagation(
                a[:, None, :, None, :, :, None],
                tau[:, None, None, :],
                path_batch_size=path_batch_size,
            )
        )

    def beat_signals(
        self,
        directions,
        simulator,
        chunk=100,
        random_phases=False,
        path_batch_size=10000,
        progress=None,
    ):
        """Beat signals at every aperture position.

        Args:
            directions: (K, 3) unit vectors from the array centre.
            simulator: an FMCW simulator built from the same spec.
            random_phases: emulate ``scat_random_phases=True`` by drawing an
                independent uniform phase per scattered path and position --
                the incoherent baseline, redrawn at every position as the
                reprojection loop does.

        Returns: (K, 1, n_rx, 1, n_tx, 1, n_samples) complex array.
        """
        dirs = np.asarray(directions, np.float32)
        out = []
        for a in range(0, len(dirs), chunk):
            d = tf.constant(dirs[a : a + chunk])
            K = int(d.shape[0])
            core, k_s, r_k, tau, same_side, ph_rx, ph_tx = self._scat(d)
            valid = tf.logical_and(
                same_side, tf.logical_not(self._blocked(k_s, r_k, K, self.S["P"]))
            )
            valid &= ~self._tx_blocked(d)
            core = core * tf.cast(valid, tf.complex64)
            if random_phases:
                th = tf.random.uniform([K, self.S["P"]], 0.0, 2 * np.pi)
                core = core * tf.complex(tf.cos(th), tf.sin(th))
            s = self._mix(simulator, core, tau, ph_rx, ph_tx, path_batch_size)
            out.append(s)
            if progress:
                progress(min(a + chunk, len(dirs)), len(dirs))
        return np.concatenate(out, 0)
