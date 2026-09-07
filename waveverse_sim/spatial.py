#!/usr/bin/env python3
"""Spatial phase-coherent FMCW imaging with a circular synthetic aperture."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from fnmatch import fnmatchcase
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

for gpu in tf.config.experimental.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

import sionna.rt as rt

from waveverse_sim import SpatialAperture, FMCWSimulator, PANORADAR_SPEC
from waveverse_sim.cuda_runtime import load_kernel, prepare_nvrtc
from waveverse_sim.config import parse_config, resolved_config
from waveverse_sim.specular import specular_signals


DEFAULT_XML = ROOT / "data/laundry_room/laundry_room.xml"
DEFAULT_SENSOR = (1.0, 2.75, 0.85)
DEFAULT_OUTPUT = ROOT / "results/spatial"

_PANO_KERNEL = None

# Relative permittivity, conductivity (S/m), and scattering coefficient at 77 GHz.
# All materials use DirectivePattern(alpha_r=10).
MATERIALS = {
    "itu_wood_ex": (1.99, 0.494, 0.50),
    "itu_paper": (2.00, 0.369, 0.45),
    "itu_metal_ex": (1.00, 1.0e7, 0.10),
    "itu_glass_ex": (6.31, 1.211, 0.15),
    "itu_plasterboard_ex": (2.73, 0.503, 0.55),
    "itu_plywood_ex": (2.71, 0.330, 0.45),
    "itu_fabric": (1.80, 0.275, 0.80),
    "itu_ceramic": (8.50, 2.753, 0.30),
}


def material_for_object(name: str) -> str:
    """Assign approximate radio materials from the exported object names."""
    n = name.lower()
    if n == "temporal_human":
        return "itu_fabric"
    if n.startswith("window_"):
        return "itu_glass_ex"
    if n.startswith("wall_") and any(
        f"_{side}_" in n for side in ("north", "south", "east", "west")
    ):
        return "itu_plasterboard_ex"
    return "itu_wood_ex"


def material_settings(overrides=None):
    settings = {
        name: dict(
            relative_permittivity=eps,
            conductivity=sigma,
            scattering_coefficient=scat,
            alpha_r=10,
        )
        for name, (eps, sigma, scat) in MATERIALS.items()
    }
    for name, values in (overrides or {}).items():
        if name not in settings or not isinstance(values, dict):
            raise ValueError(f"Unknown material or invalid properties: {name}")
        if values.keys() - settings[name].keys():
            raise ValueError(f"Unknown material properties for {name}")
        settings[name].update(values)
    for name, values in settings.items():
        for key, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name}.{key} must be a finite number")
        if (
            values["relative_permittivity"] < 1
            or values["conductivity"] < 0
            or not 0 <= values["scattering_coefficient"] <= 1
            or not isinstance(values["alpha_r"], int)
            or values["alpha_r"] < 1
        ):
            raise ValueError(f"Invalid material values for {name}")
    return settings


def configure_scene(
    xml_path: Path, sensor: np.ndarray, materials=None, material_assignments=None
):
    settings = material_settings(materials)
    for pattern, name in (material_assignments or {}).items():
        if (
            not isinstance(pattern, str)
            or not isinstance(name, str)
            or name not in settings
        ):
            raise ValueError(f"Invalid material assignment: {pattern!r}: {name!r}")
    scene = rt.load_scene(str(xml_path))
    spec = PANORADAR_SPEC.copy()
    spec.chirp_per_frame = 1
    scene.frequency = spec.fc
    scene.tx_array, scene.rx_array = spec.sionna_tx_array, spec.sionna_rx_array

    for name, values in settings.items():
        material = scene.get(name)
        if material is None:
            material = rt.RadioMaterial(name)
            scene.add(material)
        elif not isinstance(material, rt.RadioMaterial):
            raise TypeError(f"Scene item {name!r} is not a radio material")
        # The bundled backend already defines these names. Disable its
        # frequency callback to retain the configured material values.
        material.frequency_update_callback = None
        material.relative_permittivity = values["relative_permittivity"]
        material.conductivity = values["conductivity"]
        material.scattering_coefficient = values["scattering_coefficient"]
        material.scattering_pattern = rt.DirectivePattern(alpha_r=values["alpha_r"])

    assignments = {}
    for obj_name, obj in scene.objects.items():
        material_name = material_for_object(obj_name)
        for pattern, name in (material_assignments or {}).items():
            if fnmatchcase(obj_name, pattern):
                material_name = name
        obj.radio_material = material_name
        assignments[obj_name] = material_name

    tx_position, rx_position, look_at = position_sensor(scene, spec, sensor)
    return scene, spec, tx_position, rx_position, look_at, assignments


def position_sensor(scene, spec, sensor):
    """Update devices in a loaded scene before tracing a fresh scan."""
    tx_position = tf.constant(sensor, tf.float32)
    rx_position = tx_position + spec.relative_offset * spec.antenna_spacing
    # The first clockwise aperture position faces +x.  A distant point keeps
    # the TX and vertically offset RX boresights horizontal to float precision.
    look_at = tx_position + tf.constant([1000.0, 0.0, 0.0], tf.float32)
    for name, cls, position in (
        ("tx", rt.Transmitter, tx_position),
        ("rx", rt.Receiver, rx_position),
    ):
        device = scene.get(name)
        if device is None:
            scene.add(cls(name, position=position, look_at=look_at))
        else:
            device.position = position
            device.look_at(look_at)
    return tx_position, rx_position, look_at


def clockwise_directions(n_azimuths: int) -> np.ndarray:
    theta = np.linspace(0.0, -2.0 * np.pi, n_azimuths, endpoint=False)
    return np.stack(
        (np.cos(theta), np.sin(theta), np.zeros_like(theta)), axis=1
    ).astype(np.float32)


def pano_image(signal: np.ndarray) -> np.ndarray:
    """Beamform the circular aperture and compute the PanoRadar range image."""
    prepare_nvrtc()
    import cupy as cp

    n_beams = signal.shape[0]
    n_syn_ante = 301
    if n_beams < n_syn_ante:
        raise ValueError(
            f"PanoRadar imaging needs at least {n_syn_ante} azimuths; got {n_beams}"
        )
    half = (n_syn_ante - 1) // 2

    data = np.concatenate((signal[-half:], signal, signal[:half]), axis=0)
    data = data.squeeze()  # (N, 4, 2, 256)
    data = np.transpose(data, (0, 2, 1, 3))  # (N, 2, 4, 256)
    data = data.reshape(data.shape[0], 8, data.shape[-1]).astype(np.complex64)
    data = cp.asarray(data)

    spec = PANORADAR_SPEC
    sample_time = spec.adc_start_time * 1e-6 + cp.arange(
        spec.adc_samples, dtype=cp.float64
    ) / (spec.sample_rate * 1e3)
    wavelength = spec.c0 / (spec.fc + spec.slope * 1e12 * sample_time)
    # Elevation is zero, hence the vertical compensation is all ones.
    signal_bf = cp.mean(data, axis=1)
    delta_theta = 2.0 * np.pi / n_beams
    h_theta = delta_theta * cp.arange(half, -half - 1, -1)
    ante_x = 0.077 * cp.cos(h_theta).reshape(-1, 1)
    window = cp.hanning(n_syn_ante)[:, None] * cp.hanning(256)[None]
    # PanoRadar Eq. (3): 4*pi for two co-moving endpoints.
    azimuth_comp = cp.exp(4j * np.pi * ante_x / wavelength.reshape(1, -1)) * window
    # Correlate aperture windows without materializing a beam-by-window tensor.
    global _PANO_KERNEL
    if _PANO_KERNEL is None:
        _PANO_KERNEL = cp.RawKernel(
            load_kernel("imaging.cu"),
            "pano_corr",
        )
    signal_bf = cp.ascontiguousarray(signal_bf, dtype=cp.complex64)
    azimuth_comp = cp.ascontiguousarray(azimuth_comp, dtype=cp.complex64)
    correlated = cp.empty((n_beams, 256), dtype=cp.complex64)
    threads = 256
    _PANO_KERNEL(
        ((n_beams * 256 + threads - 1) // threads,),
        (threads,),
        (signal_bf, azimuth_comp, correlated, n_beams, 256, n_syn_ante),
    )
    heatmap = cp.asnumpy(correlated)
    # Only the compact beamformed array is transferred for the range FFT.
    heatmap = np.abs(np.fft.fft(heatmap, axis=1)).T
    return np.log10(heatmap * 1e4 + 1e-4) / 4.0


def save_polar(heatmap: np.ndarray, output: Path, threshold: float):
    clipped = np.clip(heatmap, threshold, None)
    radii = np.arange(clipped.shape[0]) * PANORADAR_SPEC.range_resolution
    theta = np.linspace(0.0, 2.0 * np.pi, clipped.shape[1])
    r_grid, theta_grid = np.meshgrid(radii, theta)
    fig, ax = plt.subplots(
        figsize=(5.2, 5.2), subplot_kw={"projection": "polar", "theta_direction": -1}
    )
    ax.pcolormesh(theta_grid, r_grid, clipped.T, shading="auto", cmap="viridis")
    ax.set_ylim(0.0, PANORADAR_SPEC.max_range)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)


def save_coherence_comparison(coherent, incoherent, output, threshold):
    """Compare phase controls without independently rescaling either image."""
    radii = np.arange(coherent.shape[0]) * PANORADAR_SPEC.range_resolution
    theta = np.linspace(0.0, 2.0 * np.pi, coherent.shape[1])
    vmax = max(float(coherent.max()), float(incoherent.max()), threshold + 1e-6)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5.5),
        layout="constrained",
        subplot_kw={"projection": "polar", "theta_direction": -1},
    )
    for ax, heatmap, title in zip(
        axes,
        (coherent, incoherent),
        ("With spatial phase coherence", "Without spatial phase coherence"),
    ):
        h = ax.pcolormesh(
            theta,
            radii,
            heatmap,
            shading="auto",
            cmap="viridis",
            vmin=threshold,
            vmax=vmax,
        )
        ax.set_ylim(0.0, PANORADAR_SPEC.max_range)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(title, pad=18)
    fig.colorbar(h, ax=axes, shrink=0.75, label="Log amplitude (shared scale)")
    fig.suptitle("Range FFT")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--compare-coherence",
        action=argparse.BooleanOptionalAction,
        help="also simulate a phase-randomized spatial control",
    )
    parser.add_argument("--n-azimuths", type=int, default=1200)
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--mixer-split", type=int, default=64)
    parser.add_argument("--obstruction-batch-size", type=int, default=20_000_000)
    parser.add_argument(
        "--sensor",
        type=float,
        nargs=3,
        action="append",
        metavar=("X", "Y", "Z"),
        help="sensor position in metres; repeat this option to simulate a batch in one loaded scene",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat all sensor positions, retracing each scan; amortizes process/scene startup",
    )
    parser.add_argument(
        "--no-xla", action="store_true", help="disable XLA for aperture corrections"
    )
    parser.add_argument("--no-plots", action="store_true", help="skip PNG rendering")
    parser.add_argument(
        "--compress-signal",
        action="store_true",
        help="write compressed NPZ instead of the faster uncompressed NPY",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-0.10,
        help="lower display threshold in normalized log-amplitude units",
    )
    args = parse_config(parser, "spatial", argv)
    if min(args.repeat, args.chunk, args.mixer_split, args.obstruction_batch_size) < 1:
        parser.error(
            "repeat, chunk, mixer-split and obstruction-batch-size must be positive"
        )
    if args.n_azimuths < 301:
        parser.error("PanoRadar imaging requires at least 301 azimuths")
    if min(args.num_samples, args.max_depth) < 1:
        parser.error("num-samples and max-depth must be positive")
    if not args.xml.is_file():
        parser.error(f"Scene XML does not exist: {args.xml}")
    if args.sensor and not np.isfinite(args.sensor).all():
        parser.error("Sensor coordinates must be finite")
    if not np.isfinite(args.vmin):
        parser.error("vmin must be finite")
    return args


def run_scan(args, sensor, configured=None):
    """Trace and simulate one aperture, optionally reusing the loaded scene."""
    args.output.mkdir(parents=True, exist_ok=True)
    sensor = np.asarray(sensor, np.float32)
    if sensor.shape != (3,) or not np.all(np.isfinite(sensor)):
        raise ValueError("sensor must contain three finite coordinates")
    print(f"scene: {args.xml}")
    print(f"sensor: {sensor.tolist()}")
    print(
        f"settings: radius=.077, azimuths={args.n_azimuths}, samples={args.num_samples}, "
        f"depth={args.max_depth}, scattering={args.scattering}, specular_only={args.specular_only}"
    )

    t_start = time.perf_counter()
    reused_scene = configured is not None
    if reused_scene:
        scene, spec, assignments = configured
        tx_p, rx_p, look_at = position_sensor(scene, spec, sensor)
    else:
        scene, spec, tx_p, rx_p, look_at, assignments = configure_scene(
            args.xml, sensor, args.materials, args.material_assignments
        )
        configured = (scene, spec, assignments)
    directions = clockwise_directions(args.n_azimuths)
    configure_seconds = time.perf_counter() - t_start
    t_trace = time.perf_counter()
    traced = trace_scene(scene, args) if args.scattering else None
    trace_seconds = time.perf_counter() - t_trace
    simulator = FMCWSimulator(spec)
    t_aperture_init = time.perf_counter()
    aperture = (
        SpatialAperture(
            scene,
            traced,
            spec,
            tx_p,
            rx_p,
            look_at,
            radius=0.077,
            xla=not args.no_xla,
            cuda=True,
            obstruction_batch_size=args.obstruction_batch_size,
            mixer_split=args.mixer_split,
        )
        if args.scattering
        else None
    )
    aperture_init_seconds = time.perf_counter() - t_aperture_init
    print(
        f"trace: {trace_seconds:.2f}s; retained {aperture.S['P'] if aperture else 0} scattered paths"
    )
    t_aperture = time.perf_counter()
    signal = (
        aperture.beat_signals(
            directions,
            simulator,
            chunk=args.chunk,
            random_phases=False,
            path_batch_size=1000,
            progress=lambda i, n: print(f"aperture {i}/{n}", flush=True),
        )
        if aperture
        else np.zeros(
            (args.n_azimuths, 1, spec.num_rx, 1, spec.num_tx, 1, spec.adc_samples),
            np.complex64,
        )
    )
    aperture_seconds = time.perf_counter() - t_aperture

    comparison_seconds = None
    incoherent = None
    if args.compare_coherence:
        comparison_start = time.perf_counter()
        tf.random.set_seed(739)
        incoherent = (
            aperture.beat_signals(
                directions,
                simulator,
                chunk=args.chunk,
                random_phases=True,
                path_batch_size=1000,
                progress=lambda i, n: print(f"incoherent aperture {i}/{n}", flush=True),
            )
            if aperture
            else np.zeros_like(signal)
        )
        comparison_seconds = time.perf_counter() - comparison_start

    specular_counts = []
    specular_start = time.perf_counter()
    if args.specular_only:
        rng = np.random.default_rng(739) if args.compare_coherence else None
        try:
            for i, direction in enumerate(directions):
                for name, origin in (("tx", tx_p), ("rx", rx_p)):
                    scene.get(name).position = origin + 0.077 * direction
                    scene.get(name).orientation = [
                        np.arctan2(direction[1], direction[0]),
                        0,
                        0,
                    ]
                coherent_spec, control_spec, count = specular_signals(
                    scene, spec, args.num_samples, args.max_depth, rng=rng
                )
                signal[i] += coherent_spec.reshape(signal[i].shape)
                if incoherent is not None:
                    incoherent[i] += control_spec.reshape(incoherent[i].shape)
                specular_counts.append(count)
                if (i + 1) % 50 == 0 or i + 1 == len(directions):
                    print(f"specular aperture {i + 1}/{len(directions)}", flush=True)
        finally:
            position_sensor(scene, spec, sensor)
    specular_seconds = time.perf_counter() - specular_start

    t_save = time.perf_counter()
    if args.compress_signal:
        signal_path = args.output / "signal.npz"
        np.savez_compressed(signal_path, signal=signal)
    else:
        signal_path = args.output / "signal.npy"
        np.save(signal_path, signal)
    save_seconds = time.perf_counter() - t_save
    t_imaging = time.perf_counter()
    heatmap = pano_image(signal)
    heatmap_path = args.output / "heatmap.npy"
    np.save(heatmap_path, heatmap)
    imaging_seconds = time.perf_counter() - t_imaging

    t_plot = time.perf_counter()
    spatial_png = args.output / "spatial.png"
    if not args.no_plots:
        save_polar(heatmap, spatial_png, args.vmin)
    plot_seconds = time.perf_counter() - t_plot

    if args.compare_coherence:
        comparison_start = time.perf_counter()
        if args.compress_signal:
            np.savez_compressed(
                args.output / "signal_incoherent.npz", signal=incoherent
            )
        else:
            np.save(args.output / "signal_incoherent.npy", incoherent)
        incoherent_heatmap = pano_image(incoherent)
        np.save(args.output / "heatmap_incoherent.npy", incoherent_heatmap)
        if not args.no_plots:
            save_coherence_comparison(
                heatmap,
                incoherent_heatmap,
                args.output / "spatial_comparison.png",
                args.vmin,
            )
        comparison_seconds += time.perf_counter() - comparison_start

    counts = Counter(assignments.values())
    metadata = {
        "scene": args.xml.stem,
        "coherence_comparison": args.compare_coherence,
        "incoherent_control": "same paths and path amplitudes; independent uniform phase per path/aperture position"
        if args.compare_coherence
        else None,
        "incoherent_phase_seed": 739 if args.compare_coherence else None,
        "comparison_seconds": comparison_seconds,
        "xml": str(args.xml),
        "sensor_xyz_m": sensor.tolist(),
        "radius_m": 0.077,
        "n_azimuths": args.n_azimuths,
        "rotation": "clockwise",
        "start_angle_rad": 0.0,
        "num_samples": args.num_samples,
        "max_depth": args.max_depth,
        "scat_keep_prob": 1.0,
        "scat_random_phases": False,
        "normalize_delays": False,
        "cir_los": False,
        "specular_only": args.specular_only,
        "scattering": args.scattering,
        "resolved_config": resolved_config(args),
        "materials": material_settings(args.materials),
        "specular_path_counts": specular_counts,
        "specular_seconds": specular_seconds,
        "specular_evaluation": "fresh geometry and fields at every aperture position"
        if args.specular_only
        else None,
        "move_source": True,
        "tx_motion": "moving",
        "tx_first_leg_visibility": True,
        "imaging_steering_scale": 1.0,
        "approximation": "scattering: centre-traced interaction points; fixed Fresnel/polarization; endpoint delay, direction and visibility updates",
        "frequency_hz": 77e9,
        "antenna_pattern": "tr38901,V (TX and RX)",
        "scattering_pattern": "DirectivePattern; per-material alpha_r in materials",
        "conjugate_channel": True,
        "if_convention": "TX * conj(RX); conjugate channel times chirp-delay phase, with no added carrier term",
        "propagation_phase_count": 1,
        "beamforming": "PanoRadar Eq. (3), 4*pi, using sampled chirp frequencies",
        "power_rescaling": False,
        "display_vmin": args.vmin,
        "cuda_mixer_preparation": aperture.cuda if aperture else True,
        "mixer_split": args.mixer_split,
        "obstruction_batch_size": args.obstruction_batch_size,
        "reused_scene": reused_scene,
        "timings_exclude_python_imports": True,
        "trace_seconds": trace_seconds,
        "configure_seconds": configure_seconds,
        "aperture_init_seconds": aperture_init_seconds,
        "aperture_seconds": aperture_seconds,
        "signal_save_seconds": save_seconds,
        "imaging_seconds": imaging_seconds,
        "plot_seconds": plot_seconds,
        "total_seconds": time.perf_counter() - t_start,
        "retained_scattered_paths": aperture.S["P"] if aperture else 0,
        "material_counts_local_xml": dict(sorted(counts.items())),
        "material_assignments": assignments,
        "material_assignment": "semantic mapping of object names to eight radio materials",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"signal: {signal.shape}, heatmap: {heatmap.shape}")
    print(
        "timings: "
        f"configure={configure_seconds:.2f}s trace={trace_seconds:.2f}s "
        f"aperture_init={aperture_init_seconds:.2f}s aperture={aperture_seconds:.2f}s "
        f"save={save_seconds:.2f}s imaging={imaging_seconds:.2f}s "
        f"plots={plot_seconds:.2f}s total={metadata['total_seconds']:.2f}s"
    )
    if not args.no_plots:
        print(f"figure: {spatial_png}")
        if args.compare_coherence:
            print(f"comparison: {args.output / 'spatial_comparison.png'}")
    return metadata, configured


def trace_scene(scene, args):
    """Trace the centre rays for a co-moving TX/RX aperture."""
    return scene.trace_paths(
        max_depth=args.max_depth,
        los=True,
        reflection=True,
        diffraction=False,
        scattering=True,
        scat_keep_prob=1.0,
        ris=False,
        num_samples=args.num_samples,
    )


def main():
    args = parse_args()
    sensors = args.sensor or [DEFAULT_SENSOR]
    sensors = sensors * args.repeat
    root_output = args.output
    configured = None
    records = []
    batch_start = time.perf_counter()
    for index, sensor in enumerate(sensors):
        if len(sensors) > 1:
            args.output = root_output / f"scan_{index:04d}"
        record, configured = run_scan(args, sensor, configured)
        records.append(record)
    if len(records) > 1:
        batch = {
            "scans": len(records),
            "total_seconds": time.perf_counter() - batch_start,
            "first_scan_seconds": records[0]["total_seconds"],
            "subsequent_scan_median_seconds": float(
                np.median([m["total_seconds"] for m in records[1:]])
            ),
            "each_scan_retraced": True,
            "timings_exclude_python_imports": True,
            "runs": records,
        }
        (root_output / "batch_metadata.json").write_text(
            json.dumps(batch, indent=2) + "\n"
        )
        print(
            f"batch: first={batch['first_scan_seconds']:.2f}s; "
            f"subsequent median={batch['subsequent_scan_median_seconds']:.2f}s; "
            f"total={batch['total_seconds']:.2f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
