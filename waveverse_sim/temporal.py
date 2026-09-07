#!/usr/bin/env python3
"""Temporal phase-coherent FMCW simulation with grouped human scattering."""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from waveverse_sim.spatial import (
    DEFAULT_XML,
    DEFAULT_SENSOR,
    configure_scene,
    material_settings,
)
from waveverse_sim.config import parse_config, resolved_config
from waveverse_sim.specular import specular_signals
from waveverse_sim.temporal_burst import TemporalBurst
from waveverse_sim.grouped_scattering import HumanVertexGroups
from waveverse_sim.processing import range_doppler, range_mask, velocity_limits
from waveverse_sim import PANORADAR_SPEC

MODES = ("coherent", "incoherent")


def prepare_scene(xml, pose, output, sensor, person_only=False, person_distance=0.55):
    """Place a human pose along the radar's +x direction in generated assets."""
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    human = trimesh.load(pose, process=False)
    v = np.asarray(human.vertices).copy()
    v[:, :2] += np.asarray(sensor[:2]) + [person_distance, 0.0] - v[:, :2].mean(0)
    v[:, 2] -= v[:, 2].min()
    mesh_path = (assets / "human_reference.ply").resolve()
    trimesh.Trimesh(vertices=v, faces=human.faces, process=False).export(mesh_path)
    tree = ET.parse(xml)
    if person_only:
        # Modify only the generated XML copy; source assets stay intact.
        for parent in list(tree.iter()):
            for child in list(parent):
                if child.tag == "shape":
                    parent.remove(child)
    for node in tree.iter("string"):
        if node.get("name") == "filename":
            source = Path(node.get("value"))
            if not source.is_absolute():
                node.set("value", str((xml.parent / source).resolve()))
    shape = ET.SubElement(tree.getroot(), "shape", type="ply", id="temporal_human")
    ET.SubElement(shape, "string", name="filename", value=str(mesh_path))
    ET.SubElement(shape, "ref", id="mat-itu_concrete", name="bsdf")
    scene_label = xml.stem.replace(" ", "_")
    target_xml = assets / (
        "person_only.xml" if person_only else f"{scene_label}_with_human.xml"
    )
    tree.write(target_xml, encoding="utf-8", xml_declaration=True)
    return target_xml, v.mean(0)


def plot_comparison(
    output,
    spec,
    cri,
    times,
    truth,
    displayed,
    gate,
    clutter_label="Raw Doppler (no mean subtraction)",
    clutter_removed=False,
):
    """Range-gated velocity–time comparison with one shared power reference."""
    nc = displayed[0][0].shape[0]
    effective_fc = spec.fc + spec.slope * 1e12 * (
        spec.adc_start_time * 1e-6
        + (spec.adc_samples - 1) / (2 * spec.sample_rate * 1e3)
    )
    velocities = np.fft.fftshift(np.fft.fftfreq(nc, cri)) * spec.c0 / (2 * effective_fc)
    ranges = np.arange(spec.adc_samples) * spec.range_resolution
    roi = range_mask(ranges, gate)
    spectra = [np.array([p[:, roi].sum(1) for p in arm]).T for arm in displayed]
    reference = max(float(s.max()) for s in spectra)
    view_limits = velocity_limits(velocities, truth)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), layout="constrained")
    titles = ("With temporal phase coherence", "Without temporal phase coherence")
    metrics = {}
    dv = abs(velocities[1] - velocities[0])
    for ax, mode, title, spectrum in zip(axes, MODES, titles, spectra):
        h = ax.pcolormesh(
            times,
            velocities,
            10 * np.log10(np.maximum(spectrum / max(reference, 1e-300), 1e-18)),
            cmap="magma",
            vmin=-65,
            vmax=0,
            shading="auto",
        )
        ax.set(
            ylim=view_limits,
            xlabel="Time (s)",
            ylabel="Radial velocity (m/s)",
            title=title,
        )
        ax.plot(times, truth, "w--", lw=1.3, label="Prescribed centroid velocity")
        ax.legend(loc="upper right", fontsize=8)
        peaks = velocities[np.argmax(spectrum, axis=0)]
        on_ridge = abs(velocities[:, None] - truth[None]) <= 1.5 * dv
        metrics[mode] = {
            "peak_velocity_m_s": peaks.tolist(),
            "peak_rmse_m_s": float(np.sqrt(np.mean((peaks - truth) ** 2))),
            "energy_within_1p5_bins_of_truth": float(
                (spectrum * on_ridge).sum() / max(spectrum.sum(), 1e-300)
            ),
            "spectrum_energy": float(spectrum.sum()),
            "mti_energy": float(spectrum.sum()) if clutter_removed else None,
        }
    fig.colorbar(h, ax=axes, label="Power (dB, shared reference)")
    title = clutter_label if clutter_removed else "Doppler"
    fig.suptitle(f"Range {gate[0]:g}-{gate[1]:g}m | {title}")
    fig.savefig(output / "doppler_comparison.png", dpi=180)
    plt.close(fig)
    metrics["velocity_resolution_m_s"] = dv
    metrics["effective_center_frequency_hz"] = effective_fc
    metrics["unambiguous_velocity_m_s"] = spec.c0 / (4 * effective_fc * cri)
    return metrics


def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, default=DEFAULT_XML)
    ap.add_argument(
        "--sensor",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_SENSOR,
        help="radar center position in scene coordinates (metres)",
    )
    ap.add_argument(
        "--person-only",
        action="store_true",
        help="omit all room geometry from the generated scene",
    )
    ap.add_argument(
        "--clutter-removal",
        choices=("none", "ordinary", "weighted"),
        default="none",
        help="default: raw Doppler, no mean subtraction or zero-bin notch",
    )
    ap.add_argument(
        "--range-gate",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=(0.1, 0.8),
        help="displayed range gate; default 0.1–0.8 m matches the bundled person placement",
    )
    ap.add_argument("--pose", type=Path, default=ROOT / "data/person/pose.ply")
    ap.add_argument(
        "--person-distance",
        type=float,
        default=0.55,
        help="initial horizontal distance to the human vertex centroid along +x (metres)",
    )
    ap.add_argument(
        "--correspondence", type=Path, default=ROOT / "data/person/correspondence.pkl"
    )
    ap.add_argument(
        "--no-group-expansion",
        action="store_true",
        help="track sampled hits without expanding paths across body-part vertices",
    )
    ap.add_argument("--group-batch-candidates", type=int, default=100000)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--chirps", type=int, default=128)
    ap.add_argument(
        "--chirp-batch-size",
        type=int,
        default=2,
        help="chirps processed together; reduce to lower GPU memory use",
    )
    ap.add_argument("--num-samples", type=int, default=100000)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--peak-velocity", type=float, default=1.0)
    ap.add_argument("--frequency", type=float, default=1.0)
    ap.add_argument("--frame-period", type=float, default=0.05)
    return ap


def main():
    ap = argument_parser()
    args = parse_config(ap, "temporal")
    if args.chirp_batch_size < 1:
        ap.error("chirp-batch-size must be positive")
    if not np.isfinite(args.person_distance) or args.person_distance <= 0:
        ap.error("person-distance must be finite and positive")
    if not np.isfinite(args.sensor).all():
        ap.error("sensor coordinates must be finite")
    required = [args.xml, args.pose] + (
        [] if args.no_group_expansion or not args.scattering else [args.correspondence]
    )
    for path in required:
        if not path.is_file():
            ap.error(f"Required input does not exist: {path}")
    if args.output is None:
        args.output = (
            ROOT / "results" / ("person_only" if args.person_only else "temporal")
        )
    if args.frames < 1 or args.chirps < 2 or args.num_samples < 1 or args.max_depth < 1:
        ap.error(
            "frames, num-samples and max-depth must be positive; chirps must be at least 2"
        )
    if (
        not np.isfinite([args.frequency, args.frame_period, args.peak_velocity]).all()
        or args.frequency <= 0
        or args.frame_period <= 0
        or args.peak_velocity < 0
    ):
        ap.error(
            "motion parameters must be finite; frequency and frame-period positive; peak-velocity nonnegative"
        )
    if args.group_batch_candidates < 1:
        ap.error("group-batch-candidates must be positive")
    gate = args.range_gate or [0.0, float(PANORADAR_SPEC.max_range)]
    try:
        range_mask(
            np.arange(PANORADAR_SPEC.adc_samples) * PANORADAR_SPEC.range_resolution,
            gate,
        )
    except ValueError as error:
        ap.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    sensor = np.asarray(args.sensor, np.float32)
    begin = time.perf_counter()
    xml, centroid = prepare_scene(
        args.xml, args.pose, args.output, sensor, args.person_only, args.person_distance
    )
    groups = None
    if args.scattering and not args.no_group_expansion:
        groups = HumanVertexGroups.from_correspondence(
            args.pose, args.correspondence, args.output / "assets/human_reference.ply"
        )
        groups.save(args.output / "assets/human_vertex_groups.npz")
    scene, spec, tx, rx, look, assignments = configure_scene(
        xml, sensor, args.materials, args.material_assignments
    )
    if args.person_only and set(scene.objects) != {"temporal_human"}:
        raise AssertionError("Person-only scene must contain exactly the human mesh")
    human_origin = np.asarray(scene.get("temporal_human").position).copy()
    configure_seconds = time.perf_counter() - begin
    cri = (spec.idle_time + spec.ramp_end_time) * 1e-6
    w = 2 * np.pi * args.frequency
    amplitude = args.peak_velocity / w
    raw = [[] for _ in MODES]
    displayed = [[] for _ in MODES]
    records = []
    times = []
    truth = []
    for f in range(args.frames):
        frame_start = time.perf_counter()
        t = f * args.frame_period + np.arange(args.chirps) * cri
        d = amplitude * np.sin(w * t)
        scene.get("temporal_human").position = human_origin + [d[0], 0.0, 0.0]
        trace_start = time.perf_counter()
        tr = (
            scene.trace_paths(
                max_depth=args.max_depth,
                num_samples=args.num_samples,
                scattering=True,
                reflection=True,
                los=False,
                diffraction=False,
                ris=False,
                scat_keep_prob=1.0,
            )
            if args.scattering
            else None
        )
        trace_seconds = time.perf_counter() - trace_start
        prepare_start = time.perf_counter()
        burst = (
            TemporalBurst(
                scene,
                tr,
                spec,
                tx,
                rx,
                look,
                "temporal_human",
                groups=groups,
                group_displacement=[d[0], 0, 0],
                group_batch_candidates=args.group_batch_candidates,
            )
            if args.scattering
            else None
        )
        prepare_seconds = time.perf_counter() - prepare_start
        displacement = np.zeros((len(t), 3), np.float32)
        displacement[:, 0] = d - d[0]
        record = {
            "frame": f,
            "trace_seconds": trace_seconds,
            "prepare_seconds": prepare_seconds,
            "paths": burst.G["P"] if burst else 0,
            "moving_paths": int(burst.moving.sum()) if burst else 0,
            "static_paths": int((~burst.moving).sum()) if burst else 0,
            "path_counts_scope": "scattering",
            "group_expansion": burst.group_audit if burst else None,
        }
        if args.person_only and record["static_paths"]:
            raise AssertionError(
                "Person-only run unexpectedly contains static-object paths"
            )
        signals = []
        for mode in MODES:
            signal = (
                burst.signals(
                    displacement,
                    mode=mode,
                    seed=739 + f,
                    chunk=args.chirp_batch_size,
                )
                if burst
                else np.zeros(
                    (args.chirps, spec.num_rx * spec.num_tx, spec.adc_samples),
                    np.complex64,
                )
            )
            record[mode + "_seconds"] = burst.last_seconds if burst else 0.0
            signals.append(signal)
        record["specular_path_counts"] = []
        specular_start = time.perf_counter()
        if args.specular_only:
            rng = np.random.default_rng(739 + f)
            try:
                for chirp, offset in enumerate(d):
                    scene.get("temporal_human").position = human_origin + [
                        offset,
                        0.0,
                        0.0,
                    ]
                    coherent_spec, control_spec, count = specular_signals(
                        scene,
                        spec,
                        args.num_samples,
                        args.max_depth,
                        rng=rng,
                        moving_object="temporal_human",
                    )
                    signals[0][chirp] += coherent_spec[0]
                    signals[1][chirp] += control_spec[0]
                    record["specular_path_counts"].append(count)
                    if (chirp + 1) % 32 == 0:
                        print(
                            f"frame {f}: specular chirp {chirp + 1}/{args.chirps}",
                            flush=True,
                        )
            finally:
                scene.get("temporal_human").position = human_origin + [d[0], 0.0, 0.0]
        record["specular_seconds"] = time.perf_counter() - specular_start
        for i, (mode, signal) in enumerate(zip(MODES, signals)):
            np.save(args.output / f"{mode}_frame{f:04d}.npy", signal)
            raw[i].append(range_doppler(signal))
            displayed[i].append(
                raw[i][-1]
                if args.clutter_removal == "none"
                else range_doppler(
                    signal, True, weighted_mti=args.clutter_removal == "weighted"
                )
            )
        mid = t.mean()
        times.append(mid)
        c = centroid + [amplitude * np.sin(w * mid), 0, 0]
        u_tx = (c - np.asarray(tx)) / np.linalg.norm(c - np.asarray(tx))
        u_rx = (c - np.asarray(rx)) / np.linalg.norm(c - np.asarray(rx))
        truth.append(args.peak_velocity * np.cos(w * mid) * (u_tx[0] + u_rx[0]) / 2)
        record["total_seconds"] = time.perf_counter() - frame_start
        records.append(record)
        print("FRAME", f, record, flush=True)
    processing_labels = {
        "none": "Raw Doppler (no mean subtraction)",
        "ordinary": "Ordinary-mean subtraction",
        "weighted": "Hann-weighted mean subtraction",
    }
    metrics = plot_comparison(
        args.output,
        spec,
        cri,
        np.array(times),
        np.array(truth),
        displayed,
        gate,
        clutter_label=processing_labels[args.clutter_removal],
        clutter_removed=args.clutter_removal != "none",
    )
    metadata = {
        "configuration": (
            "person only" if args.person_only else f"{args.xml.stem} + person"
        )
        + " / fixed radar Doppler burst",
        "person_only": args.person_only,
        "sensor_xyz_m": sensor.tolist(),
        "person_distance_m": args.person_distance,
        "scene_asset": str(xml.relative_to(args.output)),
        "scene_objects": list(scene.objects),
        "num_samples": args.num_samples,
        "max_depth": args.max_depth,
        "chirps": args.chirps,
        "chirp_batch_size": args.chirp_batch_size,
        "frames": args.frames,
        "chirp_interval_s": cri,
        "frame_period_s": args.frame_period,
        "schema_version": 1,
        "workflow": "temporal",
        "radar": {
            "adc_samples": spec.adc_samples,
            "channels": spec.num_tx * spec.num_rx,
            "range_resolution_m": float(spec.range_resolution),
            "max_range_m": float(spec.max_range),
            "speed_of_light_m_s": float(spec.c0),
            "effective_center_frequency_hz": float(
                metrics["effective_center_frequency_hz"]
            ),
        },
        "scat_keep_prob": 1.0,
        "scattering_pattern": "DirectivePattern; per-material alpha_r in materials",
        "normalize_delays": False,
        "specular_only": args.specular_only,
        "scattering": args.scattering,
        "resolved_config": resolved_config(args),
        "materials": material_settings(args.materials),
        "specular_evaluation": "fresh geometry and fields at every chirp"
        if args.specular_only
        else None,
        "random_phases_coherent_arm": False,
        "fmcw_convention": "TX * conj(RX); conjugated CIR times chirp-delay phase, with no added carrier term",
        "conjugate_channel": True,
        "path_evaluation": "all retained paths evaluated for every chirp",
        "controls": {
            "coherent": "rigid moving material points; deterministic phase",
            "incoherent": "same moving paths and magnitudes; random phase per moving path/chirp",
        },
        "comparison_display_db": [-65, 0],
        "sensor": sensor.tolist(),
        "target_centroid_initial": centroid.tolist(),
        "target_material": assignments["temporal_human"],
        "material_assignments": assignments,
        "peak_velocity_m_s": args.peak_velocity,
        "motion_frequency_hz": args.frequency,
        "time_s": times,
        "truth_velocity_m_s": truth,
        "metrics": metrics,
        "configure_seconds": configure_seconds,
        "group_expansion": groups is not None,
        "group_batch_candidates": args.group_batch_candidates,
        "normalization": "1/sqrt(N_valid) field weight per original candidate and prefix depth"
        if groups is not None
        else "no expansion",
        "normalization_scope": "conserves split ray sampling power, not coherent receiver power after geometry/interference",
        "clutter_removal": args.clutter_removal,
        "clutter_processing": processing_labels[args.clutter_removal],
        "doppler_range_gate_m": gate,
        "person_range_gate_m": list(gate),
        "mean_subtraction": args.clutter_removal != "none",
        "doppler_window": "Hann; zero-velocity bin retained in raw mode",
        "group_ownership": "24 semantic groups; boundary vertex assigned by its stored representative face"
        if groups is not None
        else None,
        "timings_exclude_python_imports": True,
        "frame_timings": records,
        "total_seconds": time.perf_counter() - begin,
        "approximation": "scattering: fresh trace per frame; fixed visibility/Fresnel/polarization/antenna gain within burst",
        "motion_provenance": "bundled human pose with prescribed rigid translation; not articulated walking",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("METRICS", json.dumps(metrics, indent=2), flush=True)
    print("Figure:", args.output / "doppler_comparison.png", flush=True)


if __name__ == "__main__":
    main()
