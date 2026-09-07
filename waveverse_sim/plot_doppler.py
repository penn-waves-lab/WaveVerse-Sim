#!/usr/bin/env python3
"""Standalone raw velocity-time view from saved coherent signals."""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from waveverse_sim.processing import (
    range_doppler,
    range_spectra,
    range_mask,
    velocity_limits,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--full-range",
        action="store_true",
        help="full detectable range (already the default)",
    )
    selection.add_argument(
        "--range-gate",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        help="optional range selection; default is full 0–9.6 m",
    )
    selection.add_argument(
        "--both-ranges",
        action="store_true",
        help="save full range, the recorded person range gate and a shared-scale comparison",
    )
    parser.add_argument(
        "--velocity-limit", type=float, help="symmetric displayed velocity limit in m/s"
    )
    args = parser.parse_args()
    out = args.output
    try:
        meta = json.loads((out / "metadata.json").read_text())
        if meta.get("workflow") != "temporal" or meta.get("schema_version") != 1:
            raise ValueError("Expected temporal simulation metadata from this release")
        radar = meta["radar"]
        frames, chirps = meta["frames"], meta["chirps"]
        samples, channels = radar["adc_samples"], radar["channels"]
        if (
            any(
                not isinstance(n, int) or n < 1
                for n in (frames, chirps, samples, channels)
            )
            or chirps < 2
        ):
            raise ValueError("Invalid signal dimensions in metadata")
        constants = [
            radar["range_resolution_m"],
            radar["max_range_m"],
            radar["speed_of_light_m_s"],
            radar["effective_center_frequency_hz"],
            meta["chirp_interval_s"],
        ]
        if not np.isfinite(constants).all() or min(constants) <= 0:
            raise ValueError(
                "Radar metadata must contain finite, positive timing and range values"
            )
        times = np.asarray(meta["time_s"], dtype=float)
        truth = np.asarray(meta["truth_velocity_m_s"], dtype=float)
        if (
            times.shape != (frames,)
            or truth.shape != (frames,)
            or not np.isfinite([times, truth]).all()
        ):
            raise ValueError(
                "Time and velocity arrays must match the saved frame count"
            )
        ranges = np.arange(samples) * radar["range_resolution_m"]
        full_gate = [0.0, float(radar["max_range_m"])]
        gate = args.range_gate or full_gate
        person_gate = meta.get(
            "person_range_gate_m", meta.get("doppler_range_gate_m", [0.1, 0.8])
        )
        gates = [full_gate, person_gate] if args.both_ranges else [gate]
        for selected in gates:
            range_mask(ranges, selected)
        velocity = (
            np.fft.fftshift(np.fft.fftfreq(chirps, meta["chirp_interval_s"]))
            * radar["speed_of_light_m_s"]
            / (2 * radar["effective_center_frequency_hz"])
        )
        view_limits = velocity_limits(velocity, truth, args.velocity_limit)
        rd = []
        for frame in range(frames):
            path = out / f"coherent_frame{frame:04d}.npy"
            signal = np.load(path, allow_pickle=False)
            if signal.shape != (chirps, channels, samples):
                raise ValueError(f"Signal shape does not match metadata: {path.name}")
            rd.append(range_doppler(signal, mti=False))
        rd = np.asarray(rd)
        full_power = range_spectra(rd, ranges, full_gate)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Cannot plot saved signals: {error}")
    person_only = meta.get("person_only", False)
    reference = float(full_power.max())
    if not np.isfinite(reference) or reference <= 0:
        parser.error("Power reference must be finite and positive")
    reference_label = "Power (dB, shared full-range reference)"
    name = (
        "Person only"
        if person_only
        else meta.get("configuration", "Scene + person").split(" / ", 1)[0]
    )
    prefix = "doppler"
    panels = []

    def draw(ax, power, selected):
        h = ax.pcolormesh(
            meta["time_s"],
            velocity,
            10 * np.log10(np.maximum(power / reference, 1e-18)),
            cmap="magma",
            vmin=-45,
            vmax=0,
            shading="auto",
        )
        ax.plot(
            meta["time_s"],
            meta["truth_velocity_m_s"],
            "w--",
            lw=1.5,
            label="Prescribed centroid velocity",
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.set(
            ylim=view_limits,
            xlabel="Time (s)",
            ylabel="Radial velocity (m/s, receding positive)",
            title=f"{name} — raw Doppler\nRange {selected[0]:.1f}–{selected[1]:.1f} m | No mean subtraction or zero-Doppler notch",
        )
        return h

    for selected in gates:
        try:
            power = range_spectra(rd, ranges, selected)
        except ValueError as error:
            parser.error(str(error))
        panels.append((power, selected))
        suffix = (
            "_full_range"
            if selected == full_gate
            else f"_range_{selected[0]:g}_{selected[1]:g}m".replace(".", "p")
        )
        stem = prefix + suffix
        fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
        h = draw(ax, power, selected)
        fig.colorbar(h, ax=ax, label=reference_label)
        target = out / (stem + ".png")
        fig.savefig(target, dpi=180)
        plt.close(fig)
        np.savez(
            out / (stem + ".npz"),
            power=power,
            velocity_m_s=velocity,
            time_s=meta["time_s"],
        )
        report = {
            "source": "coherent_frame*.npy",
            "simulation_rerun": False,
            "mean_subtraction": False,
            "zero_doppler_notch": False,
            "windows": "Hann on fast time and slow time",
            "range_gate_m": selected,
            "display_velocity_m_s": view_limits,
            "power_reference": reference,
            "display_db": [-45, 0],
            "power_reference_source": "full-range coherent peak",
        }
        (out / (stem + "_processing.json")).write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(target)
    if args.both_ranges:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), layout="constrained")
        for ax, (power, selected) in zip(axes, panels):
            h = draw(ax, power, selected)
        fig.colorbar(h, ax=axes, label=reference_label)
        target = out / (prefix + "_range_comparison.png")
        fig.savefig(target, dpi=180)
        plt.close(fig)
        print(target)


if __name__ == "__main__":
    main()
