# WaveVerse-Sim: Scalable RF Simulation in Generative 4D Worlds

<a href="https://arxiv.org/abs/2508.12176"><img src="https://img.shields.io/badge/arXiv-2508.12176-b31b1b.svg" alt="arXiv"></a>
<a href="https://waves.seas.upenn.edu/projects/waveverse/"><img src="https://img.shields.io/badge/Project-Website-green" alt="Project Page"></a>

WaveVerse-Sim is the **GPU-accelerated RF simulator** accompanying
*Scalable RF Simulation in Generative 4D Worlds*. It models **spatial and temporal
phase coherence** and generates FMCW radar signals from indoor scenes and moving
human meshes using GPU ray tracing, CUDA signal synthesis, and grouped human
scattering.


## 🛠️ Installation

Simulation requires an NVIDIA GPU. The examples below have been validated on an RTX 3090 Ti with 24 GB VRAM and CUDA 12.2.

```bash
git clone git@github.com:penn-waves-lab/WaveVerse-Sim.git
cd WaveVerse-Sim
conda env create -f environment.yml
conda activate waveverse-sim
```


## 📦 Example data

Both examples use the bundled laundry room. To make temporal phase coherence
easy to understand, the temporal example adds a posed human mesh and moves it
back and forth at runtime. This is a simple translation of the entire mesh,
not a recorded human-motion sequence or an articulated walking animation.

```text
data/
├── laundry_room/
│   ├── laundry_room.xml
│   ├── meshes/
│   └── scene.json
└── person/
    ├── pose.ply
    └── correspondence.pkl
```

## 🚀 Spatial phase coherence

Simulate panoramic high-resolution imaging as described in our paper, and compare
the results with and without spatial phase coherence:

```bash
python examples/spatial_coherence.py --config configs/spatial.json
```

With the bundled config, signals and range images are saved to `results/spatial/`.
Open `results/spatial/spatial_comparison.png` to compare imaging with and without
spatial phase coherence.

## 🚀 Temporal phase coherence

Compare Doppler with and without temporal phase coherence over 0.1–0.8 m:

```bash
python examples/temporal_coherence.py --config configs/temporal.json
```

With the bundled config, per-frame signals are saved to `results/temporal/`.
Open `results/temporal/doppler_comparison.png` to compare Doppler with and without
temporal phase coherence over 0.1–0.8 m.


To plot full-range 0–9.6 m Doppler from the saved signals:

```bash
python examples/plot_doppler.py results/temporal --full-range
```

The full-range figure is saved to `results/temporal/doppler_full_range.png`.

## 🙏 Acknowledgements

The codebase is built on [Sionna](https://github.com/NVlabs/sionna),
[Mitsuba](https://github.com/mitsuba-renderer/mitsuba3),
[Dr.Jit](https://github.com/mitsuba-renderer/drjit), TensorFlow and CuPy.

## 📜 Citation

```bibtex
@inproceedings{zheng2026scalable,
  title={Scalable RF Simulation in Generative 4D Worlds},
  author={Zhiwei Zheng and Dongyin Hu and Mingmin Zhao},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
}
```

## License

Code is released under [Apache-2.0](LICENSE). Bundled third-party software retains its notices.
