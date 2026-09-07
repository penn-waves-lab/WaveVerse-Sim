"""Plot full/limited raw Doppler from saved signals, without retracing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from waveverse_sim.plot_doppler import main

if __name__ == "__main__":
    main()
