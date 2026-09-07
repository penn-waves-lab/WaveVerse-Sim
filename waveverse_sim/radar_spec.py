import copy
from typing import Sequence

import sionna.rt as rt
import tensorflow as tf


class RadarSpec:
    """
    Radar timing, antenna geometry and the configuration used by both demos.
    """

    def __init__(
        self,
        c0: float = 299792458,
        num_tx: int = 0,
        num_rx: int = 0,
        fc: float = 0.0,
        slope: float = 0.0,
        adc_samples: int = 0,
        adc_start_time: float = 0,
        sample_rate: float = 0,
        idle_time: float = 0,
        ramp_end_time: float = 0,
        chirp_per_frame: int = 0,
        tx_loc: list[list[float]] = None,
        rx_loc: list[list[float]] = None,
        relative_offset: Sequence[float] = (0.0, 0.0, 0.0),
    ):
        """
        Initialize the radar specifications.

        Args:
            c0 (float, optional): Speed of light in m/s. Defaults to 299792458.
            num_tx (int, optional): Number of transmit antennas. Defaults to 0.
            num_rx (int, optional): Number of receive antennas. Defaults to 0.
            fc (float, optional): Carrier frequency in Hz. Defaults to 0.0.
            slope (float, optional): Frequency slope in MHz/us. Defaults to 0.0.
            adc_samples (int, optional): Number of ADC samples. Defaults to 0.
            adc_start_time (float, optional): ADC start time in us. Defaults to 0.
            sample_rate (float, optional): Sample rate in ksps. Defaults to 0.
            idle_time (float, optional): Idle time in us. Defaults to 0.
            ramp_end_time (float, optional): Ramp end time in us. Defaults to 0.
            chirp_per_frame (int, optional): Number of chirps per frame. Defaults to 0.
            tx_loc (list[list[float]], optional): Transmit antenna locations in half wavelengths. Defaults to None.
            rx_loc (list[list[float]], optional): Receive antenna locations in half wavelengths. Defaults to None.
            relative_offset (Sequence[float], optional): Relative offset between TX/RX centers in half wavelengths. Defaults to zero vector.

        Notes:
            In Sionna, y-axis in tx/rx locations is interpreted as the azimuth axis and z-axis is interpreted as the elevation/zenith axis.
            See the Example in https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.PlanarArray.

        See Also:
            For details on the parameters, see
            https://dr-download.ti.com/software-development/ide-configuration-compiler-or-debugger/MD-h04ItoajtS/02.01.01.00/mmwave_studio_user_guide.pdf Figure 15.1 on page 39.
        """
        self.c0 = c0
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.fc = fc
        self.slope = slope
        self.adc_samples = adc_samples
        self.adc_start_time = adc_start_time
        self.sample_rate = sample_rate
        self.idle_time = idle_time
        self.ramp_end_time = ramp_end_time
        self.chirp_per_frame = chirp_per_frame
        self.tx_loc = tx_loc if tx_loc is not None else []
        self.rx_loc = rx_loc if rx_loc is not None else []
        self.relative_offset = tf.convert_to_tensor(relative_offset)

        assert len(self.tx_loc) == self.num_tx, (
            f"#TX locations {len(self.tx_loc)} does not match num_tx {self.num_tx}"
        )
        assert len(self.rx_loc) == self.num_rx, (
            f"#RX locations {len(self.rx_loc)} does not match num_rx {self.num_rx}"
        )

    @property
    def wavelength(self):
        return self.c0 / self.fc

    @property
    def adc_sampling_time(self):
        """
        duration of a chirp being sampled in seconds
        """
        return self.adc_samples / (self.sample_rate * 1000)

    @property
    def adc_sampled_bandwidth(self):
        """
        Bandwidth of the chirp in Hz.
        """
        return self.slope * 1e12 * self.adc_sampling_time

    @property
    def antenna_spacing(self):
        return self.wavelength / 2

    @property
    def range_resolution(self):
        return self.c0 / (2 * self.adc_sampled_bandwidth)

    @property
    def max_range(self):
        return self.sample_rate * self.c0 / (2 * self.slope * 1e9)

    @property
    def tx_pos(self) -> tf.Tensor:
        return tf.convert_to_tensor(self.tx_loc) * self.antenna_spacing

    @property
    def rx_pos(self) -> tf.Tensor:
        return tf.convert_to_tensor(self.rx_loc) * self.antenna_spacing

    @property
    def sionna_tx_array(self):
        return rt.AntennaArray(
            antenna=rt.Antenna("tr38901", "V"),
            positions=self.tx_pos,
        )

    @property
    def sionna_rx_array(self):
        return rt.AntennaArray(
            antenna=rt.Antenna("tr38901", "V"),
            positions=self.rx_pos,
        )

    def copy(self):
        return copy.deepcopy(self)


PANORADAR_SPEC = RadarSpec(
    c0=299792458,
    num_tx=2,
    num_rx=4,
    fc=77e9,
    slope=79.951,
    adc_samples=256,
    adc_start_time=0.0,
    sample_rate=5120,
    idle_time=10.0,
    ramp_end_time=50.01,
    chirp_per_frame=1,
    tx_loc=[[0.0, 0.0, -2], [0.0, 0.0, 2]],
    rx_loc=[[0.0, 0.0, -1.5], [0.0, 0.0, -0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 1.5]],
    relative_offset=[0, 0, -4.5],
)
