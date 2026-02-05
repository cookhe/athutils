# from athutils.__version__ import __version__

"""Athutils - Utilities for Athena hydrodynamic code output."""

from athutils.io import Athena, AthenaSlice
from athutils.vis import (
    plot_density_timeseries,
    plot_single_slice,
    plot_field_comparison
)

__version__ = "0.1.0"

__all__ = [
    'Athena',
    'AthenaSlice',
    'plot_density_timeseries',
    'plot_single_slice',
    'plot_field_comparison',
]