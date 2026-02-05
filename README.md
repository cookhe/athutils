# AthUtils

Utilities for reading, processing, and visualizing output from the [Athena](https://princetonuniversity.github.io/Athena-Cversion/) hydrodynamic code.

## Features

- **Efficient HDF5 workflow** for large (960×960×1280+) 3D grids
- **Memory-efficient slice extraction** from distributed MPI binary files
- **Sparse data access** - load only the snapshots and fields you need
- **Publication-ready visualizations** for density, velocity, and energy fields
- **Multi-run comparisons** - compare results across different simulation parameters

## Installation

```bash
pip install --editable .
```

## Quick Start

### 1. Extract 2D Slices from 3D Data

```python
from athutils import Athena

# Initialize with your athinput file and data directory
athena = Athena(
    file_athinput='path/to/athinput.blast',
    datadir='path/to/output'
)

# Extract midplane slices and save to HDF5
athena.slice_to_hdf5(
    snapshots=[0, 10, 20, 30, 40],
    hdf5_file='midplane_slices.h5',
    x3=640  # midplane index
)
```

### 2. Load and Analyze Data

```python
from athutils import AthenaSlice

# Load a specific snapshot
slice_data = AthenaSlice.load_hdf5('midplane_slices.h5', snapshot=20)

# Access data and coordinates
rho = slice_data.data['rho']      # Density field
time = slice_data.t                # Simulation time
x1 = slice_data.coords['x1']       # X-coordinates

# Calculate statistics
mean_density = rho.mean()
max_density = rho.max()
```

### 3. Create Visualizations

```python
from athutils import plot_density_timeseries

# Compare multiple runs over time
plot_density_timeseries(
    hdf5_files=['baseline.h5', 'high_energy.h5', 'low_density.h5'],
    snapshots=[0, 10, 20, 30],
    run_labels=['Baseline', 'High Energy', 'Low Density'],
    output_file='comparison.png'
)
```

## Why HDF5?

For large simulations (e.g., 960×960×1280 grids), HDF5 offers significant advantages:

**Memory Savings:**
- **NPZ format**: 7 GB file, 7 GB memory to load (entire file in RAM)
- **HDF5 format**: 2-3 GB file (compressed), ~70 MB memory per snapshot
- **Result**: ~100× reduction in memory usage for selective access

**Features:**
- Load only specific snapshots without reading entire file
- Load only specific fields (e.g., just density, not all 5 fields)
- Built-in compression reduces storage by 2-3×
- Metadata stored with data (time, snapshot number, etc.)

## Package Structure

```
athutils/
├── io/                    # Data reading and writing
│   ├── ReadAthena.py     # Main Athena reader class
│   └── ...
├── vis/                   # Visualization tools
│   ├── plotting.py       # Core plotting functions
│   ├── plot_slice.py     # Slice-specific visualizations
│   ├── plot_mach.py      # Mach number visualizations
│   └── plot_column.py    # Column density visualizations
└── __version__.py
```

## Documentation

### Core Classes

#### `Athena`
Main interface for reading Athena simulation data.

**Methods:**
- `slice(snapshot, x3=None, x2=None, x1=None, fields=None)` - Extract single 2D slice
- `slice_to_hdf5(snapshots, hdf5_file, x3=None, x2=None, x1=None)` - Batch process to HDF5
- `parse_athinput(file_athinput)` - Parse athinput configuration file

**Example:**
```python
athena = Athena('athinput.blast', 'output/')
slice_data = athena.slice(snapshot=20, x3=640)
```

#### `AthenaSlice`
Container for 2D slice data with convenient save/load methods.

**Attributes:**
- `coords` (dict) - Coordinate arrays for the slice plane
- `data` (dict) - Data fields (rho, rux1, rux2, rux3, eng)
- `t` (float) - Simulation time
- `dt` (float) - Timestep
- `snapshot` (int) - Snapshot number
- `axis` (str) - Slicing axis ('x1', 'x2', or 'x3')
- `index` (int) - Index along slicing axis

**Methods:**
- `save_hdf5(filename, mode='a')` - Save to HDF5 file
- `load_hdf5(filename, snapshot)` - Load from HDF5 file (class method)
- `save_npz(filename)` - Save to compressed NPZ file

### Visualization Functions

#### `plot_density_timeseries(hdf5_files, snapshots, **kwargs)`
Create grid plot comparing multiple runs across time.

**Parameters:**
- `hdf5_files` (list) - HDF5 files, one per run
- `snapshots` (list) - Snapshot numbers to plot
- `run_labels` (list, optional) - Labels for each run
- `vmin`, `vmax` (float, optional) - Color scale limits
- `cmap` (str) - Matplotlib colormap
- `output_file` (str, optional) - Save path

**Returns:** `fig, axes`

#### `plot_single_slice(hdf5_file, snapshot, field='rho', **kwargs)`
Plot a single field from one snapshot.

**Parameters:**
- `hdf5_file` (str) - Path to HDF5 file
- `snapshot` (int) - Snapshot number
- `field` (str) - Field to plot ('rho', 'rux1', etc.)
- `vmin`, `vmax` (float, optional) - Color scale limits
- `cmap` (str) - Matplotlib colormap
- `output_file` (str, optional) - Save path

**Returns:** `fig, ax`

#### `plot_field_comparison(hdf5_file, snapshot, fields, **kwargs)`
Plot multiple fields side-by-side from same snapshot.

**Parameters:**
- `hdf5_file` (str) - Path to HDF5 file
- `snapshot` (int) - Snapshot number  
- `fields` (list) - Fields to plot
- `cmap` (str) - Matplotlib colormap
- `output_file` (str, optional) - Save path

**Returns:** `fig, axes`

## Example Workflows

### Memory-Efficient Time Evolution Analysis

```python
import h5py
import numpy as np

mean_densities = []
times = []

# Process snapshots one at a time (memory efficient)
with h5py.File('slices.h5', 'r') as f:
    for snap_name in sorted(f.keys()):
        grp = f[snap_name]
        
        # Load only density field
        rho = grp['data']['rho'][:]
        
        # Calculate statistics
        mean_densities.append(np.mean(rho))
        times.append(grp.attrs['t'])
        
        # Free memory immediately
        del rho

# Plot evolution
import matplotlib.pyplot as plt
plt.plot(times, mean_densities, 'o-')
plt.xlabel('Time')
plt.ylabel('Mean Density')
plt.savefig('density_evolution.png')
```

### Extract and Analyze Subregions

```python
import h5py

with h5py.File('slices.h5', 'r') as f:
    grp = f['snapshot_0020']
    
    # Load full slice
    rho = grp['data']['rho'][:]
    
    # Extract center 200×200 region
    ny, nx = rho.shape
    i0, j0 = ny//2 - 100, nx//2 - 100
    i1, j1 = ny//2 + 100, nx//2 + 100
    
    rho_center = rho[i0:i1, j0:j1]
    
    # Analyze subregion
    print(f"Center region density: {rho_center.mean():.3e}")
```

### Process Multiple Runs

```python
from athutils import Athena

runs = {
    'baseline': {'athinput': 'run1/athinput', 'datadir': 'run1/output'},
    'high_energy': {'athinput': 'run2/athinput', 'datadir': 'run2/output'},
    'low_density': {'athinput': 'run3/athinput', 'datadir': 'run3/output'}
}

snapshots = list(range(0, 50, 5))

for name, config in runs.items():
    print(f"Processing {name}...")
    athena = Athena(config['athinput'], config['datadir'])
    athena.slice_to_hdf5(
        snapshots=snapshots,
        hdf5_file=f'{name}_slices.h5',
        x3=640
    )
```

## Advanced Usage

### Custom Plotting with Logarithmic Scale

```python
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py

with h5py.File('slices.h5', 'r') as f:
    grp = f['snapshot_0020']
    rho = grp['data']['rho'][:]
    x = grp['coords']['x1'][:]
    y = grp['coords']['x2'][:]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(rho, origin='lower', cmap='magma',
               extent=[x[0], x[-1], y[0], y[-1]],
               norm=LogNorm(vmin=1e-4, vmax=1e0))

plt.colorbar(im, ax=ax, label=r'$\rho$ (log scale)')
plt.savefig('log_density.png', dpi=300)
```

### Snapshot Difference Plots

```python
import h5py
import matplotlib.pyplot as plt

with h5py.File('slices.h5', 'r') as f:
    rho_early = f['snapshot_0010']['data']['rho'][:]
    rho_late = f['snapshot_0030']['data']['rho'][:]
    x = f['snapshot_0010']['coords']['x1'][:]
    y = f['snapshot_0010']['coords']['x2'][:]

diff = rho_late - rho_early

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(diff, origin='lower', cmap='RdBu_r',
               extent=[x[0], x[-1], y[0], y[-1]],
               vmin=-abs(diff).max(), vmax=abs(diff).max())

plt.colorbar(im, ax=ax, label=r'$\Delta\rho$')
plt.savefig('density_change.png')
```

## Scripts

Example scripts are provided in the `scripts/` directory:

- **`example_workflow.py`** - Complete data processing workflows
- **`plot_examples.py`** - Visualization examples with different styles

Run them to see usage examples:
```bash
python scripts/plot_examples.py
```

## Tips for Large Datasets

1. **Always extract slices first** - Don't try to load full 3D grids
   ```python
   athena.slice_to_hdf5(snapshots, 'slices.h5', x3=640)
   ```

2. **Process snapshots sequentially** - Use HDF5's sparse reading
   ```python
   for snap in snapshots:
       data = AthenaSlice.load_hdf5('slices.h5', snap)
       # analyze...
       del data
   ```

3. **Load only needed fields** - Don't load all 5 fields if you only need density
   ```python
   with h5py.File('slices.h5', 'r') as f:
       rho = f['snapshot_0020']['data']['rho'][:]  # Only density
   ```

## Requirements

- Python ≥ 3.8
- numpy
- h5py
- matplotlib

## Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- New features include docstrings
- Large changes discussed in issues first

## License

GPL-3.0 - See LICENSE file for details.

## Citation

If you use this package in published research, please cite:

```bibtex
@software{athutils,
  author = {Cook, Harrison, E.},
  title = {AthUtils: Utilities for Athena Hydrodynamic Code},
  year = {2026},
  url = {https://github.com/cookhe/athutils}
}
```

## Acknowledgments

Built for analyzing output from [Athena](https://princetonuniversity.github.io/Athena-Cversion/), 
a grid-based code for astrophysical magnetohydrodynamics (MHD).

## Support

For bugs, feature requests, or questions:
- Open an issue on GitHub
- Contact: [your email or contact info]