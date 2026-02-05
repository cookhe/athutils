#!/usr/bin/env python3
"""Example workflow for processing Athena data to HDF5 and plotting."""

from ReadAthena_fixed import Athena, AthenaSlice
from plot_athena_slices import plot_density_timeseries, plot_single_slice
import numpy as np

# ===== STEP 1: Extract slices and save to HDF5 =====

def process_single_run():
    """Process a single run: extract slices and save to HDF5."""
    
    # Initialize Athena reader
    athena = Athena(
        file_athinput='path/to/athinput.file',
        datadir='path/to/output/data'
    )
    
    # Define which snapshots to process
    snapshots = list(range(0, 100, 5))  # Snapshots 0, 5, 10, ..., 95
    
    # Extract slices at x3=640 (midplane) and save to HDF5
    athena.slice_to_hdf5(
        snapshots=snapshots,
        hdf5_file='run1_slices.h5',
        x3=640  # or x2=480, or x1=480 depending on your slice preference
    )
    
    print("Done processing run 1!")


def process_multiple_runs():
    """Process multiple runs with different parameters."""
    
    runs = [
        {
            'name': 'baseline',
            'athinput': 'run1/athinput.baseline',
            'datadir': 'run1/output',
            'hdf5_out': 'baseline_slices.h5'
        },
        {
            'name': 'high_energy',
            'athinput': 'run2/athinput.high_energy',
            'datadir': 'run2/output',
            'hdf5_out': 'high_energy_slices.h5'
        },
        {
            'name': 'low_density',
            'athinput': 'run3/athinput.low_density',
            'datadir': 'run3/output',
            'hdf5_out': 'low_density_slices.h5'
        }
    ]
    
    snapshots = list(range(0, 50, 2))
    
    for run in runs:
        print(f"\nProcessing {run['name']}...")
        athena = Athena(
            file_athinput=run['athinput'],
            datadir=run['datadir']
        )
        
        athena.slice_to_hdf5(
            snapshots=snapshots,
            hdf5_file=run['hdf5_out'],
            x3=640  # midplane slice
        )
    
    print("\nAll runs processed!")


# ===== STEP 2: Load and analyze HDF5 data =====

def analyze_hdf5_data():
    """Example of loading and analyzing HDF5 slice data."""
    
    # Load a specific snapshot
    slice_data = AthenaSlice.load_hdf5('run1_slices.h5', snapshot=20)
    
    # Access the data
    rho = slice_data.data['rho']
    time = slice_data.t
    
    print(f"Snapshot 20:")
    print(f"  Time: {time}")
    print(f"  Density shape: {rho.shape}")
    print(f"  Density range: {rho.min():.2e} to {rho.max():.2e}")
    
    # Calculate some statistics
    mean_rho = np.mean(rho)
    std_rho = np.std(rho)
    print(f"  Mean density: {mean_rho:.2e}")
    print(f"  Std density: {std_rho:.2e}")
    
    # Access coordinates
    if 'x1' in slice_data.coords:
        x1 = slice_data.coords['x1']
        print(f"  x1 range: {x1[0]} to {x1[-1]}")


# ===== STEP 3: Create visualizations =====

def create_visualizations():
    """Create plots from the HDF5 data."""
    
    # Single snapshot, single field
    plot_single_slice(
        hdf5_file='run1_slices.h5',
        snapshot=20,
        field='rho',
        output_file='density_snapshot20.png',
        vmin=1e-4,
        vmax=1e0
    )
    
    # Timeseries comparison across multiple runs
    hdf5_files = [
        'baseline_slices.h5',
        'high_energy_slices.h5',
        'low_density_slices.h5'
    ]
    
    snapshots = [0, 10, 20, 30, 40]
    
    plot_density_timeseries(
        hdf5_files=hdf5_files,
        snapshots=snapshots,
        output_file='density_comparison.png',
        run_labels=['Baseline', 'High Energy', 'Low Density'],
        vmin=1e-4,
        vmax=1e0,
        cmap='inferno'
    )


# ===== STEP 4: Memory-efficient analysis =====

def memory_efficient_analysis():
    """Example of memory-efficient analysis using HDF5's sparse reading."""
    
    import h5py
    
    # Calculate time evolution of mean density without loading all data
    mean_densities = []
    times = []
    
    with h5py.File('run1_slices.h5', 'r') as f:
        # Get all snapshot names
        snapshot_names = sorted([k for k in f.keys() if k.startswith('snapshot_')])
        
        for snap_name in snapshot_names:
            grp = f[snap_name]
            
            # Only load density data (sparse read)
            rho = grp['data']['rho'][:]
            
            # Calculate statistics
            mean_densities.append(np.mean(rho))
            times.append(grp.attrs['t'])
            
            # Free memory immediately
            del rho
    
    # Plot evolution
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, mean_densities, 'o-')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean Density')
    ax.set_title('Density Evolution')
    ax.grid(True, alpha=0.3)
    plt.savefig('density_evolution.png', dpi=150, bbox_inches='tight')
    print("Saved density evolution plot")


# ===== STEP 5: Extract specific regions =====

def extract_subregion():
    """Extract a subregion from a slice for detailed analysis."""
    
    import h5py
    
    with h5py.File('run1_slices.h5', 'r') as f:
        grp = f['snapshot_0020']
        
        # Get full data
        rho = grp['data']['rho'][:]
        x1 = grp['coords']['x1'][:]
        x2 = grp['coords']['x2'][:]
        
        # Extract subregion (e.g., center 200x200 cells)
        nx2, nx1 = rho.shape
        i_start = nx2 // 2 - 100
        i_end = nx2 // 2 + 100
        j_start = nx1 // 2 - 100
        j_end = nx1 // 2 + 100
        
        rho_subregion = rho[i_start:i_end, j_start:j_end]
        x1_subregion = x1[j_start:j_end]
        x2_subregion = x2[i_start:i_end]
        
        print(f"Extracted subregion:")
        print(f"  Shape: {rho_subregion.shape}")
        print(f"  x1 range: {x1_subregion[0]} to {x1_subregion[-1]}")
        print(f"  x2 range: {x2_subregion[0]} to {x2_subregion[-1]}")


# ===== Main execution =====

if __name__ == '__main__':
    print("Athena Data Processing Workflow")
    print("=" * 50)
    
    # Uncomment the workflow you want to run:
    
    # Step 1: Process data
    # process_single_run()
    # process_multiple_runs()
    
    # Step 2: Analyze
    # analyze_hdf5_data()
    
    # Step 3: Visualize
    # create_visualizations()
    
    # Step 4: Memory-efficient analysis
    # memory_efficient_analysis()
    
    # Step 5: Extract subregions
    # extract_subregion()
    
    print("\nUncomment the desired workflow section to run.")