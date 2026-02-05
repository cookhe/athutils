#!/usr/bin/env python3
"""Example usage of athutils visualization tools.

This script demonstrates how to create various plots from Athena HDF5 data.
Run this after you've processed your Athena data into HDF5 format.
"""

from athutils_viz import (
    plot_density_timeseries,
    plot_single_slice,
    plot_field_comparison
)


def example_density_timeseries():
    """Example: Create a grid comparing multiple runs over time."""
    
    print("\n=== Example 1: Density Timeseries Grid ===")
    print("Creates a grid plot where:")
    print("  - Each row is a different simulation run")
    print("  - Each column is a different snapshot in time")
    
    # Define your HDF5 files (one per run)
    hdf5_files = [
        'run1_slices.h5',
        'run2_slices.h5',
        'run3_slices.h5'
    ]
    
    # Define which snapshots to plot
    snapshots = [0, 10, 20, 30, 40]
    
    # Labels for each run
    run_labels = [
        'Baseline',
        'High Energy SN',
        'Low Density'
    ]
    
    # Create the plot
    fig, axes = plot_density_timeseries(
        hdf5_files=hdf5_files,
        snapshots=snapshots,
        output_file='density_timeseries.png',
        run_labels=run_labels,
        vmin=1e-3,  # Minimum density for colorscale
        vmax=1e0,   # Maximum density for colorscale
        cmap='inferno'
    )
    
    print("✓ Saved: density_timeseries.png")


def example_single_snapshot():
    """Example: Plot a single field from one snapshot."""
    
    print("\n=== Example 2: Single Snapshot ===")
    print("Plots one field (e.g., density) from a single snapshot")
    
    fig, ax = plot_single_slice(
        hdf5_file='run1_slices.h5',
        snapshot=20,
        field='rho',
        output_file='density_snap20.png',
        vmin=1e-4,
        vmax=1e0,
        cmap='viridis',
        figsize=(10, 8)
    )
    
    print("✓ Saved: density_snap20.png")


def example_field_comparison():
    """Example: Compare multiple fields from the same snapshot."""
    
    print("\n=== Example 3: Multi-Field Comparison ===")
    print("Plots multiple fields side-by-side from one snapshot")
    
    fig, axes = plot_field_comparison(
        hdf5_file='run1_slices.h5',
        snapshot=20,
        fields=['rho', 'eng', 'rux1'],
        output_file='fields_snap20.png',
        cmap='plasma',
        figsize=(15, 4)
    )
    
    print("✓ Saved: fields_snap20.png")


def example_custom_styling():
    """Example: Customize plot appearance."""
    
    print("\n=== Example 4: Custom Styling ===")
    
    import matplotlib.pyplot as plt
    
    # Create plot
    fig, ax = plot_single_slice(
        hdf5_file='run1_slices.h5',
        snapshot=30,
        field='rho',
        cmap='RdYlBu_r',
        figsize=(12, 10)
    )
    
    # Customize after creation
    ax.set_title('Density Distribution at Snapshot 30', fontsize=16, weight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    plt.savefig('custom_density.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: custom_density.png")


def example_log_scale():
    """Example: Plot with logarithmic color scale."""
    
    print("\n=== Example 5: Logarithmic Color Scale ===")
    
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import h5py
    
    # Manually create plot with log scale
    with h5py.File('run1_slices.h5', 'r') as f:
        grp = f['snapshot_0020']
        rho = grp['data']['rho'][:]
        coords = list(grp['coords'].keys())
        x = grp['coords'][coords[0]][:]
        y = grp['coords'][coords[1]][:]
        t = grp.attrs['t']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(rho, origin='lower', cmap='magma',
                   extent=[x[0], x[-1], y[0], y[-1]],
                   norm=LogNorm(vmin=1e-4, vmax=1e0),
                   aspect='auto')
    
    ax.set_xlabel(coords[1], fontsize=12)
    ax.set_ylabel(coords[0], fontsize=12)
    ax.set_title(f'Log Density at t = {t:.2f}', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'Density $\rho$ (log scale)', fontsize=12)
    
    plt.savefig('log_density.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: log_density.png")


def example_difference_plot():
    """Example: Plot difference between two snapshots."""
    
    print("\n=== Example 6: Snapshot Difference ===")
    print("Shows how density changed between two times")
    
    import matplotlib.pyplot as plt
    import h5py
    
    # Load two snapshots
    with h5py.File('run1_slices.h5', 'r') as f:
        rho_early = f['snapshot_0010']['data']['rho'][:]
        rho_late = f['snapshot_0030']['data']['rho'][:]
        coords = list(f['snapshot_0010']['coords'].keys())
        x = f['snapshot_0010']['coords'][coords[0]][:]
        y = f['snapshot_0010']['coords'][coords[1]][:]
        t1 = f['snapshot_0010'].attrs['t']
        t2 = f['snapshot_0030'].attrs['t']
    
    # Calculate difference
    diff = rho_late - rho_early
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(diff, origin='lower', cmap='RdBu_r',
                   extent=[x[0], x[-1], y[0], y[-1]],
                   vmin=-np.abs(diff).max(), vmax=np.abs(diff).max(),
                   aspect='auto')
    
    ax.set_xlabel(coords[1], fontsize=12)
    ax.set_ylabel(coords[0], fontsize=12)
    ax.set_title(f'Density Change: t={t2:.2f} - t={t1:.2f}', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\Delta\rho$', fontsize=12)
    
    plt.savefig('density_difference.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: density_difference.png")


def main():
    """Run all examples."""
    
    print("=" * 60)
    print("Athena Visualization Examples")
    print("=" * 60)
    print("\nNote: These examples assume you have HDF5 files like:")
    print("  - run1_slices.h5")
    print("  - run2_slices.h5")
    print("  - run3_slices.h5")
    print("\nIf you don't have these files yet, process your Athena")
    print("data first using the example_workflow.py script.")
    print()
    
    # Uncomment the examples you want to run:
    
    # example_density_timeseries()
    # example_single_snapshot()
    # example_field_comparison()
    # example_custom_styling()
    # example_log_scale()
    # example_difference_plot()
    
    print("\n" + "=" * 60)
    print("Uncomment the example functions you want to run!")
    print("=" * 60)


if __name__ == '__main__':
    import numpy as np  # Needed for some examples
    main()