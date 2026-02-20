#!/usr/bin/env python3
"""Example usage of athutils visualization tools.

This script demonstrates how to create various plots from Athena HDF5 data.
Run this after you've processed your Athena data into HDF5 format.
"""

from athutils.vis.plotting import (
    plot_vertical_density_timeseries,
    plot_midplane_density_timeseries,
    plot_single_slice,
    plot_field_comparison
)


def density_field_grids(slc, zoom=False, inset=False,
                       background_color='black',
                       text_color='white',
                       fmt='png'):
    """Example: Create a grid comparing multiple runs over time."""

    if zoom and inset:
        raise ValueError("Cannot use both zoom and inset options together.")

    output_file = f'density_{slc}_timeseries.{fmt}'
    
    pathroot = '/home/harrison/code/athenaagn/runs/stampede/production'

    units_files = [
        f'{pathroot}/solver-test/hlle/slices/unit_values.json',
        f'{pathroot}/midplane1e5-zoom/slices/unit_values.json',
        f'{pathroot}/z1H/slices/unit_values.json',
    ]

    files = {
        # Define your HDF5 files (one per run)
        "xyfiles": [
            f'{pathroot}/solver-test/hlle/slices/hlle-xyslices.h5',
            f'{pathroot}/midplane1e5-zoom/slices/r1e5-L24-C16-xyslices.h5',
            f'{pathroot}/z1H/slices/z1H-xyslices.h5',
        ],
        "xzfiles": [
            f'{pathroot}/solver-test/hlle/slices/hlle-xzslices.h5',
            f'{pathroot}/midplane1e5-zoom/slices/r1e5-L24-C16-xzslices.h5',
            f'{pathroot}/z1H/slices/z1H-xzslices.h5',
        ]
    }
    
    # Define which snapshots to plot
    snapshots = {
        "xysnapshots": [
            [0, 12, 18, 30] ,
            [0, 4, 8, 12],
            [0, 6, 18, 29],
            # [0, 6, 12, 18],
        ],
        "xzsnapshots": [
            [0, 12, 18, 30],
            [0, 4, 8, 12],
            [0, 6, 18, 29],
            # [0, 6, 12, 18],
        ]
    }

    # color bar limits
    scales = {
        'xy': [1e-17, None],
        'xz': [1e-24, None]
    }

    if slc == 'xz':
        if zoom:
            print("Zoom enabled but not implemented for xzslices")
        # Create the plots
        v_fig, v_axes = plot_vertical_density_timeseries(
            hdf5_files=files['xzfiles'],
            snapshots=snapshots['xzsnapshots'],
            output_file=output_file,
            vmin=1e-24,  # Minimum density for colorscale
            # vmax=1e0,   # Maximum density for colorscale
            cmap='magma',
            units=units_files,
        )

    if slc == 'xy':
        # Create the plots
        m_fig, m_axes = plot_midplane_density_timeseries(
            hdf5_files=files['xyfiles'],
            snapshots=snapshots["xysnapshots"],
            output_file=output_file,
            # vmin=1e-15,  # Minimum density for colorscale
            # vmax=1e0,   # Maximum density for colorscale
            cmap='magma',
            units=units_files,
            zoom=zoom,
            inset=inset,
            background_color=background_color,
            text_color=text_color,
        )
    
    print(f"✓ Saved: {output_file}")


def example_single_snapshot():
    """Example: Plot a single field from one snapshot."""
    
    print("\n=== Example 2: Single Snapshot ===")
    print("Plots one field (e.g., density) from a single snapshot")
    
    # snapshots = np.range(40,44)
    snap = 30

    fig, ax = plot_single_slice(
        hdf5_file='midplane-r1e3-d1H-xzslices.h5',
        snapshot=snap,
        field='rho',
        output_file=f'density_snap{snap}.png',
        # vmin=1e-4,
        # vmax=1e0,
        slc='xz',
        log=True,
        cmap='magma',
        figsize=(10, 8)
    )
    
    print(f"✓ Saved: density_snap{snap}.png")


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

    # file = '/home/harrison/code/athenaagn/runs/stampede/production/solver-test/hlle/slices/hlle-xzslices.h5'
    file = '/home/harrison/code/athenaagn/runs/stampede/production/z1H/slices/z1H-xzslices.h5'
    n_snap = 52
    
    with h5py.File(file, 'r') as f:
        data = []
        for snap in np.arange(0, n_snap):
        # Manually create plot with log scale
            grp = f[f'snapshot_{snap:04d}']
            data.append(grp['data']['rho'][:])
        
    data_copy = np.copy(data)
    data_copy.flatten()
    vmin = np.nanmin(data_copy)
    vmax = np.nanmax(data_copy)

    with h5py.File(file, 'r') as f:
        data = []
        for snap in np.arange(0, n_snap):
        # Manually create plot with log scale
            grp = f[f'snapshot_{snap:04d}']
            rho = grp['data']['rho'][:]
            coords = list(grp['coords'].keys())
            x = grp['coords'][coords[0]][:]
            y = grp['coords'][coords[1]][:]
            t = grp.attrs['t']
    
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(rho, origin='lower', cmap='magma',
                        extent=[x[0], x[-1], y[0], y[-1]],
                        norm=LogNorm(vmin=vmin, vmax=vmax),
                        aspect='equal')
            
            ax.set_xlabel(coords[0], fontsize=12)
            ax.set_ylabel(coords[1], fontsize=12)
            ax.set_title(f'Log Density at t = {t:.2f}', fontsize=14)
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(r'$\rho$', fontsize=12)
            
            filename = f'log_density_{snap:04d}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
            plt.close()


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
    """Run plotting."""
   
    # Uncomment the examples you want to run:

    # Plot xy native grid zoom or not
    # density_field_grids(slc='xy', zoom=True)
    
    # Plot xy native grid shifted with insets
    density_field_grids(slc='xy',
                        inset=True,
                        background_color='white',
                        text_color='black',
                        fmt='pdf')
    
    # Plot xz grids no zoom.
    # density_field_grids(slc='xz',
                        # fmt='pdf')


    # example_density_timeseries()
    # example_single_snapshot()
    # example_field_comparison()
    # example_custom_styling()
    # example_log_scale()
    # example_difference_plot()
    
if __name__ == '__main__':
    import numpy as np  # Needed for some examples
    main()