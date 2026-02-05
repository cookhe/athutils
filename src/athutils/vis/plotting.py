"""Core plotting functions for Athena slice data."""

import numpy as np
import matplotlib.pyplot as plt
import h5py


def plot_density_timeseries(hdf5_files, snapshots, output_file=None, 
                            vmin=None, vmax=None, cmap='viridis',
                            figsize=None, run_labels=None):
    """Create a grid plot of density fields from multiple runs.
    
    Each row represents a different run, and each column represents
    a different snapshot in time.
    
    Parameters
    ----------
    hdf5_files : list of str
        List of HDF5 file paths, one per run
    snapshots : list of int
        List of snapshot numbers to plot (same for all runs)
    output_file : str, optional
        If provided, save figure to this path
    vmin, vmax : float, optional
        Color scale limits. If None, uses data range
    cmap : str
        Matplotlib colormap name
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated
    run_labels : list of str, optional
        Labels for each run (row)
        
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    n_runs = len(hdf5_files)
    n_snapshots = len(snapshots)
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        width = 3 * n_snapshots
        height = 3 * n_runs
        figsize = (width, height)
    
    # Create figure and axes
    fig, axes = plt.subplots(n_runs, n_snapshots, figsize=figsize,
                            squeeze=False)
    
    # Determine global vmin/vmax if not provided
    if vmin is None or vmax is None:
        all_data = []
        for hdf5_file in hdf5_files:
            with h5py.File(hdf5_file, 'r') as f:
                for snap in snapshots:
                    grp_name = f"snapshot_{snap:04d}"
                    if grp_name in f:
                        rho = f[grp_name]['data']['rho'][:]
                        all_data.append(rho)
        
        if all_data:
            all_data = np.concatenate([d.flatten() for d in all_data])
            if vmin is None:
                vmin = np.percentile(all_data, 1)
            if vmax is None:
                vmax = np.percentile(all_data, 99)
    
    # Plot each run and snapshot
    for i, hdf5_file in enumerate(hdf5_files):
        with h5py.File(hdf5_file, 'r') as f:
            for j, snap in enumerate(snapshots):
                ax = axes[i, j]
                grp_name = f"snapshot_{snap:04d}"
                
                if grp_name not in f:
                    ax.text(0.5, 0.5, f'Snapshot {snap}\nnot found',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                grp = f[grp_name]
                rho = grp['data']['rho'][:]
                t = grp.attrs['t']
                
                # Plot
                im = ax.imshow(rho, origin='lower', cmap=cmap,
                              vmin=vmin, vmax=vmax, aspect='auto')
                
                # Labels
                if i == 0:  # Top row
                    ax.set_title(f't = {t:.2f}', fontsize=10)
                
                if j == 0:  # Left column
                    if run_labels and i < len(run_labels):
                        ax.set_ylabel(run_labels[i], fontsize=10)
                
                # Remove tick labels for cleaner look
                ax.set_xticks([])
                ax.set_yticks([])
    
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=r'Density $\rho$')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_file}")
    
    return fig, axes


def plot_single_slice(hdf5_file, snapshot, field='rho', 
                     output_file=None, cmap='viridis',
                     figsize=(8, 6), vmin=None, vmax=None):
    """Plot a single field from a single snapshot.
    
    Parameters
    ----------
    hdf5_file : str
        HDF5 file path
    snapshot : int
        Snapshot number
    field : str
        Field name to plot ('rho', 'rux1', etc.)
    output_file : str, optional
        If provided, save figure to this path
    cmap : str
        Matplotlib colormap name
    figsize : tuple
        Figure size (width, height)
    vmin, vmax : float, optional
        Color scale limits
        
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    with h5py.File(hdf5_file, 'r') as f:
        grp_name = f"snapshot_{snapshot:04d}"
        if grp_name not in f:
            raise KeyError(f"Snapshot {snapshot} not found in {hdf5_file}")
        
        grp = f[grp_name]
        data = grp['data'][field][:]
        t = grp.attrs['t']
        axis = grp.attrs['axis']
        
        # Get coordinates
        coords = list(grp['coords'].keys())
        x_coord = grp['coords'][coords[0]][:]
        y_coord = grp['coords'][coords[1]][:]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(data, origin='lower', cmap=cmap, 
                   extent=[x_coord[0], x_coord[-1], y_coord[0], y_coord[-1]],
                   aspect='auto', vmin=vmin, vmax=vmax)
    
    ax.set_xlabel(coords[1])
    ax.set_ylabel(coords[0])
    ax.set_title(f'{field} at t = {t:.2f} (snapshot {snapshot})')
    
    plt.colorbar(im, ax=ax, label=field)
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_file}")
    
    return fig, ax


def plot_field_comparison(hdf5_file, snapshot, fields=['rho', 'eng'],
                         output_file=None, cmap='viridis', figsize=None):
    """Plot multiple fields from the same snapshot side by side.
    
    Parameters
    ----------
    hdf5_file : str
        HDF5 file path
    snapshot : int
        Snapshot number
    fields : list of str
        Field names to plot
    output_file : str, optional
        If provided, save figure to this path
    cmap : str
        Matplotlib colormap name
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    n_fields = len(fields)
    
    if figsize is None:
        figsize = (5 * n_fields, 4)
    
    fig, axes = plt.subplots(1, n_fields, figsize=figsize)
    if n_fields == 1:
        axes = [axes]
    
    with h5py.File(hdf5_file, 'r') as f:
        grp_name = f"snapshot_{snapshot:04d}"
        if grp_name not in f:
            raise KeyError(f"Snapshot {snapshot} not found in {hdf5_file}")
        
        grp = f[grp_name]
        t = grp.attrs['t']
        
        # Get coordinates
        coords = list(grp['coords'].keys())
        x_coord = grp['coords'][coords[0]][:]
        y_coord = grp['coords'][coords[1]][:]
        
        for i, field in enumerate(fields):
            data = grp['data'][field][:]
            
            im = axes[i].imshow(data, origin='lower', cmap=cmap,
                               extent=[x_coord[0], x_coord[-1], 
                                      y_coord[0], y_coord[-1]],
                               aspect='auto')
            
            axes[i].set_xlabel(coords[1])
            if i == 0:
                axes[i].set_ylabel(coords[0])
            axes[i].set_title(field)
            
            plt.colorbar(im, ax=axes[i])
    
    fig.suptitle(f't = {t:.2f} (snapshot {snapshot})')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_file}")
    
    return fig, axes