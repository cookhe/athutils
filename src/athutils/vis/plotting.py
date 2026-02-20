"""Core plotting functions for Athena slice data."""

import numpy as np
import matplotlib.pyplot as plt
import h5py


def plot_vertical_density_timeseries(hdf5_files, snapshots, output_file=None, 
                                    vmin=None, vmax=None, cmap='viridis',
                                    figsize=None, run_labels=None, units=None,):
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
    units : list, optional
        List of unit_values.json files for each run containing
        physical unit conversions.
    slc : str, optional
        If 'xz', applies special aspect ratio for x-z slices

        
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    from matplotlib.colors import LogNorm
    from athutils.units.agn_units import AGNUnits, Constants
    
    n_runs = len(hdf5_files)
    n_snapshots = len(snapshots[0])
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        width = 3 * n_snapshots
        height = 3 * n_runs
        figsize = (width, height)
    
    # Create figure and axes
    fig_ratio = 1.34
    fig, axes = plt.subplots(n_runs, n_snapshots, figsize=figsize,
                                gridspec_kw={
                                    'height_ratios': [fig_ratio, 1, fig_ratio],
                                #  'width_ratios': [1, 1, 1, 1]
                                },
                            squeeze=False,
                            sharey='row'
                            )
    
    # Determine global vmin/vmax if not provided
    if vmin is None or vmax is None:
        all_data = []
        for h, hdf5_file in enumerate(hdf5_files):
            if units:
                u = AGNUnits.from_json(units[h])
            with h5py.File(hdf5_file, 'r') as f:
                for snap in snapshots[h]:
                    grp_name = f"snapshot_{snap:04d}"
                    if grp_name in f:
                        rho = f[grp_name]['data']['rho'][:]
                        if units:
                            all_data.append(u.to_physical_density(rho))  
                        else:
                            all_data.append(rho)
                    # print(f"min/max units:\t", np.min(all_data[-1]), '  \t', np.max(all_data[-1]))
        
        if all_data:
            all_data = np.concatenate([d.flatten() for d in all_data])
            if vmin is None:
                vmin = np.percentile(all_data, 1)
                # print(np.min(all_data))
                # vmin = np.min(all_data)
            if vmax is None:
                vmax = np.percentile(all_data, 99)
                # print(np.max(all_data))
                # vmax = np.max(all_data)
        # print("linear:\t", vmin, vmax)
        # print("log:\t", np.log10(vmin), np.log10(vmax))
        # print(f"vmin/vmax:\t{vmin:.2e}\t{vmax:.2e}")
    
    # Plot each run and snapshot
    for i, hdf5_file in enumerate(hdf5_files):
        if units:
            u = AGNUnits.from_json(units[i])
        with h5py.File(hdf5_file, 'r') as f:
            for j, snap in enumerate(snapshots[i]):
                ax = axes[i, j]
                grp_name = f"snapshot_{snap:04d}"
                
                if grp_name not in f:
                    ax.text(0.5, 0.5, f'Snapshot {snap}\nnot found',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                grp = f[grp_name]
                rho = grp['data']['rho'][:]
                t = grp.attrs['t']
                x = grp['coords']['x1'][:]
                z = grp['coords']['x3'][:]

                # if units:
                rho = u.to_physical_density(rho)
                t = u.to_physical_time(t)
                x = u.to_physical_length(x)
                z = u.to_physical_length(z)
                H = u.H
                xH = x/u.H
                zH = z/u.H

                extent=[xH[0], xH[-1], zH[0], zH[-1]]

                # Plot
                # im = ax.imshow(rho, origin='lower', cmap=cmap,
                im = ax.imshow(np.log10(rho), origin='lower', cmap=cmap,
                              extent=extent,
                            #   norm=LogNorm(vmin=vmin, vmax=vmax),
                              vmin=np.log10(vmin), vmax=np.log10(vmax),
                              aspect='equal')
                
                if i == 1:
                    # Add time stamps in years to top left of R=1e5 Rs
                    ax.text(0.05, 0.05, f'{np.round(int(t/Constants.yr), decimals=-1)} years', fontsize=10,
                           c='white', ha='left', va='center', transform=ax.transAxes)
                else:
                    # Add time stamps in days for R=1e3 Rs runs
                    ax.text(0.05, 0.05, f'{t/Constants.day:.0f} days', fontsize=10,
                           c='white', ha='left', va='center', transform=ax.transAxes)
                if j == 0:
                    # Add radial scale in Rs and H
                    rref = u.to_physical_length(u.r_ref)
                    rref_rs = rref/u.r_s

                    # set the number of decimals to round (negative means left of decimal point)
                    decimals = -1
                    rs_string = r'$10^3$ $R_{\rm s}$'
                    if i == 1:
                        decimals = -3
                        rs_string = r'$10^5$ $R_{\rm s}$'
                    
                    H_string = r'$1\,H = $' + f'{np.round(int(H/Constants.AU), decimals=decimals)} AU\n'
                    length_string = H_string + rs_string
                    ax.text(0.5, 0.95, H_string, fontsize=10,
                           c='white', ha='center', va='top', transform=ax.transAxes)

                    ax.text(0.5, 0.1, rs_string, fontsize=10,
                           c='white', ha='center', va='bottom', transform=ax.transAxes)
                    ax.vlines(0.5, 0.0, 0.1, transform=ax.transAxes, color='white', linewidth=1)

                    # ax.text(0.95)
                if j == 3:
                    orbit_time_frac = t/u.unit_time
                    ax.text(0.95, 0.05, f'{orbit_time_frac:.2f} orbits', fontsize=10,
                           c='white', ha='right', va='center', transform=ax.transAxes)
                # add radial scale text
                # add scale height text
                # load xyz arrays


                # Labels
                # if i == 0:  # Top row
                    # ax.set_title(f't = {t/Constants.day:.2f}', fontsize=10)
                if i == 2:
                    ax.set_xlabel('x/H' if units else 'x', fontsize=10)
                
                if j == 0:  # Left column
                    if run_labels and i < len(run_labels):
                        ax.set_ylabel(run_labels[i], fontsize=10)
                    else:
                        ax.set_ylabel('y/H' if units else 'y', fontsize=10)
            
                # if j > 0:
                    # ax.set_yticks([])
    
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.83, 0.065, 0.02, 0.9175])
    fig.colorbar(im, cax=cbar_ax, label=r'$\log10\ \rho$ [g cm$^{-3}$]')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1], w_pad=-8, h_pad=0.1)
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_file}")
    
    return fig, axes

def plot_midplane_density_timeseries(hdf5_files, snapshots, output_file=None, 
                                    vmin=None, vmax=None, cmap='viridis',
                                    figsize=None, run_labels=None, units=None,
                                    slc=None, zoom=False, inset=False,
                                    background_zoom=False,
                                    background_color='black',text_color='white',):
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
    units : list, optional
        List of unit_values.json files for each run containing
        physical unit conversions.
    slc : str, optional
        If 'xz', applies special aspect ratio for x-z slices

        
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    from matplotlib.colors import LogNorm
    from athutils.units.agn_units import AGNUnits, Constants
    
    if zoom and inset:
        raise ValueError("Cannot use both zoom and inset options together.")

    n_runs = len(hdf5_files)
    n_snapshots = len(snapshots[0])
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        width = 3 * n_snapshots
        height = 3 * n_runs
        figsize = (width, height)
    
    # Create figure and axes
    fig, axes = plt.subplots(n_runs, n_snapshots, figsize=figsize,
                            squeeze=False,
                            sharey='row',
    )
                            # constrained_layout=True)

    # Determine global vmin/vmax if not provided
    if vmin is None or vmax is None:
        all_data = []
        for h, hdf5_file in enumerate(hdf5_files):
            if units:
                u = AGNUnits.from_json(units[h])
            rho0 = None
            with h5py.File(hdf5_file, 'r') as f:
                for s, snap in enumerate(snapshots[h]):
                    grp_name = f"snapshot_{snap:04d}"
                    if grp_name in f:
                        rho = f[grp_name]['data']['rho'][:]
                        if rho0 is None:
                            rho0 = rho
                        all_data.append(rho)

        if all_data:
            all_data = np.concatenate([d.flatten() for d in all_data])
            if vmin is None:
                # vmin = np.percentile(all_data, 1)
                vmin = np.nanmin(all_data)
            if vmax is None:
                # vmax = np.percentile(all_data, 99)
                vmax = np.nanmax(all_data)
    
    # Plot each run and snapshot
    for i, hdf5_file in enumerate(hdf5_files):
        if units:
            u = AGNUnits.from_json(units[i])
        with h5py.File(hdf5_file, 'r') as f:
            for j, snap in enumerate(snapshots[i]):
                ax = axes[i, j]
                grp_name = f"snapshot_{snap:04d}"
                
                if grp_name not in f:
                    ax.text(0.5, 0.5, f'Snapshot {snap}\nnot found',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                grp = f[grp_name]
                rho = grp['data']['rho'][:]
                t = grp.attrs['t']
                x = grp['coords']['x1'][:]
                y = grp['coords']['x2'][:]

                rho = u.to_physical_density(rho)
                t = u.to_physical_time(t)
                x = u.to_physical_length(x)
                y = u.to_physical_length(y)
                H = u.H
                xH = x/u.H
                yH = y/u.H

                if j == 0:
                    rho0 = np.copy(rho)
                

                if inset and background_zoom:
                    nx_shift = 280
                    ny_shift = 280
                    rho_shift = rho[ny_shift:, nx_shift:]
                    x_shift = x[nx_shift:]
                    y_shift = y[ny_shift:]
                    
                    xH_shift = x_shift/u.H
                    yH_shift = y_shift/u.H
                    
                    if j == 0:
                        rho0_shift = rho0[ny_shift:, nx_shift:]


                    extent_shift = [xH_shift[0], xH_shift[-1], yH_shift[0], yH_shift[-1]]

                    # Plot
                    im = ax.imshow(np.log10(rho_shift/rho0_shift), origin='lower', cmap=cmap,
                                extent=extent_shift,
                                vmin=np.log10(vmin), vmax=np.log10(vmax),
                                aspect='equal')

                if inset and not background_zoom:
                    nx_shift = 280
                    ny_shift = 280
                    
                    # Create padded arrays with the SAME shape as original
                    rho_shift = np.full_like(rho, np.nan)
                    
                    # Place the shifted data in the bottom-left corner
                    ny_data, nx_data = rho[ny_shift:, nx_shift:].shape
                    rho_shift[:ny_data, :nx_data] = rho[ny_shift:, nx_shift:]
                    
                    if j == 0:
                        rho0_shift = np.full_like(rho0, np.nan)
                        rho0_shift[:ny_data, :nx_data] = rho0[ny_shift:, nx_shift:]
                    
                    # Shift the coordinate arrays
                    dx = x[1] - x[0]  # Grid spacing
                    dy = y[1] - y[0]
                    
                    x_shift = x + nx_shift * dx
                    y_shift = y + ny_shift * dy
                    
                    xH_shift = x_shift / u.H
                    yH_shift = y_shift / u.H
                    
                    # Use shifted extent
                    extent_shift = [xH_shift[0], xH_shift[-1], yH_shift[0], yH_shift[-1]]
                    
                    # Plot with shifted extent
                    im = ax.imshow(np.log10(rho_shift/rho0_shift), origin='lower', cmap=cmap,
                                extent=extent_shift,
                                vmin=np.log10(vmin), vmax=np.log10(vmax),
                                aspect='equal')
                    
                    # Set the color for NaN values
                    im.cmap.set_bad(color=background_color)

                else:
                    # Plot with full data (no inset)
                    extent_full = [xH[0], xH[-1], yH[0], yH[-1]]
                    im = ax.imshow(np.log10(rho/rho0), origin='lower', cmap=cmap,
                                extent=extent_full,
                                vmin=np.log10(vmin), vmax=np.log10(vmax),
                                aspect='equal')

                # region to zoom
                if inset:
                    x1z, x2z, y1z, y2z = -5, 5, -5, 5
                    if i == 1:
                        x1z, x2z, y1z, y2z = -2.25, 2.25, -2.25, 2.25 
                    rho_zoom = rho[(yH >= y1z) & (yH <= y2z)][:, (xH >= x1z) & (xH <= x2z)]
                    rho0_zoom = rho0[(yH >= y1z) & (yH <= y2z)][:, (xH >= x1z) & (xH <= x2z)]
                    axins = ax.inset_axes(
                        [0.35, 0.35, 0.62, 0.62],
                        xlim=(x1z, x2z), ylim=(y1z, y2z),
                        # xticklabels=[], yticklabels=[],
                    )
                    axins.imshow(np.log10((rho_zoom/rho0_zoom)), origin='lower', cmap=cmap,
                                extent=[x1z, x2z, y1z, y2z],
                                vmin=np.log10(vmin), vmax=np.log10(vmax),
                                )
                    ax.indicate_inset_zoom(axins, edgecolor='black', alpha=0.25)
                    axins.tick_params(axis='both', which='both', direction='in')
                
                if j == 0:
                    x_pos = 0.95
                    ha_pos = 'right'
                # Add time stamps in years to top left of R=1e5 Rs
                else:
                    x_pos = 0.05
                    ha_pos = 'left'
                
                if i == 1:
                    time_string = f'{np.round(int(t/Constants.yr), decimals=-1)} years'
                    ax.text(x_pos, 0.05, time_string,
                           fontsize=10,
                           c=text_color,
                           ha=ha_pos,
                           va='center',
                           transform=ax.transAxes)
                else:
                    # Add time stamps in days for R=1e3 Rs runs
                    time_string = f'{t/Constants.day:.0f} days'
                    ax.text(x_pos, 0.05, time_string,
                           fontsize=10,
                           c=text_color,
                           ha=ha_pos,
                           va='center',
                           transform=ax.transAxes)
                    
                if j == 0:
                    # set the number of decimals to round (negative means left of decimal point)
                    decimals = -1
                    rs_string = r'$10^3$ $R_{\rm s}$'
                    if i == 1:
                        decimals = -3
                        rs_string = r'$10^5$ $R_{\rm s}$'
                    # Add text and line at 0 showing where this box is located in the disk
                    x_axes_0point = ax.transAxes.inverted().transform(ax.transData.transform((0, 0)))[0]
                    ax.text(x_axes_0point, 0.075, rs_string,
                           fontsize=10,
                           c=text_color,
                           ha='center',
                           va='bottom',
                           transform=ax.transAxes)
                    ax.vlines(x_axes_0point, 0.0, 0.075,
                             transform=ax.transAxes,
                             color=text_color,
                             linewidth=1)
                    
                    # Add text for the conversion of 1H to AU
                    H_string = r'$1\,H = $' + f'{np.round(int(H/Constants.AU), decimals=decimals)} AU\n'
                    length_string = H_string + rs_string
                    # if i == 1:
                    #     Hfontsize=8
                    #     x_pos = 0.975
                    # else:
                    Hfontsize=10
                    x_pos = 0.95
                    ax.text(x_pos, 0.95, H_string, fontsize=Hfontsize, zorder=6,
                           c=text_color, ha='right', va='top', transform=ax.transAxes)


                if j == 3:
                    orbit_time_frac = t/u.unit_time
                    ax.text(0.95, 0.05, f'{orbit_time_frac:.2f} orbits', fontsize=10,
                           c=text_color, ha='right', va='center', transform=ax.transAxes)

                # Labels
                if i == 2:
                    ax.set_xlabel('x/H' if units else 'x', fontsize=10)
                if j == 0:  # Left column
                    if run_labels and i < len(run_labels):
                        ax.set_ylabel(run_labels[i], fontsize=10)
                    else:
                        ax.set_ylabel('y/H' if units else 'y', fontsize=10)
                            
                if zoom:
                    ax.set_xlim(-5,5)
                    ax.set_ylim(-5,5)
                    if i == 1:
                        ax.set_xlim(-3,3)
                        ax.set_ylim(-3,3)
    
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.8925, 0.0745, 0.02, 0.89875])
    fig.colorbar(im, cax=cbar_ax, label=r'$\log10\,\rho/\rho_0$')

    # Right before tight_layout, add this:
    print("Checking axes before tight_layout:")
    for i in range(n_runs):
        for j in range(n_snapshots):
            ax = axes[i, j]
            try:
                bbox = ax.get_tightbbox(fig.canvas.get_renderer())
                # print(f"Axes[{i},{j}]: bbox OK")
            except Exception as e:
                print(f"Axes[{i},{j}]: ERROR - {e}")
                # Check what's in the axis
                print(f"  Has images: {len(ax.images)}")
                print(f"  Has lines: {len(ax.lines)}")
                print(f"  Has texts: {len(ax.texts)}")
    
    plt.tight_layout(rect=[0, 0, 0.9, 1], w_pad=0.1, h_pad=-0.1)
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_file}")
    
    return fig, axes


def plot_single_slice(hdf5_file, snapshot, field='rho', 
                     output_file=None, cmap='viridis',
                     slc='xy',
                     log=False,
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
        if log:
            data = np.log10(grp['data'][field][:])
        else:
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