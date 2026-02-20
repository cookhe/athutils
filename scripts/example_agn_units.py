#!/usr/bin/env python3
"""Example: Using AGN units with athutils for analysis."""

import numpy as np
from athutils.io.ReadAthena import Athena, AthenaSlice
from athutils.units.agn_units import AGNUnits, Constants


def example_1_load_and_convert():
    """Load data and convert to physical units."""
    
    print("="*70)
    print("Example 1: Load Data and Convert to Physical Units")
    print("="*70)
    
    # Load units from your JSON file
    units = AGNUnits.from_json('unit_values.json')
    units.summary()
    
    # Or create from parameters matching your NOTE.txt command:
    # python3 units.py -m 1e8 -d 1e-9 --sn-energy 1e51 --sn-ejecta 10 
    #                  --sn-rs 1000 --Nz 1280 --Lz 128 --aspectratio 0.005
    
    units_manual = AGNUnits.from_parameters(
        M_bh_solar=1e8,
        r_sn_rs=1000,
        rho_0=1e-9,
        aspect_ratio=0.005,
        gamma=4./3.
    )
    
    # Load a slice
    slice_data = AthenaSlice.load_hdf5('slices.h5', snapshot=20)
    
    # Convert coordinates to physical units
    x1_cm = units.to_physical_length(slice_data.coords['x1'])
    x2_cm = units.to_physical_length(slice_data.coords['x2'])
    
    x1_AU = x1_cm / Constants.AU
    x2_AU = x2_cm / Constants.AU
    
    # Convert time
    t_code = slice_data.t
    t_years = units.to_physical_time(t_code) / Constants.yr
    t_orbits = t_code  # In code units, time is in units of Omega^-1
    
    print(f"\nSnapshot {slice_data.snapshot}:")
    print(f"  Code time:     {t_code:.2f}")
    print(f"  Physical time: {t_years:.2f} years")
    print(f"  Orbital periods: {t_code * units.Omega / (2*np.pi):.2f}")
    
    print(f"\nGrid extent:")
    print(f"  x1: {x1_AU.min():.2f} to {x1_AU.max():.2f} AU")
    print(f"  x2: {x2_AU.min():.2f} to {x2_AU.max():.2f} AU")
    
    # Convert density
    rho_code = slice_data.data['rho']
    rho_physical = units.to_physical_density(rho_code)
    
    print(f"\nDensity:")
    print(f"  Min: {rho_physical.min():.3e} g/cm^3")
    print(f"  Max: {rho_physical.max():.3e} g/cm^3")
    print(f"  Mean: {rho_physical.mean():.3e} g/cm^3")


def example_2_calculate_derived_quantities():
    """Calculate temperature, Mach number, etc."""
    
    print("\n" + "="*70)
    print("Example 2: Calculate Derived Physical Quantities")
    print("="*70)
    
    units = AGNUnits.from_json('unit_values.json')
    slice_data = AthenaSlice.load_hdf5('slices.h5', snapshot=20)
    
    # Get data in code units
    rho = slice_data.data['rho']
    rux1 = slice_data.data['rux1']
    rux2 = slice_data.data['rux2']
    rux3 = slice_data.data['rux3']
    eng = slice_data.data['eng']
    
    # Calculate velocities (code units)
    vx = rux1 / rho
    vy = rux2 / rho
    vz = rux3 / rho
    v_total = np.sqrt(vx**2 + vy**2 + vz**2)
    
    # Convert to physical units
    rho_phys = units.to_physical_density(rho)
    v_phys = units.to_physical_velocity(v_total)
    eng_phys = units.to_physical_edens(eng)
    
    # Calculate kinetic energy density
    ke_phys = 0.5 * rho_phys * v_phys**2
    
    # Thermal energy density
    thermal_energy = eng_phys - ke_phys
    
    # Pressure (assuming adiabatic)
    pressure = (units.gamma - 1) * thermal_energy
    
    # Temperature
    temperature = units.calculate_temperature(pressure, rho_phys)
    
    print(f"\nVelocity:")
    print(f"  Max: {v_phys.max()/1e5:.1f} km/s")
    print(f"  Max (c): {v_phys.max()/Constants.c:.4f}")
    
    print(f"\nTemperature:")
    print(f"  Min: {temperature.min():.2e} K")
    print(f"  Max: {temperature.max():.2e} K")
    print(f"  Mean: {temperature.mean():.2e} K")
    print(f"  Mean (MK): {temperature.mean()/1e6:.2f} MK")
    
    # Mach number
    mach = units.calculate_mach_number(v_phys, temperature)
    
    print(f"\nMach number:")
    print(f"  Min: {mach.min():.2f}")
    print(f"  Max: {mach.max():.2f}")
    print(f"  Mean: {mach.mean():.2f}")
    
    # Fraction of supersonic gas
    supersonic_frac = (mach > 1).sum() / mach.size
    print(f"  Supersonic fraction: {supersonic_frac*100:.1f}%")


def example_3_supernova_properties():
    """Analyze supernova properties in physical units."""
    
    print("\n" + "="*70)
    print("Example 3: Supernova Properties")
    print("="*70)
    
    units = AGNUnits.from_json('unit_values.json')
    
    # Your supernova parameters (from NOTE.txt)
    E_sn_erg = 1e51  # erg
    M_sn_solar = 10  # solar masses
    
    # Convert to code units
    E_sn_code = units.to_code_energy(E_sn_erg)
    M_sn_code = units.to_code_mass(M_sn_solar * Constants.M_sun)
    
    print(f"Supernova in code units:")
    print(f"  Energy: {E_sn_code:.3f}")
    print(f"  Mass:   {M_sn_code:.3f}")
    
    # These should match your athinput file values!
    print(f"\nThese values should match your athinput file:")
    print(f"  snEng  = {E_sn_code:.6f}")
    print(f"  snMass = {M_sn_code:.6f}")
    
    # Gaussian sigma (from your units.py: sigma = 0.5*H)
    sigma_cm = 0.5 * units.H
    sigma_code = units.to_code_length(sigma_cm)
    
    # For Nz=1280, Lz=128 (from your NOTE.txt)
    Nz = 1280
    Lz = 128  # in code units (multiples of H)
    dz_code = Lz / Nz
    
    print(f"\nGrid resolution:")
    print(f"  dz (code): {dz_code:.6f}")
    print(f"  dz (cm):   {units.to_physical_length(dz_code):.3e}")
    print(f"  dz (AU):   {units.to_physical_length(dz_code)/Constants.AU:.4f}")
    
    print(f"\nSN Gaussian width:")
    print(f"  sigma (code): {sigma_code:.3f}")
    print(f"  sigma (H):    {sigma_cm/units.H:.3f}")
    print(f"  sigma / dz:   {sigma_code/dz_code:.1f} cells")


def example_4_time_evolution():
    """Track time evolution in physical units."""
    
    print("\n" + "="*70)
    print("Example 4: Time Evolution")
    print("="*70)
    
    import h5py
    
    units = AGNUnits.from_json('unit_values.json')
    
    # Collect evolution data
    times_code = []
    times_years = []
    times_orbits = []
    max_velocities = []
    
    with h5py.File('slices.h5', 'r') as f:
        snapshot_names = sorted([k for k in f.keys() if k.startswith('snapshot_')])
        
        for snap_name in snapshot_names[:10]:  # First 10 snapshots
            grp = f[snap_name]
            
            # Time
            t_code = grp.attrs['t']
            times_code.append(t_code)
            times_years.append(units.to_physical_time(t_code) / Constants.yr)
            times_orbits.append(t_code * units.Omega / (2*np.pi))
            
            # Velocity
            rho = grp['data']['rho'][:]
            rux1 = grp['data']['rux1'][:]
            v_code = np.abs(rux1 / rho).max()
            v_phys = units.to_physical_velocity(v_code)
            max_velocities.append(v_phys / 1e5)  # km/s
    
    print(f"\nTime evolution (first 10 snapshots):")
    print(f"{'Snap':<8} {'t_code':<10} {'Years':<12} {'Orbits':<10} {'v_max (km/s)':<15}")
    print("-" * 60)
    
    for i, (tc, ty, to, vm) in enumerate(zip(times_code, times_years, 
                                              times_orbits, max_velocities)):
        print(f"{i:<8} {tc:<10.2f} {ty:<12.2f} {to:<10.2f} {vm:<15.1f}")


def example_5_save_physical_units():
    """Save converted data for analysis."""
    
    print("\n" + "="*70)
    print("Example 5: Save Data in Physical Units")
    print("="*70)
    
    import h5py
    
    units = AGNUnits.from_json('unit_values.json')
    slice_data = AthenaSlice.load_hdf5('slices.h5', snapshot=20)
    
    # Convert everything to physical units
    x1_AU = units.to_physical_length(slice_data.coords['x1']) / Constants.AU
    x2_AU = units.to_physical_length(slice_data.coords['x2']) / Constants.AU
    
    rho_cgs = units.to_physical_density(slice_data.data['rho'])
    
    vx_kms = units.to_physical_velocity(
        slice_data.data['rux1'] / slice_data.data['rho']
    ) / 1e5
    
    # Save to new HDF5 file
    with h5py.File('snapshot_020_physical.h5', 'w') as f:
        # Coordinates
        f.create_dataset('x1_AU', data=x1_AU)
        f.create_dataset('x2_AU', data=x2_AU)
        
        # Data
        f.create_dataset('density_cgs', data=rho_cgs)
        f.create_dataset('velocity_x_kms', data=vx_kms)
        
        # Metadata
        f.attrs['time_years'] = units.to_physical_time(slice_data.t) / Constants.yr
        f.attrs['snapshot'] = slice_data.snapshot
        f.attrs['M_bh_solar'] = units.M_bh / Constants.M_sun
        f.attrs['aspect_ratio'] = units.aspect_ratio
    
    print("Saved physical units to: snapshot_020_physical.h5")


if __name__ == '__main__':
    # Run examples (uncomment to execute)
    
    example_1_load_and_convert()
    # example_2_calculate_derived_quantities()
    # example_3_supernova_properties()
    # example_4_time_evolution()
    # example_5_save_physical_units()
    
    print("\n" + "="*70)
    print("Done! Uncomment other examples to run them.")
    print("="*70)
