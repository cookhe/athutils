"""Unit conversion utilities for Athena AGN simulations.

This module provides unit conversions compatible with the AGN disk model
used in your simulations (Sirko & Goodman model).
"""

import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from astropy import constants as const


# Physical constants (CGS units)
class Constants:
    """Physical constants in CGS units."""
    
    # Fundamental constants
    c = const.c.cgs.value        # speed of light (2.998e10 cm/s)
    G = const.G.cgs.value        # gravitational constant (6.674e-8 cm^3/g/s^2)
    k_B = const.k_B.cgs.value    # Boltzmann constant (1.381e-16 erg/K)
    m_p = const.m_p.cgs.value    # proton mass (1.673e-24 g)
    amu = const.u.cgs.value      # atomic mass unit (1.661e-24 g)
    
    # Astronomical units
    M_sun = const.M_sun.cgs.value    # solar mass (1.989e33 g)
    AU = const.au.cgs.value          # astronomical unit (1.496e13 cm)
    pc = const.pc.cgs.value          # parsec (3.086e18 cm)
    day = 86400                       # day (s)
    yr = 31536000                    # year (s)
    Myr = 1e6 * yr                   # megayear (s)


@dataclass
class AGNUnits:
    """Unit system for AGN disk simulations.
    
    This follows the Sirko & Goodman disk model and matches
    the units calculated by your units.py script.
    
    Attributes
    ----------
    M_bh : float
        Black hole mass in grams
    aspect_ratio : float
        Disk aspect ratio H/r
    r_ref : float
        Reference radius in cm (typically r_sn location)
    rho_0 : float
        Midplane gas density in g/cm^3
    mmw : float
        Mean molecular weight
    gamma : float
        Adiabatic index
    
    Properties (auto-calculated)
    ---------------------------
    All unit_* properties are derived from the fundamental quantities above
    """
    
    M_bh: float
    aspect_ratio: float
    r_ref: float
    rho_0: float
    mmw: float = 1.0079764
    gamma: float = 4./3.
    
    def __post_init__(self):
        """Calculate derived unit quantities."""
        
        # Schwarzschild radius
        self.r_s = 2 * (Constants.G / Constants.c**2) * self.M_bh
        
        # Scale height at reference radius
        self.H = self.aspect_ratio * self.r_ref
        
        # Keplerian angular frequency at reference radius
        self.Omega = np.sqrt(Constants.G * self.M_bh) * self.r_ref**(-1.5)
        
        # Unit definitions (following your units.py)
        self.unit_length = self.H
        self.unit_time = 1.0 / self.Omega
        self.unit_cs = self.Omega * self.H
        self.unit_velocity = self.unit_cs
        self.unit_density = self.rho_0
        self.unit_mass = self.unit_density * self.unit_length**3
        self.unit_energy = self.unit_mass * self.unit_velocity**2
        self.unit_edens = self.unit_energy / self.unit_length**3
        self.unit_pressure = self.unit_edens
        self.unit_temp = self.unit_cs**2 * (self.mmw * Constants.amu) / self.gamma / Constants.k_B
        
        # Additional derived quantities
        self.orbital_period = 2 * np.pi / self.Omega
        self.number_density = self.unit_density / (self.mmw * Constants.amu)
    
    @classmethod
    def from_json(cls, filepath='unit_values.json'):
        """Load units from JSON file created by units.py.
        
        Parameters
        ----------
        filepath : str
            Path to JSON file
            
        Returns
        -------
        AGNUnits
            Unit system object
            
        Examples
        --------
        >>> units = AGNUnits.from_json('unit_values.json')
        >>> print(units.unit_length)
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract parameters from JSON
        M_bh = data['Mbh'] * Constants.M_sun
        aspect_ratio = data['aspectratio']
        rho_0 = data['gas_density']
        r_sn_rs = data['sn_rs']
        
        # Calculate reference radius
        r_s = 2 * (Constants.G / Constants.c**2) * M_bh
        r_ref = r_sn_rs * r_s
        
        # Get optional parameters
        mmw = data.get('mmw', 1.0079764)
        gamma = data.get('gamma', 4./3.)

        

        # print(data)
        # print(pp)
        
        return cls(
            M_bh=M_bh,
            aspect_ratio=aspect_ratio,
            r_ref=r_ref,
            rho_0=rho_0,
            mmw=mmw,
            gamma=gamma
        )
    
    @classmethod
    def from_parameters(cls, M_bh_solar, r_sn_rs, rho_0, aspect_ratio=0.005,
                       mmw=1.0079764, gamma=4./3.):
        """Create unit system from simulation parameters.
        
        Parameters
        ----------
        M_bh_solar : float
            Black hole mass in solar masses
        r_sn_rs : float
            Reference radius in Schwarzschild radii
        rho_0 : float
            Midplane density in g/cm^3
        aspect_ratio : float
            Disk aspect ratio H/r
        mmw : float
            Mean molecular weight
        gamma : float
            Adiabatic index
            
        Returns
        -------
        AGNUnits
        
        Examples
        --------
        >>> units = AGNUnits.from_parameters(
        ...     M_bh_solar=1e8,
        ...     r_sn_rs=1000,
        ...     rho_0=1e-9,
        ...     aspect_ratio=0.005
        ... )
        """
        M_bh = M_bh_solar * Constants.M_sun
        r_s = 2 * (Constants.G / Constants.c**2) * M_bh
        r_ref = r_sn_rs * r_s
        
        return cls(
            M_bh=M_bh,
            aspect_ratio=aspect_ratio,
            r_ref=r_ref,
            rho_0=rho_0,
            mmw=mmw,
            gamma=gamma
        )
    
    # Conversion methods: code -> physical
    def to_physical_length(self, code_value):
        """Convert code length to cm."""
        return code_value * self.unit_length
    
    def to_physical_time(self, code_value):
        """Convert code time to seconds."""
        return code_value * self.unit_time
    
    def to_physical_velocity(self, code_value):
        """Convert code velocity to cm/s."""
        return code_value * self.unit_velocity
    
    def to_physical_density(self, code_value):
        """Convert code density to g/cm^3."""
        return code_value * self.unit_density
    
    def to_physical_mass(self, code_value):
        """Convert code mass to g."""
        return code_value * self.unit_mass
    
    def to_physical_energy(self, code_value):
        """Convert code energy to erg."""
        return code_value * self.unit_energy
    
    def to_physical_edens(self, code_value):
        """Convert code energy density to erg/cm^3."""
        return code_value * self.unit_edens
    
    def to_physical_pressure(self, code_value):
        """Convert code pressure to erg/cm^3."""
        return code_value * self.unit_pressure
    
    # Conversion methods: physical -> code
    def to_code_length(self, physical_value):
        """Convert cm to code length."""
        return physical_value / self.unit_length
    
    def to_code_time(self, physical_value):
        """Convert seconds to code time."""
        return physical_value / self.unit_time
    
    def to_code_velocity(self, physical_value):
        """Convert cm/s to code velocity."""
        return physical_value / self.unit_velocity
    
    def to_code_density(self, physical_value):
        """Convert g/cm^3 to code density."""
        return physical_value / self.unit_density
    
    def to_code_mass(self, physical_value):
        """Convert g to code mass."""
        return physical_value / self.unit_mass
    
    def to_code_energy(self, physical_value):
        """Convert erg to code energy."""
        return physical_value / self.unit_energy
    
    # Helper methods
    def calculate_temperature(self, pressure, density):
        """Calculate temperature from pressure and density.
        
        Parameters
        ----------
        pressure : float or array
            Pressure in physical units (erg/cm^3)
        density : float or array
            Density in physical units (g/cm^3)
            
        Returns
        -------
        temperature : float or array
            Temperature in Kelvin
        """
        return pressure * self.mmw * Constants.amu / (density * Constants.k_B)
    
    def calculate_sound_speed(self, temperature):
        """Calculate sound speed from temperature.
        
        Parameters
        ----------
        temperature : float or array
            Temperature in Kelvin
            
        Returns
        -------
        sound_speed : float or array
            Sound speed in cm/s
        """
        return np.sqrt(self.gamma * Constants.k_B * temperature / (self.mmw * Constants.amu))
    
    def calculate_mach_number(self, velocity, temperature):
        """Calculate Mach number.
        
        Parameters
        ----------
        velocity : float or array
            Velocity in cm/s
        temperature : float or array
            Temperature in Kelvin
            
        Returns
        -------
        mach : float or array
            Mach number
        """
        cs = self.calculate_sound_speed(temperature)
        return velocity / cs
    
    def summary(self):
        """Print summary of unit system."""
        print("="*60)
        print("AGN Disk Unit System Summary")
        print("="*60)
        print("\nBlack Hole Properties:")
        print(f"  M_bh        = {self.M_bh/Constants.M_sun:.2e} M_sun")
        print(f"  r_s         = {self.r_s/Constants.AU:.4f} AU")
        
        print("\nDisk Properties:")
        print(f"  r_ref       = {self.r_ref/Constants.AU:.4f} AU")
        print(f"  r_ref       = {self.r_ref/self.r_s:.1f} r_s")
        print(f"  H/r         = {self.aspect_ratio:.6f}")
        print(f"  H           = {self.H/Constants.AU:.4f} AU")
        print(f"  H           = {self.H/Constants.pc:.4e} pc")
        print(f"  Omega       = {self.Omega:.4e} s^-1")
        print(f"  P_orb       = {self.orbital_period/Constants.yr:.4e} yr")
        
        print("\nCode Units:")
        print(f"  Length      = {self.unit_length:.4e} cm = {self.unit_length/Constants.AU:.4f} AU")
        print(f"  Time        = {self.unit_time:.4e} s = {self.unit_time/Constants.yr:.4e} yr")
        print(f"  Velocity    = {self.unit_velocity:.4e} cm/s = {self.unit_velocity/Constants.c:.4e} c")
        print(f"  Density     = {self.unit_density:.4e} g/cm^3")
        print(f"  Mass        = {self.unit_mass:.4e} g")
        print(f"  Energy      = {self.unit_energy:.4e} erg")
        print(f"  Temperature = {self.unit_temp:.4e} K = {self.unit_temp/1e6:.4f} MK")
        
        print("\nGas Properties:")
        print(f"  Sound speed = {self.unit_cs/1e5:.2f} km/s")
        print(f"  cs/c        = {self.unit_cs/Constants.c:.4e}")
        print(f"  n_H         = {self.number_density:.4e} cm^-3")
        print("="*60)


def read_units(filepath='unit_values.json'):
    """Load units from JSON file (backward compatible with existing function).
    
    Parameters
    ----------
    filepath : str
        Path to JSON file created by units.py
        
    Returns
    -------
    AGNUnits
        Unit system object
    """
    return AGNUnits.from_json(filepath)


# Example usage
if __name__ == '__main__':
    print("Example 1: Create units from parameters")
    print("-" * 60)
    
    units = AGNUnits.from_parameters(
        M_bh_solar=1e8,
        r_sn_rs=1000,
        rho_0=1e-9,
        aspect_ratio=0.005
    )
    
    units.summary()
    
    print("\n\nExample 2: Load from JSON file")
    print("-" * 60)
    
    # This would load from your existing unit_values.json
    # units = AGNUnits.from_json('unit_values.json')
    # units.summary()
    
    print("\n\nExample 3: Convert values")
    print("-" * 60)
    
    # Code to physical
    rho_code = 1.0
    rho_physical = units.to_physical_density(rho_code)
    print(f"Code density {rho_code} = {rho_physical:.3e} g/cm^3")
    
    t_code = 10.0
    t_physical = units.to_physical_time(t_code)
    print(f"Code time {t_code} = {t_physical/Constants.yr:.3f} years")
    
    # Physical to code
    E_sn = 1e51  # erg
    E_sn_code = units.to_code_energy(E_sn)
    print(f"SN energy {E_sn:.2e} erg = {E_sn_code:.3f} (code units)")
